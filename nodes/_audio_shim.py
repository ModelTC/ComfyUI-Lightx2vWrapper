"""In-memory shim for lightx2v.utils.audio_io.load_audio_file.

Routes ComfyUI AUDIO tensors into the runner without round-tripping through
a temp WAV file. Sentinel paths start with SENTINEL_PREFIX; the patched
loader returns a stashed tensor instead of touching disk.

Scope: only the single-AUDIO ComfyUI input path uses this shim. V3 multi-talker
padding still writes real WAV files (its inputs are external file paths / URLs,
not ComfyUI tensors), and those reads fall through to the original loader.

Patching strategy: lightx2v consumers do `from lightx2v.utils.audio_io import
load_audio_file`, which captures the original function at import time. So
patching only `lightx2v.utils.audio_io.load_audio_file` would miss them — we
rebind every known import site too. New consumers must be added to _PATCH_SITES.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Dict, Tuple

import torch

SENTINEL_PREFIX = "<lightx2v-mem-audio:"
SENTINEL_SUFFIX = ">"

_REGISTRY: Dict[str, Tuple[torch.Tensor, int]] = {}
_LOCK = threading.Lock()
_PATCHED = False

# (module dotted-path, attribute name). Each entry rebinds that module's
# `load_audio_file` attribute to the shim. Add new early-binders here as
# upstream changes.
_PATCH_SITES = (
    ("lightx2v.utils.audio_io", "load_audio_file"),
    ("lightx2v.models.runners.wan.wan_audio_runner", "load_audio_file"),
    ("lightx2v.shot_runner.rs2v_infer", "load_audio_file"),
    ("lightx2v.shot_runner.stream_infer", "load_audio_file"),
)


def register(waveform: torch.Tensor, sample_rate: int) -> str:
    """Register a [C, T] float32 waveform; return a sentinel "path" for it."""
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be [C, T]; got shape {tuple(waveform.shape)}")
    token = uuid.uuid4().hex
    sentinel = f"{SENTINEL_PREFIX}{token}{SENTINEL_SUFFIX}"
    stashed = waveform.detach().to(torch.float32).cpu().contiguous()
    with _LOCK:
        _REGISTRY[sentinel] = (stashed, int(sample_rate))
    return sentinel


def release(sentinel: str) -> None:
    with _LOCK:
        _REGISTRY.pop(sentinel, None)


def is_sentinel(path) -> bool:
    return isinstance(path, str) and path.startswith(SENTINEL_PREFIX)


def comfyui_audio_to_loader_pair(audio_dict) -> Tuple[torch.Tensor, int]:
    """ComfyUI AUDIO {"waveform": [B, C, T], "sample_rate": int} -> ([C, T], sr).

    ComfyUI gives waveform as [B, C, T] float32 in [-1, 1] (B usually 1).
    lightx2v's load_audio_file returns [C, T] (channels_first=True). We
    squeeze the batch dim here; mono-down and resample happen downstream in
    AudioProcessor / ShotRS2VPipeline so the shim stays format-agnostic.
    """
    waveform = audio_dict["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected ComfyUI AUDIO waveform shape {tuple(waveform.shape)}")
    return waveform, int(audio_dict["sample_rate"])


def install() -> None:
    """Idempotent monkey-patch. Safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return
    import importlib

    original = None
    targets = []
    for module_path, attr in _PATCH_SITES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            logging.warning(f"_audio_shim: {module_path} not importable; skipping")
            continue
        fn = getattr(module, attr, None)
        if fn is None:
            logging.warning(f"_audio_shim: {module_path}.{attr} missing; skipping")
            continue
        if original is None:
            original = fn
        targets.append((module, attr))

    if original is None:
        raise RuntimeError("_audio_shim.install: no patch sites resolved; lightx2v not installed?")

    def _patched(uri, frame_offset: int = 0, num_frames: int = -1, channels_first: bool = True):
        if is_sentinel(uri):
            with _LOCK:
                entry = _REGISTRY.get(uri)
            if entry is None:
                raise FileNotFoundError(f"Stale lightx2v in-memory audio sentinel: {uri}")
            tensor, sr = entry
            # Slice semantics mirror torchaudio.load(frame_offset, num_frames).
            if frame_offset > 0 or num_frames > 0:
                end = tensor.shape[-1] if num_frames < 0 else frame_offset + num_frames
                tensor = tensor[..., frame_offset:end]
            if not channels_first:
                tensor = tensor.transpose(0, 1).contiguous()
            return tensor, sr
        return original(uri, frame_offset=frame_offset, num_frames=num_frames, channels_first=channels_first)

    for module, attr in targets:
        setattr(module, attr, _patched)

    _PATCHED = True
    logging.info(f"_audio_shim installed at {len(targets)} site(s)")
