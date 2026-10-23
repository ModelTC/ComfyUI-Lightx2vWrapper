"""Keep ComfyUI IMAGE inputs in memory while LightX2V reads them.

LightX2V's runner API names the request field ``image_path`` and accepts a
PIL image at the final read boundary.  The wrapper used to serialize the first
ComfyUI frame to a temporary PNG only to have the runner decode it again.  This
shim keeps the path-shaped request contract while replacing that disk
round-trip with a short-lived in-memory PIL image.
"""

from __future__ import annotations

import logging
import threading
import types
import uuid
from typing import Dict

import torch
from PIL import Image

SENTINEL_PREFIX = "<lightx2v-mem-image:"
SENTINEL_SUFFIX = ">"

_REGISTRY: Dict[str, Image.Image] = {}
_LOCK = threading.Lock()


class _ImageSentinel(str):
    """A path-shaped token that releases its registry lease when discarded.

    The registry keeps a plain ``str`` key rather than this object, so the
    object can be collected even while the registry still contains the image.
    This covers failures between Combiner execution and ``generate()`` where
    the normal inference ``finally`` block has not started yet.
    """

    def __del__(self):
        try:
            release(str(self))
        except Exception:
            # Destructors can run during interpreter shutdown.
            pass


def register(images: torch.Tensor) -> str:
    """Register the first ComfyUI IMAGE frame and return a path-shaped token."""
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError(f"IMAGE must be a [B, H, W, C] tensor; got {type(images).__name__} {getattr(images, 'shape', '')}")
    if images.shape[-1] not in (1, 3, 4):
        raise ValueError(f"IMAGE must have 1, 3, or 4 channels; got shape {tuple(images.shape)}")

    frame = images[0].detach().cpu().float().clamp(0.0, 1.0)
    frame_u8 = (frame.mul(255.0).round()).to(torch.uint8).numpy()
    if frame_u8.shape[-1] == 1:
        frame_u8 = frame_u8[..., 0]
    elif frame_u8.shape[-1] == 4:
        frame_u8 = frame_u8[..., :3]

    token = f"{SENTINEL_PREFIX}{uuid.uuid4().hex}{SENTINEL_SUFFIX}"
    image = Image.fromarray(frame_u8)
    if image.mode != "RGB":
        image = image.convert("RGB")
    with _LOCK:
        _REGISTRY[token] = image
    return _ImageSentinel(token)


def release(sentinel: str | None) -> None:
    if not is_sentinel(sentinel):
        return
    with _LOCK:
        _REGISTRY.pop(str(sentinel), None)


def is_sentinel(path) -> bool:
    return isinstance(path, str) and path.startswith(SENTINEL_PREFIX)


def _lookup(sentinel: str) -> Image.Image:
    with _LOCK:
        image = _REGISTRY.get(sentinel)
    if image is None:
        raise FileNotFoundError(f"Stale LightX2V in-memory image sentinel: {sentinel}")
    return image


def install(runner) -> bool:
    """Patch one runner's image read boundary; return whether it was patched."""
    if runner is None or not hasattr(runner, "read_image_input"):
        return False
    if hasattr(runner, "_lightx2v_original_read_image_input"):
        return True

    original = runner.read_image_input

    def _read_image_input(self, image_path):
        if is_sentinel(image_path):
            image_path = _lookup(image_path)
        return original(image_path)

    runner._lightx2v_original_read_image_input = original
    runner.read_image_input = types.MethodType(_read_image_input, runner)
    logging.info("_image_shim installed on %s", type(runner).__name__)
    return True


def install_on_runner_tree(runner) -> int:
    """Patch a regular runner or every clip runner inside an rs2v pipeline."""
    targets = [runner]
    clip_generators = getattr(runner, "clip_generators", None)
    if isinstance(clip_generators, dict):
        targets.extend(clip_generators.values())
    return sum(1 for target in targets if install(target))
