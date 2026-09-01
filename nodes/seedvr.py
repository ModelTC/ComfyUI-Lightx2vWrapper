"""SeedVR2 super-resolution nodes for ComfyUI.

Split into:
- LightX2VSeedVR2Loader: pick a SeedVR2 DiT checkpoint under
  models/lightx2v/seedvr2/, load it into VRAM, return a SEEDVR_MODEL handle.
- LightX2VSeedVR2Sampler: takes SEEDVR_MODEL + IMAGE + per-call params,
  returns upscaled IMAGE frames.
- LightX2VSeedVR2FileSampler: takes a validated input video path and streams
  the restored result to ComfyUI output while preserving source audio.

The sampler installs a small shim on the runner so input frames come from the
IMAGE tensor (no temp file, no re-encode); the runner's segmenting logic still
runs and slices our in-memory tensor.
"""

import argparse
import gc
import logging
import math
import shutil
import subprocess
import tempfile
import threading
import types
import wave
from collections.abc import Mapping
from pathlib import Path

import folder_paths
import torch
from comfy.utils import ProgressBar

from .file_input import probe_video_file, resolve_input_video_path

logger = logging.getLogger(__name__)

_SEEDVR_RUN_LOCK = threading.Lock()


def _seedvr2_model_dir() -> Path:
    return Path(folder_paths.models_dir) / "lightx2v" / "seedvr2"


def _scan_seedvr2_ckpts():
    d = _seedvr2_model_dir()
    if not d.exists():
        return ["None"]
    items = sorted(f.name for f in d.iterdir() if f.is_file() and f.suffix == ".safetensors")
    return items or ["None"]


def _prepare_output_video(filename_prefix, width, height):
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
        int(width),
        int(height),
    )
    file = f"{filename}_{counter:05}_.mp4"
    full_path = Path(full_output_folder) / file
    return full_path, file, subfolder


def _split_output_filename(filename):
    raw = str(filename or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("filename is required")

    output_dir = Path(folder_paths.get_output_directory()).resolve()
    raw_path = Path(raw)
    if raw_path.is_absolute():
        full_path = raw_path.resolve()
        try:
            relative_path = full_path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError(f"Expected a file under ComfyUI output, got: {filename}") from exc
    else:
        parts = raw_path.parts
        if parts and parts[0] == "output":
            parts = parts[1:]
        relative_path = Path(*parts) if parts else Path()
        if ".." in relative_path.parts:
            raise ValueError(f"Output filename cannot contain '..': {filename}")

    if not relative_path.name:
        raise ValueError(f"Expected an output video filename, got: {filename}")

    full_path = (output_dir / relative_path).resolve()
    try:
        full_path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"Expected a file under ComfyUI output, got: {filename}") from exc

    subfolder = relative_path.parent.as_posix()
    if subfolder == ".":
        subfolder = ""
    return relative_path.name, subfolder, relative_path.as_posix(), full_path


def _output_video_file_info(filename, validate_exists=True):
    file, subfolder, relative_name, full_path = _split_output_filename(filename)
    if validate_exists and not full_path.is_file():
        raise FileNotFoundError(f"Output video does not exist: {full_path}")
    return {"filename": file, "subfolder": subfolder, "type": "output"}, relative_name


def _output_video_full_path(filename, validate_exists=True):
    _, _, _, full_path = _split_output_filename(filename)
    if validate_exists and not full_path.is_file():
        raise FileNotFoundError(f"Output video does not exist: {full_path}")
    return full_path


def _audio_to_wav(audio, wav_path):
    if not isinstance(audio, Mapping):
        logger.info("[LightX2VOutputVideoPreview] skip audio mux: unsupported AUDIO input type=%s", type(audio).__name__)
        return False
    if audio.get("waveform") is None or audio.get("sample_rate") is None:
        logger.info("[LightX2VOutputVideoPreview] skip audio mux: AUDIO has no waveform/sample_rate")
        return False

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        logger.info("[LightX2VOutputVideoPreview] skip audio mux: invalid sample_rate=%s", sample_rate)
        return False
    if not torch.is_tensor(waveform) or waveform.numel() == 0:
        logger.info("[LightX2VOutputVideoPreview] skip audio mux: empty waveform")
        return False
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        logger.info("[LightX2VOutputVideoPreview] skip audio mux: unsupported waveform shape=%s", tuple(audio["waveform"].shape))
        return False

    waveform_i16 = (waveform.detach().cpu().float().clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
    interleaved = waveform_i16.transpose(0, 1).contiguous().numpy()
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(int(waveform_i16.shape[0]))
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved.tobytes())
    return True


def _audio_mux_path(video_path):
    video_path = Path(video_path)
    return video_path.with_name(f"{video_path.stem}-audio{video_path.suffix}")


def _mux_audio_into_video(video_path, audio):
    from imageio_ffmpeg import get_ffmpeg_exe

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Output video does not exist: {video_path}")

    with tempfile.TemporaryDirectory(prefix=".lightx2v_audio_mux.", dir=str(video_path.parent)) as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        muxed_tmp_path = Path(tmp_dir) / "muxed.mp4"
        muxed_path = _audio_mux_path(video_path)
        if not _audio_to_wav(audio, wav_path):
            return None

        command = [
            get_ffmpeg_exe(),
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(wav_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(muxed_tmp_path),
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg audio mux failed: {stderr}")
        shutil.copy2(muxed_tmp_path, muxed_path)
        muxed_tmp_path.replace(video_path)
        logger.info("[LightX2VOutputVideoPreview] muxed audio into %s and updated %s", muxed_path, video_path)
        return muxed_path


def _install_tensor_input_shim(runner, frames_u8, fps):
    """Patch runner methods so input frames come from `frames_u8` instead of disk.

    frames_u8: torch.uint8 [T, C, H, W] on CPU (same format as torchvision.io.read_video output).
    """
    if not hasattr(runner, "_lightx2v_original_run_input_encoder_local_sr"):
        runner._lightx2v_original_run_input_encoder_local_sr = runner._run_input_encoder_local_sr.__func__
    if not hasattr(runner, "_lightx2v_original_run_input_encoder"):
        runner._lightx2v_original_run_input_encoder = runner.run_input_encoder

    runner._tensor_input = frames_u8
    runner._tensor_input_fps = float(fps)

    def _probe_video(self, video_path):  # noqa: ARG001
        total = self._tensor_input.shape[0]
        self._set_output_fps(self._tensor_input_fps)
        return total, self._tensor_input_fps, None

    def _read_video_segment(self, video_path, start_idx, end_idx):  # noqa: ARG001
        seg = self._tensor_input[start_idx:end_idx]
        if seg.shape[0] == 0:
            return torch.empty(0, 3, 0, 0, dtype=torch.uint8)
        return seg

    original_encoder = runner._lightx2v_original_run_input_encoder_local_sr

    def _run_input_encoder_local_sr(self):
        if getattr(self, "_sr_segment", None) is None:
            self._sr_segment = (0, self._tensor_input.shape[0])
            try:
                return original_encoder(self)
            finally:
                self._sr_segment = None
        return original_encoder(self)

    runner._probe_video = types.MethodType(_probe_video, runner)
    runner._read_video_segment = types.MethodType(_read_video_segment, runner)
    runner._run_input_encoder_local_sr = types.MethodType(_run_input_encoder_local_sr, runner)
    runner.run_input_encoder = runner._run_input_encoder_local_sr


def _clear_tensor_input_shim(runner):
    for attr in ("_tensor_input", "_tensor_input_fps"):
        if hasattr(runner, attr):
            delattr(runner, attr)
    for attr in ("_probe_video", "_read_video_segment", "_run_input_encoder_local_sr"):
        if attr in runner.__dict__:
            delattr(runner, attr)
    original_run_input_encoder = getattr(runner, "_lightx2v_original_run_input_encoder", None)
    if original_run_input_encoder is not None:
        runner.run_input_encoder = original_run_input_encoder


class LightX2VSeedVR2Loader:
    """Load a SeedVR2 DiT checkpoint from models/lightx2v/seedvr2/."""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = _scan_seedvr2_ckpts()
        return {
            "required": {
                "ckpt_name": (
                    ckpts,
                    {"default": ckpts[0], "tooltip": "DiT .safetensors under models/lightx2v/seedvr2/"},
                ),
                "precision": (
                    ["auto", "bf16", "fp8-sgl", "fp8-q8f", "fp8-vllm"],
                    {
                        "default": "auto",
                        "tooltip": "auto = bf16 for fp16/bf16 weights, fp8-sgl for fp8 weights. fp8-sgl needs sgl-kernel (H100/SM90); fp8-q8f is the 4090 path.",
                    },
                ),
                "cpu_offload": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Offload DiT blocks to CPU between forwards (slower; only needed on small VRAM)"},
                ),
                "use_tiling_vae": ("BOOLEAN", {"default": False, "tooltip": "Tile VAE to reduce peak memory; usually keep off on 32GB+ GPUs."}),
                "vae_tile_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Output-space VAE tile size when tiling is enabled. Larger is faster but uses more VRAM.",
                    },
                ),
                "vae_tile_overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "step": 8,
                        "tooltip": "Output-space VAE tile overlap when tiling is enabled. Smaller is faster but may increase tile seams.",
                    },
                ),
                "vae_causal_slice_size": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Temporal VAE slice size. 0 disables causal slicing. Larger is faster but uses more VRAM.",
                    },
                ),
                "vae_memory_limit_gb": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.25,
                        "tooltip": "Per-op VAE conv/norm memory limit in GiB. 0 disables this extra splitting.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SEEDVR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LightX2V/SeedVR"

    def load(
        self,
        ckpt_name,
        precision,
        cpu_offload,
        use_tiling_vae,
        vae_tile_size,
        vae_tile_overlap,
        vae_causal_slice_size,
        vae_memory_limit_gb,
    ):
        from ..lightx2v.lightx2v.infer import init_runner
        from ..lightx2v.lightx2v.utils.set_config import set_config

        model_dir = _seedvr2_model_dir()
        if ckpt_name == "None":
            raise FileNotFoundError(f"No .safetensors checkpoints found in {model_dir}")

        for required in ("ema_vae.pth", "pos_emb.pt", "neg_emb.pt"):
            p = model_dir / required
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing {p}. SeedVR2 needs VAE + pre-computed text embeddings (pos_emb.pt / neg_emb.pt) in the same directory as the DiT checkpoint."
                )
        ckpt_path = model_dir / ckpt_name
        if not ckpt_path.is_file():
            raise FileNotFoundError(str(ckpt_path))

        if precision == "auto":
            precision = "fp8-sgl" if "fp8" in ckpt_name.lower() else "bf16"

        config = {
            "model_cls": "seedvr2",
            "task": "sr",
            "model_path": str(model_dir),
            "infer_steps": 1,
            "fps": 16,
            "target_video_length": 81,
            "target_height": 1080,
            "target_width": 1920,
            "use_tiling_vae": bool(use_tiling_vae),
            "vae_tile_size": int(vae_tile_size),
            "vae_tile_overlap": int(vae_tile_overlap),
            "vae_causal_slice_size": int(vae_causal_slice_size),
            "vae_memory_limit_gb": float(vae_memory_limit_gb),
            "cpu_offload": bool(cpu_offload),
        }
        if "7b" in ckpt_name.lower():
            config["model_size"] = "7b"

        if precision.startswith("fp8-"):
            config["dit_quantized_ckpt"] = str(ckpt_path)
            config["dit_quant_scheme"] = precision
            config["dit_quantized"] = True
        else:
            config["dit_original_ckpt"] = str(ckpt_path)

        formatted = set_config(argparse.Namespace(**config))
        runner = init_runner(formatted)
        logger.info(
            "[SeedVR2Loader] loaded %s (%s); cpu_offload=%s, tile_vae=%s, tile=%s, overlap=%s, slice=%s, mem_limit=%sGiB",
            ckpt_name,
            precision,
            cpu_offload,
            use_tiling_vae,
            vae_tile_size,
            vae_tile_overlap,
            vae_causal_slice_size,
            vae_memory_limit_gb,
        )
        return ({"runner": runner, "precision": precision, "ckpt": ckpt_name},)


class LightX2VSeedVR2Sampler:
    """Run SeedVR2 SR on an input frame tensor; return upscaled frames as IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEEDVR_MODEL",),
                "images": ("IMAGE",),
                "target_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 8}),
                "target_height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 64,
                        "max": 4320,
                        "step": 8,
                        "tooltip": "Target output frame height. NaDiT preserves input aspect ratio; the geometric mean of target_h * target_w is the effective resolution cap.",
                    },
                ),
                "infer_steps": ("INT", {"default": 1, "min": 1, "max": 50}),
                "segment_length": (
                    "INT",
                    {"default": 81, "min": 16, "max": 512, "step": 1, "tooltip": "Frames per SR pass. Long videos are auto-segmented."},
                ),
                "segment_overlap": ("INT", {"default": 1, "min": 0, "max": 32}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": "FPS of the input frames (passed through to the runner for any internal timing logic)",
                    },
                ),
                "save_to_output_file": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Save the SR result directly under ComfyUI output and return a filename instead of returning the full IMAGE tensor.",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "lightx2v_seedvr2/SeedVR2"}),
                "color_fix": (
                    ["gpu", "off", "cpu"],
                    {
                        "default": "gpu",
                        "tooltip": "SeedVR color correction after VAE decode. gpu is faster on high-VRAM GPUs; off is fastest; cpu matches the original path.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filename")
    FUNCTION = "sample"
    CATEGORY = "LightX2V/SeedVR"

    def sample(
        self,
        model,
        images,
        target_height,
        target_width,
        infer_steps,
        segment_length,
        segment_overlap,
        seed,
        source_fps,
        save_to_output_file,
        filename_prefix,
        color_fix,
    ):
        from ..lightx2v.lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict

        runner = model["runner"]

        if images.dim() != 4 or images.shape[-1] not in (3, 4):
            raise ValueError(f"Expected IMAGE [T, H, W, C], got shape {tuple(images.shape)}")
        # ComfyUI IMAGE: [T, H, W, C] float[0,1]  →  [T, C, H, W] uint8 (read_video's contract)
        ori_h, ori_w = int(images.shape[1]), int(images.shape[2])
        frames = images[..., :3].permute(0, 3, 1, 2).contiguous()
        frames_u8 = (frames.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()

        # Derive sr_ratio from input vs target. The runner uses
        #   resolution = min(sqrt(ori_h*ori_w) * sr_ratio, sqrt(target_h*target_w))
        # so we pick sr_ratio so the min lands on the target term (clamped to >=1
        # to avoid asking the SR model to downscale).
        ori_geom = math.sqrt(ori_h * ori_w)
        target_geom = math.sqrt(target_height * target_width)
        sr_ratio = max(target_geom / ori_geom, 1.0) if ori_geom > 0 else 1.0
        if target_geom < ori_geom:
            logger.warning(f"[SeedVR2] target ({target_height}x{target_width}) smaller than input ({ori_h}x{ori_w}); SR will run at input scale.")

        _install_tensor_input_shim(runner, frames_u8, source_fps)
        save_path = ""
        output_file = ""
        output_subfolder = ""
        if save_to_output_file:
            full_path, output_file, output_subfolder = _prepare_output_video(filename_prefix, target_width, target_height)
            save_path = str(full_path)

        # runner.config is a LockableDict (locked after init); set_config uses temporarily_unlocked.
        runner.set_config(
            {
                "sr_ratio": float(sr_ratio),
                "target_height": int(target_height),
                "target_width": int(target_width),
                "target_video_length": int(segment_length),  # vestigial for SR; keep aligned with segment_length
                "sr_segment_length": int(segment_length),
                "sr_overlap": int(segment_overlap),
                "infer_steps": int(infer_steps),
                "seed": int(seed),
                "fps": float(source_fps),
                "video_path": "<tensor>",  # truthy sentinel so segmenting logic runs; shim bypasses file I/O
                "image_path": "",
                "prompt": "",
                "negative_prompt": "",
                "save_result_path": save_path,
                "return_result_tensor": not bool(save_to_output_file),
                "color_fix": str(color_fix),
            }
        )

        input_info = init_empty_input_info("sr")
        update_input_info_from_dict(
            input_info,
            {
                "video_path": "<tensor>",
                "image_path": "",
                "prompt": "",
                "negative_prompt": "",
                "seed": int(seed),
                "sr_ratio": float(sr_ratio),
                "save_result_path": save_path,
                "return_result_tensor": not bool(save_to_output_file),
            },
        )

        progress = ProgressBar(100)
        if hasattr(runner, "set_progress_callback"):
            runner.set_progress_callback(lambda cur, _tot: progress.update_absolute(cur))

        try:
            result = runner.run_pipeline(input_info)
        finally:
            _clear_tensor_input_shim(runner)
            torch.cuda.empty_cache()
            gc.collect()

        video = result.get("video") if isinstance(result, dict) else result
        if save_to_output_file:
            if not Path(save_path).is_file():
                raise RuntimeError(f"SeedVR2 did not create expected output video: {save_path}")
            placeholder = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            relative_name = str(Path(output_subfolder) / output_file) if output_subfolder else output_file
            return (placeholder, relative_name)

        if video is None or video.numel() == 0:
            raise RuntimeError("SeedVR2 returned empty result")

        # wan_vae_to_comfy already gives [T, H, W, C] float[0,1] on CPU
        video = video.detach().cpu().float().clamp(0.0, 1.0)
        return (video, "")


class LightX2VSeedVR2FileSampler:
    """Run SeedVR2 on an input video using segmented file I/O."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEEDVR_MODEL",),
                "video_path": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Absolute path produced by LightX2V Input Video Path. The file must remain under ComfyUI input.",
                    },
                ),
                "target_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 8}),
                "target_height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 64,
                        "max": 4320,
                        "step": 8,
                        "tooltip": "Target output frame height.",
                    },
                ),
                "infer_steps": ("INT", {"default": 1, "min": 1, "max": 50}),
                "segment_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 16,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Frames decoded and restored per segment. Long videos do not materialize as a full IMAGE batch.",
                    },
                ),
                "segment_overlap": ("INT", {"default": 1, "min": 0, "max": 32}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "filename_prefix": ("STRING", {"default": "lightx2v_seedvr2/SeedVR2"}),
                "color_fix": (
                    ["gpu", "off", "cpu"],
                    {
                        "default": "gpu",
                        "tooltip": "SeedVR color correction after VAE decode.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "sample"
    CATEGORY = "LightX2V/SeedVR"

    def sample(
        self,
        model,
        video_path,
        target_height,
        target_width,
        infer_steps,
        segment_length,
        segment_overlap,
        seed,
        filename_prefix,
        color_fix,
    ):
        from ..lightx2v.lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict

        input_path = resolve_input_video_path(video_path)
        source_width, source_height, source_fps = probe_video_file(input_path)
        effective_fps = source_fps if source_fps > 0 else 16.0
        source_geom = math.sqrt(source_height * source_width)
        target_geom = math.sqrt(int(target_height) * int(target_width))
        sr_ratio = max(target_geom / source_geom, 1.0) if source_geom > 0 else 1.0
        if target_geom < source_geom:
            logger.warning(
                "[SeedVR2FileSampler] target (%sx%s) is smaller than input (%sx%s); SR will run at input scale before final sizing.",
                target_width,
                target_height,
                source_width,
                source_height,
            )

        full_path, output_file, output_subfolder = _prepare_output_video(filename_prefix, target_width, target_height)
        save_path = str(full_path)
        input_info = init_empty_input_info("sr")
        update_input_info_from_dict(
            input_info,
            {
                "video_path": str(input_path),
                "image_path": "",
                "prompt": "",
                "negative_prompt": "",
                "seed": int(seed),
                "sr_ratio": float(sr_ratio),
                "save_result_path": save_path,
                "return_result_tensor": False,
            },
        )

        runner = model["runner"]
        progress = ProgressBar(100)
        logger.info(
            "[SeedVR2FileSampler] input=%s (%sx%s @ %.3f fps), target=%sx%s, segment=%s/%s",
            input_path,
            source_width,
            source_height,
            source_fps,
            target_width,
            target_height,
            segment_length,
            segment_overlap,
        )
        try:
            with _SEEDVR_RUN_LOCK:
                _clear_tensor_input_shim(runner)
                runner.set_config(
                    {
                        "sr_ratio": float(sr_ratio),
                        "target_height": int(target_height),
                        "target_width": int(target_width),
                        "target_video_length": int(segment_length),
                        "sr_segment_length": int(segment_length),
                        "sr_overlap": int(segment_overlap),
                        "stream_save_video": True,
                        "infer_steps": int(infer_steps),
                        "seed": int(seed),
                        "fps": float(effective_fps),
                        "video_path": str(input_path),
                        "image_path": "",
                        "prompt": "",
                        "negative_prompt": "",
                        "save_result_path": save_path,
                        "return_result_tensor": False,
                        "color_fix": str(color_fix),
                    }
                )
                if hasattr(runner, "set_progress_callback"):
                    runner.set_progress_callback(lambda current, _total: progress.update_absolute(current))
                runner.run_pipeline(input_info)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        if not Path(save_path).is_file():
            raise RuntimeError(f"SeedVR2 did not create expected output video: {save_path}")
        relative_name = str(Path(output_subfolder) / output_file) if output_subfolder else output_file
        return (relative_name,)


class LightX2VOutputVideoPreview:
    """Expose an existing ComfyUI output video to the history/view API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Video path under ComfyUI output, e.g. file.mp4, subfolder/file.mp4, or output/subfolder/file.mp4.",
                    },
                ),
                "validate_exists": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Check that the output video exists before creating the preview entry."},
                ),
                "mux_audio": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Mux optional AUDIO input into the output video before previewing."},
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "Optional audio from Load Video/Get Video Components to merge into the output MP4."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "LightX2V/Output"

    def preview(self, filename, validate_exists, mux_audio, audio=None):
        preview_filename = filename
        if bool(mux_audio) and audio is not None:
            video_path = _output_video_full_path(filename, bool(validate_exists))
            muxed_path = _mux_audio_into_video(video_path, audio)
            if muxed_path is not None:
                preview_filename = str(muxed_path)
        file_info, relative_name = _output_video_file_info(preview_filename, bool(validate_exists))
        return {"ui": {"images": [file_info], "animated": (True,)}, "result": (relative_name,)}
