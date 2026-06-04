"""SeedVR2 super-resolution nodes for ComfyUI.

Split into:
- LightX2VSeedVR2Loader: pick a SeedVR2 DiT checkpoint under
  models/lightx2v/seedvr2/, load it into VRAM, return a SEEDVR_MODEL handle.
- LightX2VSeedVR2Sampler: takes SEEDVR_MODEL + IMAGE + per-call params,
  returns upscaled IMAGE frames.

The sampler installs a small shim on the runner so input frames come from the
IMAGE tensor (no temp file, no re-encode); the runner's segmenting logic still
runs and slices our in-memory tensor.
"""

import argparse
import gc
import logging
import math
import types
from pathlib import Path

import folder_paths
import torch
from comfy.utils import ProgressBar

logger = logging.getLogger(__name__)


def _seedvr2_model_dir() -> Path:
    return Path(folder_paths.models_dir) / "lightx2v" / "seedvr2"


def _scan_seedvr2_ckpts():
    d = _seedvr2_model_dir()
    if not d.exists():
        return ["None"]
    items = sorted(f.name for f in d.iterdir() if f.is_file() and f.suffix == ".safetensors")
    return items or ["None"]


def _install_tensor_input_shim(runner, frames_u8, fps):
    """Patch runner methods so input frames come from `frames_u8` instead of disk.

    frames_u8: torch.uint8 [T, C, H, W] on CPU (same format as torchvision.io.read_video output).
    """
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

    original_encoder = runner._run_input_encoder_local_sr.__func__

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
                "use_tiling_vae": ("BOOLEAN", {"default": True, "tooltip": "Tile VAE to reduce peak memory"}),
            }
        }

    RETURN_TYPES = ("SEEDVR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LightX2V/SeedVR"

    def load(self, ckpt_name, precision, cpu_offload, use_tiling_vae):
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
            "cpu_offload": bool(cpu_offload),
        }
        if precision.startswith("fp8-"):
            config["dit_quantized_ckpt"] = str(ckpt_path)
            config["dit_quant_scheme"] = precision
            config["dit_quantized"] = True
        else:
            config["dit_original_ckpt"] = str(ckpt_path)

        formatted = set_config(argparse.Namespace(**config))
        runner = init_runner(formatted)
        logger.info(f"[SeedVR2Loader] loaded {ckpt_name} ({precision}); cpu_offload={cpu_offload}, tile_vae={use_tiling_vae}")
        return ({"runner": runner, "precision": precision, "ckpt": ckpt_name},)


class LightX2VSeedVR2Sampler:
    """Run SeedVR2 SR on an input frame tensor; return upscaled frames as IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEEDVR_MODEL",),
                "images": ("IMAGE",),
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
                "target_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 8}),
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"
    CATEGORY = "LightX2V/SeedVR"

    def sample(self, model, images, target_height, target_width, infer_steps, segment_length, segment_overlap, seed, source_fps):
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
                "save_result_path": "",
                "return_result_tensor": True,
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
                "save_result_path": "",
                "return_result_tensor": True,
            },
        )

        progress = ProgressBar(100)
        if hasattr(runner, "set_progress_callback"):
            runner.set_progress_callback(lambda cur, _tot: progress.update_absolute(cur))

        try:
            result = runner.run_pipeline(input_info)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        video = result.get("video") if isinstance(result, dict) else result
        if video is None or video.numel() == 0:
            raise RuntimeError("SeedVR2 returned empty result")

        # wan_vae_to_comfy already gives [T, H, W, C] float[0,1] on CPU
        video = video.detach().cpu().float().clamp(0.0, 1.0)
        return (video,)
