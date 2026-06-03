"""SeedVR2 super-resolution node."""

import gc
import hashlib
import logging

import torch
from comfy.utils import ProgressBar

from ..model_utils import get_model_base_path, get_model_full_path, scan_models


class LightX2VSeedVRSR:
    """SeedVR2 video/image super-resolution node for ComfyUI.

    Wraps the SeedVR2-3B model via LightX2V's SeedVRRunner to perform
    single-pass diffusion super-resolution on a video (mp4) or a single image.
    """

    _current_runner = None
    _current_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        available_models = scan_models()
        return {
            "required": {
                "model_name": (
                    available_models,
                    {
                        "default": available_models[0] if available_models else "None",
                        "tooltip": "SeedVR2 model directory under models/lightx2v/",
                    },
                ),
                "input_type": (
                    ["video", "image"],
                    {"default": "video", "tooltip": "Whether to SR a video file or a single image"},
                ),
                "input_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Absolute path to input .mp4 (for video) or .png/.jpg (for image). For video, also accepts a directory of frames.",
                    },
                ),
                "sr_ratio": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.5,
                        "tooltip": "Super-resolution ratio (e.g. 2.0 = 2x, 4.0 = 4x)",
                    },
                ),
                "target_height": (
                    "INT",
                    {
                        "default": 720,
                        "min": 64,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Output frame height (SeedVR NaDiT processes at native resolution)",
                    },
                ),
                "target_width": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Output frame width (must be divisible by 16 for VAE)",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": "Output FPS for video SR (input video FPS is preserved if available)",
                    },
                ),
                "segment_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 16,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Frames per segment for long video SR (1-step diffusion per segment)",
                    },
                ),
                "segment_overlap": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Overlap frames between segments to prevent seams",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": -1,
                        "max": 2**32 - 1,
                        "tooltip": "Random seed, -1 for random",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional text prompt to guide detail synthesis (SeedVR uses pre-computed embeddings; prompt mostly affects style)",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Negative prompt for guidance",
                    },
                ),
                "save_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True, also write the SR result to disk in addition to returning IMAGE tensor",
                    },
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Where to save (only used if save_output=True). Leave empty for auto-generated name next to input.",
                    },
                ),
                "unload_after_inference": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload SeedVR runner from VRAM after inference (frees ~6GB+ for other nodes)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run_seedvr"
    CATEGORY = "LightX2V/SeedVR"

    def _config_hash(
        self,
        model_name,
        input_type,
        input_path,
        sr_ratio,
        target_height,
        target_width,
        fps,
        segment_length,
        segment_overlap,
        seed,
        prompt,
        negative_prompt,
        save_output,
        output_path,
    ):
        """Hash of all parameters that should trigger runner reinit."""
        raw = (
            f"{model_name}|{input_type}|{input_path}|{sr_ratio}|"
            f"{target_height}|{target_width}|{fps}|"
            f"{segment_length}|{segment_overlap}|{seed}|"
            f"{prompt}|{negative_prompt}|{save_output}|{output_path}"
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def run_seedvr(
        self,
        model_name,
        input_type,
        input_path,
        sr_ratio,
        target_height,
        target_width,
        fps,
        segment_length,
        segment_overlap,
        seed,
        prompt,
        negative_prompt,
        save_output,
        output_path,
        unload_after_inference,
    ):
        """Run SeedVR2 super-resolution and return IMAGE tensor."""
        from ..lightx2v.lightx2v.infer import init_runner
        from ..lightx2v.lightx2v.utils.input_info import (
            init_empty_input_info,
            update_input_info_from_dict,
        )
        from ..lightx2v.lightx2v.utils.set_config import set_config

        if not model_name or model_name == "None":
            raise ValueError("model_name is required — select a SeedVR2 model directory under models/lightx2v/")
        if not input_path:
            raise ValueError("input_path is required — provide an absolute path to a video or image file")

        model_full_path = get_model_full_path(model_name)
        if not model_full_path:
            raise FileNotFoundError(
                f"Model '{model_name}' not found under models/lightx2v/. Expected directory: {get_model_base_path() / model_name}"
            )

        cfg_hash = self._config_hash(
            model_name,
            input_type,
            input_path,
            sr_ratio,
            target_height,
            target_width,
            fps,
            segment_length,
            segment_overlap,
            seed,
            prompt,
            negative_prompt,
            save_output,
            output_path,
        )

        try:
            needs_reinit = (
                getattr(self.__class__, "_current_runner", None) is None or getattr(self.__class__, "_current_config_hash", None) != cfg_hash
            )

            if needs_reinit:
                if getattr(self.__class__, "_current_runner", None) is not None:
                    del self.__class__._current_runner
                    torch.cuda.empty_cache()
                    gc.collect()

                config = {
                    "model_cls": "seedvr2",
                    "task": "sr",
                    "model_path": model_full_path,
                    "sr_ratio": float(sr_ratio),
                    "target_height": int(target_height),
                    "target_width": int(target_width),
                    "target_video_length": int(segment_length),
                    "sr_segment_length": int(segment_length),
                    "sr_overlap": int(segment_overlap),
                    "fps": float(fps),
                    "infer_steps": 1,
                    "seed": int(seed),
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }

                formatted_config = set_config(config)
                self.__class__._current_runner = init_runner(formatted_config)
                self.__class__._current_config_hash = cfg_hash

            runner = self.__class__._current_runner

            progress = ProgressBar(100)

            def _update_progress(current_step, _total):
                progress.update_absolute(current_step)

            if hasattr(runner, "set_progress_callback"):
                runner.set_progress_callback(_update_progress)

            input_info = init_empty_input_info("sr")
            update_input_info_from_dict(
                input_info,
                {
                    "video_path": input_path if input_type == "video" else "",
                    "image_path": input_path if input_type == "image" else "",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": int(seed),
                    "save_result_path": output_path if (save_output and output_path) else "",
                    "return_result_tensor": True,
                },
            )

            runner.set_config(
                {
                    "video_path": input_path if input_type == "video" else "",
                    "image_path": input_path if input_type == "image" else "",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": int(seed),
                    "save_result_path": output_path if (save_output and output_path) else "",
                    "return_result_tensor": True,
                }
            )

            result_dict = runner.run_pipeline(input_info)
            images = result_dict.get("video", None)

            if images is None or images.numel() == 0:
                raise RuntimeError("SeedVR returned empty result")

            images = images.cpu()
            if images.dtype != torch.float32:
                images = images.float()

            if images.dim() == 4 and images.shape[0] > 0:
                images = images[0]

            if unload_after_inference:
                if hasattr(self.__class__, "_current_runner"):
                    del self.__class__._current_runner
                self.__class__._current_runner = None
                self.__class__._current_config_hash = None

            torch.cuda.empty_cache()
            gc.collect()

            return (images,)

        except Exception as e:
            logging.error(f"SeedVR SR failed: {e}")
            if unload_after_inference:
                if hasattr(self.__class__, "_current_runner"):
                    del self.__class__._current_runner
                self.__class__._current_runner = None
                self.__class__._current_config_hash = None
            raise
