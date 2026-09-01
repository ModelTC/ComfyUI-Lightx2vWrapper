"""SwiftVR restoration nodes for ComfyUI."""

import argparse
import gc
import logging
import tempfile
import threading
import types
from pathlib import Path

import folder_paths
import torch
from comfy.utils import ProgressBar

from .file_input import probe_video_file, resolve_input_video_path
from .seedvr import _prepare_output_video

logger = logging.getLogger(__name__)

_SWIFTVR_RUN_LOCK = threading.Lock()
_REQUIRED_MODEL_FILES = (
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "reae.safetensors",
    "prompt_embedding.safetensors",
)
_MAX_OUTPUT_DIMENSION = 8192
_MAX_SR_RATIO = 8.0


def _swiftvr_model_root() -> Path:
    return Path(folder_paths.models_dir) / "lightx2v"


def _is_swiftvr_model(path: Path) -> bool:
    return path.is_dir() and all((path / relative_path).is_file() for relative_path in _REQUIRED_MODEL_FILES)


def _scan_swiftvr_models():
    root = _swiftvr_model_root()
    if not root.is_dir():
        return ["None"]
    models = sorted(path.name for path in root.iterdir() if _is_swiftvr_model(path))
    return models or ["None"]


def _prepare_output_image(filename_prefix, width, height):
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
        int(width),
        int(height),
    )
    file = f"{filename}_{counter:05}_.png"
    full_path = Path(full_output_folder) / file
    return full_path, file, subfolder


def _resolve_output_size(source_width: int, source_height: int, target_short_edge: int, *, require_even: bool):
    """Build an aspect-preserving public output size for SwiftVR.

    Native SwiftVR pads this public size to a multiple of 32 internally and
    crops the restored result back, so network alignment is intentionally not
    exposed through the ComfyUI interface.
    """

    source_width = int(source_width)
    source_height = int(source_height)
    target_short_edge = int(target_short_edge)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"SwiftVR source size must be positive, got {source_width}x{source_height}")
    if target_short_edge <= 0:
        raise ValueError(f"SwiftVR target_short_edge must be positive, got {target_short_edge}")

    source_short_edge = min(source_width, source_height)
    if target_short_edge < source_short_edge:
        raise ValueError(
            f"SwiftVR only performs restoration/upscaling: target_short_edge {target_short_edge} "
            f"is smaller than the aligned input short edge {source_short_edge}"
        )

    scale = target_short_edge / source_short_edge
    if scale > _MAX_SR_RATIO:
        raise ValueError(
            f"SwiftVR scale {scale:.3f}x exceeds the supported maximum {_MAX_SR_RATIO:.1f}x; "
            f"lower target_short_edge"
        )
    if source_width <= source_height:
        output_width = target_short_edge
        output_height = int(round(source_height * scale))
    else:
        output_height = target_short_edge
        output_width = int(round(source_width * scale))

    if require_even:
        output_width = max(2, (output_width + 1) // 2 * 2)
        output_height = max(2, (output_height + 1) // 2 * 2)
    if max(output_width, output_height) > _MAX_OUTPUT_DIMENSION:
        raise ValueError(
            f"SwiftVR output {output_width}x{output_height} exceeds the maximum supported dimension "
            f"{_MAX_OUTPUT_DIMENSION}; lower target_short_edge"
        )
    return output_height, output_width, scale


class _TensorVideoReader:
    """Small decord-compatible reader backed by ComfyUI IMAGE frames."""

    def __init__(self, frames: torch.Tensor, fps: float):
        self.frames = frames
        self.fps = float(fps)

    def __len__(self):
        return int(self.frames.shape[0])

    def __getitem__(self, index):
        return self.frames[index]

    def get_batch(self, indices):
        return self.frames[indices]

    def get_avg_fps(self):
        return self.fps


class _TensorVideoWriter:
    """imageio-compatible writer that keeps restored frames in memory."""

    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(torch.from_numpy(frame.copy()))

    def close(self):
        return None

    def as_images(self):
        if not self.frames:
            raise RuntimeError("SwiftVR produced no output frames")
        return torch.stack(self.frames).to(torch.float32).div_(255.0)


class LightX2VSwiftVRLoader:
    """Load a native LightX2V SwiftVR model and keep it resident."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _scan_swiftvr_models()
        return {
            "required": {
                "model_name": (
                    models,
                    {
                        "default": models[0],
                        "tooltip": "SwiftVR model directory under models/lightx2v/ containing transformer/, reae.safetensors, and prompt_embedding.safetensors.",
                    },
                ),
                "attention_backend": (
                    ["flash_attn3", "flash_attn2", "sage_attn2", "torch_sdpa"],
                    {"default": "flash_attn3", "tooltip": "flash_attn3 is recommended on H100/SM90."},
                ),
                "rope_type": (
                    ["flashinfer_rope", "torch_real_rope"],
                    {"default": "flashinfer_rope"},
                ),
                "clip_length": (
                    "INT",
                    {"default": 24, "min": 4, "max": 96, "step": 4, "tooltip": "Frames per streaming chunk; must be a multiple of 4."},
                ),
                "dit_overlap": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16, "step": 1, "tooltip": "Latent overlap retained between SwiftVR DiT chunks."},
                ),
                "reae_frame_batch_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 32, "step": 1, "tooltip": "REAE frame batch size; 0 uses the native automatic path."},
                ),
                "use_compile": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable torch.compile. First execution at a new resolution takes substantially longer."},
                ),
            }
        }

    RETURN_TYPES = ("SWIFTVR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LightX2V/SwiftVR"

    def load(self, model_name, attention_backend, rope_type, clip_length, dit_overlap, reae_frame_batch_size, use_compile):
        from ..lightx2v.lightx2v.infer import init_runner
        from ..lightx2v.lightx2v.utils.set_config import set_config

        model_path = _swiftvr_model_root() / model_name
        if model_name == "None" or not _is_swiftvr_model(model_path):
            missing = [str(model_path / relative_path) for relative_path in _REQUIRED_MODEL_FILES if not (model_path / relative_path).is_file()]
            detail = f" Missing: {', '.join(missing)}" if missing else ""
            raise FileNotFoundError(f"No complete SwiftVR model found at {model_path}.{detail}")
        if int(clip_length) % 4:
            raise ValueError(f"SwiftVR clip_length must be a multiple of 4, got {clip_length}")

        config = {
            "model_cls": "swiftvr",
            "task": "sr",
            "model_path": str(model_path),
            "attention_backend": str(attention_backend),
            "cross_attention_backend": str(attention_backend),
            "rope_type": str(rope_type),
            "clip_len": int(clip_length),
            "dit_overlap": int(dit_overlap),
            "reae_frame_batch_size": int(reae_frame_batch_size),
            "video_codec": "libx265",
            "quality": 60,
            "ffmpeg_preset": "ultrafast",
            "queue_size": 3,
            "cpu_offload": False,
            "parallel": False,
            "use_compile": bool(use_compile),
        }
        runner = init_runner(set_config(argparse.Namespace(**config)))
        logger.info(
            "[SwiftVRLoader] loaded %s; attention=%s, rope=%s, clip=%s, overlap=%s, reae_batch=%s, compile=%s",
            model_name,
            attention_backend,
            rope_type,
            clip_length,
            dit_overlap,
            reae_frame_batch_size,
            use_compile,
        )
        return ({"runner": runner, "model_name": model_name},)


class LightX2VSwiftVRSampler:
    """Restore a ComfyUI image or video-frame batch with a resident SwiftVR runner."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SWIFTVR_MODEL",),
                "images": ("IMAGE",),
                "target_short_edge": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 64,
                        "max": _MAX_OUTPUT_DIMENSION,
                        "step": 8,
                        "tooltip": "Output short edge. SwiftVR preserves aspect ratio; network padding and cropping are handled internally.",
                    },
                ),
                "source_fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "save_to_output_file": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Save directly under ComfyUI output as PNG for one image or MP4 for multiple frames. Restored images are always returned for one-image input.",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "lightx2v_swiftvr/SwiftVR"}),
                "video_codec": (["libx265", "libx264"], {"default": "libx265"}),
                "quality": ("INT", {"default": 60, "min": 0, "max": 100, "step": 1}),
                "ffmpeg_preset": (
                    ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
                    {"default": "ultrafast"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filename")
    FUNCTION = "sample"
    CATEGORY = "LightX2V/SwiftVR"

    def sample(
        self,
        model,
        images,
        target_short_edge,
        source_fps,
        save_to_output_file,
        filename_prefix,
        video_codec,
        quality,
        ffmpeg_preset,
    ):
        from lightx2v.models.runners.swiftvr import swiftvr_runner as swiftvr_module

        from ..lightx2v.lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict

        if images.dim() != 4 or images.shape[-1] not in (3, 4):
            raise ValueError(f"Expected IMAGE [T, H, W, C], got shape {tuple(images.shape)}")
        if images.shape[0] < 1:
            raise ValueError("SwiftVR requires at least one input frame")

        frames_u8 = (images[..., :3].detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().contiguous()
        is_image = int(frames_u8.shape[0]) == 1
        raw_height, raw_width = int(frames_u8.shape[1]), int(frames_u8.shape[2])
        source_height, source_width = raw_height // 8 * 8, raw_width // 8 * 8
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"SwiftVR input is too small after 8-pixel alignment: {raw_height}x{raw_width}")
        frames_u8 = frames_u8[:, :source_height, :source_width]
        output_height, output_width, sr_ratio = _resolve_output_size(
            source_width,
            source_height,
            target_short_edge,
            require_even=not is_image,
        )
        target_shape = [output_height, output_width]
        logger.info(
            "[SwiftVRSampler] aligned input=%sx%s, target_short_edge=%s, output=%sx%s, scale=%.4f",
            source_width,
            source_height,
            target_short_edge,
            output_width,
            output_height,
            sr_ratio,
        )

        runner = model["runner"]
        output_file = ""
        output_subfolder = ""
        temp_dir = None
        if save_to_output_file:
            prepare_output = _prepare_output_image if is_image else _prepare_output_video
            full_path, output_file, output_subfolder = prepare_output(filename_prefix, output_width, output_height)
            save_path = str(full_path)
            memory_writer = None
        elif is_image:
            save_path = ""
            memory_writer = None
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="lightx2v_swiftvr_")
            save_path = str(Path(temp_dir.name) / "memory-output.mp4")
            memory_writer = _TensorVideoWriter()

        image_path = "<tensor>" if is_image else ""
        video_path = "" if is_image else "<tensor>"
        return_result_tensor = is_image

        input_info = init_empty_input_info("sr")
        update_input_info_from_dict(
            input_info,
            {
                "video_path": video_path,
                "image_path": image_path,
                "sr_ratio": float(sr_ratio),
                "target_shape": target_shape,
                "save_result_path": save_path,
                "return_result_tensor": return_result_tensor,
            },
        )

        progress = ProgressBar(100)
        restored_images = None
        try:
            with _SWIFTVR_RUN_LOCK:
                original_video_reader = swiftvr_module.VideoReader
                original_mux_audio = swiftvr_module.mux_audio_from_video
                had_instance_writer = "open_video_writer" in runner.__dict__
                original_instance_writer = runner.__dict__.get("open_video_writer")
                had_instance_image_reader = "read_image_frame" in runner.__dict__
                original_instance_image_reader = runner.__dict__.get("read_image_frame")
                try:
                    runner.set_config(
                        {
                            "fps": float(source_fps),
                            "video_codec": str(video_codec),
                            "quality": int(quality),
                            "ffmpeg_preset": str(ffmpeg_preset),
                            "video_path": video_path,
                            "image_path": image_path,
                            "sr_ratio": float(sr_ratio),
                            "target_shape": target_shape,
                            "return_result_tensor": return_result_tensor,
                        }
                    )
                    if hasattr(runner, "set_progress_callback"):
                        runner.set_progress_callback(lambda current, _total: progress.update_absolute(current))
                    if is_image:
                        image_frames = frames_u8.permute(0, 3, 1, 2).contiguous()
                        runner.read_image_frame = types.MethodType(
                            lambda _runner, _path: (image_frames, source_height, source_width),
                            runner,
                        )
                    else:
                        swiftvr_module.VideoReader = lambda _path: _TensorVideoReader(frames_u8, source_fps)
                        swiftvr_module.mux_audio_from_video = lambda *_args, **_kwargs: None
                    if not is_image and memory_writer is not None:
                        runner.open_video_writer = types.MethodType(lambda _runner, _path, _fps: memory_writer, runner)
                    result = runner.run_pipeline(input_info)
                    if is_image:
                        restored_images = result.get("images") if isinstance(result, dict) else result
                finally:
                    swiftvr_module.VideoReader = original_video_reader
                    swiftvr_module.mux_audio_from_video = original_mux_audio
                    if memory_writer is not None:
                        if had_instance_writer:
                            runner.open_video_writer = original_instance_writer
                        elif "open_video_writer" in runner.__dict__:
                            del runner.open_video_writer
                    if had_instance_image_reader:
                        runner.read_image_frame = original_instance_image_reader
                    elif "read_image_frame" in runner.__dict__:
                        del runner.read_image_frame
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            torch.cuda.empty_cache()
            gc.collect()

        if is_image:
            if not torch.is_tensor(restored_images) or restored_images.numel() == 0:
                raise RuntimeError("SwiftVR produced no output image")
            restored_images = restored_images.to(device="cpu", dtype=torch.float32).clamp_(0.0, 1.0)
            if save_to_output_file:
                swiftvr_module.save_to_image(restored_images, save_path)
                if not Path(save_path).is_file():
                    raise RuntimeError(f"SwiftVR did not create expected output image: {save_path}")
                relative_name = str(Path(output_subfolder) / output_file) if output_subfolder else output_file
                return (restored_images, relative_name)
            return (restored_images, "")

        if save_to_output_file:
            if not Path(save_path).is_file():
                raise RuntimeError(f"SwiftVR did not create expected output video: {save_path}")
            placeholder = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            relative_name = str(Path(output_subfolder) / output_file) if output_subfolder else output_file
            return (placeholder, relative_name)

        return (memory_writer.as_images().clamp_(0.0, 1.0), "")


class LightX2VSwiftVRFileSampler:
    """Restore an input video from disk without materializing it as IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SWIFTVR_MODEL",),
                "video_path": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Absolute path produced by LightX2V Input Video Path. The file must remain under ComfyUI input.",
                    },
                ),
                "target_short_edge": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 64,
                        "max": _MAX_OUTPUT_DIMENSION,
                        "step": 8,
                        "tooltip": "Output short edge. SwiftVR preserves aspect ratio.",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "lightx2v_swiftvr/SwiftVR"}),
                "video_codec": (["libx265", "libx264"], {"default": "libx265"}),
                "quality": ("INT", {"default": 60, "min": 0, "max": 100, "step": 1}),
                "ffmpeg_preset": (
                    ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
                    {"default": "ultrafast"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "sample"
    CATEGORY = "LightX2V/SwiftVR"

    def sample(self, model, video_path, target_short_edge, filename_prefix, video_codec, quality, ffmpeg_preset):
        from ..lightx2v.lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict

        input_path = resolve_input_video_path(video_path)
        raw_width, raw_height, source_fps = probe_video_file(input_path)
        source_height, source_width = raw_height // 8 * 8, raw_width // 8 * 8
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"SwiftVR input is too small after 8-pixel alignment: {raw_height}x{raw_width}")

        output_height, output_width, sr_ratio = _resolve_output_size(
            source_width,
            source_height,
            target_short_edge,
            require_even=True,
        )
        full_path, output_file, output_subfolder = _prepare_output_video(filename_prefix, output_width, output_height)
        save_path = str(full_path)
        target_shape = [output_height, output_width]

        input_info = init_empty_input_info("sr")
        update_input_info_from_dict(
            input_info,
            {
                "video_path": str(input_path),
                "image_path": "",
                "sr_ratio": float(sr_ratio),
                "target_shape": target_shape,
                "save_result_path": save_path,
                "return_result_tensor": False,
            },
        )

        runner = model["runner"]
        progress = ProgressBar(100)
        logger.info(
            "[SwiftVRFileSampler] input=%s (%sx%s @ %.3f fps), output=%sx%s",
            input_path,
            source_width,
            source_height,
            source_fps,
            output_width,
            output_height,
        )
        try:
            with _SWIFTVR_RUN_LOCK:
                runner.set_config(
                    {
                        "fps": 0.0,
                        "video_codec": str(video_codec),
                        "quality": int(quality),
                        "ffmpeg_preset": str(ffmpeg_preset),
                        "video_path": str(input_path),
                        "image_path": "",
                        "sr_ratio": float(sr_ratio),
                        "target_shape": target_shape,
                        "return_result_tensor": False,
                    }
                )
                if hasattr(runner, "set_progress_callback"):
                    runner.set_progress_callback(lambda current, _total: progress.update_absolute(current))
                runner.run_pipeline(input_info)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        if not Path(save_path).is_file():
            raise RuntimeError(f"SwiftVR did not create expected output video: {save_path}")
        relative_name = str(Path(output_subfolder) / output_file) if output_subfolder else output_file
        return (relative_name,)
