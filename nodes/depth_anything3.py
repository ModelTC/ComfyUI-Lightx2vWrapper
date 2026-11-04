"""ComfyUI adapters for LightX2V's native DA3MONO-LARGE runner."""

import logging
import tempfile
import threading
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image

from ._lightx2v_runner import run_request
from ._runner_memory import get_runner, make_runner_handle
from .file_input import probe_video_file, resolve_input_video_path
from .seedvr import _prepare_output_video

logger = logging.getLogger(__name__)
_DA3_RUN_LOCK = threading.Lock()
_MODEL_ROOT = "geometry_estimation"
_PROCESS_RES_METHODS = ["upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"]


def _model_root():
    return Path(folder_paths.models_dir) / _MODEL_ROOT


def _is_mono_checkpoint(path):
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return False
    if path.name == "model.safetensors":
        name = path.parent.name.lower().replace("_", "").replace("-", "")
        return "mono" in name and "large" in name
    name = path.stem.lower().replace("_", "").replace("-", "")
    return "mono" in name and "large" in name


def _scan_mono_checkpoints():
    root = _model_root()
    if not root.is_dir():
        return ["None"]
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.safetensors") if _is_mono_checkpoint(path))
    return names or ["None"]


def _resolve_mono_checkpoint(model_name):
    if model_name == "None":
        raise FileNotFoundError(f"No DA3MONO-LARGE checkpoint found under {_model_root()}")
    root = _model_root().resolve()
    path = (root / model_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"DA3 model must be under {root}: {model_name}") from exc
    if not _is_mono_checkpoint(path):
        raise FileNotFoundError(f"DA3MONO-LARGE checkpoint not found: {path}")
    return path


def _prepare_output_file(filename_prefix, width, height, suffix):
    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory(), int(width), int(height)
    )
    name = f"{filename}_{counter:05}_.{suffix}"
    relative_name = (Path(subfolder) / name).as_posix() if subfolder else name
    return Path(full_folder) / name, relative_name


class LightX2VDA3MonoLoader:
    """Load an official or ComfyUI-format DA3MONO-LARGE checkpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _scan_mono_checkpoints()
        return {
            "required": {
                "model_name": (models, {"default": models[0], "tooltip": "DA3MONO-LARGE safetensors under models/geometry_estimation/."}),
                "process_res": ("INT", {"default": 504, "min": 14, "max": 2048, "step": 14}),
                "process_res_method": (_PROCESS_RES_METHODS, {"default": "upper_bound_resize"}),
                "attention_backend": (["torch_sdpa", "flash_attn2", "flash_attn3", "sage_attn2"], {"default": "torch_sdpa"}),
                "backbone_dtype": (["auto", "bf16", "fp32"], {"default": "auto"}),
                "norm_type": (["torch", "Triton"], {"default": "torch"}),
                "use_cuda_graph": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Capture one processed video frame size; uses extra VRAM. Images run eagerly."},
                ),
            }
        }

    RETURN_TYPES = ("DA3_MONO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LightX2V/Depth Anything 3"

    def load(self, model_name, process_res, process_res_method, attention_backend, backbone_dtype, norm_type, use_cuda_graph):
        model_path = _resolve_mono_checkpoint(model_name)
        config = {
            "model_cls": "depth_anything3",
            "model_variant": "mono_large",
            "task": "depth",
            "model_path": str(model_path),
            "process_res": int(process_res),
            "process_res_method": str(process_res_method),
            "attn_type": str(attention_backend),
            "backbone_dtype": str(backbone_dtype),
            "norm_type": str(norm_type),
            "use_cuda_graph": bool(use_cuda_graph),
            "parallel": False,
        }
        logger.info("[DA3MonoLoader] loading %s", model_path)
        return (make_runner_handle(config, model_name=model_name),)


class LightX2VDA3MonoImageSampler:
    """Return an RGB depth preview and sky mask for one ComfyUI IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DA3_MONO_MODEL",),
                "image": ("IMAGE",),
                "save_float_depth": ("BOOLEAN", {"default": False, "tooltip": "Also save the numeric depth map as NPY under ComfyUI output."}),
                "filename_prefix": ("STRING", {"default": "lightx2v_da3/DA3Mono"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("depth_preview", "sky_mask", "depth_npy")
    FUNCTION = "sample"
    CATEGORY = "LightX2V/Depth Anything 3"

    def sample(self, model, image, save_float_depth, filename_prefix):
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
            raise ValueError(f"DA3 MONO expects one IMAGE [1, H, W, RGB/RGBA], got {tuple(image.shape)}")
        height, width = int(image.shape[1]), int(image.shape[2])
        rgb = (image[0, ..., :3].detach().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        depth_path = None
        depth_name = ""
        if save_float_depth:
            depth_path, depth_name = _prepare_output_file(filename_prefix, width, height, "npy")

        with tempfile.TemporaryDirectory(prefix="lightx2v_da3_", dir=folder_paths.get_temp_directory()) as temporary:
            input_path = Path(temporary) / "input.png"
            output_path = Path(temporary) / "depth.png"
            mask_path = Path(temporary) / "sky_mask.png"
            Image.fromarray(rgb, "RGB").save(input_path)
            request = {
                "task": "depth",
                "image_path": str(input_path),
                "save_result_path": str(output_path),
                "save_sky_mask_path": str(mask_path),
            }
            if depth_path is not None:
                request["save_depth_path"] = str(depth_path)
            with _DA3_RUN_LOCK:
                run_request(get_runner(model), request)
            with Image.open(output_path) as depth_image:
                depth = np.asarray(depth_image.convert("RGB"), dtype=np.float32).copy() / 255.0
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L").resize((width, height), Image.Resampling.NEAREST), dtype=np.float32).copy() / 255.0
        return torch.from_numpy(depth).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), depth_name


class LightX2VDA3MonoVideoSampler:
    """Stream a video through DA3 and return its ComfyUI output filename."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DA3_MONO_MODEL",),
                "video_path": ("STRING", {"forceInput": True, "tooltip": "Use LightX2V Input Video Path; the file must be under ComfyUI input."}),
                "filename_prefix": ("STRING", {"default": "lightx2v_da3/DA3Mono"}),
                "save_float_depth": ("BOOLEAN", {"default": False, "tooltip": "Save per-frame numeric depth to a streaming NPZ sidecar."}),
                "save_sky_mask": ("BOOLEAN", {"default": False, "tooltip": "Save per-frame sky masks to the same NPZ sidecar."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filename", "data_npz")
    FUNCTION = "sample"
    CATEGORY = "LightX2V/Depth Anything 3"

    def sample(self, model, video_path, filename_prefix, save_float_depth, save_sky_mask):
        input_path = resolve_input_video_path(video_path)
        width, height, fps = probe_video_file(input_path)
        output_path, output_file, subfolder = _prepare_output_video(filename_prefix, width, height)
        relative_name = (Path(subfolder) / output_file).as_posix() if subfolder else output_file
        request = {"task": "depth", "video_path": str(input_path), "save_result_path": str(output_path)}
        sidecar_name = ""
        if save_float_depth or save_sky_mask:
            sidecar_path = output_path.with_name(f"{output_path.stem.removesuffix('_')}_data.npz")
            sidecar_name = (Path(subfolder) / sidecar_path.name).as_posix() if subfolder else sidecar_path.name
            if save_float_depth:
                request["save_depth_path"] = str(sidecar_path)
            if save_sky_mask:
                request["save_sky_mask_path"] = str(sidecar_path)
        logger.info("[DA3MonoVideoSampler] input=%s (%sx%s @ %.3f fps), output=%s", input_path, width, height, fps, output_path)
        with _DA3_RUN_LOCK:
            run_request(get_runner(model), request)
        if not output_path.is_file():
            raise RuntimeError(f"DA3 MONO did not create expected output video: {output_path}")
        if sidecar_name and not sidecar_path.is_file():
            raise RuntimeError(f"DA3 MONO did not create expected NPZ sidecar: {sidecar_path}")
        return relative_name, sidecar_name
