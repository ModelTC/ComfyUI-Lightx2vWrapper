"""Refactored ComfyUI nodes for LightX2V."""

import os
import torch
import gc
import logging
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast
from easydict import EasyDict
from tqdm import tqdm

import comfy.model_management as comfy_mm
from comfy.utils import ProgressBar

from .config import LightX2VConfig, ModelConfig, VideoConfig, TeaCacheConfig
from .factory import LightX2VFactory
from .models import (
    LightX2VT5Encoder,
    LightX2VClipVisionEncoder,
    LightX2VVae,
    LightX2VModel,
)

# Import original LightX2V modules
from ..lightx2v.lightx2v.utils.profiler import ProfilingContext
from ..lightx2v.lightx2v.models.schedulers.wan.scheduler import WanScheduler
from ..lightx2v.lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerTeaCaching,
)


# Coefficient values for TeaCache
TEACACHE_COEFFICIENTS = {
    "i2v-14B-480p": [
        [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
        [-3.02331670e02, 2.23948934e02, -5.25463970e01, 5.87348440e00, -2.01973289e-01],
    ],
    "i2v-14B-720p": [
        [8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02],
        [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
    ],
    "t2v-1.3B": [
        [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
        [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
    ],
    "t2v-14B": [
        [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
        [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    ],
}


class BaseNode:
    """Base class for ComfyUI nodes with common functionality."""

    CATEGORY = "LightX2V"

    @classmethod
    def get_device(cls, device_str: str) -> torch.device:
        """Convert device string to torch device."""
        if device_str == "cuda":
            return comfy_mm.get_torch_device()
        return torch.device("cpu")

    @classmethod
    def get_dtype(cls, precision_str: str) -> torch.dtype:
        """Convert precision string to torch dtype."""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return dtype_map[precision_str]


class WanVideoTeaCache(BaseNode):
    """TeaCache configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.26,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.001,
                        "tooltip": "Threshold for cache application",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start percentage for TeaCache",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End percentage for TeaCache",
                    },
                ),
                "cache_device": (
                    ["main_device", "offload_device"],
                    {"default": "offload_device", "tooltip": "Device to cache to"},
                ),
                "coefficients": (
                    list(TEACACHE_COEFFICIENTS.keys()),
                    {
                        "default": "i2v-14B-720p",
                        "tooltip": "Coefficient preset for TeaCache",
                    },
                ),
                "use_ret_steps": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mode": (
                    ["e", "e0"],
                    {
                        "default": "e",
                        "tooltip": "Time embedding mode",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LIGHT_TEACACHEARGS",)
    RETURN_NAMES = ("teacache_args",)
    FUNCTION = "process"
    EXPERIMENTAL = True

    def process(
        self,
        rel_l1_thresh: float,
        start_percent: float,
        end_percent: float,
        cache_device: str,
        coefficients: str,
        use_ret_steps: bool,
        mode: str = "e",
    ) -> Tuple[TeaCacheConfig]:
        """Create TeaCache configuration."""
        device = comfy_mm.get_torch_device() if cache_device == "main_device" else comfy_mm.unet_offload_device()

        config = TeaCacheConfig(
            rel_l1_thresh=rel_l1_thresh,
            start_percent=start_percent,
            end_percent=end_percent,
            cache_device=device,
            coefficients=TEACACHE_COEFFICIENTS[coefficients],
            use_ret_steps=use_ret_steps,
            mode=mode,
        )

        return (config,)


class Lightx2vWanVideoModelDir(BaseNode):
    """Model directory specification node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": (
                    "STRING",
                    {"default": "/mnt/aigc/users/lijiaqi2/wan_model/Wan2.1-I2V-14B-480P"},
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "process"

    def process(self, model_dir: str) -> Tuple[str]:
        """Validate and return model directory."""
        path = Path(model_dir)
        if not path.exists():
            raise ValueError(f"Model directory {model_dir} does not exist.")
        return (model_dir,)


class Lightx2vWanVideoT5EncoderLoader(BaseNode):
    """T5 encoder loader node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {"default": "models_t5_umt5-xxl-enc-bf16.pth"},
                ),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "model_dir": ("STRING", {"default": None}),
            },
        }

    RETURN_TYPES = ("LIGHT_T5_ENCODER",)
    RETURN_NAMES = ("t5_encoder",)
    FUNCTION = "load_t5_encoder"

    def load_t5_encoder(
        self,
        model_name: str,
        precision: str,
        device: str,
        model_dir: Optional[str] = None,
    ) -> Tuple[LightX2VT5Encoder]:
        """Load T5 encoder."""
        dtype = self.get_dtype(precision)
        device_obj = self.get_device(device)

        if model_dir:
            model_path = Path(model_dir) / model_name
        else:
            model_path = Path(model_name)
            if not model_path.exists():
                raise ValueError(f"T5 model path {model_path} does not exist.")

        encoder = LightX2VFactory.create_t5_encoder(
            model_path=model_path,
            dtype=dtype,
            device=device_obj,
            cpu_offload=(device == "cpu"),
        )

        return (encoder,)


class Lightx2vWanVideoT5Encoder(BaseNode):
    """T5 text encoding node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t5_encoder": ("LIGHT_T5_ENCODER",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Summer beach vacation style...",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIGHT_TEXT_EMBEDDINGS",)
    RETURN_NAMES = ("text_embeddings",)
    FUNCTION = "encode_text"

    def encode_text(
        self,
        t5_encoder: LightX2VT5Encoder,
        prompt: str,
        negative_prompt: str = "",
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Encode text with T5."""
        embeddings = t5_encoder.encode_with_negative(prompt, negative_prompt)
        return (embeddings,)


class Lightx2vWanVideoClipVisionEncoderLoader(BaseNode):
    """CLIP vision encoder loader node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {"default": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"},
                ),
                "tokenizer_path": (
                    "STRING",
                    {"default": "xlm-roberta-large"},
                ),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "model_dir": ("STRING", {"default": None}),
            },
        }

    RETURN_TYPES = ("LIGHT_CLIP_VISION_ENCODER",)
    RETURN_NAMES = ("clip_vision_encoder",)
    FUNCTION = "load_clip_vision_encoder"

    def load_clip_vision_encoder(
        self,
        model_name: str,
        tokenizer_path: str,
        precision: str,
        device: str,
        model_dir: Optional[str] = None,
    ) -> Tuple[LightX2VClipVisionEncoder]:
        """Load CLIP vision encoder."""
        dtype = self.get_dtype(precision)
        device_obj = self.get_device(device)

        if model_dir:
            model_path = Path(model_dir) / model_name
        else:
            model_path = Path(model_name)
            if not model_path.exists():
                raise ValueError(f"CLIP model path {model_path} does not exist.")

        encoder = LightX2VFactory.create_clip_vision_encoder(
            model_path=model_path,
            dtype=dtype,
            device=device_obj,
        )

        return (encoder,)


class Lightx2vWanVideoVaeLoader(BaseNode):
    """VAE loader node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {"default": "Wan2.1_VAE.pth"},
                ),
                "precision": (["bf16", "fp16", "fp32"], {"default": "fp16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "parallel": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model_dir": ("STRING", {"default": None}),
            },
        }

    RETURN_TYPES = ("LIGHT_WAN_VAE",)
    RETURN_NAMES = ("wan_vae",)
    FUNCTION = "load_vae"

    def load_vae(
        self,
        model_name: str,
        precision: str,
        device: str,
        parallel: bool,
        model_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Any]]:
        """Load VAE."""
        dtype = self.get_dtype(precision)
        device_obj = self.get_device(device)

        if model_dir:
            model_path = Path(model_dir) / model_name
        else:
            model_path = Path(model_name)
            if not model_path.exists():
                raise ValueError(f"VAE model path {model_path} does not exist.")

        vae = LightX2VFactory.create_vae(
            model_path=model_path,
            dtype=dtype,
            device=device_obj,
            parallel=parallel,
        )

        # Return in legacy format for compatibility
        return ({"vae_cls": vae, "device": device},)


class Lightx2vWanVideoVaeDecoder(BaseNode):
    """VAE decoder node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_vae": ("LIGHT_WAN_VAE",),
                "latent": ("LIGHT_LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode_latent"

    def decode_latent(
        self,
        wan_vae: Dict[str, Any],
        latent: Dict[str, Any],
    ) -> Tuple[torch.Tensor]:
        """Decode latents to images."""
        vae_instance = wan_vae["vae_cls"]
        if isinstance(vae_instance, LightX2VVae):
            vae_model = vae_instance
        else:
            # Legacy compatibility
            vae_model = vae_instance

        latents = latent["samples"]
        generator = latent["generator"]
        cpu_offload = wan_vae["device"] == "cpu"

        with torch.no_grad():
            with ProfilingContext("*decoded images*"):
                if isinstance(vae_model, LightX2VVae):
                    decoded_images = vae_model.decode(latents, generator, cpu_offload)
                else:
                    # Legacy compatibility
                    config = EasyDict({"cpu_offload": cpu_offload})
                    decoded_images = vae_model.decode(latents, generator=generator, config=config)

            # Normalize from [-1, 1] to [0, 1]
            images = (decoded_images + 1) / 2

            # Rearrange dimensions for ComfyUI [T, H, W, C]
            images = images.squeeze(0).permute(1, 2, 3, 0).cpu()
            images = torch.clamp(images, 0, 1)

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        return (images,)


class Lightx2vWanVideoImageEncoder(BaseNode):
    """Image encoding node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("LIGHT_WAN_VAE",),
                "clip_vision_encoder": ("LIGHT_CLIP_VISION_ENCODER",),
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 832,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Width of the image",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Height of the image",
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 10000,
                        "step": 4,
                        "tooltip": "Number of frames",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIGHT_IMAGE_EMBEDDINGS",)
    RETURN_NAMES = ("image_embeddings",)
    FUNCTION = "encode_image"

    def encode_image(
        self,
        vae: Dict[str, Any],
        clip_vision_encoder: LightX2VClipVisionEncoder,
        image: torch.Tensor,
        width: int,
        height: int,
        num_frames: int,
    ) -> Tuple[Dict[str, Any]]:
        """Encode image with CLIP and VAE."""
        vae_instance = vae["vae_cls"]

        # Create video configuration
        video_config = VideoConfig(
            target_width=width,
            target_height=height,
            target_video_length=num_frames,
        )

        # Convert image format
        device = comfy_mm.get_torch_device()
        img = image[0].permute(2, 0, 1).to(device)
        img = img.sub_(0.5).div_(0.5)  # Normalize to [-1, 1]

        # Encode with CLIP
        with ProfilingContext("*clip encoder*"):
            if isinstance(clip_vision_encoder, LightX2VClipVisionEncoder):
                clip_out = clip_vision_encoder.encode(img, video_config)
                clip_out = clip_out.squeeze(0).to(torch.bfloat16)
            else:
                # Legacy compatibility
                config_dict = video_config.__dict__ if hasattr(video_config, "__dict__") else video_config
                clip_out = clip_vision_encoder.visual([img[:, None, :, :]], config_dict)
                clip_out = clip_out.squeeze(0).to(torch.bfloat16)

        # Calculate dimensions
        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = video_config.max_area
        lat_h = round(np.sqrt(max_area * aspect_ratio) // video_config.vae_stride[1] // video_config.patch_size[1] * video_config.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // video_config.vae_stride[2] // video_config.patch_size[2] * video_config.patch_size[2])

        # Update config
        config_dict = video_config.to_easydict() if hasattr(video_config, "to_easydict") else EasyDict(video_config.__dict__)
        config_dict.lat_h = lat_h
        config_dict.lat_w = lat_w

        h = lat_h * video_config.vae_stride[1]
        w = lat_w * video_config.vae_stride[2]

        # Create mask
        msk = torch.ones(1, num_frames, lat_h, lat_w, device=device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # Encode with VAE
        with ProfilingContext("*vae encoder*"):
            video_tensor = torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, num_frames - 1, h, w),
                ],
                dim=1,
            ).cuda()

            if isinstance(vae_instance, LightX2VVae):
                vae_out = vae_instance.encode([video_tensor], video_config)[0]
            else:
                # Legacy compatibility
                vae_out = vae_instance.encode([video_tensor], config_dict)[0]

        vae_out = torch.concat([msk, vae_out]).to(torch.bfloat16)

        image_embeddings = {
            "clip_encoder_out": clip_out,
            "vae_encode_out": vae_out,
            "config": config_dict,
        }

        logging.info(f"Image encoder outputs - CLIP: {clip_out.shape}, VAE: {vae_out.shape}")

        return (image_embeddings,)


class Lightx2vWanVideoEmptyEmbeds(BaseNode):
    """Empty embeddings for T2V."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "num_frames": (
                    "INT",
                    {"default": 81, "min": 1, "max": 10000, "step": 4},
                ),
            }
        }

    RETURN_TYPES = ("LIGHT_IMAGE_EMBEDDINGS",)
    RETURN_NAMES = ("image_embeddings",)
    FUNCTION = "process"

    def process(self, num_frames: int, width: int, height: int) -> Tuple[Dict[str, Any]]:
        """Create empty image embeddings for T2V."""
        video_config = VideoConfig(
            target_width=width,
            target_height=height,
            target_video_length=num_frames,
        )

        return ({"config": video_config.to_easydict()},)


class Lightx2vWanVideoModelLoader(BaseNode):
    """Main model loader node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": ""}),
                "model_type": (["t2v", "i2v"], {"default": "i2v"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "attention_type": (
                    ["sdpa", "flash_attn2", "flash_attn3"],
                    {"default": "flash_attn3"},
                ),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "offload_granularity": (["block", "phase"], {"default": "phase"}),
            },
            "optional": {
                "mm_type": ("STRING", {"default": None}),
                "teacache_args": ("LIGHT_TEACACHEARGS", {"default": None}),
                "lora_path": ("STRING", {"default": None}),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "model_dir": (
                    "STRING",
                    {"default": "/mnt/aigc/users/lijiaqi2/wan_model/Wan2.1-I2V-14B-480P"},
                ),
            },
        }

    RETURN_TYPES = ("LIGHT_WAN_MODEL",)
    RETURN_NAMES = ("wan_model",)
    FUNCTION = "load_model"

    def load_model(
        self,
        model_name: str,
        model_type: str,
        precision: str,
        device: str,
        attention_type: str,
        offload_granularity: str,
        mm_type: Optional[str] = None,
        lora_path: Optional[str] = None,
        lora_strength: float = 1.0,
        cpu_offload: bool = False,
        teacache_args: Optional[TeaCacheConfig] = None,
        model_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Any]]:
        """Load the main model."""
        if model_dir:
            model_path = Path(model_dir) / model_name
        else:
            model_path = Path(model_name)
            if not model_path.exists():
                raise ValueError(f"Model path {model_path} does not exist.")

        # Parse mm_config
        mm_config = {}
        if mm_type:
            try:
                mm_config = json.loads(mm_type)
            except Exception as e:
                logging.error(f"Invalid mm_type config: {e}")

        # Create model configuration
        model_config = ModelConfig(
            model_path=model_path,
            model_type=model_type,
            precision=precision,
            device=device,
            attention_type=attention_type,
            cpu_offload=cpu_offload,
            offload_granularity=offload_granularity,
            lora_path=Path(lora_path) if lora_path and lora_path.strip() else None,
            lora_strength=lora_strength,
            mm_config=mm_config,
            feature_caching="Tea" if teacache_args else "NoCaching",
        )

        # Create model
        model = LightX2VFactory.create_model(model_config)

        # Add TeaCache config if provided
        easydict_config = model.easydict_config
        if teacache_args:
            easydict_config.teacache_thresh = teacache_args.rel_l1_thresh
            easydict_config.use_ret_steps = teacache_args.use_ret_steps
            easydict_config.coefficients = teacache_args.coefficients
            easydict_config.teacache_start_percent = teacache_args.start_percent
            easydict_config.teacache_end_percent = teacache_args.end_percent
            easydict_config.teacache_device = teacache_args.cache_device
            easydict_config.teacache_mode = teacache_args.mode

        logging.info(f"Loaded model from {model_path} with type {model_type}")

        return ({"wan_model": model._model, "config": easydict_config},)


class Lightx2vWanVideoSampler(BaseNode):
    """Video sampling node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LIGHT_WAN_MODEL",),
                "text_embeddings": ("LIGHT_TEXT_EMBEDDINGS",),
                "image_embeddings": ("LIGHT_IMAGE_EMBEDDINGS",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "shift": ("FLOAT", {"default": 5.0}),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 5, "min": 1, "max": 20.0, "step": 0.1},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("LIGHT_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"

    def sample(
        self,
        model: Dict[str, Any],
        text_embeddings: Dict[str, Any],
        steps: int,
        shift: float,
        cfg_scale: float,
        seed: int,
        image_embeddings: Dict[str, Any],
    ) -> Tuple[Dict[str, Any]]:
        """Sample video latents."""
        model_config = cast(EasyDict, model.get("config"))
        model_config.update(image_embeddings.get("config", {}))

        wan_model = model.get("wan_model")
        clip_encoder_out = image_embeddings.get("clip_encoder_out", None)
        vae_encode_out = image_embeddings.get("vae_encode_out", None)

        if model_config.task == "i2v" and (clip_encoder_out is None or vae_encode_out is None):
            raise ValueError("Image embeddings required for i2v task")

        # Update config
        model_config.infer_steps = steps
        model_config.sample_shift = shift
        model_config.sample_guide_scale = cfg_scale
        model_config.seed = seed
        model_config.enable_cfg = cfg_scale != 1.0

        # Set target shape
        num_channels_latents = model_config.get("num_channels_latents", 16)

        if model_config.task == "i2v":
            model_config.target_shape = (
                num_channels_latents,
                (model_config.target_video_length - 1) // model_config.vae_stride[0] + 1,
                model_config.lat_h,
                model_config.lat_w,
            )
        else:  # t2v
            model_config.target_shape = (
                16,
                (model_config.target_video_length - 1) // 4 + 1,
                int(model_config.target_height) // model_config.vae_stride[1],
                int(model_config.target_width) // model_config.vae_stride[2],
            )

        # Create scheduler
        if model_config.feature_caching == "NoCaching":
            scheduler = WanScheduler(model_config)
        elif model_config.feature_caching == "Tea":
            scheduler = WanSchedulerTeaCaching(model_config)
        else:
            raise NotImplementedError(f"Unsupported caching: {model_config.feature_caching}")

        wan_model.set_scheduler(scheduler)

        # Prepare inputs
        inputs = {
            "text_encoder_output": text_embeddings,
            "image_encoder_output": image_embeddings,
        }

        scheduler.prepare(inputs.get("image_encoder_output"))

        # Run sampling
        progress = ProgressBar(steps)
        for step_index in tqdm(range(scheduler.infer_steps), desc="Sampling"):
            scheduler.step_pre(step_index=step_index)
            with ProfilingContext("model.infer"):
                wan_model.infer(inputs)
            scheduler.step_post()
            progress.update(1)

        latents, generator = scheduler.latents, scheduler.generator
        scheduler.clear()

        # Cleanup
        del inputs, scheduler, text_embeddings, image_embeddings
        torch.cuda.empty_cache()

        return ({"samples": latents, "generator": generator},)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Lightx2vWanVideoModelDir": Lightx2vWanVideoModelDir,
    "Lightx2vWanVideoT5EncoderLoader": Lightx2vWanVideoT5EncoderLoader,
    "Lightx2vWanVideoT5Encoder": Lightx2vWanVideoT5Encoder,
    "Lightx2vWanVideoClipVisionEncoderLoader": Lightx2vWanVideoClipVisionEncoderLoader,
    "Lightx2vWanVideoVaeLoader": Lightx2vWanVideoVaeLoader,
    "Lightx2vTeaCache": WanVideoTeaCache,
    "Lightx2vWanVideoEmptyEmbeds": Lightx2vWanVideoEmptyEmbeds,
    "Lightx2vWanVideoImageEncoder": Lightx2vWanVideoImageEncoder,
    "Lightx2vWanVideoVaeDecoder": Lightx2vWanVideoVaeDecoder,
    "Lightx2vWanVideoModelLoader": Lightx2vWanVideoModelLoader,
    "Lightx2vWanVideoSampler": Lightx2vWanVideoSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lightx2vWanVideoModelDir": "LightX2V WAN Model Directory",
    "Lightx2vWanVideoT5EncoderLoader": "LightX2V WAN T5 Encoder Loader",
    "Lightx2vWanVideoT5Encoder": "LightX2V WAN T5 Encoder",
    "Lightx2vWanVideoClipVisionEncoderLoader": "LightX2V WAN CLIP Vision Encoder Loader",
    "Lightx2vWanVideoVaeLoader": "LightX2V WAN VAE Loader",
    "Lightx2vWanVideoImageEncoder": "LightX2V WAN Image Encoder",
    "Lightx2vWanVideoVaeDecoder": "LightX2V WAN VAE Decoder",
    "Lightx2vWanVideoModelLoader": "LightX2V WAN Model Loader",
    "Lightx2vWanVideoSampler": "LightX2V WAN Video Sampler",
    "Lightx2vTeaCache": "LightX2V WAN Tea Cache",
    "Lightx2vWanVideoEmptyEmbeds": "LightX2V WAN Video Empty Embeds",
}
