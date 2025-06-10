import os
import torch
import gc
from typing import cast, Any, no_type_check
import logging
import json
import numpy as np
import comfy.model_management as comfy_mm
from comfy.utils import ProgressBar
from pathlib import Path
import math

# import folder_paths
from tqdm import tqdm
from easydict import EasyDict


# Import LightX2V modules
from .lightx2v.lightx2v.utils.profiler import ProfilingContext
from .lightx2v.lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from .lightx2v.lightx2v.models.input_encoders.hf.xlm_roberta.model import (
    CLIPModel as ClipVisionModel,
)
from .lightx2v.lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from .lightx2v.lightx2v.models.networks.wan.model import WanModel
from .lightx2v.lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from .lightx2v.lightx2v.models.schedulers.wan.scheduler import WanScheduler
from .lightx2v.lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerTeaCaching,
)

from .lightx2v.lightx2v.common.ops import *  # noqa: F401, F403 for import global register


class LightX2VEncoderFactory:
    """编码器工厂类，统一管理所有编码器的创建"""

    @staticmethod
    def create_t5_encoder(model_path, tokenizer_path, dtype, device, cpu_offload=False):
        return T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=device,
            checkpoint_path=model_path,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            cpu_offload=cpu_offload,
        )

    @staticmethod
    def create_clip_vision_encoder(model_path, dtype, device):
        return ClipVisionModel(
            dtype=dtype,
            device=device,
            checkpoint_path=model_path,
            clip_quantized=False,
            clip_quantized_ckpt=None,
            quant_scheme=None,
        )

    @staticmethod
    def create_vae(model_path, dtype, device, parallel=False):
        return WanVAE(
            z_dim=16,
            vae_pth=str(model_path),
            dtype=dtype,
            device=device,
            parallel=parallel,
        )


def convert_dtype(dtype_str: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_str]


class WanVideoTeaCache:
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
                        "tooltip": "Threshold for to determine when to apply the cache, compromise between speed and accuracy. When using coefficients a good value range is something between 0.2-0.4 for all but 1.3B model, which should be about 10 times smaller, same as when not using coefficients.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The start percentage of the steps to use with TeaCache.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The end percentage of the steps to use with TeaCache.",
                    },
                ),
                "cache_device": (
                    ["main_device", "offload_device"],
                    {"default": "offload_device", "tooltip": "Device to cache to"},
                ),
                "coefficients": (
                    [
                        "i2v-14B-720p",
                        "i2v-14B-480p",
                        "t2v-1.3B",
                        "t2v-14B",
                    ],
                    {
                        "default": "i2v-14B-720p",
                        "tooltip": "Use coefficients for TeaCache. 'i2v-14B-720p' will use the default coefficients",
                    },
                ),
                "use_ret_steps": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mode": (
                    ["e", "e0"],
                    {
                        "default": "e",
                        "tooltip": "Choice between using e (time embeds, default) or e0 (modulated time embeds)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LIGHT_TEACACHEARGS",)
    RETURN_NAMES = ("teacache_args",)
    FUNCTION = "process"
    CATEGORY = "LightX2V"

    EXPERIMENTAL = True

    def process(
        self,
        rel_l1_thresh: float,
        start_percent: float,
        end_percent: float,
        cache_device: str,
        coefficients: str,
        use_ret_steps: bool,
        mode="e",
    ):
        if cache_device == "main_device":
            teacache_device = comfy_mm.get_torch_device()
        else:
            teacache_device = comfy_mm.unet_offload_device()

        # use_ret_steps = True is [0]
        # use_ret_steps = False is [1]
        coeff_values_map = {
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

        teacache_args = {
            "rel_l1_thresh": rel_l1_thresh,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "cache_device": teacache_device,
            "coefficients": coeff_values_map[coefficients],
            "use_ret_steps": use_ret_steps,
            "mode": mode,
        }
        return (teacache_args,)


class Lightx2vWanVideoModelDir:
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
    RETURN_NAMES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "LightX2V"

    def process(self, model_dir):
        assert Path(model_dir).exists(), f"Model directory {model_dir} does not exist."
        return (model_dir,)


class Lightx2vWanVideoT5EncoderLoader:
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
    CATEGORY = "LightX2V"

    def load_t5_encoder(
        self,
        model_name: str,
        precision: str,
        device: str,
        model_dir: str | None = None,
    ):
        dtype = convert_dtype(precision)

        if model_dir:
            model_path = Path(model_dir)
            model_path = model_path / model_name
        else:
            model_path = Path(model_name)
            assert model_path.exists(), f"T5 model path {model_path} does not exist. Please provide a valid model path or set model_dir."

        tokenizer_path = model_path.parent / "google" / "umt5-xxl"
        assert tokenizer_path.exists(), f"Tokenizer path {tokenizer_path} does not exist. Please provide a valid tokenizer path or set model_dir."

        if device == "cuda":
            init_device = comfy_mm.get_torch_device()
            cpu_offload = False
        else:
            init_device = torch.device("cpu")
            cpu_offload = True

        t5_encoder = LightX2VEncoderFactory.create_t5_encoder(model_path, tokenizer_path, dtype, init_device, cpu_offload)

        return (t5_encoder,)


class Lightx2vWanVideoT5Encoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t5_encoder": ("LIGHT_T5_ENCODER",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIGHT_TEXT_EMBEDDINGS",)
    RETURN_NAMES = ("text_embeddings",)
    FUNCTION = "encode_text"
    CATEGORY = "LightX2V"

    def encode_text(self, t5_encoder: T5EncoderModel, prompt: str, negative_prompt: str | None = None):
        context = t5_encoder.infer([prompt])
        context_null = t5_encoder.infer([negative_prompt if negative_prompt else ""])
        text_embeddings = {"context": context, "context_null": context_null}
        return (text_embeddings,)


class Lightx2vWanVideoVaeLoader:
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
    CATEGORY = "LightX2V"

    def load_vae(self, model_name: str, precision: str, device: str, parallel: bool, model_dir: str | None = None):
        dtype = convert_dtype(precision)

        if model_dir:
            model_path = Path(model_dir)
            model_path = model_path / model_name
        else:
            model_path = Path(model_name)

        assert model_path.exists(), f"VAE model path {model_path} does not exist. Please provide a valid model path or set model_dir."

        if device == "cuda":
            init_device = comfy_mm.get_torch_device()
        else:
            init_device = torch.device("cpu")

        vae = LightX2VEncoderFactory.create_vae(model_path, dtype, init_device, parallel)

        vae_result = {"vae_cls": vae, "device": device}
        return (vae_result,)


class Lightx2vWanVideoVaeDecoder:
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
    CATEGORY = "LightX2V"

    def decode_latent(self, wan_vae: dict[str, WanVAE | str], latent: dict[str, Any]):
        wan_vae_instance: WanVAE = cast(WanVAE, wan_vae["vae_cls"])
        config = EasyDict({"cpu_offload": wan_vae["device"] == "cpu"})

        latents = latent["samples"]
        generator = latent["generator"]

        with torch.no_grad():
            with ProfilingContext("*decoded images*"):
                decoded_images = wan_vae_instance.decode(latents, generator=generator, config=config)

            # 将像素值从 [-1, 1] 归一化到 [0, 1]
            images = (decoded_images + 1) / 2

            # 重新排列维度为ComfyUI标准的图像格式 [T, H, W, C]
            # 从 [1, C, T, H, W] 转换为 [T, H, W, C]
            images = images.squeeze(0).permute(1, 2, 3, 0).cpu()

            # 确保像素值在有效范围内
            images = torch.clamp(images, 0, 1)

        # 清理缓存以释放GPU内存
        torch.cuda.empty_cache()
        gc.collect()

        return (images,)


class Lightx2vWanVideoClipVisionEncoderLoader:
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
    CATEGORY = "LightX2V"

    def load_clip_vision_encoder(self, model_name: str, tokenizer_path: str, precision: str, device: str, model_dir: str | None = None):
        dtype = convert_dtype(precision)

        if model_dir:
            model_path = Path(model_dir)
            model_path = model_path / model_name
        else:
            model_path = Path(model_name)
            assert model_path.exists(), f"CLIP model path {model_path} does not exist. Please provide a valid model path or set model_dir."

        if device == "cuda":
            init_device = comfy_mm.get_torch_device()
        else:
            init_device = torch.device("cpu")

        clip_vision_encoder = LightX2VEncoderFactory.create_clip_vision_encoder(model_path, dtype, init_device)

        return (clip_vision_encoder,)


class Lightx2vWanVideoImageEncoder:
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
                        "tooltip": "Width of the image to encode",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 64,
                        "max": 29048,
                        "step": 8,
                        "tooltip": "Height of the image to encode",
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 10000,
                        "step": 4,
                        "tooltip": "Number of frames to encode",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIGHT_IMAGE_EMBEDDINGS",)
    RETURN_NAMES = ("image_embeddings",)
    FUNCTION = "encode_image"
    CATEGORY = "LightX2V"

    def encode_image(
        self,
        vae: dict[str, WanVAE | str],
        clip_vision_encoder: ClipVisionModel,
        image: torch.Tensor,
        width: int,
        height: int,
        num_frames: int,
    ):
        vae_instance: WanVAE = cast(WanVAE, vae["vae_cls"])

        config = EasyDict(
            {
                "cpu_offload": True if vae["device"] == "cpu" else False,
                "target_height": height,
                "target_width": width,
                "target_video_length": num_frames,
                "vae_stride": (4, 8, 8),
                "patch_size": (1, 2, 2),
            }
        )
        # skip lint
        config = cast(Any, config)

        # 将图像转换为期望的张量格式
        device = comfy_mm.get_torch_device()
        img = image[0].permute(2, 0, 1).to(device)  # [C, H, W]
        img = img.sub_(0.5).div_(0.5)  # 归一化到 [-1, 1]

        # 使用CLIP视觉编码器编码图像
        with ProfilingContext("*clip encoder*"):
            clip_encoder_out = clip_vision_encoder.visual([img[:, None, :, :]], config).squeeze(0).to(torch.bfloat16)

        # 计算宽高比和尺寸
        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = config.target_height * config.target_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])

        # XXX: trick
        config.lat_h = lat_h
        config.lat_w = lat_w

        h = lat_h * config.vae_stride[1]
        w = lat_w * config.vae_stride[2]

        msk = torch.ones(1, config.target_video_length, lat_h, lat_w, device=torch.device("cuda"))
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        with ProfilingContext("*vae encoder*"):
            vae_encode_out: torch.Tensor = vae_instance.encode(
                [
                    torch.concat(
                        [
                            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                            torch.zeros(3, config.target_video_length - 1, h, w),
                        ],
                        dim=1,
                    ).cuda()
                ],  # type: ignore
                config,
            )[0]

        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)

        image_embeddings = {
            "clip_encoder_out": clip_encoder_out,
            "vae_encode_out": vae_encode_out,
            "config": config,
        }

        print(f"Image Encoder Output Shape: {clip_encoder_out.shape}")
        print(f"VAE Encoder Output Shape: {vae_encode_out.shape}")
        print(f"Latent Height: {lat_h}, Latent Width: {lat_w}")
        print(f"Image Shape: {img.shape}")
        print(f"Configuration: {config}")

        return (image_embeddings,)


class Lightx2vWanVideoEmptyEmbeds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 832,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Width of the image to encode",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 64,
                        "max": 29048,
                        "step": 8,
                        "tooltip": "Height of the image to encode",
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 10000,
                        "step": 4,
                        "tooltip": "Number of frames to encode",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIGHT_IMAGE_EMBEDDINGS",)
    RETURN_NAMES = ("image_embeddings",)
    FUNCTION = "process"
    CATEGORY = "LightX2V"

    def process(self, num_frames: int, width: int, height: int):
        config = EasyDict(
            {
                "target_height": height,
                "target_width": width,
                "target_video_length": num_frames,
                "vae_stride": (4, 8, 8),
                "patch_size": (1, 2, 2),
            }
        )

        return ({"config": config},)


class Lightx2vWanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {"default": ""},
                ),
                "model_type": (["t2v", "i2v"], {"default": "i2v"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "attention_type": (
                    ["sdpa", "flash_attn2", "flash_attn3"],
                    {"default": "flash_attn3"},
                ),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "offload_granularity": (
                    ["block", "phase"],
                    {"default": "phase"},
                ),
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
    CATEGORY = "LightX2V"

    def load_model(
        self,
        model_name: str,
        model_type: str,
        precision: str,
        device: str,
        attention_type: str,
        offload_granularity: str,
        mm_type: str | None = None,
        lora_path: str | None = None,
        lora_strength: float = 1.0,
        cpu_offload: bool = False,
        teacache_args: dict[str, Any] | None = None,
        model_dir: str | None = None,
    ):
        dtype = convert_dtype(precision)

        if device == "cuda":
            init_device = comfy_mm.get_torch_device()
        else:
            init_device = torch.device("cpu")

        if model_dir:
            model_path = Path(model_dir) / model_name
        else:
            model_path = Path(model_name)
            assert model_path.exists(), f"Model path {model_path} does not exist. Please provide a valid model path or set model_dir."

        if model_path.is_dir():
            config_json_path = model_path / "config.json"
        else:
            config_json_path = model_path.parent / "config.json"

        config_json = {}
        if config_json_path.exists():
            with open(config_json_path, "r") as f:
                config_json = json.load(f)
        else:
            logging.error(f"Config file not found at {config_json_path}")
            raise FileNotFoundError(f"Config file not found at {config_json_path}")

        feature_caching = "Tea" if teacache_args is not None else "NoCaching"
        teacache_thresh = teacache_args["rel_l1_thresh"] if teacache_args else 0.26
        use_ret_steps = teacache_args["use_ret_steps"] if teacache_args else False
        coefficients = teacache_args["coefficients"] if teacache_args else []

        mm_config = {}
        try:
            if mm_type:
                mm_config = json.loads(mm_type)
        except Exception as e:
            logging.error(f"Invalid mm_type config {mm_type}, error:{e}")
            mm_config = {}

        # 创建配置字典
        config = {
            "do_mm_calib": False,
            "cpu_offload": cpu_offload,
            "parallel_attn_type": None,  # [None, "ulysses", "ring"]
            "parallel_vae": False,
            "max_area": False,
            "vae_stride": (4, 8, 8),
            "patch_size": (1, 2, 2),
            "feature_caching": feature_caching,  # ["NoCaching", "TaylorSeer", "Tea"]
            "teacache_thresh": teacache_thresh,
            "use_ret_steps": use_ret_steps,
            "coefficients": coefficients,
            "use_bfloat16": dtype == torch.bfloat16,
            "mm_config": mm_config,
            "model_path": str(model_path),
            "task": model_type,
            "model_cls": "wan2.1",
            "device": init_device,
            "attention_type": attention_type,
            "lora_path": lora_path if lora_path and lora_path.strip() else None,
            "strength_model": lora_strength,
            "offload_granularity": offload_granularity,
        }

        config.update(**config_json)
        config = EasyDict(config)  # NOTE(xxx): adapt to Lightx2v
        logging.info(f"Loaded config:\n {config}")
        logging.info(f"Loading WanModel from {model_path} with type {model_type}")
        model = WanModel(model_path, config, init_device)

        # 如果指定了LoRA路径，应用LoRA
        if lora_path and os.path.exists(lora_path):
            logging.info(f"Applying LoRA from {lora_path} with strength {lora_strength}")
            lora_wrapper = WanLoraWrapper(model)
            lora_name = lora_wrapper.load_lora(lora_path)
            lora_wrapper.apply_lora(lora_name, lora_strength)
            logging.info(f"LoRA {lora_name} applied successfully")

        wan_model = {"wan_model": model, "config": config}

        return (wan_model,)


class Lightx2vWanVideoSampler:
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
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
            }
        }

    RETURN_TYPES = ("LIGHT_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "LightX2V"

    @no_type_check
    def sample(
        self,
        model: dict[str, WanModel | dict[str, Any]],
        text_embeddings: dict[str, Any],
        steps: int,
        shift: float,
        cfg_scale: float,
        seed: int,
        image_embeddings: dict[str, Any],
    ):
        model_config = cast(EasyDict, model.get("config"))
        model_config.update(image_embeddings.get("config", {}))

        wan_model = cast(WanModel, model.get("wan_model"))
        clip_encoder_out = image_embeddings.get("clip_encoder_out", None)
        vae_encode_out = image_embeddings.get("vae_encode_out", None)

        if model_config.task == "i2v" and (clip_encoder_out is None or vae_encode_out is None):  # type: ignore
            raise ValueError("clip_encoder_out must be provided for i2v task")

        model_config.infer_steps = steps
        model_config.sample_shift = shift
        model_config.sample_guide_scale = cfg_scale
        model_config.seed = seed

        model_config.enable_cfg = False if math.isclose(cfg_scale, 1.0) else True
        logging.info(f"Loaded update config:\n {model_config}")

        # wan_runner.set_target_shape
        num_channels_latents = model_config.get("num_channels_latents", 16)

        if model_config.task == "i2v":
            model_config.target_shape = (
                num_channels_latents,
                (model_config.target_video_length - 1) // model_config.vae_stride[0] + 1,
                model_config.lat_h,
                model_config.lat_w,
            )
        elif model_config.task == "t2v":  # type: ignore
            model_config.target_shape = (
                16,
                (model_config.target_video_length - 1) // 4 + 1,
                int(model_config.target_height) // model_config.vae_stride[1],
                int(model_config.target_width) // model_config.vae_stride[2],
            )

        # wan_runner.init_scheduler
        if model_config.feature_caching == "NoCaching":
            scheduler = WanScheduler(model_config)
        elif model_config.feature_caching == "Tea":
            scheduler = WanSchedulerTeaCaching(model_config)
        else:
            raise NotImplementedError(
                f"Unsupported feature_caching type: {model_config.feature_caching}"  # type:ignore
            )

        # setup scheduler
        wan_model.set_scheduler(scheduler)

        # Set up inputs
        inputs = {
            "text_encoder_output": text_embeddings,
            "image_encoder_output": image_embeddings,
        }

        # Prepare for sampling
        scheduler.prepare(inputs.get("image_encoder_output"))

        # Run sampling
        progress = ProgressBar(steps)
        for step_index in tqdm(range(scheduler.infer_steps), desc="inference", unit="step"):
            scheduler.step_pre(step_index=step_index)
            with ProfilingContext("model.infer"):
                wan_model.infer(inputs)
            scheduler.step_post()

            progress.update(1)

        latents, generator = scheduler.latents, scheduler.generator
        scheduler.clear()
        del inputs, scheduler, text_embeddings, image_embeddings
        torch.cuda.empty_cache()

        return ({"samples": latents, "generator": generator},)


# Register the nodes
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
    "Lightx2vWanVideoClipVisionEncoder": "LightX2V WAN CLIP Vision Encoder",
    "Lightx2vWanVideoVaeLoader": "LightX2V WAN VAE Loader",
    "Lightx2vWanVideoImageEncoder": "LightX2V WAN Image Encoder",
    "Lightx2vWanVideoVaeDecoder": "LightX2V WAN VAE Decoder",
    "Lightx2vWanVideoModelLoader": "LightX2V WAN Model Loader",
    "Lightx2vWanVideoSampler": "LightX2V WAN Video Sampler",
    "Lightx2vTeaCache": "LightX2V WAN Tea Cache",
    "Lightx2vWanVideoEmptyEmbeds": "LightX2V WAN Video Empty Embeds",
}
