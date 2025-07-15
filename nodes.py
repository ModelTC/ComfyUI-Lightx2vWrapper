"""Modular ComfyUI nodes for LightX2V without presets."""

import asyncio
import gc
import logging
import os
import tempfile
from typing import Any, Dict

import numpy as np
import torch
from comfy.utils import ProgressBar
from PIL import Image

from .bridge import ModularConfigManager, get_available_attn_ops, get_available_quant_ops
from .lightx2v.lightx2v.infer import init_runner


class LightX2VInferenceConfig:
    """Basic inference configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_cls": (["wan2.1", "hunyuan"], {"default": "wan2.1", "tooltip": "模型类型"}),
                "model_path": ("STRING", {"default": "", "tooltip": "模型路径"}),
                "task": (["t2v", "i2v"], {"default": "t2v", "tooltip": "任务类型：文本到视频或图像到视频"}),
                "infer_steps": ("INT", {"default": 40, "min": 1, "max": 100, "tooltip": "推理步数"}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2**32 - 1, "tooltip": "随机种子，-1为随机"}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "CFG引导强度"}),
                "sample_shift": ("INT", {"default": 5, "min": 0, "max": 10, "tooltip": "采样偏移"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8, "tooltip": "视频高度"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "视频宽度"}),
                "video_length": ("INT", {"default": 81, "min": 16, "max": 120, "tooltip": "视频帧数"}),
                "fps": ("INT", {"default": 16, "min": 8, "max": 30, "tooltip": "每秒帧数"}),
            }
        }

    RETURN_TYPES = ("INFERENCE_CONFIG",)
    RETURN_NAMES = ("inference_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(self, model_cls, model_path, task, infer_steps, seed, cfg_scale, sample_shift, height, width, video_length, fps):
        """Create basic inference configuration."""
        config = {
            "model_cls": model_cls,
            "model_path": model_path,
            "task": task,
            "infer_steps": infer_steps,
            "seed": seed if seed != -1 else np.random.randint(0, 2**32 - 1),
            "cfg_scale": cfg_scale,
            "sample_shift": sample_shift,
            "height": height,
            "width": width,
            "video_length": video_length,
            "fps": fps,
        }
        return (config,)


class LightX2VTeaCache:
    """TeaCache configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": False, "tooltip": "启用TeaCache特征缓存"}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.26, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "缓存阈值，越低加速越多：0.1约2倍加速，0.2约3倍加速"},
                ),
                "cache_key_steps_only": ("BOOLEAN", {"default": False, "tooltip": "只缓存关键步骤以平衡质量和速度"}),
            }
        }

    RETURN_TYPES = ("TEACACHE_CONFIG",)
    RETURN_NAMES = ("teacache_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(self, enable, threshold, cache_key_steps_only):
        """Create TeaCache configuration."""
        config = {
            "enable": enable,
            "threshold": threshold,
            "cache_key_steps_only": cache_key_steps_only,
        }
        return (config,)


class LightX2VQuantization:
    """Quantization configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        # Get available quantization backends
        available_ops = get_available_quant_ops()
        quant_backends = []

        for op_name, is_available in available_ops:
            if is_available:
                quant_backends.append(op_name)

        # Always have at least one option
        if not quant_backends:
            quant_backends = ["none"]

        return {
            "required": {
                "dit_precision": (["bf16", "int8", "fp8"], {"default": "bf16", "tooltip": "DIT模型量化精度"}),
                "t5_precision": (["bf16", "int8", "fp8"], {"default": "bf16", "tooltip": "T5编码器量化精度"}),
                "clip_precision": (["fp16", "int8", "fp8"], {"default": "fp16", "tooltip": "CLIP编码器量化精度"}),
                "quant_backend": (quant_backends, {"default": quant_backends[0], "tooltip": "量化计算后端"}),
                "sensitive_layers_precision": (["fp32", "bf16"], {"default": "fp32", "tooltip": "敏感层（归一化和嵌入层）精度"}),
            }
        }

    RETURN_TYPES = ("QUANT_CONFIG",)
    RETURN_NAMES = ("quantization_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(self, dit_precision, t5_precision, clip_precision, quant_backend, sensitive_layers_precision):
        """Create quantization configuration."""
        config = {
            "dit_precision": dit_precision,
            "t5_precision": t5_precision,
            "clip_precision": clip_precision,
            "quant_backend": quant_backend,
            "sensitive_layers_precision": sensitive_layers_precision,
        }
        return (config,)


class LightX2VMemoryOptimization:
    """Memory optimization configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        # Get available attention types
        available_attn = get_available_attn_ops()
        attn_types = []

        for op_name, is_available in available_attn:
            if is_available:
                attn_types.append(op_name)

        # Always include fallback
        if "torch_sdpa" not in attn_types:
            attn_types.append("torch_sdpa")

        return {
            "required": {
                "optimization_level": (
                    ["none", "low", "medium", "high", "extreme"],
                    {"default": "none", "tooltip": "内存优化级别，越高越省内存但可能影响速度"},
                ),
                "attention_type": (attn_types, {"default": attn_types[0], "tooltip": "注意力机制类型"}),
            },
            "optional": {
                # GPU optimization
                "enable_rotary_chunk": ("BOOLEAN", {"default": False, "tooltip": "启用旋转编码分块"}),
                "rotary_chunk_size": ("INT", {"default": 100, "min": 100, "max": 10000, "step": 100}),
                "clean_cuda_cache": ("BOOLEAN", {"default": False, "tooltip": "及时清理CUDA缓存"}),
                # CPU offloading
                "enable_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "启用CPU卸载"}),
                "offload_granularity": (["block", "phase"], {"default": "phase", "tooltip": "卸载粒度"}),
                "offload_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                # Module management
                "lazy_load": ("BOOLEAN", {"default": False, "tooltip": "延迟加载模型"}),
                "unload_after_inference": ("BOOLEAN", {"default": False, "tooltip": "推理后卸载模块"}),
            },
        }

    RETURN_TYPES = ("MEMORY_CONFIG",)
    RETURN_NAMES = ("memory_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self,
        optimization_level,
        attention_type,
        enable_rotary_chunk=False,
        rotary_chunk_size=100,
        clean_cuda_cache=False,
        enable_cpu_offload=False,
        offload_granularity="phase",
        offload_ratio=1.0,
        lazy_load=False,
        unload_after_inference=False,
    ):
        """Create memory optimization configuration."""
        config = {
            "optimization_level": optimization_level,
            "attention_type": attention_type,
            "enable_rotary_chunk": enable_rotary_chunk,
            "rotary_chunk_size": rotary_chunk_size,
            "clean_cuda_cache": clean_cuda_cache,
            "enable_cpu_offload": enable_cpu_offload,
            "offload_granularity": offload_granularity,
            "offload_ratio": offload_ratio,
            "lazy_load": lazy_load,
            "unload_after_inference": unload_after_inference,
        }
        return (config,)


class LightX2VLightweightVAE:
    """Lightweight VAE configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_tiny_vae": ("BOOLEAN", {"default": False, "tooltip": "使用轻量级VAE加速解码"}),
                "use_tiling_vae": ("BOOLEAN", {"default": False, "tooltip": "使用VAE分块推理减少显存"}),
            }
        }

    RETURN_TYPES = ("VAE_CONFIG",)
    RETURN_NAMES = ("vae_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(self, use_tiny_vae, use_tiling_vae):
        """Create VAE configuration."""
        config = {
            "use_tiny_vae": use_tiny_vae,
            "use_tiling_vae": use_tiling_vae,
        }
        return (config,)


class LightX2VModularInference:
    """Modular inference node that combines all configurations."""

    def __init__(self):
        self.config_manager = ModularConfigManager()
        self._current_runner = None
        self._current_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_config": ("INFERENCE_CONFIG", {"tooltip": "基础推理配置"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "生成提示词"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "负面提示词"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "i2v任务的输入图像"}),
                "teacache_config": ("TEACACHE_CONFIG", {"tooltip": "TeaCache配置"}),
                "quantization_config": ("QUANT_CONFIG", {"tooltip": "量化配置"}),
                "memory_config": ("MEMORY_CONFIG", {"tooltip": "内存优化配置"}),
                "vae_config": ("VAE_CONFIG", {"tooltip": "VAE配置"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "LightX2V/Inference"

    def _get_config_hash(self, configs: Dict[str, Any]) -> str:
        """Generate a hash for configuration to detect changes."""
        import hashlib
        import json

        # Only hash model-related configs
        relevant_configs = {
            "model_cls": configs.get("inference", {}).get("model_cls"),
            "model_path": configs.get("inference", {}).get("model_path"),
            "quantization": configs.get("quantization"),
            "memory_lazy_load": configs.get("memory", {}).get("lazy_load"),
        }

        config_str = json.dumps(relevant_configs, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def generate(
        self,
        inference_config,
        prompt,
        negative_prompt,
        image=None,
        teacache_config=None,
        quantization_config=None,
        memory_config=None,
        vae_config=None,
        **kwargs,
    ):
        """Generate video using modular configuration."""

        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "DTYPE" not in os.environ:
            os.environ["DTYPE"] = "BF16"
        if "ENABLE_GRAPH_MODE" not in os.environ:
            os.environ["ENABLE_GRAPH_MODE"] = "false"
        if "ENABLE_PROFILING_DEBUG" not in os.environ:
            os.environ["ENABLE_PROFILING_DEBUG"] = "true"

        # Collect all configurations
        configs = {
            "inference": inference_config,
        }

        if teacache_config:
            configs["teacache"] = teacache_config
        if quantization_config:
            configs["quantization"] = quantization_config
        if memory_config:
            configs["memory"] = memory_config
        if vae_config:
            configs["vae"] = vae_config

        # Build final configuration
        config = self.config_manager.build_final_config(configs)

        # Add prompt and negative prompt
        config.prompt = prompt
        config.negative_prompt = negative_prompt

        # Check if task requires image
        if config.task == "i2v" and image is None:
            raise ValueError("i2v task requires input image")

        temp_files = []

        try:
            # Handle image input for i2v
            if config.task == "i2v" and image is not None:
                # Convert ComfyUI image to PIL
                image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)

                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pil_image.save(tmp.name)
                    config.image_path = tmp.name
                    temp_files.append(tmp.name)

            # Check if we need to reinitialize runner
            config_hash = self._get_config_hash(configs)
            needs_reinit = (
                self._current_runner is None or self._current_config_hash != config_hash or configs.get("memory", {}).get("lazy_load", False)
            )

            if needs_reinit:
                # Clear old runner
                if self._current_runner is not None:
                    del self._current_runner
                    torch.cuda.empty_cache()
                    gc.collect()

                # Initialize new runner
                self._current_runner = init_runner(config)
                self._current_config_hash = config_hash
            else:
                # Update config for existing runner
                self._current_runner.config = config

            # Set up progress callback
            total_steps = config.get("infer_steps", 40)
            progress = ProgressBar(total_steps)

            def update_progress(current_step, total):
                progress.update_absolute(current_step)

            self._current_runner.set_progress_callback(update_progress)

            # Run inference
            images = asyncio.run(self._current_runner.run_pipeline(save_video=False))

            # Clean up if requested
            if configs.get("memory", {}).get("unload_after_inference", False):
                del self._current_runner
                self._current_runner = None
                self._current_config_hash = None

            torch.cuda.empty_cache()
            gc.collect()

            # Convert output to ComfyUI format
            images = (images + 1) / 2
            images = images.squeeze(0).permute(1, 2, 3, 0).cpu()
            images = torch.clamp(images, 0, 1)

            return (images,)

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LightX2VInferenceConfig": LightX2VInferenceConfig,
    "LightX2VTeaCache": LightX2VTeaCache,
    "LightX2VQuantization": LightX2VQuantization,
    "LightX2VMemoryOptimization": LightX2VMemoryOptimization,
    "LightX2VLightweightVAE": LightX2VLightweightVAE,
    "LightX2VModularInference": LightX2VModularInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightX2VInferenceConfig": "LightX2V 推理配置",
    "LightX2VTeaCache": "LightX2V TeaCache缓存",
    "LightX2VQuantization": "LightX2V 低精度量化",
    "LightX2VMemoryOptimization": "LightX2V 内存优化",
    "LightX2VLightweightVAE": "LightX2V 轻量VAE",
    "LightX2VModularInference": "LightX2V 模块化推理",
}
