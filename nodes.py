"""Modular ComfyUI nodes for LightX2V without presets."""

import gc
import hashlib
import json
import logging
import os
import tempfile

import numpy as np
import torch
from comfy.utils import ProgressBar
from PIL import Image

from .bridge import ModularConfigManager, get_available_attn_ops, get_available_quant_ops
from .lightx2v.lightx2v.infer import init_runner
from .model_utils import get_lora_full_path, get_model_full_path, scan_loras, scan_models


class LightX2VInferenceConfig:
    @classmethod
    def INPUT_TYPES(cls):
        available_models = scan_models()

        return {
            "required": {
                "model_cls": (["wan2.1", "wan2.1_audio", "wan2.1_distill", "hunyuan"], {"default": "wan2.1", "tooltip": "Model type"}),
                "model_name": (available_models, {"default": available_models[0], "tooltip": "Select model from available models"}),
                "task": (["t2v", "i2v"], {"default": "t2v", "tooltip": "Task type: text-to-video or image-to-video"}),
                "infer_steps": ("INT", {"default": 40, "min": 1, "max": 100, "tooltip": "Inference steps"}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2**32 - 1, "tooltip": "Random seed, -1 for random"}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "CFG guidance strength"}),
                "sample_shift": ("INT", {"default": 5, "min": 0, "max": 10, "tooltip": "Sample shift"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8, "tooltip": "Video height"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Video width"}),
                "video_length": ("INT", {"default": 81, "min": 16, "max": 120, "tooltip": "Video frame count"}),
                "fps": ("INT", {"default": 16, "min": 8, "max": 30, "tooltip": "Model output frame rate (cannot be changed)"}),
            },
            "optional": {
                "denoising_steps": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Custom denoising steps for distillation models (comma-separated, e.g., '999,750,500,250'). Leave empty to use model defaults.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INFERENCE_CONFIG",)
    RETURN_NAMES = ("inference_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self, model_cls, model_name, task, infer_steps, seed, cfg_scale, sample_shift, height, width, video_length, fps, denoising_steps=""
    ):
        """Create basic inference configuration."""
        model_path = get_model_full_path(model_name)

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

        if denoising_steps and denoising_steps.strip():
            try:
                steps_list = [int(s.strip()) for s in denoising_steps.split(",")]
                config["denoising_step_list"] = steps_list
                config["infer_steps"] = len(steps_list)
            except ValueError:
                pass

        return (config,)


class LightX2VTeaCache:
    """TeaCache configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": False, "tooltip": "Enable TeaCache feature caching"}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.26,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Cache threshold, lower values provide more speedup: 0.1 ~2x speedup, 0.2 ~3x speedup",
                    },
                ),
                "use_ret_steps": ("BOOLEAN", {"default": False, "tooltip": "Only cache key steps to balance quality and speed"}),
            }
        }

    RETURN_TYPES = ("TEACACHE_CONFIG",)
    RETURN_NAMES = ("teacache_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(self, enable, threshold, use_ret_steps):
        config = {
            "enable": enable,
            "threshold": threshold,
            "use_ret_steps": use_ret_steps,
        }
        return (config,)


class LightX2VQuantization:
    @classmethod
    def INPUT_TYPES(cls):
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
                "dit_precision": (["bf16", "int8", "fp8"], {"default": "bf16", "tooltip": "DIT model quantization precision"}),
                "t5_precision": (["bf16", "int8", "fp8"], {"default": "bf16", "tooltip": "T5 encoder quantization precision"}),
                "clip_precision": (["fp16", "int8", "fp8"], {"default": "fp16", "tooltip": "CLIP encoder quantization precision"}),
                "quant_backend": (quant_backends, {"default": quant_backends[0], "tooltip": "Quantization computation backend"}),
                "sensitive_layers_precision": (
                    ["fp32", "bf16"],
                    {"default": "fp32", "tooltip": "Sensitive layers (normalization and embedding) precision"},
                ),
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
        available_attn = get_available_attn_ops()
        attn_types = []

        for op_name, is_available in available_attn:
            if is_available:
                attn_types.append(op_name)

        if "torch_sdpa" not in attn_types:
            attn_types.append("torch_sdpa")

        return {
            "required": {
                "optimization_level": (
                    ["none", "low", "medium", "high", "extreme"],
                    {"default": "none", "tooltip": "Memory optimization level, higher levels save more memory but may affect speed"},
                ),
                "attention_type": (attn_types, {"default": attn_types[0], "tooltip": "Attention mechanism type"}),
            },
            "optional": {
                # GPU optimization
                "enable_rotary_chunk": ("BOOLEAN", {"default": False, "tooltip": "Enable rotary encoding chunking"}),
                "rotary_chunk_size": ("INT", {"default": 100, "min": 100, "max": 10000, "step": 100}),
                "clean_cuda_cache": ("BOOLEAN", {"default": False, "tooltip": "Clean CUDA cache promptly"}),
                # CPU offloading
                "enable_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enable CPU offloading"}),
                "offload_granularity": (["block", "phase"], {"default": "phase", "tooltip": "Offload granularity"}),
                "offload_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                # Module management
                "lazy_load": ("BOOLEAN", {"default": False, "tooltip": "Lazy load model"}),
                "unload_after_inference": ("BOOLEAN", {"default": False, "tooltip": "Unload modules after inference"}),
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_tiny_vae": ("BOOLEAN", {"default": False, "tooltip": "Use lightweight VAE to accelerate decoding"}),
                "use_tiling_vae": ("BOOLEAN", {"default": False, "tooltip": "Use VAE tiling inference to reduce VRAM usage"}),
            }
        }

    RETURN_TYPES = ("VAE_CONFIG",)
    RETURN_NAMES = ("vae_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(self, use_tiny_vae, use_tiling_vae):
        config = {
            "use_tiny_vae": use_tiny_vae,
            "use_tiling_vae": use_tiling_vae,
        }
        return (config,)


class LightX2VLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        available_loras = scan_loras()

        return {
            "required": {
                "lora_name": (available_loras, {"default": available_loras[0], "tooltip": "Select LoRA from available LoRAs"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "LoRA strength"}),
            },
            "optional": {
                "lora_chain": ("LORA_CHAIN", {"tooltip": "Previous LoRA chain to append to"}),
            },
        }

    RETURN_TYPES = ("LORA_CHAIN",)
    RETURN_NAMES = ("lora_chain",)
    FUNCTION = "load_lora"
    CATEGORY = "LightX2V/LoRA"

    def load_lora(self, lora_name, strength, lora_chain=None):
        if lora_chain is None:
            lora_chain = []
        else:
            lora_chain = lora_chain.copy()

        lora_path = get_lora_full_path(lora_name)

        if lora_path:
            lora_config = {"path": lora_path, "strength": strength}
            lora_chain.append(lora_config)

        return (lora_chain,)


class LightX2VConfigCombiner:
    def __init__(self):
        self.config_manager = ModularConfigManager()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_config": ("INFERENCE_CONFIG", {"tooltip": "Basic inference configuration"}),
            },
            "optional": {
                "teacache_config": ("TEACACHE_CONFIG", {"tooltip": "TeaCache configuration"}),
                "quantization_config": ("QUANT_CONFIG", {"tooltip": "Quantization configuration"}),
                "memory_config": ("MEMORY_CONFIG", {"tooltip": "Memory optimization configuration"}),
                "vae_config": ("VAE_CONFIG", {"tooltip": "VAE configuration"}),
                "lora_chain": ("LORA_CHAIN", {"tooltip": "LoRA chain configuration"}),
            },
        }

    RETURN_TYPES = ("COMBINED_CONFIG",)
    RETURN_NAMES = ("combined_config",)
    FUNCTION = "combine_configs"
    CATEGORY = "LightX2V/Config"

    def combine_configs(
        self,
        inference_config,
        teacache_config=None,
        quantization_config=None,
        memory_config=None,
        vae_config=None,
        lora_chain=None,
    ):
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

        config = self.config_manager.build_final_config(configs)

        if lora_chain:
            config.lora_configs = lora_chain

        return (config,)


class LightX2VModularInference:
    def __init__(self):
        self._current_runner = None
        self._current_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combined_config": ("COMBINED_CONFIG", {"tooltip": "Combined configuration from config combiner"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Generation prompt"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Input image for i2v task"}),
                "audio": ("AUDIO", {"tooltip": "Input audio for audio-driven generation"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "LightX2V/Inference"

    def _get_config_hash(self, config) -> str:
        relevant_configs = {
            "model_cls": getattr(config, "model_cls", None),
            "model_path": getattr(config, "model_path", None),
            "dit_quantized": getattr(config, "dit_quantized", False),
            "t5_quantized": getattr(config, "t5_quantized", False),
            "clip_quantized": getattr(config, "clip_quantized", False),
            "lora_configs": getattr(config, "lora_configs", None),
            "lazy_load": getattr(config, "lazy_load", False),
        }

        config_str = json.dumps(relevant_configs, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def generate(
        self,
        combined_config,
        prompt,
        negative_prompt,
        image=None,
        audio=None,
        **kwargs,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "DTYPE" not in os.environ:
            os.environ["DTYPE"] = "BF16"
        if "ENABLE_GRAPH_MODE" not in os.environ:
            os.environ["ENABLE_GRAPH_MODE"] = "false"
        if "ENABLE_PROFILING_DEBUG" not in os.environ:
            os.environ["ENABLE_PROFILING_DEBUG"] = "true"

        config = combined_config

        config.prompt = prompt
        config.negative_prompt = negative_prompt

        if config.task == "i2v" and image is None:
            raise ValueError("i2v task requires input image")

        temp_files = []

        try:
            if config.task == "i2v" and image is not None:
                image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pil_image.save(tmp.name)
                    config.image_path = tmp.name
                    temp_files.append(tmp.name)

            if audio is not None and hasattr(config, "model_cls") and "audio" in config.model_cls:
                if isinstance(audio, tuple) and len(audio) == 2:
                    waveform, sample_rate = audio

                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.cpu().numpy()

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        try:
                            import scipy.io.wavfile as wavfile
                        except ImportError:
                            import wave

                            with wave.open(tmp.name, "wb") as wav_file:
                                wav_file.setnchannels(1 if waveform.ndim == 1 else waveform.shape[-1])
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(sample_rate)
                                if waveform.dtype != np.int16:
                                    waveform = (waveform * 32767).astype(np.int16)
                                wav_file.writeframes(waveform.tobytes())
                        else:
                            if waveform.ndim == 1:
                                wavfile.write(tmp.name, sample_rate, waveform)
                            else:
                                if waveform.shape[0] < waveform.shape[1]:
                                    waveform = waveform.T
                                wavfile.write(tmp.name, sample_rate, waveform)

                        config.audio_path = tmp.name
                        temp_files.append(tmp.name)

            config_hash = self._get_config_hash(config)
            needs_reinit = self._current_runner is None or self._current_config_hash != config_hash or getattr(config, "lazy_load", False)

            if needs_reinit:
                if self._current_runner is not None:
                    del self._current_runner
                    torch.cuda.empty_cache()
                    gc.collect()

                self._current_runner = init_runner(config)
                self._current_config_hash = config_hash
            else:
                if hasattr(self._current_runner, "config"):
                    self._current_runner.config = config

            total_steps = getattr(config, "infer_steps", 40)
            progress = ProgressBar(total_steps)

            def update_progress(current_step, total):
                progress.update_absolute(current_step)

            if hasattr(self._current_runner, "set_progress_callback"):
                self._current_runner.set_progress_callback(update_progress)

            if hasattr(self._current_runner, "run_pipeline"):
                images = self._current_runner.run_pipeline(save_video=False)
            else:
                images = self._current_runner()

            if getattr(config, "unload_after_inference", False):
                del self._current_runner
                self._current_runner = None
                self._current_config_hash = None

            torch.cuda.empty_cache()
            gc.collect()

            images = (images + 1) / 2
            images = images.squeeze(0).permute(1, 2, 3, 0).cpu()
            images = torch.clamp(images, 0, 1)

            return (images,)

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise

        # finally:
        #     for temp_file in temp_files:
        #         if os.path.exists(temp_file):
        #             try:
        #                 os.unlink(temp_file)
        #             except Exception:
        #                 pass


NODE_CLASS_MAPPINGS = {
    "LightX2VInferenceConfig": LightX2VInferenceConfig,
    "LightX2VTeaCache": LightX2VTeaCache,
    "LightX2VQuantization": LightX2VQuantization,
    "LightX2VMemoryOptimization": LightX2VMemoryOptimization,
    "LightX2VLightweightVAE": LightX2VLightweightVAE,
    "LightX2VLoRALoader": LightX2VLoRALoader,
    "LightX2VConfigCombiner": LightX2VConfigCombiner,
    "LightX2VModularInference": LightX2VModularInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightX2VInferenceConfig": "LightX2V Inference Config",
    "LightX2VTeaCache": "LightX2V TeaCache",
    "LightX2VQuantization": "LightX2V Quantization",
    "LightX2VMemoryOptimization": "LightX2V Memory Optimization",
    "LightX2VLightweightVAE": "LightX2V Lightweight VAE",
    "LightX2VLoRALoader": "LightX2V LoRA Loader",
    "LightX2VConfigCombiner": "LightX2V Config Combiner",
    "LightX2VModularInference": "LightX2V Modular Inference",
}
