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

from .bridge import (
    ModularConfigManager,
    get_available_attn_ops,
    get_available_quant_ops,
)
from .lightx2v.lightx2v.infer import init_runner
from .model_utils import (
    get_lora_full_path,
    get_model_full_path,
    scan_loras,
    scan_models,
    support_model_cls_list,
)


class LightX2VInferenceConfig:
    @classmethod
    def INPUT_TYPES(cls):
        available_models = scan_models()
        support_model_classes = support_model_cls_list()
        available_attn = get_available_attn_ops()
        attn_types = []

        for op_name, is_available in available_attn:
            if is_available:
                attn_types.append(op_name)

        if "torch_sdpa" not in attn_types:
            attn_types.append("torch_sdpa")

        return {
            "required": {
                "model_cls": (
                    support_model_classes,
                    {"default": "wan2.1", "tooltip": "Model type"},
                ),
                "model_name": (
                    available_models,
                    {
                        "default": available_models[0],
                        "tooltip": "Select model from available models",
                    },
                ),
                "task": (
                    ["t2v", "i2v"],
                    {
                        "default": "i2v",
                        "tooltip": "Task type: text-to-video or image-to-video",
                    },
                ),
                "infer_steps": (
                    "INT",
                    {"default": 4, "min": 1, "max": 100, "tooltip": "Inference steps"},
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
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "CFG guidance strength",
                    },
                ),
                "cfg_scale2": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "CFG guidance, lower noise when model cls is Wan2.2 MoE",
                    },
                ),
                "sample_shift": (
                    "INT",
                    {"default": 5, "min": 0, "max": 10, "tooltip": "Sample shift"},
                ),
                "height": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Video height",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 720,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Video width",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 999,
                        "step": 0.1,
                        "tooltip": "Video duration in seconds",
                    },
                ),
                "attention_type": (
                    attn_types,
                    {"default": attn_types[0], "tooltip": "Attention mechanism type"},
                ),
            },
            "optional": {
                "denoising_steps": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Custom denoising steps for distillation models (comma-separated, e.g., '999,750,500,250'). Leave empty to use model defaults.",
                    },
                ),
                "resize_mode": (
                    ["adaptive", "keep_ratio_fixed_area", "fixed_min_area", "fixed_max_area", "fixed_shape", "fixed_min_side"],
                    {
                        "default": "adaptive",
                        "tooltip": "Adaptive resize input image to target aspect ratio",
                    },
                ),
                "fixed_area": (
                    "STRING",
                    {
                        "default": "720p",
                        "tooltip": "Fixed shape for input image, e.g., '720p', '480p', when resize_mode is 'keep_ratio_fixed_area' or 'fixed_min_side'",
                    },
                ),
                "segment_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 16,
                        "max": 256,
                        "tooltip": "Segment length in frames for sekotalk models (target_video_length)",
                    },
                ),
                "prev_frame_length": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 16,
                        "tooltip": "Previous frame overlap for sekotalk models",
                    },
                ),
                "use_tiny_vae": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use lightweight VAE to accelerate decoding",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INFERENCE_CONFIG",)
    RETURN_NAMES = ("inference_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self,
        model_cls,
        model_name,
        task,
        infer_steps,
        seed,
        cfg_scale,
        cfg_scale2,
        sample_shift,
        height,
        width,
        duration,
        attention_type,
        denoising_steps="",
        resize_mode="adaptive",
        fixed_area="720p",
        segment_length=81,
        prev_frame_length=5,
        use_tiny_vae=False,
    ):
        """Create basic inference configuration."""
        model_path = get_model_full_path(model_name)

        if model_cls == "hunyuan":
            fps = 24
        else:
            fps = 16

        video_length = int(round(duration * fps))

        if video_length < 16:
            logging.warning("Video length is too short, setting to 16")
            video_length = 16

        remainder = (video_length - 1) % 4
        if remainder != 0:
            video_length = video_length + (4 - remainder)

        # TODO(xxx):
        use_31_block = True
        if "seko" in model_cls:
            video_length = segment_length
            use_31_block = False

        config = {
            "model_cls": model_cls,
            "model_path": model_path,
            "task": task,
            "infer_steps": infer_steps,
            "seed": seed if seed != -1 else np.random.randint(0, 2**32 - 1),
            "cfg_scale": cfg_scale,
            "cfg_scale2": cfg_scale2,
            "sample_shift": sample_shift,
            "height": height,
            "width": width,
            "video_length": video_length,
            "fps": fps,
            "video_duration": duration,
            "resize_mode": resize_mode,
            "fixed_area": fixed_area,
            "use_31_block": use_31_block,
            "attention_type": attention_type,
            "use_tiny_vae": use_tiny_vae,
        }
        if "seko" in [model_cls]:
            config["prev_frame_length"] = prev_frame_length

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
                "enable": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable TeaCache feature caching"},
                ),
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
                "use_ret_steps": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only cache key steps to balance quality and speed",
                    },
                ),
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

        supported_quant_schemes = ["bf16", "fp16", "fp8", "int8"]

        return {
            "required": {
                "quant_op": (
                    quant_backends,
                    {
                        "default": quant_backends[0],
                        "tooltip": "Quantization computation backend",
                    },
                ),
                "dit_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[0],
                        "tooltip": "DIT model quantization precision",
                    },
                ),
                "t5_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[0],
                        "tooltip": "T5 encoder quantization precision",
                    },
                ),
                "clip_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[1],
                        "tooltip": "CLIP encoder quantization precision",
                    },
                ),
                "adapter_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[0],
                        "tooltip": "Adapter quantization precision",
                    },
                ),
            }
        }

    RETURN_TYPES = ("QUANT_CONFIG",)
    RETURN_NAMES = ("quantization_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self,
        quant_op,
        dit_quant_scheme,
        t5_quant_scheme,
        clip_quant_scheme,
        adapter_quant_scheme,
    ):
        """Create quantization configuration."""
        config = {
            "dit_quant_scheme": dit_quant_scheme,
            "t5_quant_scheme": t5_quant_scheme,
            "clip_quant_scheme": clip_quant_scheme,
            "adapter_quant_scheme": adapter_quant_scheme,
            "quant_op": quant_op,
        }
        return (config,)


class LightX2VMemoryOptimization:
    """Memory optimization configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_rotary_chunk": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable rotary encoding chunking"},
                ),
                "rotary_chunk_size": (
                    "INT",
                    {"default": 100, "min": 100, "max": 10000, "step": 100},
                ),
                "clean_cuda_cache": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Clean CUDA cache promptly"},
                ),
                "cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable CPU offloading"},
                ),
                "offload_granularity": (
                    ["block", "phase", "model"],
                    {"default": "block", "tooltip": "Offload granularity"},
                ),
                "offload_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "t5_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable T5 CPU offloading"},
                ),
                "t5_offload_granularity": (
                    ["model", "block"],
                    {"default": "model", "tooltip": "T5 offload granularity"},
                ),
                "audio_encoder_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable audio encoder CPU offloading"},
                ),
                "audio_adapter_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable audio adapter CPU offloading"},
                ),
                "vae_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable VAE CPU offloading"},
                ),
                "use_tiling_vae": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable VAE tiling inference"},
                ),
                "lazy_load": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Lazy load model"},
                ),
                "unload_after_inference": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Unload modules after inference"},
                ),
            },
        }

    RETURN_TYPES = ("MEMORY_CONFIG",)
    RETURN_NAMES = ("memory_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self,
        enable_rotary_chunk=False,
        rotary_chunk_size=100,
        clean_cuda_cache=False,
        cpu_offload=False,
        offload_granularity="phase",
        offload_ratio=1.0,
        t5_cpu_offload=True,
        t5_offload_granularity="model",
        audio_encoder_cpu_offload=False,
        audio_adapter_cpu_offload=False,
        vae_cpu_offload=False,
        use_tiling_vae=False,
        lazy_load=False,
        unload_after_inference=False,
    ):
        config = {
            "enable_rotary_chunk": enable_rotary_chunk,
            "rotary_chunk_size": rotary_chunk_size,
            "clean_cuda_cache": clean_cuda_cache,
            "cpu_offload": cpu_offload,
            "offload_granularity": offload_granularity,
            "offload_ratio": offload_ratio,
            "t5_cpu_offload": t5_cpu_offload,
            "t5_offload_granularity": t5_offload_granularity,
            "audio_encoder_cpu_offload": audio_encoder_cpu_offload,
            "audio_adapter_cpu_offload": audio_adapter_cpu_offload,
            "vae_cpu_offload": vae_cpu_offload,
            "use_tiling_vae": use_tiling_vae,
            "lazy_load": lazy_load,
            "unload_after_inference": unload_after_inference,
        }
        return (config,)


class LightX2VLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        available_loras = scan_loras()

        return {
            "required": {
                "lora_name": (
                    available_loras,
                    {
                        "default": available_loras[0],
                        "tooltip": "Select LoRA from available LoRAs",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "LoRA strength",
                    },
                ),
            },
            "optional": {
                "lora_chain": (
                    "LORA_CHAIN",
                    {"tooltip": "Previous LoRA chain to append to"},
                ),
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
                "inference_config": (
                    "INFERENCE_CONFIG",
                    {"tooltip": "Basic inference configuration"},
                ),
            },
            "optional": {
                "teacache_config": (
                    "TEACACHE_CONFIG",
                    {"tooltip": "TeaCache configuration"},
                ),
                "quantization_config": (
                    "QUANT_CONFIG",
                    {"tooltip": "Quantization configuration"},
                ),
                "memory_config": (
                    "MEMORY_CONFIG",
                    {"tooltip": "Memory optimization configuration"},
                ),
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

        config = self.config_manager.build_final_config(configs)

        if lora_chain:
            config.lora_configs = lora_chain

        logging.info("lightx2v config: " + json.dumps(config, indent=2, ensure_ascii=False))

        return (config,)


class LightX2VModularInference:
    # 类变量，所有实例共享
    _current_runner = None
    _current_config_hash = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combined_config": (
                    "COMBINED_CONFIG",
                    {"tooltip": "Combined configuration from config combiner"},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Generation prompt"},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Negative prompt"},
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Input image for i2v task"}),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Input audio for audio-driven generation"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "generate"
    CATEGORY = "LightX2V/Inference"

    def _get_config_hash(self, config) -> str:
        relevant_configs = {
            "model_cls": getattr(config, "model_cls", None),
            "model_path": getattr(config, "model_path", None),
            "task": getattr(config, "task", None),
            "t5_quantized": getattr(config, "t5_quantized", False),
            "clip_quantized": getattr(config, "clip_quantized", False),
            "lora_configs": getattr(config, "lora_configs", None),
            "mm_config": getattr(config, "mm_config", None),
            "cross_attn_1_type": getattr(config, "cross_attn_1_type", None),
            "cross_attn_2_type": getattr(config, "cross_attn_2_type", None),
            "self_attn_1_type": getattr(config, "self_attn_1_type", None),
            "self_attn_2_type": getattr(config, "self_attn_2_type", None),
            "cpu_offload": getattr(config, "cpu_offload", False),
            "offload_granularity": getattr(config, "offload_granularity", None),
            "offload_ratio": getattr(config, "offload_ratio", None),
            "t5_cpu_offload": getattr(config, "t5_cpu_offload", False),
            "t5_offload_granularity": getattr(config, "t5_offload_granularity", None),
            "audio_encoder_cpu_offload": getattr(config, "audio_encoder_cpu_offload", False),
            "audio_adapter_cpu_offload": getattr(config, "audio_adapter_cpu_offload", False),
            "vae_cpu_offload": getattr(config, "vae_cpu_offload", False),
            "use_tiling_vae": getattr(config, "use_tiling_vae", False),
            "unload_after_inference": getattr(config, "unload_after_inference", False),
            "enable_rotary_chunk": getattr(config, "enable_rotary_chunk", False),
            "rotary_chunk_size": getattr(config, "rotary_chunk_size", None),
            "clean_cuda_cache": getattr(config, "clean_cuda_cache", False),
            "torch_compile": getattr(config, "torch_compile", False),
            "threshold": getattr(config, "threshold", None),
            "use_ret_steps": getattr(config, "use_ret_steps", False),
            "t5_quant_scheme": getattr(config, "t5_quant_scheme", None),
            "clip_quant_scheme": getattr(config, "clip_quant_scheme", None),
            "adapter_quant_scheme": getattr(config, "adapter_quant_scheme", None),
            "adapter_quantized": getattr(config, "adapter_quantized", False),
            "feature_caching": getattr(config, "feature_caching", None),
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
                logging.info(f"Image saved to {tmp.name}")

            if audio is not None and hasattr(config, "model_cls") and "seko" in config.model_cls:
                if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]

                    # Handle different waveform shapes
                    if isinstance(waveform, torch.Tensor):
                        if waveform.dim() == 3:  # [batch, channels, samples]
                            waveform = waveform[0]  # Take first batch
                        if waveform.dim() == 2:  # [channels, samples]
                            # Convert to [samples, channels] for wav file
                            waveform = waveform.transpose(0, 1)
                        waveform = waveform.cpu().numpy()
                elif isinstance(audio, tuple) and len(audio) == 2:
                    # Legacy format support
                    waveform, sample_rate = audio
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.cpu().numpy()
                else:
                    raise ValueError(f"Unsupported audio format: {type(audio)}")

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

                    logging.info(f"Audio saved to {tmp.name}")

            config_hash = self._get_config_hash(config)
            needs_reinit = (
                self.__class__._current_runner is None or self.__class__._current_config_hash != config_hash or getattr(config, "lazy_load", False)
            )

            logging.info(f"Needs reinit: {needs_reinit}, old config hash: {self.__class__._current_config_hash}, new config hash: {config_hash}")
            if needs_reinit:
                if self.__class__._current_runner is not None:
                    # self.__class__._current_runner.end_run()
                    del self.__class__._current_runner
                    torch.cuda.empty_cache()
                    gc.collect()

                self.__class__._current_runner = init_runner(config)
                self.__class__._current_config_hash = config_hash
            else:
                if hasattr(self.__class__._current_runner, "config"):
                    self.__class__._current_runner.config = config

            progress = ProgressBar(100)

            def update_progress(current_step, total):
                progress.update_absolute(current_step)

            if hasattr(self.__class__._current_runner, "set_progress_callback"):
                self.__class__._current_runner.set_progress_callback(update_progress)

            result_dict = self.__class__._current_runner.run_pipeline(save_video=False)
            images = result_dict.get("video", None)
            audio = result_dict.get("audio", None)

            if getattr(config, "unload_after_inference", False):
                del self.__class__._current_runner
                self.__class__._current_runner = None
                self.__class__._current_config_hash = None

            torch.cuda.empty_cache()
            gc.collect()

            return (images, audio)

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass


NODE_CLASS_MAPPINGS = {
    "LightX2VInferenceConfig": LightX2VInferenceConfig,
    "LightX2VTeaCache": LightX2VTeaCache,
    "LightX2VQuantization": LightX2VQuantization,
    "LightX2VMemoryOptimization": LightX2VMemoryOptimization,
    "LightX2VLoRALoader": LightX2VLoRALoader,
    "LightX2VConfigCombiner": LightX2VConfigCombiner,
    "LightX2VModularInference": LightX2VModularInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightX2VInferenceConfig": "LightX2V Inference Config",
    "LightX2VTeaCache": "LightX2V TeaCache",
    "LightX2VQuantization": "LightX2V Quantization",
    "LightX2VMemoryOptimization": "LightX2V Memory Optimization",
    "LightX2VLoRALoader": "LightX2V LoRA Loader",
    "LightX2VConfigCombiner": "LightX2V Config Combiner",
    "LightX2VModularInference": "LightX2V Modular Inference",
}
