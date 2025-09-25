"""Modular ComfyUI nodes for LightX2V without presets."""

import gc
import json
import logging
import os

import numpy as np
import torch
from comfy.utils import ProgressBar
from PIL import Image

from .bridge import get_available_attn_ops, get_available_quant_ops
from .config_builder import (
    ConfigBuilder,
    InferenceConfigBuilder,
    LoRAChainBuilder,
    TalkObjectConfigBuilder,
)
from .data_models import (
    InferenceConfig,
    MemoryOptimizationConfig,
    QuantizationConfig,
    TalkObjectsConfig,
    TeaCacheConfig,
)
from .file_handlers import (
    AudioFileHandler,
    ComfyUIFileResolver,
    ImageFileHandler,
    TempFileManager,
)
from .lightx2v.lightx2v.infer import init_runner
from .model_utils import scan_loras, scan_models, support_model_cls_list


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
        builder = InferenceConfigBuilder()

        config = builder.build(
            model_cls=model_cls,
            model_name=model_name,
            task=task,
            infer_steps=infer_steps,
            seed=seed,
            cfg_scale=cfg_scale,
            cfg_scale2=cfg_scale2,
            sample_shift=sample_shift,
            height=height,
            width=width,
            duration=duration,
            attention_type=attention_type,
            denoising_steps=denoising_steps,
            resize_mode=resize_mode,
            fixed_area=fixed_area,
            segment_length=segment_length,
            prev_frame_length=prev_frame_length,
            use_tiny_vae=use_tiny_vae,
        )

        return (config.to_dict(),)


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
        """Create TeaCache configuration."""
        config = TeaCacheConfig(enable=enable, threshold=threshold, use_ret_steps=use_ret_steps)
        return (config.to_dict(),)


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
        config = QuantizationConfig(
            quant_op=quant_op,
            dit_quant_scheme=dit_quant_scheme,
            t5_quant_scheme=t5_quant_scheme,
            clip_quant_scheme=clip_quant_scheme,
            adapter_quant_scheme=adapter_quant_scheme,
        )
        return (config.to_dict(),)


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
        """Create memory optimization configuration."""
        config = MemoryOptimizationConfig(
            enable_rotary_chunk=enable_rotary_chunk,
            rotary_chunk_size=rotary_chunk_size,
            clean_cuda_cache=clean_cuda_cache,
            cpu_offload=cpu_offload,
            offload_granularity=offload_granularity,
            offload_ratio=offload_ratio,
            t5_cpu_offload=t5_cpu_offload,
            t5_offload_granularity=t5_offload_granularity,
            audio_encoder_cpu_offload=audio_encoder_cpu_offload,
            audio_adapter_cpu_offload=audio_adapter_cpu_offload,
            vae_cpu_offload=vae_cpu_offload,
            use_tiling_vae=use_tiling_vae,
            lazy_load=lazy_load,
            unload_after_inference=unload_after_inference,
        )
        return (config.to_dict(),)


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
        """Load and chain LoRA configurations."""
        chain = LoRAChainBuilder.build_chain(lora_name=lora_name, strength=strength, existing_chain=lora_chain)
        return (chain,)


class TalkObjectInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "person_1", "tooltip": "说话人名称标识"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "上传的音频文件"}),
                "mask": ("MASK", {"tooltip": "上传的遮罩图像（可选）"}),
                "save_to_input": ("BOOLEAN", {"default": True, "tooltip": "是否保存到input文件夹"}),
            },
        }

    RETURN_TYPES = ("TALK_OBJECT",)
    RETURN_NAMES = ("talk_object",)
    FUNCTION = "create_talk_object"
    CATEGORY = "LightX2V/Audio"

    def create_talk_object(self, name, audio=None, mask=None, save_to_input=True):
        """Create a talk object from input data."""
        builder = TalkObjectConfigBuilder()

        talk_object = builder.build_from_input(name=name, audio=audio, mask=mask, save_to_input=save_to_input)

        if talk_object:
            return (talk_object.to_dict(),)
        return (None,)


class TalkObjectsCombiner:
    """组合多个谈话对象为配置"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {}, "optional": {}}

        # 预定义10个TALK_OBJECT输入槽
        for i in range(1, 11):
            inputs["optional"][f"talk_object_{i}"] = ("TALK_OBJECT", {"tooltip": f"谈话对象{i}"})

        return inputs

    RETURN_TYPES = ("TALK_OBJECTS_CONFIG",)
    RETURN_NAMES = ("talk_objects_config",)
    FUNCTION = "combine_talk_objects"
    CATEGORY = "LightX2V/Audio"

    def combine_talk_objects(self, **kwargs):
        config = TalkObjectsConfig()

        for i in range(1, 11):
            talk_obj = kwargs.get(f"talk_object_{i}")

            if talk_obj is not None:
                config.add_object(talk_obj)

        if not config.talk_objects:
            return (None,)

        return (config,)


class TalkObjectsFromJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_config": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": '[{"name": "person1", "audio": "/path/to/audio1.wav", "mask": "/path/to/mask1.png"}]',
                        "tooltip": "JSON格式的谈话对象配置",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TALK_OBJECTS_CONFIG",)
    RETURN_NAMES = ("talk_objects_config",)
    FUNCTION = "parse_json_config"
    CATEGORY = "LightX2V/Audio"

    def parse_json_config(self, json_config):
        builder = TalkObjectConfigBuilder()
        talk_objects_config = builder.build_from_json(json_config)
        return (talk_objects_config,)


class TalkObjectsFromFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_files": ("STRING", {"multiline": True, "default": "audio1.wav\naudio2.wav", "tooltip": "音频文件列表（每行一个）"}),
            },
            "optional": {
                "mask_files": ("STRING", {"multiline": True, "default": "mask1.png\nmask2.png", "tooltip": "遮罩文件列表（每行一个，可选）"}),
                "names": ("STRING", {"multiline": True, "default": "person1\nperson2", "tooltip": "人物名称列表（每行一个，可选）"}),
            },
        }

    RETURN_TYPES = ("TALK_OBJECTS_CONFIG",)
    RETURN_NAMES = ("talk_objects_config",)
    FUNCTION = "build_from_files"
    CATEGORY = "LightX2V/Audio"

    def build_from_files(self, audio_files, mask_files="", names=""):
        builder = TalkObjectConfigBuilder()
        talk_objects_config = builder.build_from_files(audio_files, mask_files, names)
        return (talk_objects_config,)


class LightX2VConfigCombiner:
    def __init__(self):
        self.config_builder = ConfigBuilder()

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
                "talk_objects_config": ("TALK_OBJECTS_CONFIG", {"tooltip": "Multi-person talk objects configuration"}),
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
        talk_objects_config=None,
    ):
        """Combine multiple configurations into final config."""
        # Convert dict configs back to objects if needed

        # Create objects from dicts
        inf_config = InferenceConfig(**inference_config) if isinstance(inference_config, dict) else None
        tea_config = TeaCacheConfig(**teacache_config) if teacache_config and isinstance(teacache_config, dict) else None
        quant_config = QuantizationConfig(**quantization_config) if quantization_config and isinstance(quantization_config, dict) else None
        mem_config = MemoryOptimizationConfig(**memory_config) if memory_config and isinstance(memory_config, dict) else None

        config = self.config_builder.combine_configs(
            inference_config=inf_config,
            teacache_config=tea_config,
            quantization_config=quant_config,
            memory_config=mem_config,
            lora_chain=lora_chain,
            talk_objects_config=talk_objects_config,
        )

        return (config,)


class LightX2VModularInference:
    _current_runner = None
    _current_config_hash = None

    def __init__(self):
        self.config_builder = ConfigBuilder()
        self.temp_manager = TempFileManager()
        self.image_handler = ImageFileHandler()
        self.audio_handler = AudioFileHandler()
        self.resolver = ComfyUIFileResolver()

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
        """Get hash of configuration to detect changes."""
        return self.config_builder.get_config_hash(config)

    def generate(
        self,
        combined_config,
        prompt,
        negative_prompt,
        image=None,
        audio=None,
        **kwargs,
    ):
        # config type is EasyDict
        config = combined_config
        config.prompt = prompt
        config.negative_prompt = negative_prompt

        if config.task == "i2v" and image is None:
            raise ValueError("i2v task requires input image")

        try:
            # Handle image input
            if config.task == "i2v" and image is not None:
                image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)

                temp_path = self.temp_manager.create_temp_file(suffix=".png")
                pil_image.save(temp_path)
                config.image_path = temp_path
                logging.info(f"Image saved to {temp_path}")

            # Handle audio input for seko models
            if audio is not None and hasattr(config, "model_cls") and "seko" in config.model_cls:
                temp_path = self.temp_manager.create_temp_file(suffix=".wav")
                self.audio_handler.save(audio, temp_path)
                config.audio_path = temp_path
                logging.info(f"Audio saved to {temp_path}")

            # Handle talk objects
            if hasattr(config, "talk_objects") and config.talk_objects:
                talk_objects = config.talk_objects
                processed_talk_objects = []

                for talk_obj in talk_objects:
                    processed_obj = {}

                    if "audio" in talk_obj:
                        processed_obj["audio"] = talk_obj["audio"]

                    if "mask" in talk_obj:
                        processed_obj["mask"] = talk_obj["mask"]

                    if "audio" in processed_obj:
                        processed_talk_objects.append(processed_obj)

                # Resolve paths
                for obj in processed_talk_objects:
                    if "audio" in obj and obj["audio"]:
                        audio_path = obj["audio"]
                        if not os.path.isabs(audio_path) and not audio_path.startswith("/tmp"):
                            obj["audio"] = self.resolver.resolve_input_path(audio_path)
                            logging.info(f"Resolved audio path: {audio_path} -> {obj['audio']}")
                        if not os.path.exists(obj["audio"]):
                            logging.warning(f"Audio file not found: {obj['audio']}")

                    if "mask" in obj and obj["mask"]:
                        mask_path = obj["mask"]
                        if not os.path.isabs(mask_path) and not mask_path.startswith("/tmp"):
                            obj["mask"] = self.resolver.resolve_input_path(mask_path)
                            logging.info(f"Resolved mask path: {mask_path} -> {obj['mask']}")
                        if not os.path.exists(obj["mask"]):
                            logging.warning(f"Mask file not found: {obj['mask']}")

                if processed_talk_objects:
                    config.talk_objects = processed_talk_objects
                    logging.info(f"Processed {len(processed_talk_objects)} talk objects")

            logging.info("lightx2v config: " + json.dumps(config, indent=2, ensure_ascii=False))

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
                    self.__class__._current_runner.model.config = config
                    self.__class__._current_runner.model.scheduler.config = config

            progress = ProgressBar(100)

            def update_progress(current_step, _total):
                progress.update_absolute(current_step)

            if hasattr(self.__class__._current_runner, "set_progress_callback"):
                self.__class__._current_runner.set_progress_callback(update_progress)

            result_dict = self.__class__._current_runner.run_pipeline()
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
            # Cleanup is handled by TempFileManager destructor
            pass


NODE_CLASS_MAPPINGS = {
    "LightX2VInferenceConfig": LightX2VInferenceConfig,
    "LightX2VTeaCache": LightX2VTeaCache,
    "LightX2VQuantization": LightX2VQuantization,
    "LightX2VMemoryOptimization": LightX2VMemoryOptimization,
    "LightX2VLoRALoader": LightX2VLoRALoader,
    "LightX2VConfigCombiner": LightX2VConfigCombiner,
    "LightX2VModularInference": LightX2VModularInference,
    "LightX2VTalkObjectInput": TalkObjectInput,
    "LightX2VTalkObjectsCombiner": TalkObjectsCombiner,
    "LightX2VTalkObjectsFromJSON": TalkObjectsFromJSON,
    "LightX2VTalkObjectsFromFiles": TalkObjectsFromFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightX2VInferenceConfig": "LightX2V Inference Config",
    "LightX2VTeaCache": "LightX2V TeaCache",
    "LightX2VQuantization": "LightX2V Quantization",
    "LightX2VMemoryOptimization": "LightX2V Memory Optimization",
    "LightX2VLoRALoader": "LightX2V LoRA Loader",
    "LightX2VConfigCombiner": "LightX2V Config Combiner",
    "LightX2VModularInference": "LightX2V Modular Inference",
    "LightX2VTalkObjectInput": "LightX2V Talk Object Input (Single)",
    "LightX2VTalkObjectsCombiner": "LightX2V Talk Objects Combiner",
    "LightX2VTalkObjectsFromFiles": "LightX2V Talk Objects From Files",
    "LightX2VTalkObjectsFromJSON": "LightX2V Talk Objects From JSON (API)",
}
