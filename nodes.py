import gc
import io
import json
import logging
import os
import subprocess as sp
import wave

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
    HTTPFileDownloader,
    ImageFileHandler,
    TempFileManager,
)
from .lightx2v.lightx2v.infer import init_runner
from .lightx2v.lightx2v.utils.input_info import set_input_info
from .lightx2v.lightx2v.utils.set_config import set_config
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
                    ["t2v", "i2v", "s2v"],
                    {
                        "default": "i2v",
                        "tooltip": "Task type: text-to-video or image-to-video or audio-to-video",
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
                    [
                        "adaptive",
                        "keep_ratio_fixed_area",
                        "fixed_min_area",
                        "fixed_max_area",
                        "fixed_shape",
                        "fixed_min_side",
                    ],
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

        common_schema = ["fp8", "int8"]
        supported_quant_schemes = ["Default"]
        for schema in common_schema:
            for backend in quant_backends:
                supported_quant_schemes.append(f"{schema}-{backend}")

        return {
            "required": {
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
                        "default": supported_quant_schemes[0],
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
        dit_quant_scheme,
        t5_quant_scheme,
        clip_quant_scheme,
        adapter_quant_scheme,
    ):
        """Create quantization configuration."""
        config = QuantizationConfig(
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
                "name": (
                    "STRING",
                    {"default": "person_1", "tooltip": "speaker name identifier"},
                ),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "uploaded audio file"}),
                "mask": ("MASK", {"tooltip": "uploaded mask image (optional)"}),
                "save_to_input": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "save to input folder"},
                ),
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
            return (talk_object,)
        return (None,)


class TalkObjectsCombiner:
    PREDEFINED_SLOTS = 16

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {}, "optional": {}}

        for i in range(cls.PREDEFINED_SLOTS):
            inputs["optional"][f"talk_object_{i + 1}"] = (
                "TALK_OBJECT",
                {"tooltip": f"talk object {i + 1}"},
            )

        return inputs

    RETURN_TYPES = ("TALK_OBJECTS_CONFIG",)
    RETURN_NAMES = ("talk_objects_config",)
    FUNCTION = "combine_talk_objects"
    CATEGORY = "LightX2V/Audio"

    def combine_talk_objects(self, **kwargs):
        config = TalkObjectsConfig()

        for i in range(self.PREDEFINED_SLOTS):
            talk_obj = kwargs.get(f"talk_object_{i + 1}")

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
                        "tooltip": "JSON format talk objects configuration",
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
                "audio_files": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "audio1.wav\naudio2.wav",
                        "tooltip": "audio file list (one per line)",
                    },
                ),
            },
            "optional": {
                "mask_files": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "mask1.png\nmask2.png",
                        "tooltip": "mask file list (one per line, optional)",
                    },
                ),
                "names": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "person1\nperson2",
                        "tooltip": "talk object name list (one per line, optional)",
                    },
                ),
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
        if not hasattr(self.__class__, "_current_runner"):
            self.__class__._current_runner = None
        if not hasattr(self.__class__, "_current_config_hash"):
            self.__class__._current_config_hash = None

        self.config_builder = ConfigBuilder()
        self.temp_manager = TempFileManager()
        self.image_handler = ImageFileHandler()
        self.audio_handler = AudioFileHandler()
        self.resolver = ComfyUIFileResolver()
        self.http_downloader = HTTPFileDownloader()

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
                "image": ("IMAGE", {"tooltip": "Input image for i2v or s2v task"}),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Input audio for audio-driven generation for s2v task"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "PREV_FRAMES")
    RETURN_NAMES = ("images", "audio", "prev_frames")
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

        if config.task in ["i2v", "s2v"] and image is None:
            raise ValueError(f"{config.task} task requires input image")

        try:
            # Handle image input
            if config.task in ["i2v", "s2v"] and image is not None:
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

                # Resolve paths and download URLs
                for obj in processed_talk_objects:
                    if "audio" in obj and obj["audio"]:
                        audio_path = obj["audio"]

                        # Check if it's a URL and download if needed
                        if self.http_downloader.is_url(audio_path):
                            try:
                                downloaded_path = self.http_downloader.download_if_url(audio_path, prefix="audio")
                                obj["audio"] = downloaded_path
                                logging.info(f"Downloaded audio from URL: {audio_path} -> {downloaded_path}")
                            except Exception as e:
                                logging.error(f"Failed to download audio from {audio_path}: {e}")
                                continue
                        # Handle relative paths
                        elif not os.path.isabs(audio_path) and not audio_path.startswith("/tmp"):
                            obj["audio"] = self.resolver.resolve_input_path(audio_path)
                            logging.info(f"Resolved audio path: {audio_path} -> {obj['audio']}")

                        # Check if file exists
                        if not os.path.exists(obj["audio"]):
                            logging.warning(f"Audio file not found: {obj['audio']}")

                    if "mask" in obj and obj["mask"]:
                        mask_path = obj["mask"]

                        # Check if it's a URL and download if needed
                        if self.http_downloader.is_url(mask_path):
                            try:
                                downloaded_path = self.http_downloader.download_if_url(mask_path, prefix="mask")
                                obj["mask"] = downloaded_path
                                logging.info(f"Downloaded mask from URL: {mask_path} -> {downloaded_path}")
                            except Exception as e:
                                logging.error(f"Failed to download mask from {mask_path}: {e}")
                                # Don't skip the object if mask download fails (mask is optional)
                        # Handle relative paths
                        elif not os.path.isabs(mask_path) and not mask_path.startswith("/tmp"):
                            obj["mask"] = self.resolver.resolve_input_path(mask_path)
                            logging.info(f"Resolved mask path: {mask_path} -> {obj['mask']}")

                        # Check if file exists
                        if not os.path.exists(obj["mask"]):
                            logging.warning(f"Mask file not found: {obj['mask']}")

                if processed_talk_objects:
                    config.talk_objects = processed_talk_objects
                    logging.info(f"Processed {len(processed_talk_objects)} talk objects")

            logging.info("lightx2v config: " + json.dumps(config, indent=2, ensure_ascii=False))

            config_hash = self._get_config_hash(config)

            current_runner = getattr(self.__class__, "_current_runner", None)
            current_config_hash = getattr(self.__class__, "_current_config_hash", None)

            needs_reinit = current_runner is None or current_config_hash != config_hash or getattr(config, "lazy_load", False)

            logging.info(f"Needs reinit: {needs_reinit}, old config hash: {current_config_hash}, new config hash: {config_hash}")
            if needs_reinit:
                if current_runner is not None:
                    # current_runner.end_run()
                    del self.__class__._current_runner
                    torch.cuda.empty_cache()
                    gc.collect()
                formated_config = set_config(config)
                self.__class__._current_runner = init_runner(formated_config)
                self.__class__._current_config_hash = config_hash

            progress = ProgressBar(100)

            def update_progress(current_step, _total):
                progress.update_absolute(current_step)

            current_runner = getattr(self.__class__, "_current_runner", None)

            if hasattr(current_runner, "set_progress_callback"):
                current_runner.set_progress_callback(update_progress)

            config["return_result_tensor"] = True
            config["save_result_path"] = ""
            config["negative_prompt"] = config.get("negative_prompt", "")
            input_info = set_input_info(config)
            current_runner.set_config(config)
            result_dict = current_runner.run_pipeline(input_info)
            images = result_dict.get("video", None)
            audio = result_dict.get("audio", None)
            for_next_section = result_dict.get("for_next_section", None)

            if images is not None and images.numel() > 0:
                images = images.cpu()
                if images.dtype != torch.float32:
                    images = images.float()

            if getattr(config, "unload_after_inference", False):
                if hasattr(self.__class__, "_current_runner"):
                    del self.__class__._current_runner
                self.__class__._current_runner = None
                self.__class__._current_config_hash = None

            torch.cuda.empty_cache()
            gc.collect()

            return (images, audio, for_next_section)

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise

        finally:
            # Cleanup is handled by TempFileManager destructor
            pass


class LightX2VConfigCombinerV2:
    """Config combiner that also handles data preparation (image/audio/prompts)."""

    def __init__(self):
        self.config_builder = ConfigBuilder()
        self.temp_manager = TempFileManager()
        self.image_handler = ImageFileHandler()
        self.audio_handler = AudioFileHandler()
        self.resolver = ComfyUIFileResolver()
        self.http_downloader = HTTPFileDownloader()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_config": (
                    "INFERENCE_CONFIG",
                    {"tooltip": "Basic inference configuration"},
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
                "talk_objects_config": ("TALK_OBJECTS_CONFIG", {"tooltip": "Talk objects configuration"}),
                "image": ("IMAGE", {"tooltip": "Input image for i2v or s2v task"}),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Input audio for audio-driven generation for s2v task"},
                ),
                "prev_frames": ("PREV_FRAMES", {"tooltip": "Previous frames for s2v task"}),
                "frist_image": ("IMAGE", {"tooltip": "First image for s2v task"}),
            },
        }

    RETURN_TYPES = ("PREPARED_CONFIG",)
    RETURN_NAMES = ("prepared_config",)
    FUNCTION = "prepare_config"
    CATEGORY = "LightX2V/ConfigV2"

    def prepare_config(
        self,
        inference_config,
        prompt,
        negative_prompt,
        teacache_config=None,
        quantization_config=None,
        memory_config=None,
        lora_chain=None,
        talk_objects_config=None,
        image=None,
        audio=None,
        prev_frames=None,
        frist_image=None,
    ):
        """Combine configurations and prepare data for inference."""

        # Convert dict configs back to objects if needed
        inf_config = InferenceConfig(**inference_config) if isinstance(inference_config, dict) else inference_config
        tea_config = TeaCacheConfig(**teacache_config) if teacache_config and isinstance(teacache_config, dict) else teacache_config
        quant_config = (
            QuantizationConfig(**quantization_config) if quantization_config and isinstance(quantization_config, dict) else quantization_config
        )
        mem_config = MemoryOptimizationConfig(**memory_config) if memory_config and isinstance(memory_config, dict) else memory_config

        # Build combined config
        config = self.config_builder.combine_configs(
            inference_config=inf_config,
            teacache_config=tea_config,
            quantization_config=quant_config,
            memory_config=mem_config,
            lora_chain=lora_chain,
            talk_objects_config=talk_objects_config,
        )

        # Add prompts to config
        config.prompt = prompt
        config.negative_prompt = negative_prompt

        # Validate task requirements
        if config.task in ["i2v", "s2v"] and image is None:
            raise ValueError("i2v or s2v task requires input image")

        # Handle image input
        if config.task in ["i2v", "s2v"] and image is not None:
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

        # Handle first image input
        if frist_image is not None:
            frist_image_np = (frist_image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(frist_image_np)
            temp_path = self.temp_manager.create_temp_file(suffix=".png")
            pil_image.save(temp_path)
            config.frist_image_path = temp_path
            logging.info(f"First image saved to {temp_path}")

        # Handle previous frames input
        if prev_frames is not None:
            temp_path = self.temp_manager.create_temp_file(suffix=".pt")
            torch.save(prev_frames, temp_path)
            config.prev_section_info_path = temp_path
            logging.info(f"Previous frames saved to {temp_path}")

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

            # Resolve paths and download URLs
            for obj in processed_talk_objects:
                if "audio" in obj and obj["audio"]:
                    audio_path = obj["audio"]

                    # Check if it's a URL and download if needed
                    if self.http_downloader.is_url(audio_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(audio_path, prefix="audio")
                            obj["audio"] = downloaded_path
                            logging.info(f"Downloaded audio from URL: {audio_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download audio from {audio_path}: {e}")
                            continue
                    # Handle relative paths
                    elif not os.path.isabs(audio_path) and not audio_path.startswith("/tmp"):
                        obj["audio"] = self.resolver.resolve_input_path(audio_path)
                        logging.info(f"Resolved audio path: {audio_path} -> {obj['audio']}")

                    # Check if file exists
                    if not os.path.exists(obj["audio"]):
                        logging.warning(f"Audio file not found: {obj['audio']}")

                if "mask" in obj and obj["mask"]:
                    mask_path = obj["mask"]

                    # Check if it's a URL and download if needed
                    if self.http_downloader.is_url(mask_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(mask_path, prefix="mask")
                            obj["mask"] = downloaded_path
                            logging.info(f"Downloaded mask from URL: {mask_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download mask from {mask_path}: {e}")
                            # Don't skip the object if mask download fails (mask is optional)
                    # Handle relative paths
                    elif not os.path.isabs(mask_path) and not mask_path.startswith("/tmp"):
                        obj["mask"] = self.resolver.resolve_input_path(mask_path)
                        logging.info(f"Resolved mask path: {mask_path} -> {obj['mask']}")

                    # Check if file exists
                    if not os.path.exists(obj["mask"]):
                        logging.warning(f"Mask file not found: {obj['mask']}")

            if processed_talk_objects:
                if len(processed_talk_objects) == 1 and not processed_talk_objects[0].get("mask", "").strip():
                    config.audio_path = processed_talk_objects[0]["audio"]
                    logging.info(f"Convert Processed 1 talk object to audio path: {config.audio_path}")
                else:
                    temp_dir = self.temp_manager.create_temp_dir()
                    with open(os.path.join(temp_dir, "config.json"), "w") as f:
                        json.dump({"talk_objects": processed_talk_objects}, f)
                    config.audio_path = temp_dir
                    logging.info(f"Processed {len(processed_talk_objects)} talk objects")

        logging.info("lightx2v prepared config: " + json.dumps(config, indent=2, ensure_ascii=False))

        return (config,)


class LightX2VConfigCombinerV3:
    """Config combiner that also handles data preparation (image/audio/prompts)."""

    def __init__(self):
        self.config_builder = ConfigBuilder()
        self.temp_manager = TempFileManager()
        self.image_handler = ImageFileHandler()
        self.audio_handler = AudioFileHandler()
        self.resolver = ComfyUIFileResolver()
        self.http_downloader = HTTPFileDownloader()

    @staticmethod
    def extend_mp3(input_path: str, output_path: str, duration: float) -> bool:
        """Extend or truncate MP3 audio file.

        Extend or truncate the input audio based on its duration and target
        duration:
        - If input duration > duration + 0.1, raise an error
        - If input duration is in [duration, duration + 0.1), truncate audio
        - If input duration < duration, extend audio using silence padding


        Args:
            input_path (str):
                Path to the input MP3 file.
            output_path (str):
                Path to the output MP3 file.
            duration (float):
                Target duration in seconds.

        Returns:
            bool:
                Returns True if the operation succeeds.

        Raises:
            ValueError:
                Raised when input audio duration exceeds duration + 0.1.
        """
        cmd_probe = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration,sample_rate,bit_rate,channels",
            "-of",
            "json",
            input_path,
        ]

        try:
            output = sp.check_output(cmd_probe)
            data = json.loads(output.decode("utf-8"))
            streams = data.get("streams", [])
            if not streams:
                raise ValueError(f"Failed to get audio stream information: {input_path}")

            stream_info = streams[0]
            input_duration = float(stream_info.get("duration", 0))
            sample_rate = stream_info.get("sample_rate", "44100")
            bit_rate = stream_info.get("bit_rate", "128000")
            channels = stream_info.get("channels", 2)

            if input_duration > duration:
                raise ValueError(f"Input audio duration ({input_duration:.2f}s) exceeds target duration + 0.1s ({duration + 0.1:.2f}s)")
            else:
                pad_duration = duration - input_duration
                cmd = [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-af",
                    f"apad=pad_dur={pad_duration}",
                    "-ar",
                    str(sample_rate),
                    "-b:a",
                    str(bit_rate),
                    "-ac",
                    str(channels),
                    "-c:a",
                    "libmp3lame",
                    "-y",
                    output_path,
                ]

            sp.run(cmd, capture_output=True, text=True, check=True)
            return True

        except sp.CalledProcessError as e:
            if e.stderr:
                logging.error(f"Subprocess execution failed, stderr: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse audio information: {input_path}")
        except Exception as e:
            raise

    @staticmethod
    def get_audio_duration(input_path: str) -> float:
        """Get the duration of an audio file.

        Uses ffprobe to extract audio stream information and returns the
        duration in seconds.


        Args:
            input_path (str):
                Path to the audio file.

        Returns:
            float:
                Audio duration in seconds.

        Raises:
            ValueError:
                Raised when audio stream information cannot be retrieved or
                parsed.
        """
        cmd_probe = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration,sample_rate,bit_rate,channels",
            "-of",
            "json",
            input_path,
        ]
        try:
            output = sp.check_output(cmd_probe)
            data = json.loads(output.decode("utf-8"))
            streams = data.get("streams", [])
            if not streams:
                raise ValueError(f"Failed to get audio stream information: {input_path}")

            stream_info = streams[0]
            input_duration = float(stream_info.get("duration", 0))
            return input_duration

        except sp.CalledProcessError as e:
            if e.stderr:
                logging.error(f"Subprocess execution failed, stderr: {e.stderr}")
            raise e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse audio information: {input_path}") from e
        except Exception as e:
            raise e

    @staticmethod
    def generate_white_noise(
        duration: float, framerate: int, n_channels: int = 1, rms: float = None, std_dev: float = None, seed: int = None
    ) -> np.ndarray:
        """Generate white noise audio.

        Generate white noise audio data with optional normalization using
        RMS or standard deviation. The noise is generated using a normal
        distribution and can be normalized to a target RMS value or standard
        deviation.


        Args:
            duration (float):
                Audio duration in seconds.
            framerate (int):
                Sample rate in Hz.
            n_channels (int, optional):
                Number of audio channels. Defaults to 1 (mono).
            rms (float, optional):
                Target RMS value for normalization. If provided, the noise
                will be normalized to this RMS value. Defaults to None.
            std_dev (float, optional):
                Target standard deviation for normalization. If provided, the
                noise will be normalized to this standard deviation.
                Defaults to None.
            seed (int, optional):
                Random seed for reproducible generation. Defaults to None.

        Returns:
            np.ndarray:
                Generated audio data with shape (n_samples, n_channels) for
                multi-channel or (n_samples,) for mono channel, where
                n_samples = duration * framerate.
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = int(duration * framerate)

        if n_channels == 1:
            noise = np.random.normal(0, 1, n_samples).astype(np.float32)
        else:
            noise = np.random.normal(0, 1, (n_samples, n_channels)).astype(np.float32)

        if std_dev is not None:
            current_std = np.std(noise)
            if current_std > 0:
                noise = noise * (std_dev / current_std)
        elif rms is not None:
            current_rms = np.sqrt(np.mean(noise**2))
            if current_rms > 0:
                noise = noise * (rms / current_rms)
        return noise

    @staticmethod
    def save_wav_file(audio_data: np.ndarray, output_path: str | io.BytesIO, framerate: int, sample_width: int = 2) -> None:
        """Save audio data as WAV file or BytesIO object.

        Convert normalized float audio data to integer format and save as
        WAV file. Supports mono and multi-channel audio with configurable
        sample width.


        Args:
            audio_data (np.ndarray):
                Audio data with shape (n_samples,) for mono or
                (n_samples, n_channels) for multi-channel. Values should
                be in the range [-1.0, 1.0].
            output_path (str | io.BytesIO):
                Output file path as string or BytesIO object.
            framerate (int):
                Sample rate in Hz.
            sample_width (int, optional):
                Sample width in bytes. Supported values are 1 (8-bit),
                2 (16-bit), and 4 (32-bit). Defaults to 2.
        """
        if audio_data.ndim == 1:
            n_channels = 1
            audio_data = audio_data.reshape(-1, 1)
        else:
            n_channels = audio_data.shape[1]

        audio_data = np.clip(audio_data, -1.0, 1.0)

        if sample_width == 1:
            audio_int = ((audio_data + 1.0) * 127.5).astype(np.uint8)
        elif sample_width == 2:
            audio_int = (audio_data * 32767).astype(np.int16)
        elif sample_width == 4:
            audio_int = (audio_data * 2147483647).astype(np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels == 1:
            audio_int = audio_int.flatten()
        else:
            audio_int = audio_int.reshape(-1, n_channels)

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(framerate)
            wav_file.writeframes(audio_int.tobytes())

    @staticmethod
    def generate_background_mask(positive_mask_paths: list[str]) -> io.BytesIO:
        """Generate background mask from positive mask images.

        Generate a background mask by finding pixels that are zero (or
        below threshold) in all input positive mask images. The resulting
        mask marks background regions (all masks are zero) as white (255)
        and foreground regions (any mask has non-zero values) as black (0).


        Args:
            positive_mask_paths (list[str]):
                List of paths to positive mask image files. All images
                must have the same width and height.

        Returns:
            io.BytesIO:
                BytesIO object containing the background mask image in JPEG
                format. The mask is a grayscale image where white (255)
                represents background regions and black (0) represents
                foreground regions.

        Raises:
            ValueError:
                Raised when mask images have different dimensions.
        """
        width = None
        height = None
        opened_imgs = list()
        for path in positive_mask_paths:
            img = Image.open(path)
            if width is None:
                width = img.width
            elif width != img.width:
                raise ValueError(f"Widths of masks are not the same: {width} != {img.width}")
            if height is None:
                height = img.height
            elif height != img.height:
                raise ValueError(f"Heights of masks are not the same: {height} != {img.height}")
            opened_imgs.append(img)
        img_arrays = []
        for img in opened_imgs:
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = img_array[:, :, np.newaxis]
            img_arrays.append(img_array)

        threshold = 1
        zero_masks = []
        for img_array in img_arrays:
            if img_array.shape[-1] == 1:
                zero_mask = img_array[:, :, 0] <= threshold
            else:
                zero_mask = np.all(img_array <= threshold, axis=-1)
            zero_masks.append(zero_mask)

        if zero_masks:
            all_zero_mask = np.logical_and.reduce(zero_masks)
            bg_array = np.where(all_zero_mask, 255, 0).astype(np.uint8)
        else:
            bg_array = np.full((height, width), 255, dtype=np.uint8)

        bg_img = Image.fromarray(bg_array, mode="L")
        img_io = io.BytesIO()
        bg_img.save(img_io, format="JPEG")
        img_io.seek(0)
        for img in opened_imgs:
            img.close()
        return img_io

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_config": (
                    "INFERENCE_CONFIG",
                    {"tooltip": "Basic inference configuration"},
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
                "talk_objects_config": ("TALK_OBJECTS_CONFIG", {"tooltip": "Talk objects configuration"}),
                "image": ("IMAGE", {"tooltip": "Input image for i2v or s2v task"}),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Input audio for audio-driven generation for s2v task"},
                ),
            },
        }

    RETURN_TYPES = ("PREPARED_CONFIG",)
    RETURN_NAMES = ("prepared_config",)
    FUNCTION = "prepare_config"
    CATEGORY = "LightX2V/ConfigV2"

    def prepare_config(
        self,
        inference_config,
        prompt,
        negative_prompt,
        teacache_config=None,
        quantization_config=None,
        memory_config=None,
        lora_chain=None,
        talk_objects_config=None,
        image=None,
        audio=None,
    ):
        """Combine configurations and prepare data for inference."""

        # Convert dict configs back to objects if needed
        inf_config = InferenceConfig(**inference_config) if isinstance(inference_config, dict) else inference_config
        tea_config = TeaCacheConfig(**teacache_config) if teacache_config and isinstance(teacache_config, dict) else teacache_config
        quant_config = (
            QuantizationConfig(**quantization_config) if quantization_config and isinstance(quantization_config, dict) else quantization_config
        )
        mem_config = MemoryOptimizationConfig(**memory_config) if memory_config and isinstance(memory_config, dict) else memory_config

        # Build combined config
        config = self.config_builder.combine_configs(
            inference_config=inf_config,
            teacache_config=tea_config,
            quantization_config=quant_config,
            memory_config=mem_config,
            lora_chain=lora_chain,
            talk_objects_config=talk_objects_config,
        )

        # Add prompts to config
        config.prompt = prompt
        config.negative_prompt = negative_prompt

        # Validate task requirements
        if config.task in ["i2v", "s2v"] and image is None:
            raise ValueError("i2v or s2v task requires input image")

        # Handle image input
        if config.task in ["i2v", "s2v"] and image is not None:
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
            src_talk_objects = []

            for talk_obj in talk_objects:
                src_obj = {}

                if "audio" in talk_obj:
                    src_obj["audio"] = talk_obj["audio"]

                if "mask" in talk_obj:
                    src_obj["mask"] = talk_obj["mask"]

                if "audio" in src_obj:
                    src_talk_objects.append(src_obj)

            # Resolve paths and download URLs,
            # record the max duration of the src talk objects
            max_src_duration = None
            for obj in src_talk_objects:
                if "audio" in obj and obj["audio"]:
                    audio_path = obj["audio"]

                    # Check if it's a URL and download if needed
                    if self.http_downloader.is_url(audio_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(audio_path, prefix="audio")
                            obj["audio"] = downloaded_path
                            logging.info(f"Downloaded audio from URL: {audio_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download audio from {audio_path}: {e}")
                            continue
                    # Handle relative paths
                    elif not os.path.isabs(audio_path) and not audio_path.startswith("/tmp"):
                        obj["audio"] = self.resolver.resolve_input_path(audio_path)
                        logging.info(f"Resolved audio path: {audio_path} -> {obj['audio']}")

                    # Check if file exists
                    if not os.path.exists(obj["audio"]):
                        logging.warning(f"Audio file not found: {obj['audio']}")
                    duration = self.get_audio_duration(obj["audio"])
                    obj["duration"] = duration
                    if max_src_duration is None or duration > max_src_duration:
                        max_src_duration = duration

                if "mask" in obj and obj["mask"]:
                    mask_path = obj["mask"]

                    # Check if it's a URL and download if needed
                    if self.http_downloader.is_url(mask_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(mask_path, prefix="mask")
                            obj["mask"] = downloaded_path
                            logging.info(f"Downloaded mask from URL: {mask_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download mask from {mask_path}: {e}")
                            # Don't skip the object if mask download fails (mask is optional)
                    # Handle relative paths
                    elif not os.path.isabs(mask_path) and not mask_path.startswith("/tmp"):
                        obj["mask"] = self.resolver.resolve_input_path(mask_path)
                        logging.info(f"Resolved mask path: {mask_path} -> {obj['mask']}")

                    # Check if file exists
                    if not os.path.exists(obj["mask"]):
                        logging.warning(f"Mask file not found: {obj['mask']}")

            if len(src_talk_objects) > 1:
                # extend audio duration to the max duration of the src talk objects
                processed_talk_objects: list[dict[str, str]] = list()
                mask_img_paths = list()
                extend_count = 0
                for obj in src_talk_objects:
                    dst_obj = dict()
                    src_audio_path = obj["audio"]
                    src_audio_duration = obj["duration"]
                    if max_src_duration - src_audio_duration > 0.1:
                        dst_audio_path = self.temp_manager.create_temp_file(suffix=".mp3")
                        self.extend_mp3(src_audio_path, dst_audio_path, max_src_duration)
                        extend_count += 1
                        dst_obj["audio"] = dst_audio_path
                    else:
                        dst_obj["audio"] = src_audio_path
                    src_mask = obj.get("mask", None)
                    if src_mask:
                        dst_obj["mask"] = src_mask
                        mask_img_paths.append(src_mask)
                    processed_talk_objects.append(dst_obj)
                logging.info(f"Extended {extend_count} audio files")
                # generate background mask and audio
                bg_mask_io = self.generate_background_mask(mask_img_paths)
                bg_mask_path = self.temp_manager.create_temp_file(suffix=".jpg")
                with open(bg_mask_path, "wb") as f:
                    f.write(bg_mask_io.getvalue())
                bg_noise_data = self.generate_white_noise(
                    duration=max_src_duration,
                    framerate=16000,
                    n_channels=1,
                    rms=0.00232,
                    std_dev=0.00232,
                )
                wav_io = io.BytesIO()
                self.save_wav_file(audio_data=bg_noise_data, output_path=wav_io, framerate=16000, sample_width=2)
                bg_audio_path = self.temp_manager.create_temp_file(suffix=".wav")
                with open(bg_audio_path, "wb") as f:
                    f.write(wav_io.getvalue())
                bg_obj = dict(
                    audio=bg_audio_path,
                    mask=bg_mask_path,
                )
                processed_talk_objects.append(bg_obj)
                logging.info(f"Generated background mask and audio: {bg_mask_path}, {bg_audio_path}")
            else:
                processed_talk_objects = src_talk_objects

            if processed_talk_objects:
                if len(processed_talk_objects) == 1 and not processed_talk_objects[0].get("mask", "").strip():
                    config.audio_path = processed_talk_objects[0]["audio"]
                    logging.info(f"Convert Processed 1 talk object to audio path: {config.audio_path}")
                else:
                    temp_dir = self.temp_manager.create_temp_dir()
                    with open(os.path.join(temp_dir, "config.json"), "w") as f:
                        json.dump({"talk_objects": processed_talk_objects}, f)
                    config.audio_path = temp_dir
                    logging.info(f"Processed {len(processed_talk_objects)} talk objects")

        logging.info("lightx2v prepared config: " + json.dumps(config, indent=2, ensure_ascii=False))

        return (config,)


class LightX2VModularInferenceV2:
    """Pure inference node that takes prepared config and runs inference."""

    _current_runner = None
    _current_config_hash = None

    def __init__(self):
        if not hasattr(self.__class__, "_current_runner"):
            self.__class__._current_runner = None
        if not hasattr(self.__class__, "_current_config_hash"):
            self.__class__._current_config_hash = None

        self.config_builder = ConfigBuilder()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared_config": (
                    "PREPARED_CONFIG",
                    {"tooltip": "Fully prepared configuration from ConfigCombinerV2"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "PREV_FRAMES")
    RETURN_NAMES = ("images", "audio", "prev_frames")
    FUNCTION = "generate"
    CATEGORY = "LightX2V/InferenceV2"

    def _get_config_hash(self, config) -> str:
        """Get hash of configuration to detect changes."""
        return self.config_builder.get_config_hash(config)

    def generate(self, prepared_config):
        """Run inference with prepared configuration."""
        config = prepared_config

        try:
            config_hash = self._get_config_hash(config)

            current_runner = getattr(self.__class__, "_current_runner", None)
            current_config_hash = getattr(self.__class__, "_current_config_hash", None)

            needs_reinit = current_runner is None or current_config_hash != config_hash or getattr(config, "lazy_load", False)

            logging.info(f"Needs reinit: {needs_reinit}, old config hash: {current_config_hash}, new config hash: {config_hash}")
            if needs_reinit:
                if current_runner is not None:
                    # current_runner.end_run()
                    del self.__class__._current_runner
                    torch.cuda.empty_cache()
                    gc.collect()
                formatted_config = set_config(config)
                self.__class__._current_runner = init_runner(formatted_config)
                self.__class__._current_config_hash = config_hash

            progress = ProgressBar(100)

            def update_progress(current_step, _total):
                progress.update_absolute(current_step)

            current_runner = self.__class__._current_runner

            if hasattr(current_runner, "set_progress_callback"):
                current_runner.set_progress_callback(update_progress)

            config["return_result_tensor"] = True
            config["save_result_path"] = ""
            config["negative_prompt"] = config.get("negative_prompt", "")
            config["prev_section_info_path"] = config.get("prev_section_info_path", "")
            input_info = set_input_info(config)
            current_runner.set_config(config)

            result_dict = current_runner.run_pipeline(input_info)

            images = result_dict.get("video", None)
            audio = result_dict.get("audio", None)
            prev_frames = result_dict.get("for_next_section", None)

            if images is not None and images.numel() > 0:
                images = images.cpu()
                if images.dtype != torch.float32:
                    images = images.float()

            if getattr(config, "unload_after_inference", False):
                if hasattr(self.__class__, "_current_runner"):
                    del self.__class__._current_runner
                self.__class__._current_runner = None
                self.__class__._current_config_hash = None

            torch.cuda.empty_cache()
            gc.collect()

            return (images, audio, prev_frames)

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
    "LightX2VConfigCombinerV2": LightX2VConfigCombinerV2,
    "LightX2VConfigCombinerV3": LightX2VConfigCombinerV3,
    "LightX2VModularInferenceV2": LightX2VModularInferenceV2,
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
    "LightX2VConfigCombinerV2": "LightX2V Config Combiner V2",
    "LightX2VConfigCombinerV3": "LightX2V Config Combiner V3",
    "LightX2VModularInferenceV2": "LightX2V Modular Inference V2",
    "LightX2VTalkObjectInput": "LightX2V Talk Object Input (Single)",
    "LightX2VTalkObjectsCombiner": "LightX2V Talk Objects Combiner",
    "LightX2VTalkObjectsFromFiles": "LightX2V Talk Objects From Files",
    "LightX2VTalkObjectsFromJSON": "LightX2V Talk Objects From JSON (API)",
}
