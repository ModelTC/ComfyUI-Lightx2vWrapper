import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from easydict import EasyDict

from .bridge import (
    ModularConfigManager,
)
from .data_models import (
    CombinedConfig,
    InferenceConfig,
    LoRAConfig,
    MemoryOptimizationConfig,
    QuantizationConfig,
    TalkObject,
    TalkObjectsConfig,
    TeaCacheConfig,
)
from .file_handlers import (
    AudioFileHandler,
    ComfyUIFileResolver,
    MaskFileHandler,
    TempFileManager,
)
from .model_utils import get_lora_full_path, get_model_full_path


class ConfigValidator:
    """Validator for configuration parameters."""

    @staticmethod
    def validate_inference_config(config: InferenceConfig) -> InferenceConfig:
        """Validate and adjust inference configuration."""
        # Ensure video length is valid
        if config.video_length < 16:
            logging.warning("Video length is too short, setting to 16")
            config.video_length = 16

        # Adjust video length to be compatible with model requirements
        remainder = (config.video_length - 1) % 4
        if remainder != 0:
            config.video_length = config.video_length + (4 - remainder)

        # Set FPS based on model class
        if config.model_cls == "hunyuan":
            config.fps = 24
        else:
            config.fps = 16

        # Handle random seed
        if config.seed == -1:
            config.seed = np.random.randint(0, 2**32 - 1)

        return config

    @staticmethod
    def validate_dimensions(height: int, width: int) -> Tuple[int, int]:
        """Validate and adjust video dimensions."""
        # Ensure dimensions are multiples of 8
        height = (height // 8) * 8
        width = (width // 8) * 8

        # Ensure minimum dimensions
        height = max(64, height)
        width = max(64, width)

        # Ensure maximum dimensions
        height = min(2048, height)
        width = min(2048, width)

        return height, width


class InferenceConfigBuilder:
    """Builder for inference configuration."""

    def __init__(self):
        self.validator = ConfigValidator()

    def build(
        self,
        model_cls: str,
        model_name: str,
        task: str,
        infer_steps: int,
        seed: int,
        cfg_scale: float,
        cfg_scale2: float,
        sample_shift: int,
        height: int,
        width: int,
        duration: float,
        attention_type: str,
        **optional_params,
    ) -> InferenceConfig:
        """Build inference configuration from parameters."""
        # Get model path
        model_path = get_model_full_path(model_name)

        # Calculate video length from duration
        fps = 24 if model_cls == "hunyuan" else 16
        video_length = int(round(duration * fps))

        # Validate dimensions
        height, width = self.validator.validate_dimensions(height, width)

        # Create base config
        config = InferenceConfig(
            model_cls=model_cls,
            model_path=model_path,
            task=task,
            infer_steps=infer_steps,
            seed=seed,
            cfg_scale=cfg_scale,
            cfg_scale2=cfg_scale2,
            sample_shift=sample_shift,
            height=height,
            width=width,
            video_length=video_length,
            fps=fps,
            video_duration=duration,
            attention_type=attention_type,
        )

        # Handle optional parameters
        self._apply_optional_params(config, optional_params)

        # Validate final config
        config = self.validator.validate_inference_config(config)

        return config

    def _apply_optional_params(self, config: InferenceConfig, optional_params: Dict[str, Any]):
        """Apply optional parameters to config."""
        # Handle denoising steps
        if "denoising_steps" in optional_params:
            steps_str = optional_params["denoising_steps"]
            if steps_str and steps_str.strip():
                try:
                    steps_list = [int(s.strip()) for s in steps_str.split(",")]
                    config.denoising_step_list = steps_list
                    config.infer_steps = len(steps_list)
                except ValueError:
                    logging.warning(f"Invalid denoising steps: {steps_str}")

        # Handle other optional params
        for param in [
            "resize_mode",
            "fixed_area",
            "segment_length",
            "prev_frame_length",
            "use_tiny_vae",
        ]:
            if param in optional_params:
                setattr(config, param, optional_params[param])

        # Special handling for seko models
        if "seko" in config.model_cls:
            config.video_length = optional_params.get("segment_length", 81)
            config.use_31_block = False
            if "prev_frame_length" in optional_params:
                config.prev_frame_length = optional_params["prev_frame_length"]


class TalkObjectConfigBuilder:
    """Builder for talk object configurations."""

    def __init__(self):
        self.audio_handler = AudioFileHandler()
        self.mask_handler = MaskFileHandler()
        self.resolver = ComfyUIFileResolver()

    def build_from_input(
        self,
        name: str,
        audio: Optional[Any] = None,
        mask: Optional[Any] = None,
        save_to_input: bool = True,
    ) -> TalkObject:
        if audio is None:
            return None

        talk_object = TalkObject(name=name)

        if save_to_input and audio is not None:
            audio_path = self._save_audio_to_input(name, audio)
            if audio_path:
                talk_object.audio = audio_path

        else:
            talk_object.audio = audio

        if mask is not None:
            if save_to_input:
                mask_path = self._save_mask_to_input(name, mask)
                if mask_path:
                    talk_object.mask = mask_path
            else:
                talk_object.mask = mask

        return talk_object

    def build_from_json(self, json_config: str) -> Optional[TalkObjectsConfig]:
        """Build talk objects configuration from JSON."""
        try:
            objects_data = json.loads(json_config)
            if not isinstance(objects_data, list):
                logging.error("JSON config must be a list")
                return None

            config = TalkObjectsConfig()

            for obj_data in objects_data:
                if not isinstance(obj_data, dict) or "audio" not in obj_data:
                    continue

                talk_obj = TalkObject(
                    name=obj_data.get("name", "unknown"),
                    audio=obj_data["audio"],
                    mask=obj_data.get("mask"),
                )
                config.add_object(talk_obj)

            return config if config.talk_objects else None

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")

    def build_from_files(self, audio_files: str, mask_files: str = "", names: str = "") -> Optional[TalkObjectsConfig]:
        """Build talk objects configuration from file lists."""
        audio_list = [f.strip() for f in audio_files.split("\n") if f.strip()]
        if not audio_list:
            return None

        mask_list = [f.strip() for f in mask_files.split("\n") if f.strip()] if mask_files else []
        name_list = [n.strip() for n in names.split("\n") if n.strip()] if names else []

        config = TalkObjectsConfig()

        for i, audio_file in enumerate(audio_list):
            talk_obj = TalkObject(
                name=name_list[i] if i < len(name_list) else f"person_{i + 1}",
                audio=audio_file,
                mask=mask_list[i] if i < len(mask_list) else None,
            )
            config.add_object(talk_obj)

        return config

    def _save_audio_to_input(self, name: str, audio_data: Any) -> Optional[str]:
        try:
            filename = f"{name}_audio_{uuid.uuid4().hex[:8]}.wav"
            return self.resolver.save_to_input(audio_data, filename, self.audio_handler)
        except Exception as e:
            logging.error(f"Failed to save audio: {e}")
            return None

    def _save_mask_to_input(self, name: str, mask_data: Any) -> Optional[str]:
        try:
            filename = f"{name}_mask_{uuid.uuid4().hex[:8]}.png"
            return self.resolver.save_to_input(mask_data, filename, self.mask_handler)
        except Exception as e:
            logging.error(f"Failed to save mask: {e}")
            return None


class ConfigBuilder:
    """Main configuration builder that combines all configs."""

    def __init__(self):
        self.manager = ModularConfigManager()

    def combine_configs(
        self,
        inference_config: InferenceConfig,
        teacache_config: Optional[TeaCacheConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        memory_config: Optional[MemoryOptimizationConfig] = None,
        lora_chain: Optional[List[Dict[str, Any]]] = None,
        talk_objects_config: Optional[TalkObjectsConfig] = None,
    ) -> EasyDict:
        # Create combined configuration
        combined = CombinedConfig(
            inference=inference_config,
            teacache=teacache_config,
            quantization=quantization_config,
            memory=memory_config,
            talk_objects=talk_objects_config,
        )

        # Process LoRA configs if provided
        if lora_chain:
            for lora_dict in lora_chain:
                lora_config = LoRAConfig(path=lora_dict["path"], strength=lora_dict.get("strength", 1.0))
                combined.lora_configs.append(lora_config)

        # Build final configuration from combined config
        final_config = self.manager.build_final_config_from_combined(combined)

        return final_config

    def get_config_hash(self, config: EasyDict) -> str:
        """Generate hash for configuration to detect changes."""
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


class LoRAChainBuilder:
    """Builder for LoRA chain configurations."""

    @staticmethod
    def build_chain(lora_name: str, strength: float, existing_chain: Optional[List[Dict]] = None) -> List[Dict]:
        """Build or extend a LoRA chain."""
        if existing_chain is None:
            chain = []
        else:
            chain = existing_chain.copy()

        lora_path = get_lora_full_path(lora_name)
        if lora_path:
            lora_config = {"path": lora_path, "strength": strength}
            chain.append(lora_config)

        return chain
