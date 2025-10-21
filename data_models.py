"""Data models for LightX2V ComfyUI wrapper."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class TalkObject:
    """Single talk object containing audio and optional mask."""

    name: str
    audio: Optional[Union[str, Dict[str, Any], torch.Tensor, np.ndarray]] = None
    mask: Optional[Union[str, torch.Tensor, np.ndarray]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline."""
        result = {"name": self.name}

        if isinstance(self.audio, str):
            result["audio"] = self.audio
        elif self.audio is not None:
            result["audio_data"] = self.audio

        if isinstance(self.mask, str):
            result["mask"] = self.mask
        elif self.mask is not None:
            result["mask_data"] = self.mask

        return result


@dataclass
class InferenceConfig:
    """Basic inference configuration."""

    model_cls: str = "wan2.1"
    model_path: str = ""
    task: str = "i2v"
    infer_steps: int = 4
    seed: int = 42
    cfg_scale: float = 5.0
    cfg_scale2: float = 5.0
    sample_shift: int = 5
    height: int = 1280
    width: int = 720
    video_length: int = 81
    fps: int = 16
    video_duration: float = 5.0
    attention_type: str = "torch_sdpa"
    use_31_block: bool = True

    # Optional parameters
    denoising_step_list: Optional[List[int]] = None
    resize_mode: str = "adaptive"
    fixed_area: str = "720p"
    segment_length: int = 81
    prev_frame_length: int = 5
    use_tiny_vae: bool = False

    # Runtime parameters
    prompt: str = ""
    negative_prompt: str = ""
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class TeaCacheConfig:
    """TeaCache configuration."""

    enable: bool = False
    threshold: float = 0.26
    use_ret_steps: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable": self.enable,
            "threshold": self.threshold,
            "use_ret_steps": self.use_ret_steps,
        }


@dataclass
class QuantizationConfig:
    """Quantization configuration."""

    dit_quant_scheme: str = "Default"
    t5_quant_scheme: str = "Default"
    clip_quant_scheme: str = "Default"
    adapter_quant_scheme: str = "Default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dit_quant_scheme": self.dit_quant_scheme,
            "t5_quant_scheme": self.t5_quant_scheme,
            "clip_quant_scheme": self.clip_quant_scheme,
            "adapter_quant_scheme": self.adapter_quant_scheme,
        }


@dataclass
class MemoryOptimizationConfig:
    """Memory optimization configuration."""

    enable_rotary_chunk: bool = False
    rotary_chunk_size: int = 100
    clean_cuda_cache: bool = False
    cpu_offload: bool = True
    offload_granularity: str = "block"
    offload_ratio: float = 1.0
    t5_cpu_offload: bool = True
    t5_offload_granularity: str = "model"
    audio_encoder_cpu_offload: bool = True
    audio_adapter_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    use_tiling_vae: bool = True
    lazy_load: bool = False
    unload_after_inference: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class LoRAConfig:
    """LoRA configuration."""

    path: str
    strength: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "strength": self.strength}


@dataclass
class TalkObjectsConfig:
    talk_objects: List[TalkObject] = field(default_factory=list)

    def add_object(self, talk_object: TalkObject):
        self.talk_objects.append(talk_object)

    def to_dict(self) -> Dict[str, Any]:
        return {"talk_objects": [obj.to_dict() for obj in self.talk_objects]}

    def to_list(self) -> List[Dict[str, Any]]:
        return [obj.to_dict() for obj in self.talk_objects]


@dataclass
class CombinedConfig:
    """Combined configuration for all modules."""

    inference: Optional[InferenceConfig] = None
    teacache: Optional[TeaCacheConfig] = None
    quantization: Optional[QuantizationConfig] = None
    memory: Optional[MemoryOptimizationConfig] = None
    lora_configs: List[LoRAConfig] = field(default_factory=list)
    talk_objects: Optional[TalkObjectsConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline."""
        result = {}

        if self.inference:
            result.update(self.inference.to_dict())

        if self.teacache:
            result["teacache"] = self.teacache.to_dict()

        if self.quantization:
            result["quantization"] = self.quantization.to_dict()

        if self.memory:
            result["memory"] = self.memory.to_dict()

        if self.lora_configs:
            result["lora_configs"] = [lora.to_dict() for lora in self.lora_configs]

        if self.talk_objects:
            result["talk_objects"] = self.talk_objects.to_list()

        return result
