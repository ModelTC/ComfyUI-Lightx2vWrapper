"""Model wrappers for LightX2V components."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .config import EncoderConfig, ModelConfig, VideoConfig


class BaseModel(ABC):
    """Base class for all LightX2V models."""
    
    def __init__(self, config: Union[EncoderConfig, ModelConfig]):
        self.config = config
    
    @abstractmethod
    def to(self, device: torch.device) -> "BaseModel":
        """Move model to device."""
        pass


class LightX2VT5Encoder(BaseModel):
    """Wrapper for T5 text encoder."""
    
    def __init__(self, t5_model: Any, config: EncoderConfig):
        super().__init__(config)
        self._model = t5_model
    
    def encode(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text prompts."""
        context = self._model.infer(prompts)
        return {"context": context}
    
    def encode_with_negative(
        self, 
        prompt: str, 
        negative_prompt: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode prompt with negative prompt."""
        context = self._model.infer([prompt])
        context_null = self._model.infer([negative_prompt if negative_prompt else ""])
        return {
            "context": context,
            "context_null": context_null
        }
    
    def to(self, device: torch.device) -> "LightX2VT5Encoder":
        """Move encoder to device."""
        # T5 model handles device internally
        return self
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self.config.device


class LightX2VClipVisionEncoder(BaseModel):
    """Wrapper for CLIP vision encoder."""
    
    def __init__(self, clip_model: Any, config: EncoderConfig):
        super().__init__(config)
        self._model = clip_model
    
    def encode(
        self, 
        images: torch.Tensor, 
        video_config: Optional[VideoConfig] = None
    ) -> torch.Tensor:
        """Encode images with CLIP."""
        if video_config:
            # Convert VideoConfig to dict format expected by CLIP
            config_dict = {
                "target_height": video_config.height,
                "target_width": video_config.width,
                "target_video_length": video_config.num_frames,
                "vae_stride": video_config.vae_stride,
                "patch_size": video_config.patch_size,
            }
        else:
            config_dict = {}
        
        # Ensure images are in correct format [B, C, T, H, W]
        if images.dim() == 3:  # [C, H, W]
            images = images.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
        elif images.dim() == 4:  # [B, C, H, W]
            images = images.unsqueeze(2)  # [B, C, 1, H, W]
        
        return self._model.visual(images, config_dict)
    
    def to(self, device: torch.device) -> "LightX2VClipVisionEncoder":
        """Move encoder to device."""
        # CLIP model handles device internally
        return self
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self.config.device


class LightX2VVae(BaseModel):
    """Wrapper for VAE."""
    
    def __init__(self, vae_model: Any, config: EncoderConfig):
        super().__init__(config)
        self._model = vae_model
    
    def encode(
        self, 
        videos: List[torch.Tensor], 
        video_config: Optional[VideoConfig] = None,
        cpu_offload: bool = False
    ) -> List[torch.Tensor]:
        """Encode videos to latent space."""
        config_dict = {"cpu_offload": cpu_offload}
        
        if video_config:
            config_dict.update({
                "target_height": video_config.height,
                "target_width": video_config.width,
                "target_video_length": video_config.num_frames,
                "vae_stride": video_config.vae_stride,
                "patch_size": video_config.patch_size,
            })
        
        from easydict import EasyDict
        return self._model.encode(videos, EasyDict(config_dict))
    
    def decode(
        self, 
        latents: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        cpu_offload: bool = False
    ) -> torch.Tensor:
        """Decode latents to video."""
        from easydict import EasyDict
        config = EasyDict({"cpu_offload": cpu_offload})
        
        return self._model.decode(latents, generator=generator, config=config)
    
    def to(self, device: torch.device) -> "LightX2VVae":
        """Move VAE to device."""
        # VAE model handles device internally
        return self
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self.config.device


class LightX2VModel(BaseModel):
    """Wrapper for main LightX2V model."""
    
    def __init__(self, wan_model: Any, config: ModelConfig, easydict_config: Any):
        super().__init__(config)
        self._model = wan_model
        self._easydict_config = easydict_config
        self._scheduler = None
    
    def set_scheduler(self, scheduler: Any):
        """Set the scheduler for the model."""
        self._scheduler = scheduler
        self._model.set_scheduler(scheduler)
    
    def infer(self, inputs: Dict[str, Any]):
        """Run inference."""
        return self._model.infer(inputs)
    
    def prepare_inputs(
        self,
        text_embeddings: Dict[str, torch.Tensor],
        image_embeddings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare inputs for inference."""
        inputs = {
            "text_encoder_output": text_embeddings,
            "image_encoder_output": image_embeddings or {},
        }
        return inputs
    
    def to(self, device: torch.device) -> "LightX2VModel":
        """Move model to device."""
        # Model handles device internally
        return self
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self.config.to_device()
    
    @property
    def easydict_config(self) -> Any:
        """Get EasyDict config for compatibility."""
        return self._easydict_config