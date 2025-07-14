"""Factory pattern for creating LightX2V components."""
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import logging

from .config import EncoderConfig, ModelConfig, VideoConfig
from .models import (
    LightX2VT5Encoder,
    LightX2VClipVisionEncoder,
    LightX2VVae,
    LightX2VModel,
)

# Import original LightX2V modules
from ..lightx2v.lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from ..lightx2v.lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel as ClipVisionModel
from ..lightx2v.lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from ..lightx2v.lightx2v.models.networks.wan.model import WanModel
from ..lightx2v.lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper


class LightX2VFactory:
    """Factory for creating LightX2V components with proper configuration."""
    
    @staticmethod
    def create_t5_encoder(
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        cpu_offload: bool = False,
    ) -> LightX2VT5Encoder:
        """Create a T5 encoder with configuration."""
        model_path = Path(model_path)
        
        # Auto-detect tokenizer path if not provided
        if tokenizer_path is None:
            tokenizer_path = model_path.parent / "google" / "umt5-xxl"
            if not tokenizer_path.exists():
                raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        
        config = EncoderConfig(
            model_path=model_path,
            tokenizer_path=Path(tokenizer_path),
            dtype=dtype or torch.bfloat16,
            device=device or torch.device("cuda"),
            cpu_offload=cpu_offload,
        )
        
        # Create underlying T5 model
        t5_model = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.dtype,
            device=config.device,
            checkpoint_path=str(config.model_path),
            tokenizer_path=str(config.tokenizer_path),
            shard_fn=None,
            cpu_offload=config.cpu_offload,
        )
        
        return LightX2VT5Encoder(t5_model, config)
    
    @staticmethod
    def create_clip_vision_encoder(
        model_path: Union[str, Path],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        clip_quantized: bool = False,
        clip_quantized_ckpt: Optional[Union[str, Path]] = None,
        quant_scheme: Optional[str] = None,
    ) -> LightX2VClipVisionEncoder:
        """Create a CLIP vision encoder with configuration."""
        config = EncoderConfig(
            model_path=Path(model_path),
            dtype=dtype or torch.float16,
            device=device or torch.device("cuda"),
            clip_quantized=clip_quantized,
            clip_quantized_ckpt=Path(clip_quantized_ckpt) if clip_quantized_ckpt else None,
            quant_scheme=quant_scheme,
        )
        
        # Create underlying CLIP model
        clip_model = ClipVisionModel(
            dtype=config.dtype,
            device=config.device,
            checkpoint_path=str(config.model_path),
            clip_quantized=config.clip_quantized,
            clip_quantized_ckpt=str(config.clip_quantized_ckpt) if config.clip_quantized_ckpt else None,
            quant_scheme=config.quant_scheme,
        )
        
        return LightX2VClipVisionEncoder(clip_model, config)
    
    @staticmethod
    def create_vae(
        model_path: Union[str, Path],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        parallel: bool = False,
        z_dim: int = 16,
    ) -> LightX2VVae:
        """Create a VAE with configuration."""
        config = EncoderConfig(
            model_path=Path(model_path),
            dtype=dtype or torch.float16,
            device=device or torch.device("cuda"),
            parallel=parallel,
            z_dim=z_dim,
        )
        
        # Create underlying VAE model
        vae_model = WanVAE(
            z_dim=config.z_dim,
            vae_pth=str(config.model_path),
            dtype=config.dtype,
            device=config.device,
            parallel=config.parallel,
        )
        
        return LightX2VVae(vae_model, config)
    
    @staticmethod
    def create_model(
        config: ModelConfig,
        video_config: Optional[VideoConfig] = None,
    ) -> LightX2VModel:
        """Create a complete LightX2V model with configuration."""
        # Load config.json if it exists
        config_json_path = config.model_path / "config.json"
        config_json = {}
        
        if config_json_path.exists():
            import json
            with open(config_json_path, "r") as f:
                config_json = json.load(f)
        else:
            logging.warning(f"Config file not found at {config_json_path}")
        
        # Create model configuration dict
        model_config_dict = {
            "model_path": str(config.model_path),
            "task": config.model_type,
            "dtype": config.to_dtype(),
            "device": config.to_device(),
            "attention_type": config.attention_type,
            "cpu_offload": config.cpu_offload,
            "offload_granularity": config.offload_granularity,
            "mm_config": config.mm_config,
            "model_cls": "wan2.1",
            "do_mm_calib": False,
            "parallel_attn_type": None,
            "parallel_vae": False,
            "use_bfloat16": config.to_dtype() == torch.bfloat16,
        }
        
        # Add video config if provided
        if video_config:
            model_config_dict.update({
                "target_height": video_config.height,
                "target_width": video_config.width,
                "target_video_length": video_config.num_frames,
                "vae_stride": video_config.vae_stride,
                "patch_size": video_config.patch_size,
                "max_area": video_config.max_area,
            })
        
        # Merge with config.json
        model_config_dict.update(config_json)
        
        # Create EasyDict for compatibility
        from easydict import EasyDict
        easydict_config = EasyDict(model_config_dict)
        
        # Create underlying model
        wan_model = WanModel(
            str(config.model_path),
            easydict_config,
            config.to_device()
        )
        
        # Apply LoRA if specified
        if config.lora_path and config.lora_path.exists():
            logging.info(f"Applying LoRA from {config.lora_path}")
            lora_wrapper = WanLoraWrapper(wan_model)
            lora_name = lora_wrapper.load_lora(str(config.lora_path))
            lora_wrapper.apply_lora(lora_name, config.lora_strength)
            logging.info(f"LoRA {lora_name} applied successfully")
        
        return LightX2VModel(wan_model, config, easydict_config)
    
    @staticmethod
    def create_from_paths(
        model_dir: Union[str, Path],
        model_name: str,
        model_type: str = "i2v",
        precision: str = "bf16",
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method to create all components from a model directory."""
        model_dir = Path(model_dir)
        
        # Create model config
        model_config = ModelConfig(
            model_path=model_dir / model_name,
            model_type=model_type,
            precision=precision,
            device=device,
            **kwargs
        )
        
        # Create components
        components = {
            "model": LightX2VFactory.create_model(model_config),
        }
        
        # Try to create encoders if paths exist
        t5_path = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        if t5_path.exists():
            components["t5_encoder"] = LightX2VFactory.create_t5_encoder(
                t5_path,
                dtype=model_config.to_dtype(),
                device=model_config.to_device(),
            )
        
        clip_path = model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        if clip_path.exists():
            components["clip_encoder"] = LightX2VFactory.create_clip_vision_encoder(
                clip_path,
                dtype=torch.float16,  # CLIP typically uses fp16
                device=model_config.to_device(),
            )
        
        vae_path = model_dir / "Wan2.1_VAE.pth"
        if vae_path.exists():
            components["vae"] = LightX2VFactory.create_vae(
                vae_path,
                dtype=torch.float16,  # VAE typically uses fp16
                device=model_config.to_device(),
            )
        
        return components