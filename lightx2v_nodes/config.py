"""Configuration management for LightX2V."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import torch
from easydict import EasyDict
import importlib.util


@dataclass
class TeaCacheConfig:
    """Configuration for TeaCache optimization."""

    rel_l1_thresh: float = 0.26
    start_percent: float = 0.1
    end_percent: float = 1.0
    cache_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    coefficients: List[List[float]] = field(default_factory=list)
    use_ret_steps: bool = False
    mode: str = "e"


@dataclass
class VideoConfig:
    """Video generation configuration."""

    target_width: int = 832
    target_height: int = 480
    target_video_length: int = 81
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    patch_size: Tuple[int, int, int] = (1, 2, 2)

    @property
    def max_area(self) -> int:
        return self.target_height * self.target_width


@dataclass
class ModelConfig:
    """Model loading and inference configuration."""

    model_path: Path
    model_type: str = "i2v"  # "t2v" or "i2v"
    precision: str = "bf16"  # "bf16", "fp16", "fp32"
    device: str = "cuda"
    attention_type: str = "flash_attn3"
    cpu_offload: bool = False
    offload_granularity: str = "phase"  # "block" or "phase"

    # Optional configurations
    lora_path: Optional[Path] = None
    lora_strength: float = 1.0
    mm_config: Dict[str, Any] = field(default_factory=dict)

    # Inference settings
    steps: int = 20
    shift: float = 5.0
    cfg_scale: float = 5.0
    seed: int = 42
    feature_caching: str = "NoCaching"

    def to_dtype(self) -> torch.dtype:
        """Convert precision string to torch dtype."""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return dtype_map[self.precision]

    def to_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "cuda":
            return torch.device("cuda")
        return torch.device("cpu")


@dataclass
class EncoderConfig:
    """Encoder configuration."""

    model_path: Path
    dtype: torch.dtype
    device: torch.device

    # T5 specific
    text_len: int = 512
    tokenizer_path: Optional[Path] = None
    cpu_offload: bool = False

    # CLIP specific
    clip_quantized: bool = False
    clip_quantized_ckpt: Optional[Path] = None
    quant_scheme: Optional[str] = None

    # VAE specific
    z_dim: int = 16
    parallel: bool = False


@dataclass
class LightX2VConfig:
    """Main configuration container for LightX2V."""

    model: ModelConfig
    video: VideoConfig
    teacache: Optional[TeaCacheConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LightX2VConfig":
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        video_config = VideoConfig(**config_dict.get("video", {}))

        teacache_config = None
        if "teacache" in config_dict:
            teacache_config = TeaCacheConfig(**config_dict["teacache"])

        return cls(model=model_config, video=video_config, teacache=teacache_config)

    def to_easydict(self) -> EasyDict:
        """Convert to EasyDict for legacy compatibility."""
        config_dict = {
            "model_path": str(self.model.model_path),
            "task": self.model.model_type,
            "dtype": self.model.to_dtype(),
            "device": self.model.to_device(),
            "attention_type": self.model.attention_type,
            "cpu_offload": self.model.cpu_offload,
            "offload_granularity": self.model.offload_granularity,
            "target_height": self.video.target_height,
            "target_width": self.video.target_width,
            "target_video_length": self.video.target_video_length,
            "vae_stride": self.video.vae_stride,
            "patch_size": self.video.patch_size,
            "infer_steps": self.model.steps,
            "sample_shift": self.model.shift,
            "sample_guide_scale": self.model.cfg_scale,
            "seed": self.model.seed,
            "enable_cfg": self.model.cfg_scale != 1.0,
            "mm_config": self.model.mm_config,
            "feature_caching": self.model.feature_caching,
        }

        if self.teacache:
            config_dict.update(
                {
                    "feature_caching": "Tea",
                    "teacache_thresh": self.teacache.rel_l1_thresh,
                    "use_ret_steps": self.teacache.use_ret_steps,
                    "coefficients": self.teacache.coefficients,
                }
            )
        else:
            config_dict["feature_caching"] = "NoCaching"

        if self.model.lora_path:
            config_dict["lora_path"] = str(self.model.lora_path)
            config_dict["strength_model"] = self.model.lora_strength

        return EasyDict(config_dict)


def is_fp8_supported_gpu():
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major == 8 and minor == 9) or (major >= 9)


def is_module_installed(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ModuleNotFoundError:
        return False


def get_available_quant_ops():
    available_ops = []

    vllm_installed = is_module_installed("vllm")
    if vllm_installed:
        available_ops.append(("vllm", True))
    else:
        available_ops.append(("vllm", False))

    sgl_installed = is_module_installed("sgl_kernel")
    if sgl_installed:
        available_ops.append(("sgl", True))
    else:
        available_ops.append(("sgl", False))

    q8f_installed = is_module_installed("q8_kernels")
    if q8f_installed:
        available_ops.append(("q8f", True))
    else:
        available_ops.append(("q8f", False))

    return available_ops


def get_available_attn_ops():
    available_ops = []

    vllm_installed = is_module_installed("flash_attn")
    if vllm_installed:
        available_ops.append(("flash_attn2", True))
    else:
        available_ops.append(("flash_attn2", False))

    sgl_installed = is_module_installed("flash_attn_interface")
    if sgl_installed:
        available_ops.append(("flash_attn3", True))
    else:
        available_ops.append(("flash_attn3", False))

    q8f_installed = is_module_installed("sageattention")
    if q8f_installed:
        available_ops.append(("sage_attn2", True))
    else:
        available_ops.append(("sage_attn2", False))

    torch_installed = is_module_installed("torch")
    if torch_installed:
        available_ops.append(("torch_sdpa", True))
    else:
        available_ops.append(("torch_sdpa", False))

    return available_ops
