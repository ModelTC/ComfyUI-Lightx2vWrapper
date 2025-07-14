"""Universal bridge between LightX2V and ComfyUI for automatic adaptation."""

import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from PIL import Image
import tempfile
from easydict import EasyDict
import asyncio
import gc

# from .config import get_available_attn_ops, get_available_quant_ops

from ..lightx2v.lightx2v.utils.set_config import get_default_config
from ..lightx2v.lightx2v.infer import init_runner


class LightX2VBridge:
    """Universal bridge for LightX2V and ComfyUI integration."""

    def __init__(self):
        self._model_registry = None
        self._config_registry = None
        self._quantization_registry = None
        self._current_runner = None  # Only store one runner at a time
        self._current_runner_key = None  # Track current runner identity

    @property
    def model_registry(self):
        """Lazy load model registry."""
        if self._model_registry is None:
            self._model_registry = self._discover_models()
        return self._model_registry

    @property
    def config_registry(self):
        """Lazy load config registry."""
        if self._config_registry is None:
            self._config_registry = self._discover_configs()
        return self._config_registry

    @property
    def quantization_registry(self):
        """Lazy load quantization registry."""
        if self._quantization_registry is None:
            self._quantization_registry = self._discover_quantization_configs()
        return self._quantization_registry

    def _discover_models(self) -> List[str]:
        """Discover available model classes from LightX2V."""
        return ["wan2.1", "hunyuan", "wan2.1_audio", "wan2.1_distill"]

    def _discover_configs(self) -> Dict[str, Dict]:
        """Discover all available config files for single-card inference."""
        configs = {}
        base_dir = Path(__file__).parent.parent / "lightx2v" / "configs"

        if base_dir.exists():
            for config_file in base_dir.rglob("*.json"):
                # Create a descriptive key
                relative_path = config_file.relative_to(base_dir)
                path_str = str(relative_path)

                # 排除分布式推理和部署相关的配置
                if any(exclude in path_str for exclude in ["deploy", "causvid", "skyreels", "cogvideox"]):
                    continue

                # 包含蒸馏模型配置
                if "distill" in path_str:
                    config_info = {
                        "path": config_file,
                        "model_type": "wan2.1_distill",
                        "task": self._extract_task_from_path(path_str),
                        "is_quantized": False,
                        "is_distilled": True,
                        "description": self._generate_config_description(relative_path, is_distilled=True),
                    }
                    key = str(relative_path).replace("/", "_").replace(".json", "")
                    configs[key] = config_info
                    continue

                # 排除量化配置（单独处理）
                if "quantization" in path_str:
                    continue

                # 创建配置信息
                config_info = {
                    "path": config_file,
                    "model_type": relative_path.parts[0],  # wan, hunyuan, etc.
                    "task": self._extract_task_from_path(path_str),
                    "is_quantized": False,
                    "description": self._generate_config_description(relative_path),
                }

                key = str(relative_path).replace("/", "_").replace(".json", "")
                configs[key] = config_info

        return configs

    def _discover_quantization_configs(self) -> Dict[str, Dict]:
        """Discover quantization configurations."""
        quant_configs = {}
        base_dir = Path(__file__).parent.parent / "lightx2v" / "configs" / "quantization"

        if base_dir.exists():
            for config_file in base_dir.rglob("*.json"):
                relative_path = config_file.relative_to(base_dir.parent)
                path_str = str(relative_path)

                config_info = {
                    "path": config_file,
                    "model_type": relative_path.parts[1],  # wan, hunyuan, etc.
                    "task": self._extract_task_from_path(path_str),
                    "is_quantized": True,
                    "description": self._generate_config_description(relative_path, is_quantized=True),
                }

                key = f"quant_{str(relative_path).replace('/', '_').replace('.json', '')}"
                quant_configs[key] = config_info

        return quant_configs

    def _extract_task_from_path(self, path_str: str) -> str:
        """Extract task type from config path."""
        if "i2v" in path_str:
            return "i2v"
        elif "t2v" in path_str:
            return "t2v"
        else:
            return "unknown"

    def _generate_config_description(
        self,
        relative_path: Path,
        is_quantized: bool = False,
        is_distilled: bool = False,
    ) -> str:
        """Generate human-readable description for config."""
        parts = relative_path.parts
        model_type = parts[0] if not is_quantized else parts[1]
        task = self._extract_task_from_path(str(relative_path))

        desc = f"{model_type.upper()} {task.upper()}"
        if is_quantized:
            desc += " (Quantized)"
        if is_distilled:
            desc += " (Distilled 4-step)"

        return desc

    def get_quantization_model_path(self, model_type: str, mm_type: str) -> str:
        """Get quantization model path based on model type and mm_type."""
        # 定义量化模型的固定路径结构
        base_path = Path(__file__).parent.parent / "lightx2v" / "models" / "quantized"

        # 根据mm_type确定具体的模型文件名
        mm_type_to_filename = {
            "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm": "int8_dynamic_vllm.safetensors",
            "W-int8-channel-sym-A-fp16-dynamic-Vllm": "int8_fp16_dynamic_vllm.safetensors",
            "W-fp16-A-fp16-dynamic-Vllm": "fp16_dynamic_vllm.safetensors",
        }

        filename = mm_type_to_filename.get(mm_type, "default_quantized.safetensors")
        model_path = base_path / model_type / filename

        return str(model_path)

    def load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load a config file."""
        with open(config_path, "r") as f:
            return json.load(f)

    def get_runner(self, config: EasyDict):
        """Get or create a runner for the given config."""
        # Create a simple key based on essential parameters only
        # Only model_cls and model_path are essential for runner identity
        key = f"{config.model_cls}_{config.model_path}"  # type:ignore

        # If we have a different runner, clear the old one first
        if self._current_runner_key != key:
            self.clear_cache()
            self._current_runner = init_runner(config)
            self._current_runner_key = key

        return self._current_runner

    def clear_cache(self):
        """Clear cached runner to free memory."""
        if self._current_runner is not None:
            # Clean up the runner if it has cleanup methods
            self._current_runner = None
            self._current_runner_key = None
            torch.cuda.empty_cache()


class InputOutputConverter:
    """Convert between ComfyUI and LightX2V formats."""

    @staticmethod
    def comfy_image_to_pil(comfy_image: torch.Tensor) -> Image.Image:
        """Convert ComfyUI image tensor to PIL Image.

        ComfyUI format: [B, H, W, C] float32 0-1
        """
        # Take first image from batch
        image_np = (comfy_image[0].cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    @staticmethod
    def comfy_audio_to_path(comfy_audio: Dict) -> str:
        """Convert ComfyUI audio to file path."""
        # ComfyUI audio format: {"waveform": tensor, "sample_rate": int}
        waveform = comfy_audio["waveform"]
        sample_rate = comfy_audio["sample_rate"]

        # Save to temporary file
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, waveform, sample_rate)
            return tmp.name

    @staticmethod
    def video_to_latent(video_path: str) -> Dict[str, Any]:
        """Convert video path to ComfyUI latent format."""
        # For now, return the path wrapped in latent format
        # This allows downstream nodes to handle the video
        return {"samples": video_path, "type": "video", "format": "path"}

    @staticmethod
    def tensor_to_latent(video_tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert video tensor to ComfyUI latent format."""
        return {"samples": video_tensor, "type": "video", "format": "tensor"}


class ConfigManager:
    """Manage configuration merging and parameter mapping."""

    # Mapping from ComfyUI parameter names to LightX2V config keys
    PARAM_MAPPING = {
        "steps": "infer_steps",
        "cfg_scale": "sample_guide_scale",
        "seed": "seed",
        "height": "target_height",
        "width": "target_width",
        "video_length": "target_video_length",
    }

    # 用户可配置的参数（在ComfyUI中显示）
    USER_CONFIGURABLE_PARAMS = {
        "steps": {
            "type": "INT",
            "default": 20,
            "min": 1,
            "max": 200,
            "tooltip": "推理步数 (-1使用默认值)",
        },
        "cfg_scale": {
            "type": "FLOAT",
            "default": 7,
            "min": 0.1,
            "max": 30,
            "step": 0.1,
            "tooltip": "CFG引导强度 (-1使用默认值)",
        },
        "seed": {
            "type": "INT",
            "default": 42,
            "min": -1,
            "max": 2**32 - 1,
            "tooltip": "随机种子 (-1随机)",
        },
        "height": {
            "type": "INT",
            "default": 640,
            "min": 64,
            "max": 2048,
            "step": 8,
            "tooltip": "视频高度 (-1使用默认值)",
        },
        "width": {
            "type": "INT",
            "default": 640,
            "min": 64,
            "max": 2048,
            "step": 8,
            "tooltip": "视频宽度 (-1使用默认值)",
        },
        "video_length": {
            "type": "INT",
            "default": 1,
            "min": 1,
            "max": 300,
            "tooltip": "视频帧数 (-1使用默认值)",
        },
        "sample_shift": {
            "type": "INT",
            "default": 5,
            "min": 0,
            "max": 20,
            "tooltip": "采样偏移 (-1使用默认值)",
        },
    }

    @classmethod
    def create_config(
        cls,
        model_cls: str,
        model_path: str,
        task: str,
        base_config: Dict,
        overrides: Dict,
        quantization_config: Optional[Dict] = None,
    ) -> EasyDict:
        """Create a complete config for LightX2V runner."""
        # Start with default config
        config = get_default_config()

        # Add required fields
        config.update(
            {
                "model_cls": model_cls,
                "model_path": model_path,
                "task": task,
                "mode": "infer",
            }
        )

        # Apply base config from file
        config.update(base_config)

        # Apply quantization config if provided
        if quantization_config:
            config.update(quantization_config)
            # 自动设置量化模型路径
            if "mm_config" in quantization_config and "mm_type" in quantization_config["mm_config"]:
                mm_type = quantization_config["mm_config"]["mm_type"]
                config["dit_quantized_ckpt"] = cls._get_quantization_model_path(model_cls, mm_type)

        # Apply ComfyUI overrides
        for comfy_key, value in overrides.items():
            if value is not None and value != -1:  # -1 means use default
                if comfy_key in cls.PARAM_MAPPING:
                    lightx2v_key = cls.PARAM_MAPPING[comfy_key]
                    config[lightx2v_key] = value
                else:
                    # Direct mapping for unknown parameters
                    config[comfy_key] = value

        # 检查并加载模型路径中的config.json
        config = cls._load_model_config(config)

        return EasyDict(config)

    @classmethod
    def _load_model_config(cls, config: Dict) -> Dict:
        """Load and merge config.json from model path if it exists."""
        model_path = config.get("model_path", "")
        if not model_path:
            return config

        model_config_path = os.path.join(model_path, "config.json")
        print(f"Loading model config from: {model_config_path}", flush=True)
        if os.path.exists(model_config_path):
            try:
                with open(model_config_path, "r") as f:
                    model_config = json.load(f)
                # 合并模型配置，模型配置优先级更高
                config.update(model_config)
                print(f"Loaded model config from: {model_config_path}")
            except Exception as e:
                print(f"Failed to load model config from {model_config_path}: {e}")

        return config

    @classmethod
    def _get_quantization_model_path(cls, model_cls: str, mm_type: str) -> str:
        """Get quantization model path based on model class and mm_type."""
        bridge = LightX2VBridge()
        return bridge.get_quantization_model_path(model_cls, mm_type)

    @classmethod
    def parse_custom_config(cls, custom_config_str: str) -> Dict:
        """Parse custom config string."""
        if not custom_config_str.strip():
            return {}

        try:
            return json.loads(custom_config_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse custom config: {e}")
            return {}


class LightX2VConfigBuilder:
    """Build LightX2V configuration from various sources."""

    def __init__(self):
        self.bridge = LightX2VBridge()

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for the config builder."""
        bridge = LightX2VBridge()

        # Get available models and configs
        model_choices = bridge.model_registry

        # 构建配置选项
        config_choices = ["custom"]
        config_descriptions = {}

        # 添加普通配置
        for key, config_info in bridge.config_registry.items():
            config_choices.append(key)
            config_descriptions[key] = config_info["description"]

        required_inputs = {
            "model_cls": (model_choices, {"tooltip": "选择模型类型"}),
            "model_path": (
                "STRING",
                {"default": "", "tooltip": "模型权重路径（留空使用默认路径）"},
            ),
            "task": (
                ["t2v", "i2v"],
                {
                    "default": "t2v",
                    "tooltip": "任务类型: 文本到视频或图像到视频",
                },
            ),
            "config_preset": (
                config_choices,
                {
                    "default": "custom",
                    "tooltip": "配置预设或'custom'自定义",
                },
            ),
        }

        # 添加用户可配置参数
        for param_name, param_config in ConfigManager.USER_CONFIGURABLE_PARAMS.items():
            required_inputs[param_name] = (
                param_config["type"],
                {
                    "default": param_config["default"],
                    "min": param_config["min"],
                    "max": param_config["max"],
                    "tooltip": param_config["tooltip"],
                },
            )
            if "step" in param_config:
                required_inputs[param_name][1]["step"] = param_config["step"]

        optional_inputs = {
            "custom_config": (
                "STRING",
                {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "自定义配置（JSON格式）",
                },
            ),
            "quantization_config": (
                "LIGHTX2V_CONFIG",
                {"tooltip": "量化配置"},
            ),
            "attention_config": (
                "LIGHTX2V_CONFIG",
                {"tooltip": "注意力机制配置"},
            ),
            "caching_config": (
                "LIGHTX2V_CONFIG",
                {"tooltip": "缓存配置"},
            ),
            "distill_config": (
                "LIGHTX2V_CONFIG",
                {"tooltip": "蒸馏配置"},
            ),
            "lora_path": (
                "STRING",
                {"default": "", "tooltip": "LoRA权重路径"},
            ),
            "strength_model": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "LoRA模型强度",
                },
            ),
        }

        return {
            "required": required_inputs,
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("LIGHTX2V_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "LightX2V/Config"

    def build_config(
        self,
        model_cls,
        model_path,
        task,
        config_preset,
        steps,
        cfg_scale,
        seed,
        height,
        width,
        video_length,
        sample_shift,
        custom_config="{}",
        quantization_config=None,
        attention_config=None,
        caching_config=None,
        distill_config=None,
        lora_path="",
        strength_model=1.0,
        **kwargs,
    ):
        """Build configuration for LightX2V inference."""

        # 加载基础配置
        if config_preset == "custom":
            base_config = {}
        else:
            config_info = self.bridge.config_registry.get(config_preset)
            if config_info:
                base_config = self.bridge.load_config(config_info["path"])
            else:
                raise ValueError(f"配置预设 '{config_preset}' 未找到")

        # 解析自定义配置
        custom_cfg = ConfigManager.parse_custom_config(custom_config)
        base_config.update(custom_cfg)

        # 合并其他配置
        if quantization_config:
            base_config.update(quantization_config)
        if attention_config:
            base_config.update(attention_config)
        if caching_config:
            base_config.update(caching_config)
        if distill_config:
            base_config.update(distill_config)

        # 准备覆盖参数
        overrides = {
            "seed": seed if seed != -1 else None,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "video_length": video_length,
            "sample_shift": sample_shift,
            "lora_path": lora_path if lora_path else None,
            "strength_model": strength_model,
        }

        # 创建最终配置
        config = ConfigManager.create_config(model_cls, model_path, task, base_config, overrides)

        return (config,)


class LightX2VQuantizationConfig:
    """Configuration node for quantization settings."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for quantization config."""
        bridge = LightX2VBridge()

        # Get available quantization configs
        quant_choices = ["custom"]
        for key, config_info in bridge.quantization_registry.items():
            quant_choices.append(key)

        return {
            "required": {
                "quantization_preset": (
                    quant_choices,
                    {
                        "default": "custom",
                        "tooltip": "量化配置预设",
                    },
                ),
            },
            "optional": {
                "mm_type": (
                    [
                        "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm",
                        "W-int8-channel-sym-A-fp16-dynamic-Vllm",
                        "W-fp16-A-fp16-dynamic-Vllm",
                    ],
                    {
                        "default": "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm",
                        "tooltip": "量化类型",
                    },
                ),
                "dit_quantized_ckpt": (
                    "STRING",
                    {"default": "", "tooltip": "量化模型路径"},
                ),
            },
        }

    RETURN_TYPES = ("LIGHTX2V_CONFIG",)
    RETURN_NAMES = ("quantization_config",)
    FUNCTION = "build_quantization_config"
    CATEGORY = "LightX2V/Config"

    def build_quantization_config(
        self,
        quantization_preset,
        mm_type="W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm",
        dit_quantized_ckpt="",
    ):
        """Build quantization configuration."""
        bridge = LightX2VBridge()

        if quantization_preset == "custom":
            config: dict[str, Any] = {
                "mm_config": {"mm_type": mm_type},
            }
            if dit_quantized_ckpt:
                config["dit_quantized_ckpt"] = dit_quantized_ckpt
        else:
            config_info = bridge.quantization_registry.get(quantization_preset)
            if config_info:
                config = bridge.load_config(config_info["path"])
            else:
                raise ValueError(f"量化配置 '{quantization_preset}' 未找到")

        return (config,)


class LightX2VAttentionConfig:
    """Configuration node for attention mechanisms."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for attention config."""
        return {
            "required": {
                "self_attn_1_type": (
                    ["flash_attn3", "sage_attn2", "radial_attn", "sparge_attn"],
                    {
                        "default": "flash_attn3",
                        "tooltip": "自注意力类型",
                    },
                ),
                "cross_attn_1_type": (
                    ["flash_attn3", "sage_attn2", "radial_attn", "sparge_attn"],
                    {
                        "default": "flash_attn3",
                        "tooltip": "交叉注意力1类型",
                    },
                ),
                "cross_attn_2_type": (
                    ["flash_attn3", "sage_attn2", "radial_attn", "sparge_attn"],
                    {
                        "default": "flash_attn3",
                        "tooltip": "交叉注意力2类型",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LIGHTX2V_CONFIG",)
    RETURN_NAMES = ("attention_config",)
    FUNCTION = "build_attention_config"
    CATEGORY = "LightX2V/Config"

    def build_attention_config(
        self,
        self_attn_1_type,
        cross_attn_1_type,
        cross_attn_2_type,
    ):
        """Build attention configuration."""
        config = {
            "self_attn_1_type": self_attn_1_type,
            "cross_attn_1_type": cross_attn_1_type,
            "cross_attn_2_type": cross_attn_2_type,
        }

        return (config,)


class LightX2VCachingConfig:
    """Configuration node for caching mechanisms."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for caching config."""
        return {
            "required": {
                "feature_caching": (
                    ["NoCaching", "Tea", "TaylorSeer", "Ada", "Custom"],
                    {
                        "default": "NoCaching",
                        "tooltip": "特征缓存类型",
                    },
                ),
            },
            "optional": {
                "teacache_thresh": (
                    "FLOAT",
                    {
                        "default": 0.26,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "TeaCache阈值",
                    },
                ),
                "use_ret_steps": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "使用返回步骤",
                    },
                ),
                "coefficients": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "系数配置（JSON格式）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LIGHTX2V_CONFIG",)
    RETURN_NAMES = ("caching_config",)
    FUNCTION = "build_caching_config"
    CATEGORY = "LightX2V/Config"

    def build_caching_config(
        self,
        feature_caching,
        teacache_thresh=0.26,
        use_ret_steps=True,
        coefficients="",
    ):
        """Build caching configuration."""
        config = {}

        if feature_caching != "NoCaching":
            config["feature_caching"] = feature_caching

            if feature_caching == "Tea":
                config["teacache_thresh"] = teacache_thresh
                config["use_ret_steps"] = use_ret_steps

                if coefficients:
                    try:
                        config["coefficients"] = json.loads(coefficients)
                    except json.JSONDecodeError:
                        print(f"Failed to parse coefficients: {coefficients}")
        else:
            config["feature_caching"] = "NoCaching"

        return (config,)


class LightX2VInference:
    """LightX2V inference node that generates images directly."""

    def __init__(self):
        self.bridge = LightX2VBridge()
        self.converter = InputOutputConverter()

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for the inference node."""
        return {
            "required": {
                "config": ("LIGHTX2V_CONFIG", {"tooltip": "LightX2V配置"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "生成提示词",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "负面提示词"},
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "i2v任务的输入图像"}),
                "audio": (
                    "AUDIO",
                    {"tooltip": "音频驱动生成的输入音频"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "LightX2V/Inference"

    def generate(
        self,
        config,
        prompt,
        negative_prompt,
        image=None,
        audio=None,
        **kwargs,
    ):
        """Generate images using LightX2V."""

        # 直接使用传入的config (已经是EasyDict)
        # 添加运行时需要的参数
        config.prompt = prompt
        config.negative_prompt = negative_prompt
        config.mode = "infer"

        # 设置环境变量（从run_wan_i2v.sh中提取）
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "DTYPE" not in os.environ:
            os.environ["DTYPE"] = "BF16"  # 可通过配置更改
        if "ENABLE_GRAPH_MODE" not in os.environ:
            os.environ["ENABLE_GRAPH_MODE"] = "false"
        if "ENABLE_PROFILING_DEBUG" not in os.environ:
            os.environ["ENABLE_PROFILING_DEBUG"] = "true"

        # 临时文件列表，用于清理
        temp_files = []

        try:
            # 处理i2v任务的图像输入
            if config.task == "i2v" and image is not None:
                # 保存图像到临时文件
                pil_image = self.converter.comfy_image_to_pil(image)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pil_image.save(tmp.name)
                    config.image_path = tmp.name
                    temp_files.append(tmp.name)
            elif config.task == "i2v" and image is None:
                raise ValueError("i2v任务需要输入图像")

            # 处理音频输入
            if audio is not None and "audio" in config.model_cls:
                audio_path = self.converter.comfy_audio_to_path(audio)
                config.audio_path = audio_path
                temp_files.append(audio_path)

            # 获取或创建runner
            runner = self.bridge.get_runner(config)

            if runner is None:
                raise RuntimeError("Failed to initialize runner")

            # 运行生成 - 我们需要创建一个自定义的pipeline来获取latents
            # 由于原始的run_pipeline方法会删除latents和generator，我们需要分步执行
            if asyncio.iscoroutinefunction(runner.run_pipeline):
                # 在同步环境中运行异步方法
                try:
                    # 使用现有的事件循环（如果存在）或创建新的
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                try:
                    # 分步执行pipeline，在删除latents前获取它们
                    # 这里我们使用try/except来捕获可能的属性访问错误
                    try:
                        # 尝试分步执行
                        if hasattr(runner, "run_input_encoder"):
                            runner.inputs = loop.run_until_complete(runner.run_input_encoder())
                        if hasattr(runner, "set_target_shape"):
                            kwargs = runner.set_target_shape()
                        if hasattr(runner, "run_dit"):
                            latents, generator = loop.run_until_complete(runner.run_dit(kwargs))
                            images = loop.run_until_complete(runner.run_vae_decoder(latents, generator))
                            # 直接解码latents为图像
                            return self._decode_latents_to_images(images)
                    except (AttributeError, TypeError):
                        raise RuntimeError("Failed to get latents from generation")
                except Exception as e:
                    print(f"Error during pipeline execution: {e}")
                    raise
            else:
                try:
                    if hasattr(runner, "run_input_encoder"):
                        runner.inputs = runner.run_input_encoder()
                    if hasattr(runner, "set_target_shape"):
                        kwargs = runner.set_target_shape()
                    if hasattr(runner, "run_dit"):
                        latents, generator = runner.run_dit(kwargs)
                        # 直接解码latents为图像
                        images = runner.run_vae_decoder(latents, generator)
                        return self._decode_latents_to_images(images)
                except (AttributeError, TypeError):
                    raise RuntimeError("Failed to get latents from generation")

        except Exception as e:
            print(f"Error in LightX2V generation: {e}")
            raise
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass

    def _decode_latents_to_images(self, decoded_images):
        """Decode latents to images using VAE."""

        # 归一化从 [-1, 1] 到 [0, 1]
        images = (decoded_images + 1) / 2

        # 重新排列维度为ComfyUI格式 [T, H, W, C]
        images = images.squeeze(0).permute(1, 2, 3, 0).cpu()
        images = torch.clamp(images, 0, 1)

        return (images,)


class LightX2VDistillConfig:
    """Configuration node for distillation settings."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for distillation config."""
        return {
            "required": {
                "enable_distill": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "启用蒸馏模型",
                    },
                ),
                "infer_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 50,
                        "tooltip": "推理步数（蒸馏模型通常使用4步）",
                    },
                ),
                "denoising_steps": (
                    "STRING",
                    {
                        "default": "[999, 750, 500, 250]",
                        "tooltip": "去噪步骤列表（JSON格式）",
                    },
                ),
            },
            "optional": {
                "enable_cfg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "启用CFG（分类器自由引导）",
                    },
                ),
                "enable_dynamic_cfg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "启用动态CFG",
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.1,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": "动态CFG缩放比例",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LIGHTX2V_CONFIG",)
    RETURN_NAMES = ("distill_config",)
    FUNCTION = "build_distill_config"
    CATEGORY = "LightX2V/Config"

    def build_distill_config(
        self,
        enable_distill,
        infer_steps=4,
        denoising_steps="[999, 750, 500, 250]",
        enable_cfg=False,
        enable_dynamic_cfg=False,
        cfg_scale=4.0,
    ):
        """Build distillation configuration."""
        config = {}

        if enable_distill:
            config["infer_steps"] = infer_steps
            config["enable_cfg"] = enable_cfg
            config["enable_dynamic_cfg"] = enable_dynamic_cfg

            if enable_dynamic_cfg:
                config["cfg_scale"] = cfg_scale

            try:
                config["denoising_step_list"] = json.loads(denoising_steps)
                # 确保去噪步骤列表长度与推理步数匹配
                if len(config["denoising_step_list"]) != infer_steps:
                    print(f"Warning: denoising_step_list length ({len(config['denoising_step_list'])}) doesn't match infer_steps ({infer_steps})")
            except json.JSONDecodeError:
                print(f"Failed to parse denoising steps, using default: {denoising_steps}")
                config["denoising_step_list"] = [999, 750, 500, 250]

        return (config,)


# Node class mapping
NODE_CLASS_MAPPINGS = {
    # Modular nodes
    "LightX2VConfigBuilder": LightX2VConfigBuilder,
    "LightX2VQuantizationConfig": LightX2VQuantizationConfig,
    "LightX2VAttentionConfig": LightX2VAttentionConfig,
    "LightX2VCachingConfig": LightX2VCachingConfig,
    "LightX2VDistillConfig": LightX2VDistillConfig,
    "LightX2VInference": LightX2VInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Modular nodes
    "LightX2VConfigBuilder": "LightX2V Config Builder",
    "LightX2VQuantizationConfig": "LightX2V Quantization Config",
    "LightX2VAttentionConfig": "LightX2V Attention Config",
    "LightX2VCachingConfig": "LightX2V Caching Config",
    "LightX2VDistillConfig": "LightX2V Distill Config",
    "LightX2VInference": "LightX2V Inference",
}
