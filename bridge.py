import copy
import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import torch
from easydict import EasyDict


def get_gpu_capability():
    if not torch.cuda.is_available():
        return None, None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception as e:
        logging.warning(f"Failed to get GPU capability: {e}")
        return None, None


def is_fp8_supported_gpu():
    major, minor = get_gpu_capability()
    if major is None:
        return False
    return (major == 8 and minor == 9) or (major >= 9)


def is_ada_architecture_gpu():
    major, minor = get_gpu_capability()
    if major is None:
        return False
    return major == 8 and minor == 9


def is_module_installed(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ModuleNotFoundError:
        return False


def get_available_ops(op_mapping):
    available_ops = []
    for op_name, module_name in op_mapping.items():
        is_available = is_module_installed(module_name)
        available_ops.append((op_name, is_available))
    return available_ops


def get_available_quant_ops():
    quant_mapping = {
        "sgl": "sgl_kernel",
        "vllm": "vllm",
        "q8f": "q8_kernels",
        "torchao": "torchao",
    }

    available_ops = get_available_ops(quant_mapping)

    # Prefer q8f for Ada architecture GPUs
    if is_ada_architecture_gpu():
        q8f_available = next((op for op in available_ops if op[0] == "q8f" and op[1]), None)
        if q8f_available:
            available_ops.remove(q8f_available)
            available_ops.insert(0, q8f_available)

    return available_ops


def get_available_attn_ops():
    attn_mapping = {
        "sage_attn2": "sageattention",
        "flash_attn3": "flash_attn_interface",
        "flash_attn2": "flash_attn",
        "torch_sdpa": "torch",
    }

    return get_available_ops(attn_mapping)


class LightX2VDefaultConfig:
    """Central default configuration for LightX2V."""

    DEFAULT_ATTENTION_TYPE = "flash_attn3"
    DEFAULT_QUANTIZATION_SCHEMES = {
        "dit": "bf16",
        "t5": "bf16",
        "clip": "fp16",
        "adapter": "bf16",
    }
    DEFAULT_VIDEO_PARAMS = {
        "height": 480,
        "width": 832,
        "length": 81,
        "fps": 16,
        "vae_stride": [4, 8, 8],
        "patch_size": [1, 2, 2],
    }

    DEFAULT_CONFIG = {
        # Model Configuration
        "model_cls": "wan2.1",
        "model_path": "",
        "task": "t2v",
        # Inference Parameters
        "infer_steps": 40,
        "seed": 42,
        "sample_guide_scale": 5.0,
        "sample_shift": 5,
        "enable_cfg": True,
        "prompt": "",
        "negative_prompt": "",
        # Video Parameters
        "target_height": DEFAULT_VIDEO_PARAMS["height"],
        "target_width": DEFAULT_VIDEO_PARAMS["width"],
        "target_video_length": DEFAULT_VIDEO_PARAMS["length"],
        "fps": DEFAULT_VIDEO_PARAMS["fps"],
        "vae_stride": DEFAULT_VIDEO_PARAMS["vae_stride"],
        "patch_size": DEFAULT_VIDEO_PARAMS["patch_size"],
        # TeaCache
        "feature_caching": "NoCaching",
        "teacache_thresh": 0.26,
        "coefficients": None,
        "use_ret_steps": False,
        # Quantization
        "t5_quant_scheme": DEFAULT_QUANTIZATION_SCHEMES["t5"],
        "clip_quant_scheme": DEFAULT_QUANTIZATION_SCHEMES["clip"],
        "adapter_quant_scheme": DEFAULT_QUANTIZATION_SCHEMES["adapter"],
        "mm_config": {"mm_type": "Default"},
        # Memory Optimization
        "rotary_chunk": False,
        "rotary_chunk_size": 100,
        "clean_cuda_cache": False,
        "torch_compile": False,
        "self_attn_1_type": DEFAULT_ATTENTION_TYPE,
        "cross_attn_1_type": DEFAULT_ATTENTION_TYPE,
        "cross_attn_2_type": DEFAULT_ATTENTION_TYPE,
        # CPU Offloading
        "cpu_offload": False,
        "offload_granularity": "block",
        "offload_ratio": 1.0,
        "t5_cpu_offload": False,
        "t5_offload_granularity": "model",
        "lazy_load": False,
        "unload_modules": False,
        # VAE Settings
        "use_tiling_vae": False,
        # Other Settings
        "do_mm_calib": False,
        "max_area": False,
        "use_prompt_enhancer": False,
        "text_len": 512,
        "use_31_block": True,
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "audio_sr": 16000,
        "return_video": True,
        "talk_objects": None,
    }


class CoefficientCalculator:
    """Calculate TeaCache coefficients based on model and resolution."""

    COEFFICIENTS = {
        "t2v": {
            "1.3b": {
                "default": [
                    [
                        -5.21862437e04,
                        9.23041404e03,
                        -5.28275948e02,
                        1.36987616e01,
                        -4.99875664e-02,
                    ],
                    [
                        2.39676752e03,
                        -1.31110545e03,
                        2.01331979e02,
                        -8.29855975e00,
                        1.37887774e-01,
                    ],
                ]
            },
            "14b": {
                "default": [
                    [
                        -3.03318725e05,
                        4.90537029e04,
                        -2.65530556e03,
                        5.87365115e01,
                        -3.15583525e-01,
                    ],
                    [
                        -5784.54975374,
                        5449.50911966,
                        -1811.16591783,
                        256.27178429,
                        -13.02252404,
                    ],
                ]
            },
        },
        "i2v": {
            "720p": [
                [
                    8.10705460e03,
                    2.13393892e03,
                    -3.72934672e02,
                    1.66203073e01,
                    -4.17769401e-02,
                ],
                [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
            ],
            "480p": [
                [
                    2.57151496e05,
                    -3.54229917e04,
                    1.40286849e03,
                    -1.35890334e01,
                    1.32517977e-01,
                ],
                [
                    -3.02331670e02,
                    2.23948934e02,
                    -5.25463970e01,
                    5.87348440e00,
                    -2.01973289e-01,
                ],
            ],
        },
    }

    @classmethod
    def get_coefficients(
        cls,
        task: str,
        model_size: str,
        resolution: Tuple[int, int],
        use_ret_steps: bool,
    ) -> List[List[float]]:
        """Get appropriate coefficients for TeaCache."""
        if task == "t2v":
            coeffs = cls.COEFFICIENTS["t2v"].get(model_size, {}).get("default", None)
        else:  # i2v
            width, height = resolution
            if height >= 720 or width >= 720:
                coeffs = cls.COEFFICIENTS["i2v"]["720p"]
            else:
                coeffs = cls.COEFFICIENTS["i2v"]["480p"]

        if coeffs:
            return coeffs[0] if use_ret_steps else coeffs[1]
        raise ValueError(
            f"No coefficients found for task: {task}, model_size: {model_size}, resolution: {resolution}, use_ret_steps: {use_ret_steps}"
        )


class ModularConfigManager:
    """Manages modular configuration without presets."""

    def __init__(self):
        self.base_config = copy.deepcopy(LightX2VDefaultConfig.DEFAULT_CONFIG)
        self._available_attn_ops = None
        self._available_quant_ops = None

    def _get_available_ops(self, ops_list: List[Tuple[str, bool]], fallback: str = None) -> List[str]:
        available = [op_name for op_name, is_available in ops_list if is_available]
        if fallback and fallback not in available:
            available.append(fallback)
        return available

    @property
    def available_attention_types(self) -> List[str]:
        """Get available attention types."""
        if self._available_attn_ops is None:
            self._available_attn_ops = get_available_attn_ops()
        return self._get_available_ops(self._available_attn_ops, "torch_sdpa")

    @property
    def available_quant_schemes(self) -> List[str]:
        """Get available quantization schemes."""
        if self._available_quant_ops is None:
            self._available_quant_ops = get_available_quant_ops()
        return self._get_available_ops(self._available_quant_ops)

    def _update_from_config(self, updates: Dict, config: Dict, mappings: Dict[str, str]) -> None:
        for config_key, update_key in mappings.items():
            if config_key in config:
                if config_key == "seed" and config[config_key] == -1:
                    continue
                updates[update_key] = config[config_key]

    def apply_inference_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        updates = {}

        basic_mappings = {
            "model_cls": "model_cls",
            "model_path": "model_path",
            "task": "task",
            "infer_steps": "infer_steps",
            "seed": "seed",
            "sample_shift": "sample_shift",
            "height": "target_height",
            "width": "target_width",
            "video_length": "target_video_length",
            "fps": "fps",
            "video_duration": "video_duration",
            "resize_mode": "resize_mode",
            "denoising_step_list": "denoising_step_list",
            "use_31_block": "use_31_block",
            "prev_frame_length": "prev_frame_length",
            "fixed_area": "fixed_area",
        }

        self._update_from_config(updates, config, basic_mappings)

        if "cfg_scale" in config:
            updates["sample_guide_scale"] = config["cfg_scale"]
            updates["enable_cfg"] = config["cfg_scale"] != 1.0

        if config["model_cls"] == "wan2.2_moe":
            updates["sample_guide_scale"] = [config["cfg_scale2"], config["cfg_scale2"]]
            updates["boundary"] = 0.9
        if "wan2.2" in config["model_cls"]:
            updates["use_image_encoder"] = False

        attention_type = config.get("attention_type", LightX2VDefaultConfig.DEFAULT_ATTENTION_TYPE)
        for attn_key in [
            "attention_type",
            "self_attn_1_type",
            "cross_attn_1_type",
            "cross_attn_2_type",
        ]:
            updates[attn_key] = attention_type

        if config.get("use_tiny_vae", False):
            updates.update(
                {
                    "use_tiny_vae": True,
                    "tiny_vae": True,
                    "tiny_vae_path": os.path.join(config["model_path"], "taew2_1.pth"),
                }
            )

        return updates

    def apply_teacache_config(self, config: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply TeaCache configuration."""
        updates = {}

        if config.get("enable", False):
            updates["feature_caching"] = "Tea"
            updates["teacache_thresh"] = config.get("threshold", 0.26)
            updates["use_ret_steps"] = config.get("use_ret_steps", False)

            task = model_info.get("task", "t2v")
            model_size = "14b" if "14b" in model_info.get("model_cls", "") else "1.3b"
            resolution = (
                model_info.get("target_width", 832),
                model_info.get("target_height", 480),
            )

            coeffs = CoefficientCalculator.get_coefficients(task, model_size, resolution, updates["use_ret_steps"])
            updates["coefficients"] = coeffs
        else:
            updates["feature_caching"] = "NoCaching"

        return updates

    def _get_mm_type(self, dit_scheme: str, quant_backend: str) -> str:
        if dit_scheme == "bf16":
            return "Default"

        base_pattern = f"W-{dit_scheme}-channel-sym-A-{dit_scheme}-channel-sym-dynamic"

        if quant_backend == "vllm":
            return f"{base_pattern}-Vllm"
        elif quant_backend == "sgl":
            suffix = "-Sgl-ActVllm" if dit_scheme == "int8" else "-Sgl"
            return f"{base_pattern}{suffix}"
        elif quant_backend == "q8f":
            return f"{base_pattern}-Q8F"
        elif quant_backend == "torchao":
            return f"{base_pattern}-Torchao"
        else:
            return "Default"

    def apply_quantization_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quantization configuration."""
        updates = {}
        defaults = LightX2VDefaultConfig.DEFAULT_QUANTIZATION_SCHEMES

        dit_scheme = config.get("dit_quant_scheme", defaults["dit"])
        t5_scheme = config.get("t5_quant_scheme", defaults["t5"])
        clip_scheme = config.get("clip_quant_scheme", defaults["clip"])
        adapter_scheme = config.get("adapter_quant_scheme", defaults["adapter"])
        quant_backend = config.get("quant_op", "vllm")

        updates.update(
            {
                "clip_quantized": clip_scheme != defaults["clip"],
                "clip_quant_scheme": clip_scheme,
                "t5_quant_scheme": t5_scheme,
                "t5_quantized": t5_scheme != defaults["t5"],
                "adapter_quantized": adapter_scheme != defaults["adapter"],
                "adapter_quant_scheme": adapter_scheme,
            }
        )

        if updates.get("t5_quantized") and quant_backend == "q8f":
            updates["t5_quant_scheme"] = f"{t5_scheme}-q8f"
        if updates.get("clip_quantized") and quant_backend == "q8f":
            updates["clip_quant_scheme"] = f"{clip_scheme}-q8f"

        mm_type = self._get_mm_type(dit_scheme, quant_backend)
        updates["mm_config"] = {"mm_type": mm_type}

        return updates

    def apply_memory_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory optimization settings."""
        updates = {}

        direct_mappings = {
            "enable_rotary_chunk": "rotary_chunk",
            "clean_cuda_cache": "clean_cuda_cache",
            "cpu_offload": "cpu_offload",
            "t5_cpu_offload": "t5_cpu_offload",
            "vae_cpu_offload": "vae_cpu_offload",
            "audio_encoder_cpu_offload": "audio_encoder_cpu_offload",
            "audio_adapter_cpu_offload": "audio_adapter_cpu_offload",
            "lazy_load": "lazy_load",
            "unload_after_inference": "unload_modules",
            "use_tiling_vae": "use_tiling_vae",
        }

        for config_key, update_key in direct_mappings.items():
            updates[update_key] = config.get(config_key, config.get("cpu_offload", False))

        if updates.get("rotary_chunk"):
            updates["rotary_chunk_size"] = config.get("rotary_chunk_size", 100)

        if updates.get("cpu_offload"):
            updates.update(
                {
                    "offload_granularity": config.get("offload_granularity", "phase"),
                    "offload_ratio": config.get("offload_ratio", 1.0),
                }
            )

        if updates.get("t5_cpu_offload"):
            updates["t5_offload_granularity"] = config.get("t5_offload_granularity", "model")

        return updates

    def _load_model_config(self, model_path: str) -> Dict[str, Any]:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return {}

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load model config: {e}")
            return {}

    def build_final_config(self, configs: Dict[str, Dict[str, Any]]) -> EasyDict:
        """Build final configuration from module configs."""
        final_config = copy.deepcopy(self.base_config)

        config_modules = [("inference", self.apply_inference_config)]

        for module_name, apply_func in config_modules:
            if module_name in configs:
                final_config.update(apply_func(configs[module_name]))

        if "memory" in configs:
            memory_updates = self.apply_memory_optimization(configs["memory"])
            final_config.update(memory_updates)

        if "teacache" in configs:
            teacache_updates = self.apply_teacache_config(configs["teacache"], final_config)
            final_config.update(teacache_updates)

        if "quantization" in configs:
            quant_updates = self.apply_quantization_config(configs["quantization"])
            final_config.update(quant_updates)

        model_config = self._load_model_config(final_config.get("model_path", ""))
        for key, value in model_config.items():
            if key not in final_config or final_config[key] is None:
                final_config[key] = value

        return EasyDict(final_config)
