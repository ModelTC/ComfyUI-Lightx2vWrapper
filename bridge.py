"""Modular configuration system for LightX2V ComfyUI integration."""

import copy
import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import torch
from easydict import EasyDict


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


class LightX2VDefaultConfig:
    """Central default configuration for LightX2V."""

    DEFAULT_CONFIG = {
        # ========== Model Configuration ==========
        "model_cls": "wan2.1",
        "model_path": "",
        "task": "t2v",
        "mode": "infer",
        # ========== Inference Parameters ==========
        "infer_steps": 40,
        "seed": 42,
        "sample_guide_scale": 5.0,
        "sample_shift": 5,
        "enable_cfg": True,
        "prompt": "",
        "negative_prompt": "",
        # ========== Video Parameters ==========
        "target_height": 480,
        "target_width": 832,
        "target_video_length": 81,
        "fps": 16,
        "vae_stride": [4, 8, 8],
        "patch_size": [1, 2, 2],
        # ========== Feature Caching (TeaCache) ==========
        "feature_caching": "NoCaching",
        "teacache_thresh": 0.26,
        "coefficients": None,  # Auto-calculated
        "use_ret_steps": False,
        # ========== Quantization ==========
        "dit_quant_scheme": "bf16",
        "t5_quant_scheme": "bf16",
        "clip_quant_scheme": "fp16",
        "quant_op": "vllm",
        "precision_mode": "fp32",
        "dit_quantized_ckpt": None,
        "t5_quantized_ckpt": None,
        "clip_quantized_ckpt": None,
        "mm_config": {"mm_type": "Default"},
        # ========== GPU Memory Optimization ==========
        "rotary_chunk": False,
        "rotary_chunk_size": 100,
        "clean_cuda_cache": False,
        "torch_compile": False,
        "attention_type": "flash_attn3",
        "self_attn_1_type": "flash_attn3",
        "cross_attn_1_type": "flash_attn3",
        "cross_attn_2_type": "flash_attn3",
        # ========== Async Offloading ==========
        "cpu_offload": False,
        "offload_granularity": "phase",
        "offload_ratio": 1.0,
        "t5_cpu_offload": False,
        "t5_offload_granularity": "model",
        "lazy_load": False,
        "unload_modules": False,
        # ========== Lightweight VAE ==========
        "use_tiny_vae": False,
        "tiny_vae": False,
        "tiny_vae_path": None,
        "use_tiling_vae": False,
        # ========== Other Settings ==========
        "lora_path": None,
        "strength_model": 1.0,
        "do_mm_calib": False,
        "parallel_attn_type": None,
        "parallel_vae": False,
        "max_area": False,
        "use_prompt_enhancer": False,
        "text_len": 512,
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

    @property
    def available_attention_types(self) -> List[str]:
        """Get available attention types."""
        if self._available_attn_ops is None:
            self._available_attn_ops = get_available_attn_ops()

        available = []
        for op_name, is_available in self._available_attn_ops:
            if is_available:
                available.append(op_name)

        if "torch_sdpa" not in available:
            available.append("torch_sdpa")

        return available

    @property
    def available_quant_schemes(self) -> List[str]:
        """Get available quantization schemes."""
        if self._available_quant_ops is None:
            self._available_quant_ops = get_available_quant_ops()

        available = []
        for op_name, is_available in self._available_quant_ops:
            if is_available:
                available.append(op_name)

        return available

    def apply_inference_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic inference configuration."""
        updates = {}

        if "model_cls" in config:
            updates["model_cls"] = config["model_cls"]
        if "model_path" in config:
            updates["model_path"] = config["model_path"]
        if "task" in config:
            updates["task"] = config["task"]

        if "infer_steps" in config:
            updates["infer_steps"] = config["infer_steps"]
        if "seed" in config and config["seed"] != -1:
            updates["seed"] = config["seed"]
        if "cfg_scale" in config:
            updates["sample_guide_scale"] = config["cfg_scale"]
            updates["enable_cfg"] = config["cfg_scale"] != 1.0
        if "sample_shift" in config:
            updates["sample_shift"] = config["sample_shift"]

        if "height" in config:
            updates["target_height"] = config["height"]
        if "width" in config:
            updates["target_width"] = config["width"]
        if "video_length" in config:
            updates["target_video_length"] = config["video_length"]
        if "fps" in config:
            updates["fps"] = config["fps"]

        if "denoising_step_list" in config:
            updates["denoising_step_list"] = config["denoising_step_list"]

        return updates

    def apply_teacache_config(
        self, config: Dict[str, Any], model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
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

            coeffs = CoefficientCalculator.get_coefficients(
                task, model_size, resolution, updates["use_ret_steps"]
            )
            updates["coefficients"] = coeffs
        else:
            updates["feature_caching"] = "NoCaching"

        return updates

    def apply_quantization_config(
        self, config: Dict[str, Any], model_path: str
    ) -> Dict[str, Any]:
        """Apply quantization configuration."""
        updates = {}

        dit_scheme = config.get("dit_precision", "bf16")
        updates["dit_quant_scheme"] = dit_scheme
        if dit_scheme != "bf16":
            updates["dit_quantized_ckpt"] = os.path.join(model_path, dit_scheme)

        t5_scheme = config.get("t5_precision", "bf16")
        updates["t5_quant_scheme"] = t5_scheme
        updates["t5_quantized"] = t5_scheme != "bf16"
        if t5_scheme != "bf16":
            t5_path = os.path.join(model_path, t5_scheme)
            updates["t5_quantized_ckpt"] = os.path.join(
                t5_path, f"models_t5_umt5-xxl-enc-{t5_scheme}.pth"
            )

        clip_scheme = config.get("clip_precision", "fp16")
        updates["clip_quant_scheme"] = clip_scheme
        updates["clip_quantized"] = clip_scheme != "fp16"
        if clip_scheme != "fp16":
            clip_path = os.path.join(model_path, clip_scheme)
            updates["clip_quantized_ckpt"] = os.path.join(
                clip_path, f"clip-{clip_scheme}.pth"
            )

        quant_backend = config.get("quant_backend", "vllm")
        updates["quant_op"] = quant_backend

        if dit_scheme != "bf16":
            if quant_backend == "vllm":
                mm_type = f"W-{dit_scheme}-channel-sym-A-{dit_scheme}-channel-sym-dynamic-Vllm"
            elif quant_backend == "sgl":
                if dit_scheme == "int8":
                    mm_type = f"W-{dit_scheme}-channel-sym-A-{dit_scheme}-channel-sym-dynamic-Sgl-ActVllm"
                else:
                    mm_type = f"W-{dit_scheme}-channel-sym-A-{dit_scheme}-channel-sym-dynamic-Sgl"
            elif quant_backend == "q8f":
                mm_type = (
                    f"W-{dit_scheme}-channel-sym-A-{dit_scheme}-channel-sym-dynamic-Q8F"
                )
            else:
                mm_type = "Default"

            updates["mm_config"] = {"mm_type": mm_type}
        else:
            updates["mm_config"] = {"mm_type": "Default"}

        updates["precision_mode"] = config.get("sensitive_layers_precision", "fp32")

        return updates

    def apply_memory_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory optimization settings."""
        updates = {}

        level = config.get("optimization_level", "none")

        # GPU optimization
        if config.get("enable_rotary_chunk", False) or level in ["high", "extreme"]:
            updates["rotary_chunk"] = True
            updates["rotary_chunk_size"] = config.get("rotary_chunk_size", 100)

        if config.get("clean_cuda_cache", False) or level == "extreme":
            updates["clean_cuda_cache"] = True

        # CPU offloading
        if config.get("enable_cpu_offload", False) or level in [
            "medium",
            "high",
            "extreme",
        ]:
            updates["cpu_offload"] = True
            updates["offload_granularity"] = config.get("offload_granularity", "phase")
            updates["offload_ratio"] = config.get("offload_ratio", 1.0)

        # T5 offloading
        if level in ["high", "extreme"]:
            updates["t5_cpu_offload"] = True
            updates["t5_offload_granularity"] = (
                "block" if level == "extreme" else "model"
            )

        # Module management
        if config.get("lazy_load", False) or level == "extreme":
            updates["lazy_load"] = True

        if config.get("unload_after_inference", False) or level == "extreme":
            updates["unload_modules"] = True

        # Attention type
        attention_type = config.get("attention_type", "flash_attn3")
        updates["attention_type"] = attention_type
        updates["self_attn_1_type"] = attention_type
        updates["cross_attn_1_type"] = attention_type
        updates["cross_attn_2_type"] = attention_type

        return updates

    def apply_vae_config(
        self, config: Dict[str, Any], model_path: str
    ) -> Dict[str, Any]:
        """Apply VAE configuration."""
        updates = {}

        if config.get("use_tiny_vae", False):
            updates["use_tiny_vae"] = True
            updates["tiny_vae"] = True
            updates["tiny_vae_path"] = os.path.join(model_path, "taew2_1.pth")

        if config.get("use_tiling_vae", False):
            updates["use_tiling_vae"] = True

        return updates

    def build_final_config(self, configs: Dict[str, Dict[str, Any]]) -> EasyDict:
        """Build final configuration from module configs."""
        final_config = copy.deepcopy(self.base_config)

        if "inference" in configs:
            final_config.update(self.apply_inference_config(configs["inference"]))

        if "teacache" in configs:
            teacache_updates = self.apply_teacache_config(
                configs["teacache"],
                final_config,
            )
            final_config.update(teacache_updates)

        if "quantization" in configs:
            model_path = final_config.get("model_path", "")
            quant_updates = self.apply_quantization_config(
                configs["quantization"], model_path
            )
            final_config.update(quant_updates)

        if "memory" in configs:
            final_config.update(self.apply_memory_optimization(configs["memory"]))

        if "vae" in configs:
            model_path = final_config.get("model_path", "")
            final_config.update(self.apply_vae_config(configs["vae"], model_path))

        model_config_path = os.path.join(final_config["model_path"], "config.json")
        if os.path.exists(model_config_path):
            try:
                with open(model_config_path, "r") as f:
                    model_config = json.load(f)
                for key, value in model_config.items():
                    if key not in final_config or final_config[key] is None:
                        final_config[key] = value
            except Exception as e:
                logging.warning(f"Failed to load model config: {e}")

        return EasyDict(final_config)
