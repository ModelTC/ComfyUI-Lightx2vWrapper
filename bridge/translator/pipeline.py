"""Orchestrate per-feature translators into a final lightx2v config dict.

Flow:
    1. start from ``LightX2VDefaultConfig.DEFAULT_CONFIG``
    2. apply inference / memory / teacache / quantization translators in order
       (teacache runs after inference so it can see the resolved task/resolution)
    3. attach LoRA chain and talk_objects (no rename needed)
    4. shallow-merge the model's own ``config.json`` for keys still unset
       (lightx2v's ``set_config`` will further read its own model config later)
    5. wrap as ``EasyDict`` so consumers can use attribute access

NOTE: ``input_info`` (the per-call dataclass in ``lightx2v.utils.input_info``)
is NOT built here. The inference node constructs it dynamically from this
config plus the runtime image/audio paths, because lightx2v itself distinguishes
"persistent config" from "per-call input_info".
"""

import copy
import json
import logging
import os
from typing import Any, Dict

from easydict import EasyDict

from ..capability import get_available_attn_ops, get_available_quant_ops
from ..defaults import LightX2VDefaultConfig
from .inference import apply_inference_config
from .memory import apply_memory_optimization
from .quant import apply_quantization_config
from .teacache import apply_teacache_config


class ModularConfigManager:
    """Compose translators into a final lightx2v config."""

    def __init__(self):
        self.base_config = copy.deepcopy(LightX2VDefaultConfig.DEFAULT_CONFIG)
        self._available_attn_ops = None
        self._available_quant_ops = None

    @staticmethod
    def _filter_available(ops_list, fallback=None):
        available = [name for name, ok in ops_list if ok]
        if fallback and fallback not in available:
            available.append(fallback)
        return available

    @property
    def available_attention_types(self):
        if self._available_attn_ops is None:
            self._available_attn_ops = get_available_attn_ops()
        return self._filter_available(self._available_attn_ops, "torch_sdpa")

    @property
    def available_quant_schemes(self):
        if self._available_quant_ops is None:
            self._available_quant_ops = get_available_quant_ops()
        return self._filter_available(self._available_quant_ops)

    # Exposed for tests/debugging; the public entrypoint is build_final_config_from_combined.
    apply_inference_config = staticmethod(apply_inference_config)
    apply_teacache_config = staticmethod(apply_teacache_config)
    apply_quantization_config = staticmethod(apply_quantization_config)
    apply_memory_optimization = staticmethod(apply_memory_optimization)

    @staticmethod
    def _load_model_config(model_path: str) -> Dict[str, Any]:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load model config: {e}")
            return {}

    def build_final_config_from_combined(self, combined_config) -> EasyDict:
        """Build the final lightx2v config from a CombinedConfig dataclass."""
        final_config = copy.deepcopy(self.base_config)

        if combined_config.inference:
            final_config.update(apply_inference_config(combined_config.inference.to_dict()))

        if combined_config.memory:
            final_config.update(apply_memory_optimization(combined_config.memory.to_dict()))

        # teacache reads the (already-resolved) task and resolution off final_config.
        if combined_config.teacache:
            final_config.update(apply_teacache_config(combined_config.teacache.to_dict(), final_config))

        if combined_config.quantization:
            final_config.update(apply_quantization_config(combined_config.quantization.to_dict()))

        if combined_config.lora_configs:
            final_config["lora_configs"] = [lora.to_dict() for lora in combined_config.lora_configs]

        if combined_config.talk_objects:
            final_config.update(combined_config.talk_objects.to_dict())

        # Shallow-merge the model's own config.json for keys still unset.
        # lightx2v's own set_config.auto_calc_config will do its own deeper
        # merge of model_path/config.json — this just gives translators a
        # chance to see model-side hints (e.g. text_len) when they run.
        model_config = self._load_model_config(final_config.get("model_path", ""))
        for key, value in model_config.items():
            if key not in final_config or final_config[key] is None:
                final_config[key] = value

        return EasyDict(final_config)
