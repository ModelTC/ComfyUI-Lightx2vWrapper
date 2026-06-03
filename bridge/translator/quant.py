"""Translate ``LightX2VQuantization`` widget values into lightx2v config keys.

Each of dit/t5/clip/adapter contributes two keys to lightx2v:
``{component}_quantized`` (bool) and ``{component}_quant_scheme`` (str).
``"Default"`` means "leave as-is".
"""

from typing import Any, Dict

from ..defaults import LightX2VDefaultConfig

_COMPONENTS = ("dit", "t5", "clip", "adapter")


def apply_quantization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate quantization widget values."""
    updates: Dict[str, Any] = {}
    defaults = LightX2VDefaultConfig.DEFAULT_QUANTIZATION_SCHEMES

    for component in _COMPONENTS:
        scheme = config.get(f"{component}_quant_scheme", defaults[component])
        updates[f"{component}_quantized"] = scheme != "Default"
        updates[f"{component}_quant_scheme"] = scheme

    return updates
