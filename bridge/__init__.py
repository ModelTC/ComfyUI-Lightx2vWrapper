"""Bridge between ComfyUI widget values and lightx2v's internal config schema.

Submodules:
  - ``capability``        GPU + backend-op detection (pure functions)
  - ``defaults``          ``LightX2VDefaultConfig`` — wrapper-side starting values
  - ``teacache_coeffs``   ``CoefficientCalculator`` — polynomial constants
  - ``translator/``       per-feature wrapper-key -> lightx2v-key translators,
                          plus ``ModularConfigManager`` that orchestrates them

Public surface (re-exported here for backward compat with existing imports
``from .bridge import …``):
"""

from .capability import (
    get_available_attn_ops,
    get_available_ops,
    get_available_quant_ops,
    get_gpu_capability,
    is_ada_architecture_gpu,
    is_fp8_supported_gpu,
    is_module_installed,
)
from .defaults import LightX2VDefaultConfig
from .teacache_coeffs import CoefficientCalculator
from .translator import ModularConfigManager

__all__ = [
    # capability
    "get_gpu_capability",
    "is_fp8_supported_gpu",
    "is_ada_architecture_gpu",
    "is_module_installed",
    "get_available_ops",
    "get_available_quant_ops",
    "get_available_attn_ops",
    # defaults / coeffs
    "LightX2VDefaultConfig",
    "CoefficientCalculator",
    # translator orchestrator
    "ModularConfigManager",
]
