"""Field translators: ComfyUI widget value -> lightx2v config dict.

Each module here mirrors one ComfyUI Config node in ``nodes/config.py``:

    LightX2VInferenceConfig      <-> translator/inference.py
    LightX2VTeaCache             <-> translator/teacache.py
    LightX2VQuantization         <-> translator/quant.py
    LightX2VMemoryOptimization   <-> translator/memory.py

``translator/pipeline.py`` orchestrates them and adds the model's own
``config.json`` (read from disk by lightx2v's ``set_config``).
"""

from .inference import apply_inference_config
from .memory import apply_memory_optimization
from .pipeline import ModularConfigManager
from .quant import apply_quantization_config
from .teacache import apply_teacache_config

__all__ = [
    "apply_inference_config",
    "apply_teacache_config",
    "apply_quantization_config",
    "apply_memory_optimization",
    "ModularConfigManager",
]
