"""ComfyUI node definitions for LightX2V.

Each submodule groups a category of nodes:
- ``config``    : per-feature configuration nodes (inference / teacache / quant / memory)
- ``lora``      : LoRA chain loader
- ``talk``      : talk-object input/combiner nodes
- ``combiner``  : config combiners (V1/V2/V3) that aggregate the above
- ``inference`` : the modular inference runner
- ``seedvr``    : SeedVR2 super-resolution runner
"""

from .combiner import (
    LightX2VConfigCombinerV2,
    LightX2VConfigCombinerV3,
)
from .config import (
    LightX2VInferenceConfig,
    LightX2VMemoryOptimization,
    LightX2VQuantization,
    LightX2VTeaCache,
)
from .inference import LightX2VModularInferenceV2
from .lora import LightX2VLoRALoader
from .seedvr import LightX2VSeedVR2Loader, LightX2VSeedVR2Sampler
from .talk import (
    TalkObjectInput,
    TalkObjectsCombiner,
    TalkObjectsFromFiles,
    TalkObjectsFromJSON,
)

NODE_CLASS_MAPPINGS = {
    "LightX2VInferenceConfig": LightX2VInferenceConfig,
    "LightX2VTeaCache": LightX2VTeaCache,
    "LightX2VQuantization": LightX2VQuantization,
    "LightX2VMemoryOptimization": LightX2VMemoryOptimization,
    "LightX2VLoRALoader": LightX2VLoRALoader,
    "LightX2VConfigCombinerV2": LightX2VConfigCombinerV2,
    "LightX2VConfigCombinerV3": LightX2VConfigCombinerV3,
    "LightX2VModularInferenceV2": LightX2VModularInferenceV2,
    "LightX2VSeedVR2Loader": LightX2VSeedVR2Loader,
    "LightX2VSeedVR2Sampler": LightX2VSeedVR2Sampler,
    "LightX2VTalkObjectInput": TalkObjectInput,
    "LightX2VTalkObjectsCombiner": TalkObjectsCombiner,
    "LightX2VTalkObjectsFromJSON": TalkObjectsFromJSON,
    "LightX2VTalkObjectsFromFiles": TalkObjectsFromFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightX2VInferenceConfig": "LightX2V Inference Config",
    "LightX2VTeaCache": "LightX2V TeaCache",
    "LightX2VQuantization": "LightX2V Quantization",
    "LightX2VMemoryOptimization": "LightX2V Memory Optimization",
    "LightX2VLoRALoader": "LightX2V LoRA Loader",
    "LightX2VConfigCombinerV2": "LightX2V Config Combiner V2",
    "LightX2VConfigCombinerV3": "LightX2V Config Combiner V3",
    "LightX2VModularInferenceV2": "LightX2V Modular Inference V2",
    "LightX2VSeedVR2Loader": "LightX2V SeedVR2 Loader",
    "LightX2VSeedVR2Sampler": "LightX2V SeedVR2 Sampler",
    "LightX2VTalkObjectInput": "LightX2V Talk Object Input (Single)",
    "LightX2VTalkObjectsCombiner": "LightX2V Talk Objects Combiner",
    "LightX2VTalkObjectsFromFiles": "LightX2V Talk Objects From Files",
    "LightX2VTalkObjectsFromJSON": "LightX2V Talk Objects From JSON (API)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
