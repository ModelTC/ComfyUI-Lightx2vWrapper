"""ComfyUI node definitions for LightX2V.

Each submodule groups a category of nodes:
- ``config``    : per-feature configuration nodes (inference / teacache / quant / memory)
- ``lora``      : LoRA chain loader
- ``talk``      : talk-object input/combiner nodes
- ``combiner``  : config combiners (V1/V2/V3) that aggregate the above
- ``inference`` : the modular inference runner
- ``file_input``: validated, upload-backed media paths
- ``seedvr``    : SeedVR2 super-resolution runner
- ``swiftvr``   : SwiftVR restoration runner
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
from .file_input import LightX2VInputVideoPath
from .inference import LightX2VModularInferenceV2
from .lora import LightX2VLoRALoader
from .seedvr import LightX2VOutputVideoPreview, LightX2VSeedVR2FileSampler, LightX2VSeedVR2Loader, LightX2VSeedVR2Sampler
from .swiftvr import LightX2VSwiftVRFileSampler, LightX2VSwiftVRLoader, LightX2VSwiftVRSampler
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
    "LightX2VInputVideoPath": LightX2VInputVideoPath,
    "LightX2VSeedVR2Loader": LightX2VSeedVR2Loader,
    "LightX2VSeedVR2Sampler": LightX2VSeedVR2Sampler,
    "LightX2VSeedVR2FileSampler": LightX2VSeedVR2FileSampler,
    "LightX2VSwiftVRLoader": LightX2VSwiftVRLoader,
    "LightX2VSwiftVRSampler": LightX2VSwiftVRSampler,
    "LightX2VSwiftVRFileSampler": LightX2VSwiftVRFileSampler,
    "LightX2VOutputVideoPreview": LightX2VOutputVideoPreview,
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
    "LightX2VInputVideoPath": "LightX2V Input Video Path (Upload)",
    "LightX2VSeedVR2Loader": "LightX2V SeedVR2 Loader",
    "LightX2VSeedVR2Sampler": "LightX2V SeedVR2 Sampler",
    "LightX2VSeedVR2FileSampler": "LightX2V SeedVR2 File Sampler",
    "LightX2VSwiftVRLoader": "LightX2V SwiftVR Loader",
    "LightX2VSwiftVRSampler": "LightX2V SwiftVR Sampler",
    "LightX2VSwiftVRFileSampler": "LightX2V SwiftVR File Sampler",
    "LightX2VOutputVideoPreview": "LightX2V Output Video Preview",
    "LightX2VTalkObjectInput": "LightX2V Talk Object Input (Single)",
    "LightX2VTalkObjectsCombiner": "LightX2V Talk Objects Combiner",
    "LightX2VTalkObjectsFromFiles": "LightX2V Talk Objects From Files",
    "LightX2VTalkObjectsFromJSON": "LightX2V Talk Objects From JSON (API)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
