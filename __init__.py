import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ENABLE_GRAPH_MODE"] = "false"
os.environ["ENABLE_PROFILING_DEBUG"] = "true"
# os.environ["SENSITIVE_LAYER_DTYPE"] = "FP32"
os.environ["PROFILING_DEBUG_LEVEL"] = "2"
os.environ["DTYPE"] = "BF16"

current_path = Path(__file__).parent.absolute()
print("Current path set to:", current_path)
sys.path.insert(0, os.path.join(current_path, "lightx2v"))  # Adjust the path as needed

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
