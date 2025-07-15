import os
import sys
from pathlib import Path

current_path = Path(__file__).parent.absolute()
print("Current path set to:", current_path)
sys.path.insert(0, os.path.join(current_path, "lightx2v"))  # Adjust the path as needed

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
