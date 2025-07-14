"""
Refactored nodes.py that uses the new modular structure while maintaining backward compatibility.
"""
# Import refactored modules
from .lightx2v_refactored.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Export the mappings
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]