"""ComfyUI-Lightx2vWrapper entrypoint.

ComfyUI discovers custom nodes by importing this package and reading
``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS``. The actual node
classes live under the ``nodes/`` subpackage.
"""

import os
import sys
from pathlib import Path


def _setup_env() -> None:
    """Set environment variables consumed by the bundled lightx2v engine.

    Done in a function (instead of bare module-level statements) so the
    side effects are explicit and easy to audit. ComfyUI imports this
    module exactly once at startup, which is when these need to be set.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PROFILING_DEBUG_LEVEL", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ENABLE_GRAPH_MODE", "false")
    os.environ.setdefault("ENABLE_PROFILING_DEBUG", "true")
    os.environ.setdefault("DTYPE", "BF16")


def _register_lightx2v_submodule() -> None:
    """Expose the bundled ``lightx2v/`` git submodule on ``sys.path``.

    The submodule ships its own top-level package also named ``lightx2v``;
    putting the outer directory on ``sys.path`` lets internal modules import
    ``lightx2v.xxx`` directly (as they do, e.g. ``lightx2v.common.ops``).
    Our own nodes import via the relative path ``..lightx2v.lightx2v.xxx``
    and do not depend on this entry, but third-party / lightx2v-internal
    code does.
    """
    submodule_root = Path(__file__).parent.absolute() / "lightx2v"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))


_setup_env()
_register_lightx2v_submodule()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
