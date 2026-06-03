"""GPU and backend-op capability detection.

Pure functions — no state, no module-level side effects. Cheap to call
(the underlying ``torch.cuda`` / ``importlib`` probes are fast).
"""

import importlib.util
import logging
from typing import List, Tuple

import torch


def get_gpu_capability():
    if not torch.cuda.is_available():
        return None, None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception as e:
        logging.warning(f"Failed to get GPU capability: {e}")
        return None, None


def is_fp8_supported_gpu() -> bool:
    major, minor = get_gpu_capability()
    if major is None:
        return False
    return (major == 8 and minor == 9) or (major >= 9)


def is_ada_architecture_gpu() -> bool:
    major, minor = get_gpu_capability()
    if major is None:
        return False
    return major == 8 and minor == 9


def is_module_installed(module_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ModuleNotFoundError:
        return False


def get_available_ops(op_mapping: dict) -> List[Tuple[str, bool]]:
    return [(op_name, is_module_installed(module_name)) for op_name, module_name in op_mapping.items()]


_QUANT_OP_MAPPING = {
    "sgl": "sgl_kernel",
    "vllm": "vllm",
    "q8f": "q8_kernels",
    "torchao": "torchao",
}

_ATTN_OP_MAPPING = {
    "sage_attn2": "sageattention",
    "sage_attn3": "sageattn3",
    "flash_attn3": "flash_attn_interface",
    "flash_attn2": "flash_attn",
    "torch_sdpa": "torch",
}


def get_available_quant_ops() -> List[Tuple[str, bool]]:
    available_ops = get_available_ops(_QUANT_OP_MAPPING)

    # Prefer q8f on Ada (sm_8.9) GPUs — best perf/precision tradeoff there.
    if is_ada_architecture_gpu():
        q8f_available = next((op for op in available_ops if op[0] == "q8f" and op[1]), None)
        if q8f_available:
            available_ops.remove(q8f_available)
            available_ops.insert(0, q8f_available)

    return available_ops


def get_available_attn_ops() -> List[Tuple[str, bool]]:
    return get_available_ops(_ATTN_OP_MAPPING)
