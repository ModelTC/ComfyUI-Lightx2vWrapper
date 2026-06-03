"""Translate ``LightX2VMemoryOptimization`` widget values into lightx2v config keys.

Several toggles only matter when their parent is enabled (e.g. ``offload_granularity``
only when ``cpu_offload=True``). Those nested keys are written only on the
true-branch to keep the resulting config dict minimal.
"""

from typing import Any, Dict

# Direct rename: wrapper key -> lightx2v key.
WRAPPER_TO_LIGHTX2V_FIELDS: Dict[str, str] = {
    "enable_rotary_chunk": "rotary_chunk",
    "clean_cuda_cache": "clean_cuda_cache",
    "cpu_offload": "cpu_offload",
    "t5_cpu_offload": "t5_cpu_offload",
    "vae_cpu_offload": "vae_cpu_offload",
    "audio_encoder_cpu_offload": "audio_encoder_cpu_offload",
    "audio_adapter_cpu_offload": "audio_adapter_cpu_offload",
    "lazy_load": "lazy_load",
    "unload_after_inference": "unload_modules",
    "use_tiling_vae": "use_tiling_vae",
}


def apply_memory_optimization(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate memory-optimization widget values."""
    updates: Dict[str, Any] = {}

    # NOTE: legacy behavior — when a specific offload key is missing, fall back
    # to the global ``cpu_offload`` flag. This means if the user only sets
    # ``cpu_offload=True``, every sub-offload (T5/VAE/audio…) silently follows.
    # Preserved as-is for backward compat; revisit when audio_* offloads
    # become widget-exposed everywhere.
    global_cpu_offload = config.get("cpu_offload", False)
    for wrapper_key, lightx2v_key in WRAPPER_TO_LIGHTX2V_FIELDS.items():
        updates[lightx2v_key] = config.get(wrapper_key, global_cpu_offload)

    if updates.get("rotary_chunk"):
        updates["rotary_chunk_size"] = config.get("rotary_chunk_size", 100)

    if updates.get("cpu_offload"):
        updates["offload_granularity"] = config.get("offload_granularity", "phase")
        updates["offload_ratio"] = config.get("offload_ratio", 1.0)

    if updates.get("t5_cpu_offload"):
        updates["t5_offload_granularity"] = config.get("t5_offload_granularity", "model")

    return updates
