"""Translate ``LightX2VInferenceConfig`` widget values into lightx2v config keys.

The wrapper's widget naming follows ComfyUI conventions (``height``, ``width``,
``video_length``, ``cfg_scale`` …). lightx2v's internal naming is different
(``target_height``, ``target_width``, ``target_video_length``,
``sample_guide_scale`` …). The single source of truth for that translation
is the ``WRAPPER_TO_LIGHTX2V_FIELDS`` table below — when adding a new field,
add a row there rather than burying the rename inside the function body.
"""

import os
from typing import Any, Dict

from ..defaults import LightX2VDefaultConfig

# Direct rename map: wrapper-side key -> lightx2v-side key.
# A row of ("foo", "foo") means the name matches but we still want to forward
# the value explicitly (rather than relying on the default config).
WRAPPER_TO_LIGHTX2V_FIELDS: Dict[str, str] = {
    # Model selection — names match.
    "model_cls": "model_cls",
    "model_path": "model_path",
    "task": "task",
    # Inference loop.
    "infer_steps": "infer_steps",
    "seed": "seed",
    "sample_shift": "sample_shift",
    # Output shape — wrapper uses bare names, lightx2v prefixes with target_.
    "height": "target_height",
    "width": "target_width",
    "video_length": "target_video_length",
    "fps": "target_fps",
    "video_duration": "video_duration",
    # Image preprocessing.
    "resize_mode": "resize_mode",
    "fixed_area": "fixed_area",
    # Sekotalk-specific.
    "prev_frame_length": "prev_frame_length",
    "f2v_process": "f2v_process",
    # Distillation.
    "denoising_step_list": "denoising_step_list",
    "use_31_block": "use_31_block",
}

# Attention type fans out to three internal slots in lightx2v.
_ATTN_TYPE_SLOTS = ("self_attn_1_type", "cross_attn_1_type", "cross_attn_2_type")


def apply_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate inference widget values to a partial lightx2v config dict."""
    updates: Dict[str, Any] = {}

    # Bulk rename via the table.
    for wrapper_key, lightx2v_key in WRAPPER_TO_LIGHTX2V_FIELDS.items():
        if wrapper_key not in config:
            continue
        # seed=-1 means "use lightx2v's default / random"; leave it out.
        if wrapper_key == "seed" and config[wrapper_key] == -1:
            continue
        updates[lightx2v_key] = config[wrapper_key]

    # cfg_scale -> sample_guide_scale (and toggle enable_cfg).
    if "cfg_scale" in config:
        updates["sample_guide_scale"] = config["cfg_scale"]
        updates["enable_cfg"] = config["cfg_scale"] != 1.0

    # Wan2.2 MoE has two CFG scales (high/low noise) and a boundary param.
    model_cls = config.get("model_cls", "")
    if "wan2.2_moe" in model_cls:
        updates["boundary"] = 0.9
        updates["sample_guide_scale"] = [config.get("cfg_scale"), config.get("cfg_scale2")]
    if "wan2.2" in model_cls:
        updates["use_image_encoder"] = False

    # One widget value drives three lightx2v attention slots.
    attention_type = config.get("attention_type", LightX2VDefaultConfig.DEFAULT_ATTENTION_TYPE)
    for slot in _ATTN_TYPE_SLOTS:
        updates[slot] = attention_type

    # TAEW2.1 lightweight VAE lives next to the model.
    if config.get("use_tiny_vae", False):
        updates["use_tiny_vae"] = True
        updates["tiny_vae"] = True
        updates["tiny_vae_path"] = os.path.join(config["model_path"], "taew2_1.pth")

    return updates
