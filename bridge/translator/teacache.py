"""Translate ``LightX2VTeaCache`` widget values into lightx2v config keys.

Wrapper-side ``enable / threshold / use_ret_steps`` -> lightx2v-side
``feature_caching / teacache_thresh / use_ret_steps`` + polynomial coefficients
picked from the calibration table.
"""

from typing import Any, Dict

from ..teacache_coeffs import CoefficientCalculator


def apply_teacache_config(config: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Translate TeaCache widget values.

    ``model_info`` is the partially-built lightx2v config so we can pick
    coefficients matched to the actual task and output resolution.
    """
    if not config.get("enable", False):
        return {"feature_caching": "NoCaching"}

    use_ret_steps = config.get("use_ret_steps", False)
    task = model_info.get("task", "t2v")
    model_size = "14b" if "14b" in model_info.get("model_cls", "") else "1.3b"
    resolution = (
        model_info.get("target_width", 832),
        model_info.get("target_height", 480),
    )

    return {
        "feature_caching": "Tea",
        "teacache_thresh": config.get("threshold", 0.26),
        "use_ret_steps": use_ret_steps,
        "coefficients": CoefficientCalculator.get_coefficients(task, model_size, resolution, use_ret_steps),
    }
