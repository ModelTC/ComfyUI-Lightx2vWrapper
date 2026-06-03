"""TeaCache polynomial coefficients per (task, model size, resolution).

These constants come from the upstream TeaCache calibration runs (one set per
task/resolution bucket). They are pure data; no logic here other than picking
the right bucket.
"""

from typing import List, Tuple


class CoefficientCalculator:
    """Pick TeaCache polynomial coefficients for a given task/model/resolution."""

    COEFFICIENTS = {
        "t2v": {
            "1.3b": {
                "default": [
                    [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
                    [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
                ]
            },
            "14b": {
                "default": [
                    [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
                    [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                ]
            },
        },
        "i2v": {
            "720p": [
                [8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02],
                [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
            ],
            "480p": [
                [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
                [-3.02331670e02, 2.23948934e02, -5.25463970e01, 5.87348440e00, -2.01973289e-01],
            ],
        },
    }

    @classmethod
    def get_coefficients(
        cls,
        task: str,
        model_size: str,
        resolution: Tuple[int, int],
        use_ret_steps: bool,
    ) -> List[float]:
        """Pick the right coefficient row for this (task, model_size, resolution).

        ``use_ret_steps`` selects between the two calibration runs (cache key
        steps only vs. cache all steps).
        """
        if task == "t2v":
            coeffs = cls.COEFFICIENTS["t2v"].get(model_size, {}).get("default", None)
        else:  # i2v
            width, height = resolution
            coeffs = cls.COEFFICIENTS["i2v"]["720p"] if height >= 720 or width >= 720 else cls.COEFFICIENTS["i2v"]["480p"]

        if coeffs:
            return coeffs[0] if use_ret_steps else coeffs[1]
        raise ValueError(
            f"No coefficients found for task: {task}, model_size: {model_size}, resolution: {resolution}, use_ret_steps: {use_ret_steps}"
        )
