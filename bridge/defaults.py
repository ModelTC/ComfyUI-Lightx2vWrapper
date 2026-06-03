"""Default config values that the wrapper provides to lightx2v.

These are the wrapper's *starting point* — lightx2v's own ``set_config`` will
further merge from ``config_json`` and the model's own ``config.json`` on disk
(see ``lightx2v/utils/set_config.py:set_config``). Anything lightx2v sets
internally (``vae_stride``, ``patch_size``, etc.) should NOT be duplicated here.
"""


class LightX2VDefaultConfig:
    """Central default configuration for LightX2V."""

    DEFAULT_ATTENTION_TYPE = "flash_attn3"
    DEFAULT_QUANTIZATION_SCHEMES = {
        "dit": "Default",
        "t5": "Default",
        "clip": "Default",
        "adapter": "Default",
    }

    DEFAULT_CONFIG = {
        # Model
        "model_cls": "wan2.1",
        "model_path": "",
        "task": "t2v",
        # Inference
        "infer_steps": 40,
        "seed": 42,
        "sample_guide_scale": 5.0,
        "sample_shift": 5,
        "enable_cfg": True,
        "prompt": "",
        "negative_prompt": "",
        # Video / Image output (lightx2v field names — see translator/inference.py)
        "target_height": 480,
        "target_width": 832,
        "target_video_length": 81,
        "fps": 16,
        # TeaCache
        "feature_caching": "NoCaching",
        "teacache_thresh": 0.26,
        "coefficients": None,
        "use_ret_steps": False,
        # Quantization
        "dit_quant_scheme": DEFAULT_QUANTIZATION_SCHEMES["dit"],
        "t5_quant_scheme": DEFAULT_QUANTIZATION_SCHEMES["t5"],
        "clip_quant_scheme": DEFAULT_QUANTIZATION_SCHEMES["clip"],
        "adapter_quant_scheme": DEFAULT_QUANTIZATION_SCHEMES["adapter"],
        # Attention
        "self_attn_1_type": DEFAULT_ATTENTION_TYPE,
        "cross_attn_1_type": DEFAULT_ATTENTION_TYPE,
        "cross_attn_2_type": DEFAULT_ATTENTION_TYPE,
        # Memory / offload
        "rotary_chunk": False,
        "rotary_chunk_size": 100,
        "clean_cuda_cache": False,
        "torch_compile": False,
        "cpu_offload": False,
        "offload_granularity": "block",
        "offload_ratio": 1.0,
        "t5_cpu_offload": False,
        "t5_offload_granularity": "model",
        "lazy_load": False,
        "unload_modules": False,
        # VAE
        "use_tiling_vae": False,
        # Misc
        "do_mm_calib": False,
        "max_area": False,
        "use_prompt_enhancer": False,
        "text_len": 512,
        "use_31_block": True,
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "audio_sr": 16000,
        "talk_objects": None,
        "boundary_step_index": 2,
        "rope_type": "torch",
    }
