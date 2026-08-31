"""Per-feature configuration nodes: inference / teacache / quantization / memory."""

from ..bridge import get_available_attn_ops, get_available_quant_ops
from ..config_builder import InferenceConfigBuilder
from ..data_models import (
    MemoryOptimizationConfig,
    QuantizationConfig,
    TeaCacheConfig,
)
from ..model_utils import scan_models, support_model_cls_list


class LightX2VInferenceConfig:
    @classmethod
    def INPUT_TYPES(cls):
        available_models = scan_models()
        support_model_classes = support_model_cls_list()
        available_attn = get_available_attn_ops()
        attn_types = []

        for op_name, is_available in available_attn:
            if is_available:
                attn_types.append(op_name)

        if "torch_sdpa" not in attn_types:
            attn_types.append("torch_sdpa")

        return {
            "required": {
                "model_cls": (
                    support_model_classes,
                    {"default": "wan2.1", "tooltip": "Model type"},
                ),
                "model_name": (
                    available_models,
                    {
                        "default": available_models[0],
                        "tooltip": "Select model from available models",
                    },
                ),
                "task": (
                    ["t2v", "i2v", "s2v", "rs2v"],
                    {
                        "default": "i2v",
                        "tooltip": "Task type: text-to-video or image-to-video or reference_image and audio to video (shot)",
                    },
                ),
                "infer_steps": (
                    "INT",
                    {"default": 4, "min": 1, "max": 100, "tooltip": "Inference steps"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": -1,
                        "max": 2**32 - 1,
                        "tooltip": "Random seed, -1 for random",
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "CFG guidance strength",
                    },
                ),
                "cfg_scale2": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "CFG guidance, lower noise when model cls is Wan2.2 MoE",
                    },
                ),
                "sample_shift": (
                    "INT",
                    {"default": 5, "min": 0, "max": 10, "tooltip": "Sample shift"},
                ),
                "height": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Video height",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 720,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Video width",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 999,
                        "step": 0.1,
                        "tooltip": "Video duration in seconds",
                    },
                ),
                "attention_type": (
                    attn_types,
                    {"default": attn_types[0], "tooltip": "Attention mechanism type"},
                ),
            },
            "optional": {
                "denoising_steps": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Custom denoising steps for distillation models (comma-separated, e.g., '999,750,500,250'). Leave empty to use model defaults.",
                    },
                ),
                "resize_mode": (
                    [
                        "adaptive",
                        "keep_ratio_fixed_area",
                        "fixed_min_area",
                        "fixed_max_area",
                        "fixed_shape",
                        "fixed_min_side",
                    ],
                    {
                        "default": "adaptive",
                        "tooltip": "Adaptive resize input image to target aspect ratio",
                    },
                ),
                "fixed_area": (
                    "STRING",
                    {
                        "default": "720p",
                        "tooltip": "Fixed shape for input image, e.g., '720p', '480p', when resize_mode is 'keep_ratio_fixed_area' or 'fixed_min_side'",
                    },
                ),
                "segment_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 16,
                        "max": 256,
                        "tooltip": "Segment length in frames for sekotalk models (target_video_length)",
                    },
                ),
                "prev_frame_length": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 16,
                        "tooltip": "Previous frame overlap for sekotalk models",
                    },
                ),
                "use_tiny_vae": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use lightweight VAE to accelerate decoding",
                    },
                ),
                "f2v_process": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use frame-to-video processing for SekoTalk",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INFERENCE_CONFIG",)
    RETURN_NAMES = ("inference_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self,
        model_cls,
        model_name,
        task,
        infer_steps,
        seed,
        cfg_scale,
        cfg_scale2,
        sample_shift,
        height,
        width,
        duration,
        attention_type,
        denoising_steps="",
        resize_mode="adaptive",
        fixed_area="720p",
        segment_length=81,
        prev_frame_length=5,
        use_tiny_vae=False,
        f2v_process=False,
    ):
        """Create basic inference configuration."""
        builder = InferenceConfigBuilder()

        config = builder.build(
            model_cls=model_cls,
            model_name=model_name,
            task=task,
            infer_steps=infer_steps,
            seed=seed,
            cfg_scale=cfg_scale,
            cfg_scale2=cfg_scale2,
            sample_shift=sample_shift,
            height=height,
            width=width,
            duration=duration,
            attention_type=attention_type,
            denoising_steps=denoising_steps,
            resize_mode=resize_mode,
            fixed_area=fixed_area,
            segment_length=segment_length,
            prev_frame_length=prev_frame_length,
            use_tiny_vae=use_tiny_vae,
            f2v_process=f2v_process,
        )

        return (config.to_dict(),)


class LightX2VTeaCache:
    """TeaCache configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable TeaCache feature caching"},
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.26,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Cache threshold, lower values provide more speedup: 0.1 ~2x speedup, 0.2 ~3x speedup",
                    },
                ),
                "use_ret_steps": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only cache key steps to balance quality and speed",
                    },
                ),
            }
        }

    RETURN_TYPES = ("TEACACHE_CONFIG",)
    RETURN_NAMES = ("teacache_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(self, enable, threshold, use_ret_steps):
        """Create TeaCache configuration."""
        config = TeaCacheConfig(enable=enable, threshold=threshold, use_ret_steps=use_ret_steps)
        return (config.to_dict(),)


class LightX2VQuantization:
    @classmethod
    def INPUT_TYPES(cls):
        available_ops = get_available_quant_ops()
        quant_backends = []

        for op_name, is_available in available_ops:
            if is_available:
                quant_backends.append(op_name)

        common_schema = ["fp8", "int8"]
        supported_quant_schemes = ["Default"]
        for schema in common_schema:
            for backend in quant_backends:
                supported_quant_schemes.append(f"{schema}-{backend}")

        return {
            "required": {
                "dit_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[0],
                        "tooltip": "DIT model quantization precision",
                    },
                ),
                "t5_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[0],
                        "tooltip": "T5 encoder quantization precision",
                    },
                ),
                "clip_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[0],
                        "tooltip": "CLIP encoder quantization precision",
                    },
                ),
                "adapter_quant_scheme": (
                    supported_quant_schemes,
                    {
                        "default": supported_quant_schemes[0],
                        "tooltip": "Adapter quantization precision",
                    },
                ),
            }
        }

    RETURN_TYPES = ("QUANT_CONFIG",)
    RETURN_NAMES = ("quantization_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self,
        dit_quant_scheme,
        t5_quant_scheme,
        clip_quant_scheme,
        adapter_quant_scheme,
    ):
        """Create quantization configuration."""
        config = QuantizationConfig(
            dit_quant_scheme=dit_quant_scheme,
            t5_quant_scheme=t5_quant_scheme,
            clip_quant_scheme=clip_quant_scheme,
            adapter_quant_scheme=adapter_quant_scheme,
        )
        return (config.to_dict(),)


class LightX2VMemoryOptimization:
    """Memory optimization configuration node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_rotary_chunk": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable rotary encoding chunking"},
                ),
                "rotary_chunk_size": (
                    "INT",
                    {"default": 100, "min": 100, "max": 10000, "step": 100},
                ),
                "clean_cuda_cache": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Clean CUDA cache promptly"},
                ),
                "cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable CPU offloading"},
                ),
                "offload_granularity": (
                    ["block", "phase", "model"],
                    {"default": "block", "tooltip": "Offload granularity"},
                ),
                "offload_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "t5_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable T5 CPU offloading"},
                ),
                "t5_offload_granularity": (
                    ["model", "block"],
                    {"default": "model", "tooltip": "T5 offload granularity"},
                ),
                "audio_encoder_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable audio encoder CPU offloading"},
                ),
                "audio_adapter_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable audio adapter CPU offloading"},
                ),
                "vae_cpu_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable VAE CPU offloading"},
                ),
                "use_tiling_vae": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable VAE tiling inference"},
                ),
                "lazy_load": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Lazy load model"},
                ),
                "unload_after_inference": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Unload modules after inference"},
                ),
            },
        }

    RETURN_TYPES = ("MEMORY_CONFIG",)
    RETURN_NAMES = ("memory_config",)
    FUNCTION = "create_config"
    CATEGORY = "LightX2V/Config"

    def create_config(
        self,
        enable_rotary_chunk=False,
        rotary_chunk_size=100,
        clean_cuda_cache=False,
        cpu_offload=False,
        offload_granularity="phase",
        offload_ratio=1.0,
        t5_cpu_offload=True,
        t5_offload_granularity="model",
        audio_encoder_cpu_offload=False,
        audio_adapter_cpu_offload=False,
        vae_cpu_offload=False,
        use_tiling_vae=False,
        lazy_load=False,
        unload_after_inference=False,
    ):
        """Create memory optimization configuration."""
        config = MemoryOptimizationConfig(
            enable_rotary_chunk=enable_rotary_chunk,
            rotary_chunk_size=rotary_chunk_size,
            clean_cuda_cache=clean_cuda_cache,
            cpu_offload=cpu_offload,
            offload_granularity=offload_granularity,
            offload_ratio=offload_ratio,
            t5_cpu_offload=t5_cpu_offload,
            t5_offload_granularity=t5_offload_granularity,
            audio_encoder_cpu_offload=audio_encoder_cpu_offload,
            audio_adapter_cpu_offload=audio_adapter_cpu_offload,
            vae_cpu_offload=vae_cpu_offload,
            use_tiling_vae=use_tiling_vae,
            lazy_load=lazy_load,
            unload_after_inference=unload_after_inference,
        )
        return (config.to_dict(),)
