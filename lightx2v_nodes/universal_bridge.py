"""Universal bridge between LightX2V and ComfyUI for automatic adaptation."""

import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from PIL import Image
import tempfile
from easydict import EasyDict

# Import LightX2V modules
from ..lightx2v.utils.set_config import set_config, get_default_config
from ..lightx2v.utils.registry_factory import RUNNER_REGISTER
from ..lightx2v.infer import init_runner


class LightX2VBridge:
    """Universal bridge for LightX2V and ComfyUI integration."""

    def __init__(self):
        self._model_registry = None
        self._config_registry = None
        self._runners = {}  # Cache initialized runners

    @property
    def model_registry(self):
        """Lazy load model registry."""
        if self._model_registry is None:
            self._model_registry = self._discover_models()
        return self._model_registry

    @property
    def config_registry(self):
        """Lazy load config registry."""
        if self._config_registry is None:
            self._config_registry = self._discover_configs()
        return self._config_registry

    def _discover_models(self) -> List[str]:
        """Discover available model classes from LightX2V."""
        # These are the known model classes from the infer.py
        return ["wan2.1", "hunyuan", "wan2.1_distill", "wan2.1_causvid", "wan2.1_skyreels_v2_df", "cogvideox", "wan2.1_audio"]

    def _discover_configs(self) -> Dict[str, Path]:
        """Discover all available config files."""
        configs = {}
        base_dir = Path(__file__).parent.parent / "lightx2v" / "configs"

        if base_dir.exists():
            for config_file in base_dir.rglob("*.json"):
                # Create a descriptive key
                relative_path = config_file.relative_to(base_dir)
                key = str(relative_path).replace("/", "_").replace(".json", "")
                configs[key] = config_file

        return configs

    def load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load a config file."""
        with open(config_path, "r") as f:
            return json.load(f)

    def get_runner(self, config: EasyDict):
        """Get or create a runner for the given config."""
        # Create a unique key for this configuration
        key = f"{config.model_cls}_{config.model_path}_{hash(str(config))}"

        if key not in self._runners:
            self._runners[key] = init_runner(config)

        return self._runners[key]

    def clear_cache(self):
        """Clear cached runners to free memory."""
        self._runners.clear()
        torch.cuda.empty_cache()


class InputOutputConverter:
    """Convert between ComfyUI and LightX2V formats."""

    @staticmethod
    def comfy_image_to_pil(comfy_image: torch.Tensor) -> Image.Image:
        """Convert ComfyUI image tensor to PIL Image.

        ComfyUI format: [B, H, W, C] float32 0-1
        """
        # Take first image from batch
        image_np = (comfy_image[0].cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    @staticmethod
    def comfy_audio_to_path(comfy_audio: Dict) -> str:
        """Convert ComfyUI audio to file path."""
        # ComfyUI audio format: {"waveform": tensor, "sample_rate": int}
        waveform = comfy_audio["waveform"]
        sample_rate = comfy_audio["sample_rate"]

        # Save to temporary file
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, waveform, sample_rate)
            return tmp.name

    @staticmethod
    def video_to_latent(video_path: str) -> Dict[str, Any]:
        """Convert video path to ComfyUI latent format."""
        # For now, return the path wrapped in latent format
        # This allows downstream nodes to handle the video
        return {"samples": video_path, "type": "video", "format": "path"}

    @staticmethod
    def tensor_to_latent(video_tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert video tensor to ComfyUI latent format."""
        return {"samples": video_tensor, "type": "video", "format": "tensor"}


class ConfigManager:
    """Manage configuration merging and parameter mapping."""

    # Mapping from ComfyUI parameter names to LightX2V config keys
    PARAM_MAPPING = {
        "steps": "infer_steps",
        "cfg_scale": "sample_guide_scale",
        "seed": "seed",
        "height": "target_height",
        "width": "target_width",
        "video_length": "target_video_length",
    }

    @classmethod
    def create_config(cls, model_cls: str, model_path: str, task: str, base_config: Dict, overrides: Dict) -> EasyDict:
        """Create a complete config for LightX2V runner."""
        # Start with default config
        config = get_default_config()

        # Add required fields
        config.update(
            {
                "model_cls": model_cls,
                "model_path": model_path,
                "task": task,
                "mode": "infer",
            }
        )

        # Apply base config from file
        config.update(base_config)

        # Apply ComfyUI overrides
        for comfy_key, value in overrides.items():
            if value is not None and value != -1:  # -1 means use default
                if comfy_key in cls.PARAM_MAPPING:
                    lightx2v_key = cls.PARAM_MAPPING[comfy_key]
                    config[lightx2v_key] = value
                else:
                    # Direct mapping for unknown parameters
                    config[comfy_key] = value

        return EasyDict(config)

    @classmethod
    def parse_custom_config(cls, custom_config_str: str) -> Dict:
        """Parse custom config string."""
        if not custom_config_str.strip():
            return {}

        try:
            return json.loads(custom_config_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse custom config: {e}")
            return {}


class UniversalLightX2VNode:
    """Universal ComfyUI node for all LightX2V models."""

    def __init__(self):
        self.bridge = LightX2VBridge()
        self.converter = InputOutputConverter()

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for the node."""
        bridge = LightX2VBridge()

        # Get available models and configs
        model_choices = bridge.model_registry
        config_choices = ["custom"] + list(bridge.config_registry.keys())

        return {
            "required": {
                "model_cls": (model_choices, {"tooltip": "Model class to use"}),
                "model_path": ("STRING", {"default": "", "tooltip": "Path to model weights"}),
                "task": (["t2v", "i2v"], {"default": "t2v", "tooltip": "Task type: text-to-video or image-to-video"}),
                "config_preset": (config_choices, {"default": "custom", "tooltip": "Configuration preset or 'custom'"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt for generation"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1, "tooltip": "Random seed (-1 for random)"}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 200, "tooltip": "Inference steps (-1 for default)"}),
                "cfg_scale": ("FLOAT", {"default": -1, "min": -1, "max": 30, "step": 0.1, "tooltip": "Guidance scale (-1 for default)"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Input image for i2v task"}),
                "audio": ("AUDIO", {"tooltip": "Input audio for audio-driven generation"}),
                "custom_config": ("STRING", {"multiline": True, "default": "{}", "tooltip": "Custom configuration in JSON format"}),
                "lora_path": ("STRING", {"default": "", "tooltip": "Path to LoRA weights"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Model strength for LoRA"}),
                "height": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 8, "tooltip": "Video height (-1 for default)"}),
                "width": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 8, "tooltip": "Video width (-1 for default)"}),
                "video_length": ("INT", {"default": -1, "min": -1, "max": 300, "tooltip": "Number of frames (-1 for default)"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "LightX2V/Universal"

    def generate(
        self,
        model_cls,
        model_path,
        task,
        config_preset,
        prompt,
        negative_prompt,
        seed,
        steps,
        cfg_scale,
        image=None,
        audio=None,
        custom_config="{}",
        lora_path="",
        strength_model=1.0,
        height=-1,
        width=-1,
        video_length=-1,
        **kwargs,
    ):
        """Generate video using LightX2V."""

        # Load base configuration
        if config_preset == "custom":
            base_config = {}
        else:
            config_path = self.bridge.config_registry.get(config_preset)
            if config_path:
                base_config = self.bridge.load_config(config_path)
            else:
                raise ValueError(f"Config preset '{config_preset}' not found")

        # Parse custom config
        custom_cfg = ConfigManager.parse_custom_config(custom_config)
        base_config.update(custom_cfg)

        # Prepare overrides
        overrides = {
            "seed": seed if seed != -1 else None,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "video_length": video_length,
            "lora_path": lora_path if lora_path else None,
            "strength_model": strength_model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }

        # Handle image input for i2v
        if task == "i2v" and image is not None:
            # Save image to temporary file
            pil_image = self.converter.comfy_image_to_pil(image)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp.name)
                overrides["image_path"] = tmp.name

        # Handle audio input
        if audio is not None and "audio" in model_cls:
            audio_path = self.converter.comfy_audio_to_path(audio)
            overrides["audio_path"] = audio_path

        # Create final config
        config = ConfigManager.create_config(model_cls, model_path, task, base_config, overrides)

        # Create a temporary config file (required by set_config)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(base_config, tmp)
            config.config_json = tmp.name

        # Get or create runner
        try:
            runner = self.bridge.get_runner(config)

            # Run generation
            result = runner.run_pipeline()

            # Convert output
            if hasattr(result, "save_video_path"):
                return (self.converter.video_to_latent(result.save_video_path),)
            else:
                # Assume tensor output
                return (self.converter.tensor_to_latent(result),)

        finally:
            # Cleanup temporary files
            if "image_path" in overrides and os.path.exists(overrides["image_path"]):
                os.unlink(overrides["image_path"])
            if "audio_path" in overrides and os.path.exists(overrides["audio_path"]):
                os.unlink(overrides["audio_path"])
            if hasattr(config, "config_json") and os.path.exists(config.config_json):
                os.unlink(config.config_json)


# Node class mapping
NODE_CLASS_MAPPINGS = {
    "UniversalLightX2V": UniversalLightX2VNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalLightX2V": "LightX2V Universal Generator",
}
