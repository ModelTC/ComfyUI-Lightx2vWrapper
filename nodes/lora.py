"""LoRA chain loader node."""

from ..config_builder import LoRAChainBuilder
from ..model_utils import scan_loras


class LightX2VLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        available_loras = scan_loras()

        return {
            "required": {
                "lora_name": (
                    available_loras,
                    {
                        "default": available_loras[0],
                        "tooltip": "Select LoRA from available LoRAs",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "LoRA strength",
                    },
                ),
            },
            "optional": {
                "lora_chain": (
                    "LORA_CHAIN",
                    {"tooltip": "Previous LoRA chain to append to"},
                ),
            },
        }

    RETURN_TYPES = ("LORA_CHAIN",)
    RETURN_NAMES = ("lora_chain",)
    FUNCTION = "load_lora"
    CATEGORY = "LightX2V/LoRA"

    def load_lora(self, lora_name, strength, lora_chain=None):
        """Load and chain LoRA configurations."""
        chain = LoRAChainBuilder.build_chain(lora_name=lora_name, strength=strength, existing_chain=lora_chain)
        return (chain,)
