# coding: utf-8

import json
from pathlib import Path
from typing import Dict, List

import folder_paths


def get_model_base_path() -> Path:
    models_base = folder_paths.models_dir
    lightx2v_path = Path(models_base) / "lightx2v"
    if not lightx2v_path.exists():
        lightx2v_path.mkdir(parents=True, exist_ok=True)
    return lightx2v_path


def scan_models() -> List[str]:
    models = []
    base_path = get_model_base_path()

    if base_path.exists():
        for item in base_path.iterdir():
            if item.is_dir() and item.name != "loras":
                models.append(item.name)

    models.sort()

    return ["None"] + models if models else ["None"]


def support_model_cls_list() -> List[str]:
    return [
        "wan2.1",
        "wan2.1_distill",
        "wan2.1_vace",
        "cogvideox",
        "seko_talk",
        "wan2.2_moe",
        "wan2.2",
        "wan2.2_moe_audio",
        "wan2.2_audio",
        "wan2.2_moe_distill",
        "qwen_image",
    ]


def get_loras_models(model_path: Path) -> List[str]:
    loras = []
    if model_path.exists():
        for item in model_path.iterdir():
            if item.is_file():
                if item.suffix.lower() in [".safetensors", ".pt", ".pth", ".ckpt"]:
                    loras.append(item.name)
    return loras


def scan_loras() -> List[str]:
    base_path = get_model_base_path()
    loras_path = base_path / "loras"
    loras = get_loras_models(loras_path)

    models_base = folder_paths.models_dir
    loras_path = Path(models_base) / "loras"

    loras2 = get_loras_models(loras_path)
    loras.extend(loras2)

    loras.sort()

    return ["None"] + loras if loras else ["None"]


def get_model_full_path(model_name: str) -> str:
    if model_name == "None" or not model_name:
        return ""

    base_path = get_model_base_path()
    model_path = base_path / model_name

    if model_path.exists():
        return str(model_path)
    return ""


def get_lora_full_path(lora_name: str) -> str:
    if lora_name == "None" or not lora_name:
        return ""

    base_path = get_model_base_path()
    lora_path = base_path / "loras" / lora_name

    if lora_path.exists():
        return str(lora_path)

    models_base = folder_paths.models_dir
    loras_path = Path(models_base) / "loras" / lora_name

    if loras_path.exists():
        return str(loras_path)

    return ""


def get_model_info(model_name: str) -> Dict:
    if model_name == "None" or not model_name:
        return {}

    base_path = get_model_base_path()
    config_path = base_path / model_name / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}
