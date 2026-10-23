[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-brightgreen)](https://github.com/comfyanonymous/ComfyUI)
[![Models](https://img.shields.io/badge/models-LightX2V-yellow)](https://huggingface.co/lightx2v)

# ComfyUI-Lightx2vWrapper

ComfyUI custom nodes for [LightX2V](https://github.com/ModelTC/LightX2V). The
wrapper exposes LightX2V's modular configuration and runner APIs, together
with ComfyUI-friendly file inputs and output previews.

[中文文档](README_CN.md)

## Supported workflows

The repository includes runnable workflow examples under
[`example_workflows/`](example_workflows/):

- [Wan 2.2](example_workflows/lighx2v_wan2.2.json) text/image-to-video
- [SekoTalk](example_workflows/lightx2v_seko_talk.json) and [static
  SekoTalk](example_workflows/lightx2v_seko_talk_static.json)
- [Multi-person SekoTalk](example_workflows/lightx2v_seko_talk_multi_talk.json)
- [SeedVR2](example_workflows/lightx2v_seedvr.json) video restoration
- [SwiftVR image](example_workflows/lightx2v_swiftvr_image.json) and [SwiftVR
  video](example_workflows/lightx2v_swiftvr_video.json) restoration

The examples are versioned with the wrapper and are the best reference for
current node names and connections.

## Installation

Clone the repository with its LightX2V submodule into ComfyUI's
`custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/ModelTC/ComfyUI-Lightx2vWrapper.git
cd ComfyUI-Lightx2vWrapper
python -m pip install -r lightx2v/requirements.txt
python -m pip install -r requirements.txt
```

For an existing checkout, update the submodule with:

```bash
git submodule update --init --recursive
```

Download the model files from [LightX2V on Hugging Face](https://huggingface.co/lightx2v)
and place them under `ComfyUI/models/lightx2v/`.

## Node groups

- **Configuration:** `LightX2VInferenceConfig`, `LightX2VTeaCache`,
  `LightX2VQuantization`, and `LightX2VMemoryOptimization`.
- **Composition:** `LightX2VConfigCombinerV2` and `LightX2VConfigCombinerV3`.
- **Inference:** `LightX2VModularInferenceV2`.
- **Talk inputs:** single-object, file-list, JSON, and multi-object talk
  combiners.
- **Video inputs and restoration:** `LightX2VInputVideoPath`, SeedVR2 and
  SwiftVR loaders/samplers, plus `LightX2VOutputVideoPreview`.
- **LoRA:** `LightX2VLoRALoader`.

The inference node currently accepts the model classes exposed by
`model_utils.support_model_cls_list`: Wan 2.1/2.2 variants, SekoTalk,
Qwen Image, and SeedVR2. The retired `*_distill` values remain selectable as
compatibility aliases for saved ComfyUI workflows and are normalized before
LightX2V's runner is built.

## Model files

Place each model in its own directory under `ComfyUI/models/lightx2v/`. LoRA
files can be placed in either `ComfyUI/models/lightx2v/loras/` or
`ComfyUI/models/loras/`.

```text
ComfyUI/models/lightx2v/
├── Wan2.2-I2V-A14B-distill-fp8/
├── seedvr2_ema_3b_fp8/
└── loras/
```

The exact directory names depend on the downloaded checkpoint. The model
picker only lists directories that exist on the local ComfyUI installation.

## Development

Install the development hook and run the same checks used by CI:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for pull request and workflow
validation guidelines.
