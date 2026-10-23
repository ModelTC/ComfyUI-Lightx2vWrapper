[![许可证](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-自定义节点-brightgreen)](https://github.com/comfyanonymous/ComfyUI)
[![模型](https://img.shields.io/badge/models-LightX2V-yellow)](https://huggingface.co/lightx2v)

# ComfyUI-Lightx2vWrapper

[English](README.md) | 中文

[LightX2V](https://github.com/ModelTC/LightX2V) 的 ComfyUI 自定义节点封装，
提供模块化配置、ComfyUI 文件输入以及视频输出预览。

## 已提供的工作流

仓库在 [`example_workflows/`](example_workflows/) 中提供了与当前节点注册表
同步的示例：

- [Wan 2.2](example_workflows/lighx2v_wan2.2.json) 文生视频/图生视频
- [SekoTalk](example_workflows/lightx2v_seko_talk.json) 和[静态
  SekoTalk](example_workflows/lightx2v_seko_talk_static.json)
- [多人 SekoTalk](example_workflows/lightx2v_seko_talk_multi_talk.json)
- [SeedVR2](example_workflows/lightx2v_seedvr.json) 视频修复
- [SwiftVR 图像](example_workflows/lightx2v_swiftvr_image.json) 和 [SwiftVR
  视频](example_workflows/lightx2v_swiftvr_video.json) 修复

示例工作流会随封装器一起更新，可作为当前节点名称和连线方式的参考。

## 安装

将仓库和 LightX2V 子模块一起克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/ModelTC/ComfyUI-Lightx2vWrapper.git
cd ComfyUI-Lightx2vWrapper
python -m pip install -r lightx2v/requirements.txt
python -m pip install -r requirements.txt
```

如果已有仓库但还没有子模块：

```bash
git submodule update --init --recursive
```

从 [Hugging Face 的 LightX2V 页面](https://huggingface.co/lightx2v) 下载模型，
放到 `ComfyUI/models/lightx2v/` 下。

## 节点分组

- **配置：** `LightX2VInferenceConfig`、`LightX2VTeaCache`、
  `LightX2VQuantization`、`LightX2VMemoryOptimization`
- **组合：** `LightX2VConfigCombinerV2`、`LightX2VConfigCombinerV3`
- **推理：** `LightX2VModularInferenceV2`
- **Talk 输入：** 单对象、文件列表、JSON 和多人 Talk 组合节点
- **视频输入与修复：** `LightX2VInputVideoPath`、SeedVR2/SwiftVR 加载器与
  采样器、`LightX2VOutputVideoPreview`
- **LoRA：** `LightX2VLoRALoader`

推理节点当前支持 `model_utils.support_model_cls_list` 中的 Wan 2.1/2.2
变体、SekoTalk、Qwen Image 和 SeedVR2。旧 workflow 使用的 `*_distill`
值仍作为兼容别名保留，构建 LightX2V runner 前会自动归一化。

## 模型目录

每个模型放在 `ComfyUI/models/lightx2v/` 下的独立目录中。LoRA 可以放在
`ComfyUI/models/lightx2v/loras/` 或 `ComfyUI/models/loras/`：

```text
ComfyUI/models/lightx2v/
├── Wan2.2-I2V-A14B-distill-fp8/
├── seedvr2_ema_3b_fp8/
└── loras/
```

具体目录名取决于下载的 checkpoint；模型选择器只会列出当前 ComfyUI 中存在
的目录。

## 开发

安装 pre-commit 并运行 CI 使用的检查：

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

贡献流程和 workflow 验证要求见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
