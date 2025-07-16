[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/) [![ComfyUI](https://img.shields.io/badge/ComfyUI-自定义节点-brightgreen)](https://github.com/comfyanonymous/ComfyUI) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20模型-lightx2v-yellow)](https://huggingface.co/lightx2v) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/GacLove/ComfyUI-Lightx2vWrapper)

# ComfyUI-Lightx2vWrapper

[English](./README.md) | 中文版

LightX2V 的 ComfyUI 自定义节点封装，通过模块化配置实现视频生成和高级优化功能。

## 功能特性

- **模块化配置系统**：为视频生成的各个方面提供独立的配置节点
- **文本生成视频（T2V）和图像生成视频（I2V）**：支持两种生成模式
- **高级优化功能**：
  - TeaCache 加速（最高可达 3 倍速度提升）
  - 量化支持（int8、fp8）
  - 内存优化与 CPU 卸载
  - 轻量级 VAE 选项
- **LoRA 支持**：可串联多个 LoRA 模型进行自定义
- **多模型支持**：wan2.1、hunyuan 架构

## 安装说明

1. 将此仓库克隆到 ComfyUI 的 `custom_nodes` 目录，包含子模块：

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/gaclove/ComfyUI-Lightx2vWrapper.git
```

如果已经克隆但未包含子模块，请初始化子模块：

```bash
cd ComfyUI-Lightx2vWrapper
git submodule update --init --recursive
```

2. 安装依赖：

```bash
cd ComfyUI-Lightx2vWrapper
# 安装 lightx2v 子模块依赖
pip install -r lightx2v/requirements.txt
# 安装 ComfyUI 封装器依赖
pip install -r requirements.txt
```

3. 下载模型并放置在 `ComfyUI/models/lightx2v/` 目录

## 节点概览

### 配置节点

#### 1. LightX2V Inference Config（推理配置）

视频生成的基础推理配置。

- **输入**：model（模型）、task_type（任务类型）、inference_steps（推理步数）、seed（种子）、cfg_scale（CFG 缩放）、width（宽度）、height（高度）、video_length（视频长度）、fps（帧率）
- **输出**：基础配置对象

#### 2. LightX2V TeaCache（TeaCache 配置）

特征缓存加速配置。

- **输入**：enable（启用）、threshold（阈值 0.0-1.0）、use_ret_steps（使用返回步数）
- **输出**：TeaCache 配置
- **注意**：阈值越低 = 速度提升越大（0.1 约 2 倍，0.2 约 3 倍）

#### 3. LightX2V Quantization（量化配置）

模型量化设置，提升内存效率。

- **输入**：dit_precision（DIT 精度）、t5_precision（T5 精度）、clip_precision（CLIP 精度）、backend（后端）、sensitive_layers_precision（敏感层精度）
- **输出**：量化配置
- **后端**：自动检测（vllm、sgl、q8f）

#### 4. LightX2V Memory Optimization（内存优化）

内存管理策略。

- **输入**：optimization_level（优化级别）、attention_type（注意力类型）、enable_rotary_chunking（启用旋转分块）、cpu_offload（CPU 卸载）、unload_after_generate（生成后卸载）
- **输出**：内存优化配置

#### 5. LightX2V Lightweight VAE（轻量级 VAE）

VAE 优化选项。

- **输入**：use_tiny_vae（使用微型 VAE）、use_tiling_vae（使用平铺 VAE）
- **输出**：VAE 配置

#### 6. LightX2V LoRA Loader（LoRA 加载器）

加载并串联 LoRA 模型。

- **输入**：lora_name（LoRA 名称）、strength（强度 0.0-2.0）、lora_chain（LoRA 链，可选）
- **输出**：LoRA 链配置

### 组合节点

#### 7. LightX2V Config Combiner（配置组合器）

将所有配置模块组合成单一配置。

- **输入**：所有配置类型（可选）
- **输出**：组合后的配置对象

### 推理节点

#### 8. LightX2V Modular Inference（模块化推理）

视频生成的主推理节点。

- **输入**：combined_config（组合配置）、prompt（提示词）、negative_prompt（负面提示词）、image（图像，可选）、audio（音频，可选）
- **输出**：生成的视频帧

## 使用示例

### 基础 T2V 工作流

1. 创建 LightX2V Inference Config（task_type: "t2v"）
2. 使用 LightX2V Config Combiner
3. 连接到 LightX2V Modular Inference，输入文本提示词
4. 保存视频输出

### 带优化的 I2V

1. 加载输入图像
2. 创建 LightX2V Inference Config（task_type: "i2v"）
3. 添加 LightX2V TeaCache（threshold: 0.26）
4. 添加 LightX2V Memory Optimization
5. 使用 LightX2V Config Combiner 组合配置
6. 运行 LightX2V Modular Inference

### 使用 LoRA

1. 创建基础配置
2. 使用 LightX2V LoRA Loader 加载 LoRA
3. 如需要可串联多个 LoRA
4. 组合所有配置
5. 运行推理

## 模型目录结构

从以下地址下载模型：<https://huggingface.co/lightx2v>

模型应放置在：

```txt
ComfyUI/models/lightx2v/
├── Wan2.1-I2V-14B-720P-xxx/     # 主模型检查点
├── Wan2.1-I2V-14B-480P-xxx/     # 主模型检查点
├── loras/          # LoRA 模型
```

## 使用技巧

- 从默认设置开始，根据硬件情况调整
- 使用 TeaCache，阈值设为 0.1-0.2 可显著提速
- 在显存有限时启用内存优化
- 量化可减少内存使用但可能影响质量
- 串联多个 LoRA 实现复杂风格组合

## 故障排除

- **内存不足**：启用内存优化或使用量化
- **生成速度慢**：启用 TeaCache 或减少推理步数
- **找不到模型**：检查 `ComfyUI/models/lightx2v/` 中的模型路径

## 示例工作流

`examples/` 目录中提供了示例工作流 JSON 文件：

- `wan_i2v.json` - 基础图像生成视频
- `wan_i2v_with_distill_lora.json` - 带蒸馏 LoRA 的 I2V
- `wan_t2v_with_distill_lora.json` - 带蒸馏 LoRA 的 T2V