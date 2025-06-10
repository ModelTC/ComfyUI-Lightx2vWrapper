# ComfyUI-Lightx2vWrapper

## 介绍

ComfyUI-Lightx2vWrapper 是一个用于 ComfyUI 的 [Lightx2v](https://github.com/ModelTC/lightx2v) 推理包装器，支持文本生成视频(T2V)和图像生成视频(I2V)功能。该插件基于Wan模型架构，提供高质量的视频生成能力。

## 功能特性

- 🎬 **文本生成视频(T2V)**: 根据文本描述生成视频
- 🖼️ **图像生成视频(I2V)**: 基于输入图像生成动态视频  
- 🚀 **TeaCache加速**: 支持TeaCache功能缓存优化，提升推理速度
- 🔧 **LoRA支持**: 支持LoRA微调模型加载
- 💾 **内存优化**: 支持CPU卸载和显存管理
- ⚡ **多种注意力机制**: 支持flash_attn2/flash_attn3等高效注意力实现

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gaclove/ComfyUI-Lightx2vWrapper.git
cd ComfyUI-Lightx2vWrapper
git submodule update --init --recursive
cd lightx2v
pip install -r lightx2v/requirements.txt # Install dependencies for lightx2v
```

## 模型准备

### 模型目录结构

确保你的模型目录结构如下：

```
your_model_dir/
├── config.json                                    # 模型配置文件
├── models_t5_umt5-xxl-enc-bf16.pth                # T5文本编码器
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP视觉编码器
├── Wan2.1_VAE.pth                                 # VAE模型
├── google/
│   └── umt5-xxl/                                  # T5 tokenizer目录
└── [其他模型文件]
```

### 推荐模型

- **Wan2.1-I2V-14B-480P**: 图像生成视频模型(480p分辨率)
- **Wan2.1-I2V-14B-720P**: 图像生成视频模型(720p分辨率) 
- **Wan2.1-T2V-14B**: 文本生成视频模型

## 使用方法

### 图像生成视频(I2V)工作流

1. **设置模型目录**
   - 使用 `Lightx2vWanVideoModelDir` 节点设置模型路径

2. **加载编码器**
   - `Lightx2vWanVideoT5EncoderLoader`: 加载T5文本编码器
   - `Lightx2vWanVideoClipVisionEncoderLoader`: 加载CLIP视觉编码器
   - `Lightx2vWanVideoVaeLoader`: 加载VAE编码器

3. **文本编码**
   - `Lightx2vWanVideoT5Encoder`: 编码正负提示词

4. **图像编码**
   - `Lightx2vWanVideoImageEncoder`: 编码输入图像

5. **模型加载**
   - `Lightx2vWanVideoModelLoader`: 加载主要的Wan模型

6. **视频生成**
   - `Lightx2vWanVideoSampler`: 执行采样生成

7. **解码输出**
   - `Lightx2vWanVideoVaeDecoder`: 将潜在表示解码为视频帧

### 文本生成视频(T2V)工作流

T2V工作流与I2V类似，但使用 `Lightx2vWanVideoEmptyEmbeds` 替代图像编码器。

## 节点说明

### 核心节点

#### Lightx2vWanVideoModelDir
- **功能**: 设置模型目录路径
- **输入**: 
  - `model_dir`: 模型目录路径

#### Lightx2vWanVideoT5EncoderLoader  
- **功能**: 加载T5文本编码器
- **输入**:
  - `model_name`: T5模型文件名
  - `precision`: 精度(bf16/fp16/fp32)
  - `device`: 设备(cuda/cpu)

#### Lightx2vWanVideoT5Encoder
- **功能**: 编码文本提示词
- **输入**:
  - `t5_encoder`: T5编码器实例
  - `prompt`: 正面提示词
  - `negative_prompt`: 负面提示词

#### Lightx2vWanVideoVaeLoader
- **功能**: 加载VAE模型
- **输入**:
  - `model_name`: VAE模型文件名
  - `precision`: 精度设置
  - `device`: 设备选择
  - `parallel`: 是否并行处理

#### Lightx2vWanVideoImageEncoder
- **功能**: 编码输入图像用于I2V生成
- **输入**:
  - `vae`: VAE模型实例
  - `clip_vision_encoder`: CLIP视觉编码器
  - `image`: 输入图像
  - `width/height`: 目标分辨率
  - `num_frames`: 生成帧数

#### Lightx2vWanVideoModelLoader
- **功能**: 加载主要的Wan生成模型
- **输入**:
  - `model_type`: 模型类型(t2v/i2v)
  - `precision`: 精度设置
  - `attention_type`: 注意力机制类型
  - `cpu_offload`: 是否CPU卸载
  - `lora_path`: LoRA模型路径(可选)
  - `teacache_args`: TeaCache参数(可选)

#### Lightx2vWanVideoSampler
- **功能**: 执行视频生成采样
- **输入**:
  - `model`: Wan模型实例
  - `text_embeddings`: 文本嵌入
  - `image_embeddings`: 图像嵌入
  - `steps`: 采样步数
  - `cfg_scale`: CFG引导强度
  - `seed`: 随机种子

#### Lightx2vWanVideoVaeDecoder
- **功能**: 解码潜在表示为视频帧
- **输入**:
  - `wan_vae`: VAE模型实例
  - `latent`: 潜在表示

### 优化节点

#### Lightx2vTeaCache
- **功能**: TeaCache加速配置
- **输入**:
  - `rel_l1_thresh`: 缓存阈值
  - `start_percent/end_percent`: 缓存使用范围
  - `coefficients`: 预设系数
  - `cache_device`: 缓存设备

#### Lightx2vWanVideoEmptyEmbeds
- **功能**: 为T2V任务提供空图像嵌入
- **输入**:
  - `width/height`: 目标分辨率
  - `num_frames`: 生成帧数

## 参数说明

### 分辨率设置
- **480P**: width=832, height=480
- **720P**: width=1280, height=720
- 确保宽高为8的倍数

### 帧数设置
- 推荐帧数: 81帧 (约5秒 @ 16fps)
- 帧数应为4的倍数+1 (如: 81, 85, 89等)

### 精度选择
- **bf16**: 推荐用于主模型，平衡精度和性能
- **fp16**: 适用于VAE和CLIP，节省显存
- **fp32**: 最高精度，显存占用大

### CFG引导
- **CFG Scale**: 1.0-20.0，值越高越遵循提示词
- **推荐值**: 5.0-8.0

## 示例工作流

参考 `examples/i2v_workflow.json` 获取完整的I2V工作流配置。

## 性能优化建议

1. **启用TeaCache**: 使用 `Lightx2vTeaCache` 节点加速推理
2. **CPU卸载**: 显存不足时启用 `cpu_offload`
3. **合适精度**: 根据显存情况选择bf16或fp16
4. **批处理**: 多个视频生成时考虑批处理

## 故障排除

### 常见问题

1. **模型文件缺失**: 确保所有必需的模型文件都在正确位置
2. **显存不足**: 降低精度或启用CPU卸载
3. **分辨率错误**: 确保宽高为8的倍数
4. **帧数错误**: 使用4的倍数+1的帧数

### 调试技巧

- 检查模型目录结构是否正确
- 验证config.json文件格式
- 监控GPU显存使用情况
- 查看ComfyUI控制台错误信息

## 更新日志

- 支持Wan2.1模型架构
- 集成TeaCache加速功能  
- 添加LoRA微调支持
- 优化内存管理和性能
