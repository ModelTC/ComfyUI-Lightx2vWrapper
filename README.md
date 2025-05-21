# ComfyUI-Lightx2vWrapper

## 介绍

ComfyUI-Lightx2vWrapper 是一个用于 ComfyUI 的 [Lightx2v](https://github.com/ModelTC/lightx2v) 推理包装器。

## 安装

cd ComfyUI/custom_nodes
# Option 1: If lightx2v is a submodule
# git clone --recursive https://github.com/gaopeng123456/ComfyUI-Lightx2vWrapper.git
# cd ComfyUI-Lightx2vWrapper
# pip install -r lightx2v/requirements.txt # Install dependencies for lightx2v
# pip install -r requirements.txt # If wrapper has its own (please specify)

# Option 2: If lightx2v needs to be cloned manually
git clone https://github.com/gaopeng123456/ComfyUI-Lightx2vWrapper.git
cd ComfyUI-Lightx2vWrapper
git clone https://github.com/ModelTC/lightx2v.git lightx2v
pip install -r lightx2v/requirements.txt # Install dependencies for lightx2v
# pip install -r requirements.txt # If wrapper has its own (please specify)

## 使用

在 ComfyUI 的 `custom_nodes` 目录下，找到 `lightx2v_wrapper` 文件夹，然后按照 `lightx2v` 的说明进行使用。
