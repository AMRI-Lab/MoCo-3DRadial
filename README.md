# MoCo-3DRadial

This repository contains the implementations of our manuscript "Unsupervised high-resolution 3D MRI motion correction via physics-informed implicit neural representations".

We propose a rigid-body motion correction methodology for high-resolution 3D radial MRI built upon implicit neural representation.

## Setup

1. Python 3.10.11
2. PyTorch 2.4.1
3. h5py, numpy, nibabel, tqdm, torchkbnufft, sigpy, cupy
4. tiny-cuda-nn

## Files Description

SUMMIT/

    ├── ReadMe.md           // 帮助文档
    
    ├── AutoCreateDDS.py    // 合成DDS的 python脚本文件
    
    ├── DDScore             // DDS核心文件库，包含各版本的include、src、lib文件夹，方便合并
