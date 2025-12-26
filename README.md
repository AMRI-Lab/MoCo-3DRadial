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
├── run_demo.py # Entry script for running the motion-compensated reconstruction demo
├── model.so # Compiled motion-compensated implicit neural modeling module
├── utils.so # Compiled utility functions for NUFFT, motion handling, and data processing
└── README.md # Documentation and usage instructions

Data/
└── rawdata.h5 # Motion-corrupted k-space data used in the demo
