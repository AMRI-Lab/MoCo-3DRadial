# MoCo-3DRadial

This repository contains the implementations of our manuscript "Unsupervised high-resolution 3D MRI motion correction via physics-informed implicit neural representations".

We propose a rigid-body motion correction methodology for high-resolution 3D radial MRI built upon implicit neural representation.

## Setup

1. Python 3.10.11
2. PyTorch 2.4.1
3. h5py, numpy, nibabel, tqdm, torchkbnufft, sigpy, cupy
4. tiny-cuda-nn

## Files Description
    MoCo/
    ├── config.yaml              # Network and training parameters
    ├── kspace_correct.py        # K-space correction based on estimated motion parameters 
    ├── run_demo.py              # Entry script for running the demo
    ├── train.pyc                # Model and training process
    ├── utils.pyc                # Utility functions
    └── data/
        ├── gt_mot               # The ground truth of motion parameters
        ├── recon                # The reconstruction result
        ├── kdata.h5             # Simulated stack-of-stars k-space data
        └── rotAngle.mat         # The rotation angle for trajectory calculation
## Usage

You can run "run_demo.py" to test the performance of our method.
Data for running the demo are available at [Google Drive](https://drive.google.com/drive/folders/1lmnFnkr0NMPpT82Xo85xwhp0uXUtQPXp?usp=sharing)
