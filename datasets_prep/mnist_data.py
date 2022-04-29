# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import numpy as np
from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms

  

def _data_transforms_mnist():
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    return train_transform, valid_transform
