#!/usr/bin/env python3

"""
RESNET-50 MODEL
===============

The following program is the model definition for __ Classification
"""

# %%
# Importing Libraries
import os
import pickle
from PIL import Image
import numpy as np
import torch
import torchvision as tv
from torchvision.models import resnet50, ResNet50_Weights

# %%
# Metadata
dummy_input_shape = (1,3,384,384)

# %%
# Training Transforms
random_transforms = tv.transforms.RandomApply(
    torch.nn.ModuleList([
        tv.transforms.ColorJitter(brightness=0.5, hue=0.3),
        tv.transforms.GaussianBlur(kernel_size=(3, 3)),
        tv.transforms.RandomInvert(),
        tv.transforms.RandomAdjustSharpness(sharpness_factor=2),
        tv.transforms.RandomAutocontrast(),
        tv.transforms.RandomEqualize(),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.RandomVerticalFlip(p=0.5)
    ]), p=0.5
)

training_transforms = tv.transforms.Compose([
    random_transforms,
    tv.transforms.Resize((dummy_input_shape[2], dummy_input_shape[3])),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Inference Transforms
inference_transforms = tv.transforms.Compose([
    tv.transforms.Resize((dummy_input_shape[2], dummy_input_shape[3])),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Main Model Definition
class Model(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        """
        This method is used to initialize Model

        Method Input
        =============
        num_classes : Number of classes to make classification model for

        Method Output
        ==============
        None
        """
        super(Model, self).__init__()
        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2, progress=False)
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.Tanh()
        )

        self.classifier = torch.nn.Linear(512, num_classes)
        self.resnet.requires_grad_(True)
    
    def forward(self, x) -> tuple:
        """
        This method is used to perform forward propagation on input data

        Method Input
        =============
        x : Input data as image batch ( Batch x Channel x Height x Width)

        Method Output
        ==============
        Output results after forward propagation
        """
        embed = self.resnet(x)
        res = self.classifier(embed)
        return embed, res
