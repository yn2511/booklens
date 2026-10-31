"""BookLens AI Model Architecture Definitions.
Curriculum Mapping: Week 8 (CNN, Scratch vs Feature Extractor vs Fine-Tuning).
"""
import torch
import torch.nn as nn
from torchvision import models

CLASSES = ["picture_book", "text_book", "comic_book"]
CLASS_KR = {"picture_book": "그림책", "text_book": "줄글책", "comic_book": "만화책"}

def build_small_cnn(num_classes=3):
    """Week 8-3 Baseline: Lightweight 4-block CNN built from scratch."""
    def block(ci, co):
        return [
            nn.Conv2d(ci, co, kernel_size=3, padding=1),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
    return nn.Sequential(
        *block(3, 32),
        *block(32, 64),
        *block(64, 128),
        *block(128, 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )

def build_resnet18_transfer(num_classes=3, freeze_backbone=False, weights_path=None):
    """Week 8-3 ResNet-18 Transfer Learning Model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_path is None else None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    return model
