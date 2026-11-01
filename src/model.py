"""BookLens AI Model Architecture Definitions.
Curriculum Mapping: Week 8 (CNN, Scratch vs Feature Extractor vs Fine-Tuning)
& Advanced Regularization (Dropout, Label Smoothing, Differential Learning Rate).
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

def build_resnet18_transfer(num_classes=3, freeze_backbone=False, dropout_p=0.0, weights_path=None):
    """Week 8-3 ResNet-18 Transfer Learning Model with optional Classifier Dropout."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_path is None else None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    if dropout_p > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    return model

def build_differential_optimizer(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-2):
    """Discriminative Fine-Tuning: Separate learning rates for backbone and classifier head."""
    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    head_params = model.fc.parameters()
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay}
    ])
