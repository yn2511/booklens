import torch
import torch.nn as nn
from torchvision import models

class MultiTaskFashionResNet(nn.Module):
    """
    FitFinder Core Architecture:
    Pretrained ResNet-18 Shared Backbone + Dual Classification Heads
    """
    def __init__(self, num_categories=5, num_fits=3, freeze_backbone=False):
        super(MultiTaskFashionResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Head A: 의류 카테고리 (티셔츠, 후드, 셔츠, 청바지, 아우터)
        self.fc_category = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_categories)
        )
        
        # Head B: 실루엣 핏 (오버핏, 레귤러, 슬림핏)
        self.fc_fit = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_fits)
        )

    def forward(self, x):
        features = self.backbone(x)
        out_category = self.fc_category(features)
        out_fit = self.fc_fit(features)
        return out_category, out_fit

if __name__ == "__main__":
    model = MultiTaskFashionResNet()
    dummy = torch.randn(2, 3, 224, 224)
    cat, fit = model(dummy)
    print(f"Model instantiated successfully! Shapes: Category {cat.shape}, Fit {fit.shape}")
