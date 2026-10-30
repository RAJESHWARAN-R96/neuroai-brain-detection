"""
Deep Learning Model Architectures for Brain Tumor MRI Detection and Classification.
Includes:
- EfficientNet-B0 / B3
- ResNet-50 / ResNet-101
- DenseNet-121
- AttentionBrainCNN with CBAM (Channel & Spatial Attention Modules)
- Metaheuristic-Weighted Multi-Model Ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional, Dict, Union


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation / CBAM Channel Attention).
    Applies both Average Pooling and Max Pooling to capture global channel correlations.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM).
    Highlights spatial lesion regions (tumor margins, necrosis, edema).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module combining Channel & Spatial Attention.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class AttentionBrainCNN(nn.Module):
    """
    Custom Deep Attention Convolutional Neural Network tailored for Brain MRI Scans.
    Uses multi-scale receptive field convolutions combined with CBAM attention.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3, feature_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Mish(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Mish(inplace=False),
            nn.MaxPool2d(2, 2)  # 112x112
        )
        self.cbam1 = CBAMBlock(64)
        
        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Mish(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Mish(inplace=False),
            nn.MaxPool2d(2, 2)  # 56x56
        )
        self.cbam2 = CBAMBlock(128)
        
        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Mish(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Mish(inplace=False),
            nn.MaxPool2d(2, 2)  # 28x28
        )
        self.cbam3 = CBAMBlock(256)
        
        # Stage 4 (Deep Semantic Features)
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=False),
            nn.MaxPool2d(2, 2)  # 14x14
        )
        self.cbam4 = CBAMBlock(512)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection head
        self.feature_layer = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Mish(inplace=False)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.Mish(inplace=False),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.cbam1(x)
        x = self.stage2(x)
        x = self.cbam2(x)
        x = self.stage3(x)
        x = self.cbam3(x)
        x = self.stage4(x)
        x = self.cbam4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        feat = self.feature_layer(x)
        return feat

    def get_last_conv_layer(self) -> nn.Module:
        return self.stage4[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B0 / B3 Backbone with custom classification head & embedding extractor.
    """
    def __init__(self, num_classes: int = 4, version: str = 'b0', pretrained: bool = True, feature_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        if version == 'b3':
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b3(weights=weights)
            in_features = 1536
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = 1280
            
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.Mish(inplace=False),
            nn.Dropout(dropout_rate * 0.65),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.feature_proj(x)
        return feat

    def get_last_conv_layer(self) -> nn.Module:
        if hasattr(self.features[-1], '__getitem__') and len(self.features[-1]) > 0:
            return self.features[-1][0]
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


class ResNetBackbone(nn.Module):
    """
    ResNet-50 Backbone with custom projection and classification head.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, feature_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_proj = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.Mish(inplace=False),
            nn.Dropout(dropout_rate * 0.65),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.feature_proj(x)
        return feat

    def get_last_conv_layer(self) -> nn.Module:
        return self.layer4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


class DenseNetBackbone(nn.Module):
    """
    DenseNet-121 Backbone for dense feature reuse across brain MRI slices.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, feature_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)
        
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_proj = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.Mish(inplace=False),
            nn.Dropout(dropout_rate * 0.65),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        feat = self.feature_proj(out)
        return feat

    def get_last_conv_layer(self) -> nn.Module:
        last_block = list(self.features.denseblock4.children())[-1]
        if hasattr(last_block, 'conv2'):
            return last_block.conv2
        return last_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


class MetaheuristicEnsembleModel(nn.Module):
    """
    Metaheuristic-Optimized Weighted Ensemble combining multiple Deep Learning Backbones:
    Probability Output: P(y|x) = sum_i (w_i * Softmax(Model_i(x)))
    where sum_i(w_i) = 1.0 and w_i are optimized via GWO / PSO / WOA / HHO.
    """
    def __init__(self, model_list: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(model_list)
        n = len(model_list)
        if weights is None or len(weights) != n:
            weights = [1.0 / n] * n
        # Normalize weights
        total = sum(weights)
        norm_weights = [w / total for w in weights]
        self.register_buffer('ensemble_weights', torch.tensor(norm_weights, dtype=torch.float32))

    def set_weights(self, weights: List[float]):
        total = sum(weights)
        norm_weights = [w / total for w in weights]
        self.ensemble_weights = torch.tensor(norm_weights, dtype=torch.float32, device=self.ensemble_weights.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.zeros((x.size(0), self.models[0].num_classes), device=x.device)
        for idx, model in enumerate(self.models):
            logits = model(x)
            p = F.softmax(logits, dim=-1)
            probs += self.ensemble_weights[idx] * p
        # Convert probabilities back to log-scale logits for numerical stability in loss functions
        eps = 1e-7
        probs = torch.clamp(probs, min=eps, max=1.0 - eps)
        return torch.log(probs)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def get_last_conv_layer(self) -> nn.Module:
        # Default to first model's last conv layer for Grad-CAM
        return self.models[0].get_last_conv_layer()


def build_model(
    model_name: str = 'attention_cnn',
    num_classes: int = 4,
    pretrained: bool = True,
    feature_dim: int = 512,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Factory function to instantiate models by name.
    Supported: 'attention_cnn', 'efficientnet_b0', 'efficientnet_b3', 'resnet50', 'densenet121', 'ensemble'
    """
    name = model_name.lower().replace('-', '_')
    if name in ('attention_cnn', 'cbam_cnn', 'attention'):
        return AttentionBrainCNN(num_classes=num_classes, feature_dim=feature_dim, dropout_rate=dropout_rate)
    elif name in ('efficientnet_b0', 'efficientnet'):
        return EfficientNetBackbone(num_classes=num_classes, version='b0', pretrained=pretrained, feature_dim=feature_dim, dropout_rate=dropout_rate)
    elif name in ('efficientnet_b3',):
        return EfficientNetBackbone(num_classes=num_classes, version='b3', pretrained=pretrained, feature_dim=feature_dim, dropout_rate=dropout_rate)
    elif name in ('resnet50', 'resnet'):
        return ResNetBackbone(num_classes=num_classes, pretrained=pretrained, feature_dim=feature_dim, dropout_rate=dropout_rate)
    elif name in ('densenet121', 'densenet'):
        return DenseNetBackbone(num_classes=num_classes, pretrained=pretrained, feature_dim=feature_dim, dropout_rate=dropout_rate)
    elif name in ('ensemble', 'metaheuristic_ensemble'):
        m1 = EfficientNetBackbone(num_classes=num_classes, version='b0', pretrained=pretrained, feature_dim=feature_dim)
        m2 = ResNetBackbone(num_classes=num_classes, pretrained=pretrained, feature_dim=feature_dim)
        m3 = DenseNetBackbone(num_classes=num_classes, pretrained=pretrained, feature_dim=feature_dim)
        m4 = AttentionBrainCNN(num_classes=num_classes, feature_dim=feature_dim)
        # Default starting weights
        return MetaheuristicEnsembleModel([m1, m2, m3, m4], weights=[0.30, 0.25, 0.25, 0.20])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
