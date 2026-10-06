# -*- coding: utf-8 -*-
"""Improved ResNet50 for thyroid nodule benign/malignant classification.

Modules
-------
Bottleneck        : ResNet50 basic residual block (expansion = 4)
ASPP              : Atrous Spatial Pyramid Pooling, 5 parallel branches
                    (1x1 conv / dilation 6, 12, 18 / global average pooling)
CBAM              : Convolutional Block Attention Module
                    (channel attention + spatial attention)
ResNet50_ASPP_CBAM: backbone + ASPP + CBAM + classifier
MultimodalThyroidModel : image branch + clinical numeric branch, concat fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """ResNet50 bottleneck block: 1x1 down -> 3x3 -> 1x1 up, expansion = 4."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: capture multi-scale context.

    Five parallel branches are concatenated and fused by a 1x1 conv:
    1x1 conv, 3x3 conv with dilation 6 / 12 / 18, and a global average
    pooling branch that is bilinearly upsampled back to the input size.
    """

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        self.fuse_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.shape[2:]

        feat1 = self.relu(self.bn1(self.conv1(x)))
        feat2 = self.relu(self.bn2(self.conv2(x)))
        feat3 = self.relu(self.bn3(self.conv3(x)))
        feat4 = self.relu(self.bn4(self.conv4(x)))
        feat5 = F.interpolate(self.global_avg_pool(x), size=size,
                              mode='bilinear', align_corners=False)

        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.relu(self.fuse_bn(self.fuse_conv(out)))
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention)."""

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        # --- channel attention ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # --- spatial attention ---
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        x = x * self.sigmoid_channel(avg_out + max_out)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid_spatial(self.conv_spatial(spatial_input))


class ResNet50_ASPP_CBAM(nn.Module):
    """ResNet50 backbone with ASPP + CBAM inserted after layer4."""

    def __init__(self, block, layers, num_classes=2):
        super(ResNet50_ASPP_CBAM, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # improvement 1: multi-scale context
        self.aspp = ASPP(in_channels=2048, out_channels=2048)
        # improvement 2: channel + spatial attention
        self.cbam = CBAM(channels=2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B, 2048, 7, 7)
        x = self.aspp(x)            # multi-scale features
        x = self.cbam(x)            # attention re-weighting
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def resnet50_improved(num_classes=2):
    """Factory: improved ResNet50 (ASPP + CBAM)."""
    return ResNet50_ASPP_CBAM(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


class MultimodalThyroidModel(nn.Module):
    """Image branch (ResNet50+ASPP+CBAM) fused with a clinical numeric branch.

    Note: this branch is provided for architecture reference only. The public
    TN5000 annotations contain no usable clinical fields, so the reported
    results in this repo are from the image-only model.
    """

    def __init__(self, num_classes=2, clinical_dim=1):
        super(MultimodalThyroidModel, self).__init__()
        self.image_backbone = resnet50_improved(num_classes=num_classes)
        self.image_feature_dim = 2048

        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        fused_dim = self.image_feature_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, clinical_data):
        backbone = self.image_backbone
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(image))))
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        x = backbone.aspp(x)
        x = backbone.cbam(x)
        img_features = torch.flatten(backbone.avgpool(x), 1)     # (B, 2048)

        if clinical_data.dim() == 1:
            clinical_data = clinical_data.unsqueeze(1)
        clin_features = self.clinical_branch(clinical_data)      # (B, 32)

        fused = torch.cat([img_features, clin_features], dim=1)  # (B, 2080)
        return self.classifier(fused)
