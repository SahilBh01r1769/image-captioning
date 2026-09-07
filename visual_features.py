"""Shared frozen ResNet feature extraction for controlled model comparisons."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config


def global_average_pool(spatial_features: torch.Tensor) -> torch.Tensor:
    """Pool ``(batch, locations, channels)`` features into one image vector."""
    if spatial_features.ndim != 3:
        raise ValueError("spatial_features must have shape (batch, locations, channels)")
    return spatial_features.mean(dim=1)


class SpatialResNetBackbone(nn.Module):
    """Return the final ResNet50 convolutional grid without model-specific projection.

    Keeping this representation shared lets the baseline globally pool the exact
    same visual tensor over which the attention model learns spatial weights.
    """

    output_dim = config.CNN_FEAT_DIM

    def __init__(self, fine_tune: bool = False, pretrained: bool = True):
        super().__init__()
        if pretrained:
            try:
                resnet = tv_models.resnet50(
                    weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not load pretrained ResNet50 weights. Connect to the internet "
                    "for first-time extraction or use pretrained=False for tests."
                ) from exc
        else:
            resnet = tv_models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.set_fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # A frozen BatchNorm backbone must also stay in evaluation mode; merely
        # disabling gradients would still update its running statistics.
        if not self.fine_tune:
            self.backbone.eval()
        with torch.set_grad_enabled(self.fine_tune):
            features = self.backbone(images)
        return features.flatten(2).transpose(1, 2).contiguous()

    def set_fine_tune(self, enable: bool) -> None:
        self.fine_tune = enable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = enable
        if not enable:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.fine_tune:
            self.backbone.eval()
        return self
