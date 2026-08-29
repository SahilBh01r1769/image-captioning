"""Baseline image-captioning model: global ResNet50 vector + stacked LSTM.

This module is intentionally kept as an ablation baseline. The primary model
lives in ``attention_model.py`` and preserves spatial image features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

import config


class CNNEncoder(nn.Module):
    """ResNet50 encoder that collapses the image to one projected vector."""

    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        fine_tune: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        if pretrained:
            try:
                resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
            except Exception as exc:
                raise RuntimeError(
                    "Could not load pretrained ResNet50 weights. Connect to the internet "
                    "for first-time training or instantiate with pretrained=False for tests."
                ) from exc
        else:
            resnet = tv_models.resnet50(weights=None)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.CNN_FEAT_DIM, embed_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )
        self.set_fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.fine_tune):
            features = self.backbone(images)
        return self.projection(features)

    def set_fine_tune(self, enable: bool) -> None:
        self.fine_tune = enable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = enable
        for parameter in self.projection.parameters():
            parameter.requires_grad = True


class LSTMDecoder(nn.Module):
    """Teacher-forced stacked-LSTM caption decoder for the baseline model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        glove_matrix: Optional[np.ndarray] = None,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if glove_matrix is not None:
            self._init_glove(glove_matrix)
        self.init_h = nn.Linear(embed_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(embed_dim, hidden_dim * num_layers)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def _init_glove(self, glove_matrix: np.ndarray) -> None:
        tensor = torch.from_numpy(glove_matrix).float()
        if tensor.shape[1] == self.embed_dim:
            projected = tensor
        else:
            generator = torch.Generator().manual_seed(config.SEED)
            projection = torch.randn(
                tensor.shape[1], self.embed_dim, generator=generator
            ) / max(1.0, tensor.shape[1] ** 0.5)
            projected = tensor @ projection
        with torch.no_grad():
            self.embedding.weight.copy_(projected)

    def _init_lstm_state(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = image_features.size(0)
        hidden = torch.tanh(self.init_h(image_features))
        cell = torch.tanh(self.init_c(image_features))
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        cell = cell.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return hidden, cell

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embedding(captions[:, :-1]))
        hidden, cell = self._init_lstm_state(image_features)
        output, _ = self.lstm(embeddings, (hidden, cell))
        return self.classifier(self.dropout(output))


class ImageCaptioningModel(nn.Module):
    """Original global-vector architecture retained as a baseline."""

    architecture_name = "baseline"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        glove_matrix: Optional[np.ndarray] = None,
        pad_idx: int = 0,
        fine_tune_cnn: bool = False,
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = CNNEncoder(
            embed_dim=embed_dim,
            fine_tune=fine_tune_cnn,
            pretrained=pretrained_encoder,
        )
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            glove_matrix=glove_matrix,
            pad_idx=pad_idx,
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images), captions)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def set_cnn_fine_tune(self, enable: bool) -> None:
        self.encoder.set_fine_tune(enable)

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_groups(self, lr: float, cnn_lr_factor: float) -> list[dict]:
        cnn_params = list(self.encoder.backbone.parameters())
        cnn_ids = {id(parameter) for parameter in cnn_params}
        other = [
            parameter for parameter in self.parameters()
            if id(parameter) not in cnn_ids and parameter.requires_grad
        ]
        return [
            {"params": other, "lr": lr},
            {"params": [parameter for parameter in cnn_params if parameter.requires_grad], "lr": lr * cnn_lr_factor},
        ]
