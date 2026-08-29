"""Explainable image captioning with spatial additive attention.

The baseline model in ``model.py`` collapses a ResNet feature map into one
vector before decoding. This module preserves the final 7x7 spatial grid,
attends to a different region at every generated word, and exposes those
weights for visualisation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

import config


class SpatialCNNEncoder(nn.Module):
    """ResNet50 encoder that preserves a projected spatial feature grid."""

    def __init__(
        self,
        encoder_dim: int = config.ATTENTION_ENCODER_DIM,
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

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Sequential(
            nn.Conv2d(config.CNN_FEAT_DIM, encoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoder_dim),
            nn.ReLU(inplace=True),
        )
        self.encoder_dim = encoder_dim
        self.set_fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.fine_tune):
            features = self.backbone(images)
        features = self.projection(features)
        return features.flatten(2).transpose(1, 2).contiguous()

    def set_fine_tune(self, enable: bool) -> None:
        self.fine_tune = enable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = enable
        for parameter in self.projection.parameters():
            parameter.requires_grad = True


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention over image locations."""

    def __init__(self, encoder_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.score(
            self.activation(self.encoder_att(encoder_out) + self.decoder_att(hidden).unsqueeze(1))
        ).squeeze(-1)
        alpha = self.softmax(energy)
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)
        return context, alpha


class AttentionLSTMDecoder(nn.Module):
    """LSTM decoder whose recurrent state selects a visual region each step."""

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = config.ATTENTION_ENCODER_DIM,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        attention_dim: int = config.ATTENTION_DIM,
        dropout: float = config.DROPOUT,
        glove_matrix: Optional[np.ndarray] = None,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if glove_matrix is not None:
            self._init_glove(glove_matrix)

        self.attention = AdditiveAttention(encoder_dim, hidden_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
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

    def init_state(self, encoder_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_features = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(mean_features)), torch.tanh(self.init_c(mean_features))

    def step(
        self,
        token: torch.Tensor,
        encoder_out: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        hidden, cell = state
        context, alpha = self.attention(encoder_out, hidden)
        embedding = self.embedding(token)
        hidden, cell = self.decode_step(torch.cat([embedding, context], dim=-1), (hidden, cell))
        logits = self.classifier(self.dropout(hidden))
        return logits, (hidden, cell), alpha

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_out.size(0)
        steps = captions.size(1) - 1
        locations = encoder_out.size(1)
        logits = encoder_out.new_zeros(batch_size, steps, self.vocab_size)
        alphas = encoder_out.new_zeros(batch_size, steps, locations)
        state = self.init_state(encoder_out)

        for step_idx in range(steps):
            step_logits, state, alpha = self.step(captions[:, step_idx], encoder_out, state)
            logits[:, step_idx] = step_logits
            alphas[:, step_idx] = alpha
        return logits, alphas


class ExplainableCaptioningModel(nn.Module):
    """Spatial ResNet encoder + additive-attention LSTM decoder."""

    architecture_name = "attention"

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = config.ATTENTION_ENCODER_DIM,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        attention_dim: int = config.ATTENTION_DIM,
        dropout: float = config.DROPOUT,
        glove_matrix: Optional[np.ndarray] = None,
        pad_idx: int = 0,
        fine_tune_cnn: bool = False,
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = SpatialCNNEncoder(
            encoder_dim=encoder_dim,
            fine_tune=fine_tune_cnn,
            pretrained=pretrained_encoder,
        )
        self.decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            glove_matrix=glove_matrix,
            pad_idx=pad_idx,
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def attention_coverage_loss(alphas: torch.Tensor, valid_steps: torch.Tensor | None = None) -> torch.Tensor:
    """Doubly-stochastic attention regularization over non-padding time steps."""
    if valid_steps is not None:
        if valid_steps.ndim != 2 or valid_steps.shape[:2] != alphas.shape[:2]:
            raise ValueError("valid_steps must have shape (batch, decoder_steps)")
        alphas = alphas * valid_steps.to(alphas.dtype).unsqueeze(-1)
    return ((1.0 - alphas.sum(dim=1)) ** 2).mean()
