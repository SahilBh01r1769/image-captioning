# model.py
"""
Image Captioning Model — CNN Encoder + LSTM Decoder

Architecture:
  Encoder : ResNet50 (pretrained, final FC removed)
              → GlobalAvgPool (2048,)
              → Linear projection (embed_dim,)

  Decoder : Embedding(vocab_size, embed_dim)  ← initialised with GloVe
              → LSTM(hidden_dim, num_layers)
              → Linear(vocab_size)             ← next-word prediction

Training uses Teacher Forcing.
Inference supports Greedy and Beam Search decoding.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import numpy as np
from typing import Optional

import config


# ══════════════════════════════════════════════════════════════════════════════
#  CNN Encoder
# ══════════════════════════════════════════════════════════════════════════════

class CNNEncoder(nn.Module):
    """
    ResNet50 encoder that outputs a single feature vector per image.

    Parameters
    ----------
    embed_dim   : dimension of the projected feature vector
    fine_tune   : if True, CNN backbone weights are updated during training
    """

    def __init__(self, embed_dim: int = config.EMBED_DIM,
                 fine_tune: bool = False):
        super().__init__()

        # Load pretrained ResNet50 and drop the final FC layer
        try:
            resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            resnet = tv_models.resnet50(weights=None)
            print("⚠  Could not download ImageNet weights — ResNet50 randomly initialised.")
        modules = list(resnet.children())[:-1]          # remove avgpool + fc
        # Actually ResNet50: remove only the final FC (index -1), keep avgpool
        modules = list(resnet.children())[:-1]          # keeps avgpool → (B,2048,1,1)
        self.backbone = nn.Sequential(*modules)

        # Project 2048 → embed_dim
        self.projection = nn.Sequential(
            nn.Flatten(),                                # (B,2048)
            nn.Linear(config.CNN_FEAT_DIM, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT),
        )

        self.set_fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : (B, 3, H, W)

        Returns
        -------
        features : (B, embed_dim)
        """
        with torch.set_grad_enabled(self.fine_tune):
            features = self.backbone(images)             # (B, 2048, 1, 1)
        return self.projection(features)                 # (B, embed_dim)

    def set_fine_tune(self, enable: bool) -> None:
        """Freeze or unfreeze backbone parameters."""
        self.fine_tune = enable
        for param in self.backbone.parameters():
            param.requires_grad = enable
        # Projection layer always trains
        for param in self.projection.parameters():
            param.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM Decoder
# ══════════════════════════════════════════════════════════════════════════════

class LSTMDecoder(nn.Module):
    """
    LSTM-based caption decoder with teacher forcing.

    The image feature vector initialises both h₀ and c₀ of the LSTM via
    learned linear projections. At each step the LSTM receives the embedding
    of the current (or predicted) word.

    Parameters
    ----------
    vocab_size    : size of vocabulary
    embed_dim     : word embedding dimension (matches encoder output)
    hidden_dim    : LSTM hidden state size
    num_layers    : number of stacked LSTM layers
    dropout       : dropout probability
    glove_matrix  : optional numpy array (vocab_size, embed_dim) for init
    """

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int = config.EMBED_DIM,
        hidden_dim:   int = config.HIDDEN_DIM,
        num_layers:   int = config.NUM_LAYERS,
        dropout:      float = config.DROPOUT,
        glove_matrix: Optional[np.ndarray] = None,
        pad_idx:      int = 0,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # ── Word embeddings ──────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                       padding_idx=pad_idx)
        if glove_matrix is not None:
            self._init_glove(glove_matrix, embed_dim)

        # ── Image feature → initial LSTM state ──────────────────────────
        # embed_dim → hidden_dim * num_layers  (one per layer)
        self.init_h = nn.Linear(embed_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(embed_dim, hidden_dim * num_layers)

        # ── LSTM ─────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    # ── GloVe init ────────────────────────────────────────────────────────────
    def _init_glove(self, glove_matrix: np.ndarray, embed_dim: int) -> None:
        """
        Initialise embedding weights from GloVe.
        If GloVe dim ≠ embed_dim, project down/up via a linear layer.
        """
        glove_dim = glove_matrix.shape[1]
        glove_tensor = torch.from_numpy(glove_matrix)

        if glove_dim == embed_dim:
            with torch.no_grad():
                self.embedding.weight.copy_(glove_tensor)
            print(f"  Embedding initialised from GloVe ({glove_dim}d).")
        else:
            # Project GloVe → embed_dim at init time
            proj = nn.Linear(glove_dim, embed_dim, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            projected = proj(glove_tensor).detach()
            with torch.no_grad():
                self.embedding.weight.copy_(projected)
            print(f"  GloVe {glove_dim}d projected → {embed_dim}d for embedding init.")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _init_lstm_state(self, img_features: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Derive (h₀, c₀) from image features.

        img_features : (B, embed_dim)
        Returns      : h (num_layers, B, hidden_dim),
                       c (num_layers, B, hidden_dim)
        """
        B = img_features.size(0)
        h = torch.tanh(self.init_h(img_features))  # (B, hidden_dim * num_layers)
        c = torch.tanh(self.init_c(img_features))

        h = h.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h, c

    # ── forward (teacher forcing) ─────────────────────────────────────────────
    def forward(self, img_features: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Parameters
        ----------
        img_features : (B, embed_dim)  — from CNNEncoder
        captions     : (B, max_length) — integer token ids incl. <START>

        Returns
        -------
        logits : (B, max_length-1, vocab_size)
                 predictions for positions 1 … max_length
        """
        # Drop last token: we predict captions[:,1:] from captions[:,:-1]
        embeddings = self.embedding(captions[:, :-1])     # (B, T-1, embed_dim)
        embeddings = self.dropout(embeddings)

        h, c = self._init_lstm_state(img_features)
        lstm_out, _ = self.lstm(embeddings, (h, c))       # (B, T-1, hidden_dim)
        logits = self.classifier(self.dropout(lstm_out))  # (B, T-1, vocab_size)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
#  Full Model wrapper
# ══════════════════════════════════════════════════════════════════════════════

class ImageCaptioningModel(nn.Module):
    """Combines CNNEncoder + LSTMDecoder into a single module."""

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int   = config.EMBED_DIM,
        hidden_dim:   int   = config.HIDDEN_DIM,
        num_layers:   int   = config.NUM_LAYERS,
        dropout:      float = config.DROPOUT,
        glove_matrix: Optional[np.ndarray] = None,
        pad_idx:      int   = 0,
        fine_tune_cnn: bool = False,
    ):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim, fine_tune=fine_tune_cnn)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            glove_matrix=glove_matrix,
            pad_idx=pad_idx,
        )

    def forward(self, images: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images   : (B, 3, H, W)
        captions : (B, max_length)

        Returns
        -------
        logits : (B, max_length-1, vocab_size)
        """
        img_features = self.encoder(images)
        logits       = self.decoder(img_features, captions)
        return logits

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return image features only (used during inference)."""
        return self.encoder(images)

    def set_cnn_fine_tune(self, enable: bool) -> None:
        self.encoder.set_fine_tune(enable)

    def trainable_parameters(self):
        """Return only parameters that require grad (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_groups(self, lr: float = config.LEARNING_RATE,
                          cnn_lr_factor: float = config.CNN_LR_FACTOR
                          ) -> list[dict]:
        """
        Return param groups: CNN backbone gets a reduced LR.
        Use this with an optimizer when CNN fine-tuning is active.
        """
        cnn_params     = list(self.encoder.backbone.parameters())
        cnn_param_ids  = {id(p) for p in cnn_params}
        other_params   = [p for p in self.parameters()
                          if id(p) not in cnn_param_ids and p.requires_grad]
        return [
            {"params": other_params,                       "lr": lr},
            {"params": [p for p in cnn_params
                        if p.requires_grad],               "lr": lr * cnn_lr_factor},
        ]
