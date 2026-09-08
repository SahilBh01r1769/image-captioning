"""Versioned experiment configurations for the controlled CaptionLab runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    architecture: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    embedding_dim: int
    hidden_dim: int
    attention_encoder_dim: int
    attention_dim: int
    dropout: float
    coverage_lambda: float
    label_smoothing: float
    clip_grad_norm: float
    early_stopping_patience: int
    schema_version: int = EXPERIMENT_SCHEMA_VERSION

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExperimentConfig":
        config = ExperimentConfig(**payload)
        config.validate()
        return config

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return ExperimentConfig.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("unsupported experiment configuration version")
        if self.architecture not in {"baseline", "attention"}:
            raise ValueError("architecture must be 'baseline' or 'attention'")
        if not self.run_name or any(char in self.run_name for char in "/\\ "):
            raise ValueError("run_name must be a non-empty path-safe name")
        for name in ("epochs", "batch_size", "embedding_dim", "hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.dropout < 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError("dropout and label_smoothing must be in [0, 1)")
        if self.coverage_lambda < 0:
            raise ValueError("coverage_lambda must be non-negative")
        if self.architecture == "baseline" and self.coverage_lambda != 0:
            raise ValueError("coverage regularization only applies to attention")
        if self.clip_grad_norm <= 0 or self.early_stopping_patience < 1:
            raise ValueError("gradient clipping and early stopping patience must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def controlled_signature(self) -> dict[str, Any]:
        """Return fields that must match across the three primary runs."""
        excluded = {"run_name", "architecture", "coverage_lambda"}
        return {key: value for key, value in self.to_dict().items() if key not in excluded}
