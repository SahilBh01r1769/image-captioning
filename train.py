"""Reproducible runner for CaptionLab's three controlled experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from attention_model import ExplainableCaptioningModel, attention_coverage_loss
from data_protocol import captions_for_images
from dataset import parse_captions, resolve_dataset_split
from experiment import ExperimentConfig
from feature_cache import build_cached_feature_dataloaders
from model import ImageCaptioningModel
from run_artifacts import (
    atomic_torch_save,
    atomic_write_json,
    git_revision,
    load_trainable_model_state,
    prepare_run_directory,
    stable_hash,
    trainable_model_state,
)
from vocabulary import TOKENIZER_VERSION, Vocabulary


DEFAULT_EXPERIMENT = os.path.join(config.BASE_DIR, "experiments", "attention.json")
DEFAULT_RUNS_DIR = os.path.join(config.BASE_DIR, "runs")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled CaptionLab experiment")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--feature_cache", default=config.FEATURE_CACHE_PATH)
    parser.add_argument("--runs_dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild_vocab", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch with two train/validation batches in a separate run directory",
    )
    parser.add_argument(
        "--overwrite_smoke",
        action="store_true",
        help="Allow replacing only the disposable smoke-run directory",
    )
    return parser.parse_args()


def build_vocabulary(
    image_captions: dict[str, list[str]],
    split_manifest: dict,
    *,
    rebuild: bool = False,
) -> Vocabulary:
    """Build vocabulary from training captions only and bind it to this split."""
    train_keys = split_manifest["splits"]["train"]
    captions = captions_for_images(image_captions, train_keys)
    expected_metadata = {
        "protocol_version": split_manifest["protocol_version"],
        "dataset_fingerprint": split_manifest["dataset_fingerprint"],
        "split_fingerprint": split_manifest["split_fingerprint"],
        "training_images": len(train_keys),
        "minimum_word_frequency": config.MIN_WORD_FREQ,
        "tokenizer_version": TOKENIZER_VERSION,
        "source": "training captions only",
    }
    if os.path.exists(config.VOCAB_PATH) and not rebuild:
        vocab = Vocabulary.load(config.VOCAB_PATH)
        if getattr(vocab, "metadata", {}) != expected_metadata:
            raise RuntimeError(
                "Cached vocabulary does not match the frozen data protocol. "
                "Re-run with --rebuild_vocab after reviewing the split change."
            )
        return vocab

    vocab = Vocabulary()
    vocab.build_from_captions(captions)
    vocab.metadata = expected_metadata
    vocab.save(config.VOCAB_PATH)
    return vocab


def vocabulary_diagnostics(
    vocab: Vocabulary,
    image_captions: dict[str, list[str]],
    split_manifest: dict,
) -> dict[str, float | int]:
    splits = split_manifest["splits"]
    return {
        "vocabulary_size": len(vocab),
        "validation_unknown_token_rate": vocab.unknown_token_rate(
            captions_for_images(image_captions, splits["validation"])
        ),
        "test_unknown_token_rate": vocab.unknown_token_rate(
            captions_for_images(image_captions, splits["test"])
        ),
    }


def build_model(experiment: ExperimentConfig, vocab: Vocabulary):
    common = dict(
        vocab_size=len(vocab),
        embed_dim=experiment.embedding_dim,
        hidden_dim=experiment.hidden_dim,
        dropout=experiment.dropout,
        glove_matrix=None,
        pad_idx=vocab[config.PAD_TOKEN],
        fine_tune_cnn=False,
        pretrained_encoder=False,
    )
    if experiment.architecture == "attention":
        return ExplainableCaptioningModel(
            **common,
            encoder_dim=experiment.attention_encoder_dim,
            attention_dim=experiment.attention_dim,
        )
    return ImageCaptioningModel(**common, num_layers=1)


def run_epoch(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    pad_idx: int,
    device: torch.device,
    experiment: ExperimentConfig,
    split: str,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Return separately observable caption, coverage, and total losses."""
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"caption": 0.0, "total": 0.0, "tokens": 0, "coverage": 0.0, "batches": 0}

    with tqdm(loader, desc=split, leave=False, unit="batch") as bar:
        for batch_index, (features, captions, _) in enumerate(bar):
            if max_batches is not None and batch_index >= max_batches:
                break
            features = features.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            targets = captions[:, 1:].contiguous()
            valid_steps = targets != pad_idx

            with torch.set_grad_enabled(is_train):
                output = model.forward_from_features(features, captions)
                if experiment.architecture == "attention":
                    logits, alphas = output
                    coverage_loss = attention_coverage_loss(alphas, valid_steps)
                else:
                    logits = output
                    coverage_loss = logits.new_tensor(0.0)
                batch, steps, vocab_size = logits.shape
                caption_loss = criterion(
                    logits.reshape(batch * steps, vocab_size),
                    targets.reshape(batch * steps),
                )
                total_loss = caption_loss + experiment.coverage_lambda * coverage_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), experiment.clip_grad_norm)
                optimizer.step()

            non_pad = int(valid_steps.sum().item())
            totals["caption"] += float(caption_loss.item()) * non_pad
            totals["total"] += float(total_loss.item()) * non_pad
            totals["tokens"] += non_pad
            totals["coverage"] += float(coverage_loss.item())
            totals["batches"] += 1
            bar.set_postfix(caption=f"{caption_loss.item():.4f}", total=f"{total_loss.item():.4f}")

    return {
        "caption_loss": totals["caption"] / max(totals["tokens"], 1),
        "coverage_loss": totals["coverage"] / max(totals["batches"], 1),
        "total_loss": totals["total"] / max(totals["tokens"], 1),
    }


def _checkpoint_payload(
    model,
    optimizer: optim.Optimizer,
    experiment: ExperimentConfig,
    epoch: int,
    best_val_caption_loss: float,
    stale_epochs: int,
    history: list[dict],
    identities: dict[str, str],
) -> dict:
    return {
        "checkpoint_version": 1,
        "epoch": epoch,
        "model_state": trainable_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "best_val_caption_loss": best_val_caption_loss,
        "stale_epochs": stale_epochs,
        "history": history,
        "architecture": experiment.architecture,
        "experiment": experiment.to_dict(),
        "experiment_fingerprint": experiment.fingerprint,
        "identities": identities,
        "vocab_size": model.decoder.vocab_size,
        "frozen_backbone_excluded": True,
    }


def _restore_checkpoint(
    path: Path,
    model,
    optimizer: optim.Optimizer,
    experiment: ExperimentConfig,
    identities: dict[str, str],
) -> tuple[int, float, int, list[dict]]:
    checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)
    if checkpoint.get("experiment_fingerprint") != experiment.fingerprint:
        raise RuntimeError("Resume checkpoint uses a different experiment configuration")
    if checkpoint.get("identities") != identities:
        raise RuntimeError("Resume checkpoint uses a different split, vocabulary, or feature cache")
    load_trainable_model_state(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_val_caption_loss"]),
        int(checkpoint["stale_epochs"]),
        list(checkpoint["history"]),
    )


def _provenance(
    experiment: ExperimentConfig,
    identities: dict[str, str],
    diagnostics: dict[str, float | int],
) -> dict:
    device_name = (
        torch.cuda.get_device_name(0)
        if config.DEVICE.type == "cuda"
        else platform.processor() or "CPU"
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_revision(),
        "command": sys.argv,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(config.DEVICE),
        "device_name": device_name,
        "experiment_fingerprint": experiment.fingerprint,
        "identities": identities,
        "data_diagnostics": diagnostics,
        "declared_constraints": {
            "encoder": "frozen torchvision ResNet50 IMAGENET1K_V1",
            "visual_input": "shared cached 7x7 spatial feature grid",
            "glove": False,
            "label_smoothing": experiment.label_smoothing,
        },
    }


def train(args: argparse.Namespace) -> Path:
    experiment = ExperimentConfig.load(args.experiment)
    if experiment.seed != config.SEED:
        raise ValueError("Experiment seed must match the frozen split seed")
    if args.smoke:
        experiment = ExperimentConfig.from_dict(
            {**experiment.to_dict(), "run_name": f"{experiment.run_name}_smoke", "epochs": 1}
        )
    if args.overwrite_smoke and not args.smoke:
        raise ValueError("--overwrite_smoke is accepted only together with --smoke")

    seed_everything(experiment.seed)
    image_captions = parse_captions()
    split_manifest = resolve_dataset_split(image_captions)
    vocab = build_vocabulary(image_captions, split_manifest, rebuild=args.rebuild_vocab)
    diagnostics = vocabulary_diagnostics(vocab, image_captions, split_manifest)
    train_loader, val_loader, _, _ = build_cached_feature_dataloaders(
        vocab,
        args.feature_cache,
        batch_size=experiment.batch_size,
        num_workers=args.workers,
    )
    cache_metadata = train_loader.dataset.cache_metadata
    identities = {
        "split_fingerprint": split_manifest["split_fingerprint"],
        "vocabulary_fingerprint": stable_hash(vocab.word2idx),
        "feature_cache_fingerprint": stable_hash(cache_metadata),
    }

    run_dir = prepare_run_directory(
        args.runs_dir,
        experiment.run_name,
        resume=args.resume,
        overwrite=args.overwrite_smoke,
    )
    atomic_write_json(run_dir / "config.json", experiment.to_dict())
    if not args.resume:
        atomic_write_json(run_dir / "provenance.json", _provenance(experiment, identities, diagnostics))

    model = build_model(experiment, vocab).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab[config.PAD_TOKEN],
        label_smoothing=experiment.label_smoothing,
    )
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
    )
    start_epoch, best_loss, stale_epochs, history = 1, math.inf, 0, []
    if args.resume:
        start_epoch, best_loss, stale_epochs, history = _restore_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            experiment,
            identities,
        )
    atomic_write_json(
        run_dir / "status.json",
        {"state": "running", "last_completed_epoch": start_epoch - 1},
    )

    max_batches = 2 if args.smoke else None
    try:
        for epoch in range(start_epoch, experiment.epochs + 1):
            started = time.time()
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, vocab[config.PAD_TOKEN],
                config.DEVICE, experiment, "train", max_batches=max_batches,
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, None, vocab[config.PAD_TOKEN],
                config.DEVICE, experiment, "validation", max_batches=max_batches,
            )
            improved = val_metrics["caption_loss"] < best_loss
            if improved:
                best_loss = val_metrics["caption_loss"]
                stale_epochs = 0
            else:
                stale_epochs += 1
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "best_val_caption_loss": best_loss,
                "improved": improved,
                "seconds": round(time.time() - started, 2),
            }
            history.append(record)
            print(json.dumps(record))
            payload = _checkpoint_payload(
                model, optimizer, experiment, epoch, best_loss, stale_epochs, history, identities
            )
            atomic_torch_save(run_dir / "checkpoints" / "last.pt", payload)
            if improved:
                atomic_torch_save(run_dir / "checkpoints" / "best.pt", payload)
            atomic_write_json(run_dir / "history.json", history)
            atomic_write_json(
                run_dir / "status.json", {"state": "running", "last_completed_epoch": epoch}
            )
            if stale_epochs >= experiment.early_stopping_patience:
                break
    except BaseException:
        atomic_write_json(
            run_dir / "status.json",
            {"state": "interrupted", "last_completed_epoch": history[-1]["epoch"] if history else 0},
        )
        raise

    atomic_write_json(
        run_dir / "status.json",
        {
            "state": "completed",
            "last_completed_epoch": history[-1]["epoch"] if history else 0,
            "best_val_caption_loss": best_loss,
            "stopped_early": len(history) < experiment.epochs,
        },
    )
    return run_dir


if __name__ == "__main__":
    completed_dir = train(parse_args())
    print(f"Artifacts written to {completed_dir}")
