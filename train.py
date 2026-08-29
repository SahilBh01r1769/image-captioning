"""Training entrypoint for baseline and explainable attention captioners."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from attention_model import ExplainableCaptioningModel, attention_coverage_loss
from dataset import build_dataloaders, parse_captions
from model import ImageCaptioningModel
from vocabulary import Vocabulary


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image-captioning model")
    parser.add_argument("--architecture", choices=["attention", "baseline"], default=config.DEFAULT_ARCHITECTURE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_glove", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def build_vocabulary() -> Vocabulary:
    if os.path.exists(config.VOCAB_PATH):
        return Vocabulary.load()
    image_captions = parse_captions()
    captions = [caption for group in image_captions.values() for caption in group]
    vocab = Vocabulary()
    vocab.build_from_captions(captions)
    vocab.save()
    return vocab


def build_model(
    architecture: str,
    vocab: Vocabulary,
    glove_matrix: np.ndarray | None,
    *,
    dropout: float = config.DROPOUT,
    pretrained_encoder: bool = True,
):
    common = dict(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=dropout,
        glove_matrix=glove_matrix,
        pad_idx=vocab[config.PAD_TOKEN],
        fine_tune_cnn=False,
        pretrained_encoder=pretrained_encoder,
    )
    if architecture == "attention":
        return ExplainableCaptioningModel(
            **common,
            encoder_dim=config.ATTENTION_ENCODER_DIM,
            attention_dim=config.ATTENTION_DIM,
        )
    return ImageCaptioningModel(**common, num_layers=config.NUM_LAYERS)


def build_optimizer(model, lr: float, cnn_fine_tuned: bool) -> optim.Optimizer:
    if cnn_fine_tuned:
        return optim.Adam(
            model.parameter_groups(lr, config.CNN_LR_FACTOR),
            weight_decay=config.WEIGHT_DECAY,
        )
    return optim.Adam(
        model.trainable_parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )


def checkpoint_payload(model, optimizer, epoch: int, val_loss: float, architecture: str, vocab: Vocabulary) -> dict:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "vocab_size": len(vocab),
        "architecture": architecture,
        "cnn_fine_tuned": bool(epoch > config.FREEZE_CNN_EPOCHS),
        "model_config": {
            "embed_dim": config.EMBED_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "attention_encoder_dim": config.ATTENTION_ENCODER_DIM,
            "attention_dim": config.ATTENTION_DIM,
        },
    }


def inspect_checkpoint(path: str, expected_architecture: str) -> dict:
    checkpoint = torch.load(path, map_location=config.DEVICE)
    architecture = checkpoint.get("architecture", "baseline")
    if architecture != expected_architecture:
        raise ValueError(
            f"Checkpoint architecture is {architecture!r}, but --architecture is "
            f"{expected_architecture!r}. Use the matching architecture."
        )
    return checkpoint


def restore_checkpoint(checkpoint: dict, model, optimizer: optim.Optimizer) -> tuple[int, float]:
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["val_loss"])


def run_epoch(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    pad_idx: int,
    device: torch.device,
    architecture: str,
    split: str,
) -> tuple[float, float]:
    """Return token-normalized total loss and attention regularization."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    total_attention = 0.0
    batches = 0

    with tqdm(loader, desc=split, leave=False, unit="batch") as bar:
        for images, captions, _ in bar:
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            targets = captions[:, 1:].contiguous()
            valid_steps = targets != pad_idx

            with torch.set_grad_enabled(is_train):
                output = model(images, captions)
                if architecture == "attention":
                    logits, alphas = output
                    attn_loss = attention_coverage_loss(alphas, valid_steps)
                else:
                    logits = output
                    attn_loss = logits.new_tensor(0.0)

                batch, steps, vocab_size = logits.shape
                token_loss = criterion(
                    logits.reshape(batch * steps, vocab_size),
                    targets.reshape(batch * steps),
                )
                loss = token_loss + config.ATTENTION_REGULARIZATION * attn_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                optimizer.step()

            non_pad = int(valid_steps.sum().item())
            total_loss += float(loss.item()) * non_pad
            total_tokens += non_pad
            total_attention += float(attn_loss.item())
            batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tokens, 1), total_attention / max(batches, 1)


def train(args: argparse.Namespace) -> None:
    seed_everything()
    vocab = build_vocabulary()
    pad_idx = vocab[config.PAD_TOKEN]

    glove_matrix = None
    if config.USE_GLOVE and not args.no_glove and os.path.exists(config.GLOVE_FILE):
        glove_matrix = vocab.build_glove_matrix()

    train_loader, val_loader, _, _ = build_dataloaders(
        vocab,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = build_model(args.architecture, vocab, glove_matrix).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    start_epoch = 1
    best_val_loss = math.inf
    resumed_checkpoint = None
    cnn_fine_tuned = False

    if args.resume:
        resumed_checkpoint = inspect_checkpoint(args.resume, args.architecture)
        checkpoint_epoch = int(resumed_checkpoint["epoch"])
        # New checkpoints store the stage explicitly; old checkpoints infer it
        # from the epoch so resumes remain backwards compatible.
        cnn_fine_tuned = bool(
            resumed_checkpoint.get(
                "cnn_fine_tuned",
                checkpoint_epoch > config.FREEZE_CNN_EPOCHS,
            )
        )
        model.set_cnn_fine_tune(cnn_fine_tuned)

    optimizer = build_optimizer(model, args.lr, cnn_fine_tuned)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    if resumed_checkpoint is not None:
        start_epoch, best_val_loss = restore_checkpoint(
            resumed_checkpoint,
            model,
            optimizer,
        )

    history: list[dict] = []
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()

        if not cnn_fine_tuned and epoch == config.FREEZE_CNN_EPOCHS + 1:
            cnn_fine_tuned = True
            model.set_cnn_fine_tune(True)
            optimizer = build_optimizer(model, args.lr, cnn_fine_tuned=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
            )

        train_loss, train_attn = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            pad_idx,
            config.DEVICE,
            args.architecture,
            "train",
        )
        val_loss, val_attn = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            pad_idx,
            config.DEVICE,
            args.architecture,
            "val",
        )
        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_attention_regularizer": train_attn,
            "val_attention_regularizer": val_attn,
            "cnn_fine_tuned": cnn_fine_tuned,
            "seconds": round(time.time() - started, 2),
        }
        history.append(record)
        print(json.dumps(record))

        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            val_loss,
            args.architecture,
            vocab,
        )
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                payload,
                os.path.join(config.MODELS_DIR, f"{args.architecture}_epoch_{epoch}.pth"),
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(payload, config.BEST_MODEL_PATH)

    summary = {
        "architecture": args.architecture,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(
        os.path.join(config.OUTPUTS_DIR, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    train(parse_args())
