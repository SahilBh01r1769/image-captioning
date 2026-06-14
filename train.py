# train.py
"""
Training pipeline for Image Captioning (CNN Encoder + LSTM Decoder).

Features
--------
- Teacher forcing
- GloVe embedding initialisation
- Progressive CNN fine-tuning (frozen for first N epochs, then unfrozen)
- Gradient clipping
- Checkpoint saving + resuming
- Training/validation loss curves saved to outputs/
"""

import os
import argparse
import pickle
import time
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from model import ImageCaptioningModel
from utils.vocabulary import Vocabulary
from utils.dataset import parse_captions, split_dataset, Flickr8kDataset, build_dataloaders
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train Image Captioning Model")
    p.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint .pth to resume training")
    p.add_argument("--no_glove",    action="store_true",
                   help="Skip GloVe initialisation even if file exists")
    p.add_argument("--workers",     type=int,   default=4)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path: str, model: ImageCaptioningModel,
                    optimizer: optim.Optimizer
                    ) -> tuple[int, float]:
    """Load checkpoint. Returns (start_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Resumed from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")
    return ckpt["epoch"] + 1, ckpt["val_loss"]


def plot_losses(train_losses: list[float], val_losses: list[float]) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Train Loss")
    plt.plot(epochs, val_losses,   "r-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_DIR, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Loss curve saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  One epoch
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model: ImageCaptioningModel,
              loader: DataLoader,
              criterion: nn.Module,
              optimizer: optim.Optimizer | None,
              pad_idx: int,
              device: torch.device,
              split: str = "train") -> float:
    """
    Run one epoch.  If optimizer is None → evaluation mode.
    Returns average loss over all non-padding tokens.
    """
    is_train = (optimizer is not None)
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0

    with tqdm(loader, desc=f"  {split:5s}", leave=False, unit="batch") as bar:
        for images, captions, lengths in bar:
            images   = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            with torch.set_grad_enabled(is_train):
                logits = model(images, captions)          # (B, T-1, V)
                # Target: captions[:, 1:]  (shift by one)
                targets = captions[:, 1:].contiguous()    # (B, T-1)

                B, T, V = logits.shape
                loss = criterion(
                    logits.view(B * T, V),
                    targets.view(B * T)
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                          config.CLIP_GRAD_NORM)
                optimizer.step()

            # Count non-padding tokens for accurate mean
            non_pad = (targets != pad_idx).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad
            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    print(f"\n{'═'*55}")
    print(f"  Image Captioning — Training")
    print(f"  Device : {config.DEVICE}")
    print(f"  Epochs : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}")
    print(f"{'═'*55}\n")

    # ── 1. Vocabulary ──────────────────────────────────────────────────────
    if os.path.exists(config.VOCAB_PATH):
        vocab = Vocabulary.load()
    else:
        image_captions = parse_captions()
        all_captions   = [c for caps in image_captions.values() for c in caps]
        vocab = Vocabulary()
        vocab.build_from_captions(all_captions)
        vocab.save()

    pad_idx = vocab[config.PAD_TOKEN]

    # ── 2. GloVe matrix ────────────────────────────────────────────────────
    glove_matrix = None
    if config.USE_GLOVE and not args.no_glove:
        if os.path.exists(config.GLOVE_FILE):
            glove_matrix = vocab.build_glove_matrix()
        else:
            print(f"⚠  GloVe file not found at {config.GLOVE_FILE}. "
                  "Proceeding without GloVe (random init).")

    # ── 3. DataLoaders ─────────────────────────────────────────────────────
    train_loader, val_loader, _, _ = build_dataloaders(
        vocab, batch_size=args.batch_size, num_workers=args.workers
    )

    # ── 4. Model ───────────────────────────────────────────────────────────
    model = ImageCaptioningModel(
        vocab_size   = len(vocab),
        embed_dim    = config.EMBED_DIM,
        hidden_dim   = config.HIDDEN_DIM,
        num_layers   = config.NUM_LAYERS,
        dropout      = config.DROPOUT,
        glove_matrix = glove_matrix,
        pad_idx      = pad_idx,
        fine_tune_cnn = False,               # start frozen
    ).to(config.DEVICE)

    print(f"\nModel parameters:")
    total = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total      : {total:,}")
    print(f"  Trainable  : {train_params:,}  (CNN frozen)")

    # ── 5. Loss & Optimiser ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # ── 6. Resume? ─────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = math.inf
    train_losses, val_losses = [], []

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)

    # ── 7. Epoch loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Progressive CNN fine-tuning
        if epoch == config.FREEZE_CNN_EPOCHS + 1:
            print(f"\n→ Epoch {epoch}: Unfreezing CNN backbone (lr × {config.CNN_LR_FACTOR})")
            model.set_cnn_fine_tune(True)
            optimizer = optim.Adam(
                model.parameter_groups(args.lr, config.CNN_LR_FACTOR),
                weight_decay=config.WEIGHT_DECAY,
            )

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = run_epoch(model, train_loader, criterion,
                                optimizer, pad_idx, config.DEVICE, "Train")
        val_loss   = run_epoch(model, val_loader,   criterion,
                                None,      pad_idx, config.DEVICE, "Val")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"  Train Loss : {train_loss:.4f} | "
              f"Val Loss : {val_loss:.4f} | "
              f"Time : {elapsed:.1f}s")

        # Save periodic checkpoint
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = os.path.join(config.MODELS_DIR,
                                     f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "vocab_size":      len(vocab),
            }, ckpt_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "vocab_size":      len(vocab),
            }, config.BEST_MODEL_PATH)
            print(f"  ★ New best model  (val_loss={best_val_loss:.4f})")

    # ── 8. Save plots & summary ────────────────────────────────────────────
    plot_losses(train_losses, val_losses)

    summary = {
        "epochs":          args.epochs,
        "best_val_loss":   best_val_loss,
        "final_train_loss": train_losses[-1],
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }
    summary_path = os.path.join(config.OUTPUTS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary → {summary_path}")
    print(f"Best val loss   : {best_val_loss:.4f}")
    print("Done! ✓")


if __name__ == "__main__":
    args = parse_args()
    train(args)
