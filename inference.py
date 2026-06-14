# inference.py
"""
Caption generation for Image Captioning model.

Supports
--------
  Greedy decoding  — fast, picks argmax at each step
  Beam Search      — higher quality, keeps top-k hypotheses
"""

import os
import argparse
import glob
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import ImageCaptioningModel
from utils.vocabulary import Vocabulary


# ══════════════════════════════════════════════════════════════════════════════
#  Image preprocessing
# ══════════════════════════════════════════════════════════════════════════════

_INFER_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(config.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
])


def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load, preprocess and return image tensor with batch dim."""
    img = Image.open(path).convert("RGB")
    return _INFER_TRANSFORM(img).unsqueeze(0).to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str,
               vocab: Vocabulary,
               device: torch.device) -> ImageCaptioningModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ImageCaptioningModel(
        vocab_size  = len(vocab),
        embed_dim   = config.EMBED_DIM,
        hidden_dim  = config.HIDDEN_DIM,
        num_layers  = config.NUM_LAYERS,
        dropout     = 0.0,          # disabled at inference
        pad_idx     = vocab[config.PAD_TOKEN],
        fine_tune_cnn = False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Greedy decoding
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(model: ImageCaptioningModel,
                  img_tensor: torch.Tensor,
                  vocab: Vocabulary,
                  max_len: int = config.MAX_GEN_LEN) -> str:
    """
    Generate a caption using greedy (argmax) decoding.

    At each step we pick the highest-probability next token.
    """
    device    = img_tensor.device
    start_idx = vocab[config.START_TOKEN]
    end_idx   = vocab[config.END_TOKEN]
    pad_idx   = vocab[config.PAD_TOKEN]

    img_feat = model.encode(img_tensor)     # (1, embed_dim)
    h, c     = model.decoder._init_lstm_state(img_feat)

    token    = torch.tensor([[start_idx]], device=device)   # (1, 1)
    tokens   = []

    for _ in range(max_len):
        embed  = model.decoder.embedding(token)              # (1, 1, embed_dim)
        out, (h, c) = model.decoder.lstm(embed, (h, c))     # (1, 1, hidden)
        logits = model.decoder.classifier(out.squeeze(1))   # (1, vocab_size)
        pred   = logits.argmax(dim=-1)                       # (1,)

        idx = pred.item()
        if idx == end_idx:
            break
        if idx != pad_idx:
            tokens.append(idx)
        token = pred.unsqueeze(1)

    return vocab.decode(tokens)


# ══════════════════════════════════════════════════════════════════════════════
#  Beam Search decoding
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def beam_search_decode(model: ImageCaptioningModel,
                        img_tensor: torch.Tensor,
                        vocab: Vocabulary,
                        beam_size: int = config.BEAM_SIZE,
                        max_len:   int = config.MAX_GEN_LEN) -> list[tuple[str, float]]:
    """
    Beam search decoding.

    Returns
    -------
    List of (caption_string, score) sorted by score descending.
    Score = log-probability (higher is better, all values ≤ 0).
    """
    device    = img_tensor.device
    start_idx = vocab[config.START_TOKEN]
    end_idx   = vocab[config.END_TOKEN]
    pad_idx   = vocab[config.PAD_TOKEN]

    img_feat = model.encode(img_tensor)           # (1, embed_dim)
    h, c     = model.decoder._init_lstm_state(img_feat)

    # Each beam: (log_prob, token_ids, h, c)
    beams = [(0.0, [start_idx], h, c)]
    completed = []

    for step in range(max_len):
        new_beams = []
        for log_prob, tokens, h_t, c_t in beams:
            last_token = torch.tensor([[tokens[-1]]], device=device)
            embed      = model.decoder.embedding(last_token)     # (1,1,E)
            out, (h_new, c_new) = model.decoder.lstm(embed, (h_t, c_t))
            log_probs  = F.log_softmax(
                model.decoder.classifier(out.squeeze(1)), dim=-1
            ).squeeze(0)   # (vocab_size,)

            # Top-k candidates
            topk_lp, topk_idx = log_probs.topk(beam_size)
            for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                new_tokens   = tokens + [idx]
                new_log_prob = log_prob + lp

                if idx == end_idx or step == max_len - 1:
                    completed.append((new_log_prob, new_tokens))
                else:
                    new_beams.append((new_log_prob, new_tokens, h_new, c_new))

        # Keep top beam_size beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

        if not beams:
            break

    if not completed:
        completed = [(lp, toks) for lp, toks, _, _ in beams]

    completed.sort(key=lambda x: x[0], reverse=True)

    results = []
    for lp, token_ids in completed[:beam_size]:
        # Remove START token before decoding
        caption = vocab.decode(token_ids[1:])
        results.append((caption, lp))

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def show_captioned_image(image_path: str,
                          caption: str,
                          save_path: Optional[str] = None) -> None:
    """Display or save an image with its generated caption."""
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=13, wrap=True,
                 fontdict={"family": "DejaVu Sans"},
                 pad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generate captions for images")
    p.add_argument("--image",      type=str, default=None,
                   help="Path to a single image file")
    p.add_argument("--image_dir",  type=str, default=None,
                   help="Path to a folder of images")
    p.add_argument("--model",      type=str, default=config.BEST_MODEL_PATH,
                   help="Path to model checkpoint")
    p.add_argument("--beam_size",  type=int, default=config.BEAM_SIZE,
                   help="Beam size (1 = greedy)")
    p.add_argument("--save_dir",   type=str, default=config.OUTPUTS_DIR,
                   help="Where to save captioned images")
    return p.parse_args()


def caption_one(image_path: str,
                model: ImageCaptioningModel,
                vocab: Vocabulary,
                beam_size: int,
                save_dir: str) -> None:
    img_tensor = load_image(image_path, config.DEVICE)

    if beam_size == 1:
        caption = greedy_decode(model, img_tensor, vocab)
        print(f"\n📷 {os.path.basename(image_path)}")
        print(f"   Caption : {caption}")
    else:
        results = beam_search_decode(model, img_tensor, vocab, beam_size)
        caption = results[0][0]
        print(f"\n📷 {os.path.basename(image_path)}")
        print(f"   Best caption (beam={beam_size}) : {caption}")
        for i, (cap, score) in enumerate(results[:3], 1):
            print(f"   Beam {i} (score={score:.3f}): {cap}")

    # Save visualisation
    os.makedirs(save_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{stem}_captioned.png")
    show_captioned_image(image_path, caption, save_path=out_path)


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found at {args.model}. Please train first.")
        return

    vocab = Vocabulary.load()
    model = load_model(args.model, vocab, config.DEVICE)
    print(f"Model loaded from {args.model}")

    if args.image:
        caption_one(args.image, model, vocab, args.beam_size, args.save_dir)
    elif args.image_dir:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        if not paths:
            print(f"No images found in {args.image_dir}")
            return
        for path in sorted(paths):
            caption_one(path, model, vocab, args.beam_size, args.save_dir)
    else:
        print("Please provide --image or --image_dir")


if __name__ == "__main__":
    main()
