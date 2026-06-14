# evaluate.py
"""
Corpus-level evaluation on the Flickr8k test split.

Metrics
-------
  BLEU-1, BLEU-2, BLEU-3, BLEU-4  (cumulative)
  METEOR

Usage
-----
  python evaluate.py
  python evaluate.py --model models/checkpoint_epoch_10.pth --beam_size 5
"""

import os
import argparse
import json

import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

import config
from model import ImageCaptioningModel
from utils.vocabulary import Vocabulary
from utils.dataset import parse_captions, split_dataset
from utils.metrics import corpus_bleu, corpus_meteor, print_scores
from inference import load_image, greedy_decode, beam_search_decode, load_model


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     type=str, default=config.BEST_MODEL_PATH)
    p.add_argument("--beam_size", type=int, default=config.BEAM_SIZE,
                   help="Beam size for decoding (1 = greedy)")
    p.add_argument("--max_images", type=int, default=None,
                   help="Limit test images (for quick debugging)")
    return p.parse_args()


def evaluate(args):
    print(f"\n{'═'*50}")
    print(f"  Evaluation on Flickr8k Test Split")
    print(f"  Model      : {args.model}")
    print(f"  Beam size  : {args.beam_size}")
    print(f"  Device     : {config.DEVICE}")
    print(f"{'═'*50}\n")

    # ── Load vocab & model ────────────────────────────────────────────────
    vocab = Vocabulary.load()
    model = load_model(args.model, vocab, config.DEVICE)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Build test split ──────────────────────────────────────────────────
    image_captions = parse_captions()
    _, _, test_keys = split_dataset(image_captions)

    if args.max_images:
        test_keys = test_keys[:args.max_images]
        print(f"  (Limited to {args.max_images} images for quick eval)")

    # ── Generate captions ─────────────────────────────────────────────────
    hypotheses: list[str]       = []
    references_list: list[list[str]] = []
    failed = 0

    print(f"\nGenerating captions for {len(test_keys)} images …")
    for img_name in tqdm(test_keys):
        img_path = os.path.join(config.IMAGES_DIR, img_name)
        refs     = image_captions.get(img_name, [])

        if not os.path.exists(img_path) or not refs:
            failed += 1
            continue

        try:
            img_tensor = load_image(img_path, config.DEVICE)

            if args.beam_size == 1:
                caption = greedy_decode(model, img_tensor, vocab)
            else:
                results = beam_search_decode(
                    model, img_tensor, vocab, args.beam_size
                )
                caption = results[0][0]

            hypotheses.append(caption)
            references_list.append(refs)

        except Exception as e:
            failed += 1
            print(f"  ⚠ Error on {img_name}: {e}")
            continue

    if failed:
        print(f"\n  ⚠ Skipped {failed} images (missing/error)")

    if not hypotheses:
        print("No captions generated — cannot evaluate.")
        return

    # ── Compute metrics ───────────────────────────────────────────────────
    print(f"\nComputing BLEU & METEOR over {len(hypotheses)} hypotheses …")
    bleu   = corpus_bleu(hypotheses, references_list)
    meteor = corpus_meteor(hypotheses, references_list)

    print_scores(bleu, meteor)

    # ── Show sample captions ──────────────────────────────────────────────
    print("Sample captions:")
    for i in range(min(5, len(hypotheses))):
        print(f"\n  [{i+1}] Generated : {hypotheses[i]}")
        print(f"       Reference  : {references_list[i][0]}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "model":      args.model,
        "beam_size":  args.beam_size,
        "n_images":   len(hypotheses),
        "bleu1":      bleu.get("bleu1_cumulative", 0),
        "bleu2":      bleu.get("bleu2_cumulative", 0),
        "bleu3":      bleu.get("bleu3_cumulative", 0),
        "bleu4":      bleu.get("bleu4_cumulative", 0),
        "meteor":     meteor,
        "brevity_penalty": bleu.get("brevity_penalty", 1.0),
    }
    out_path = os.path.join(config.OUTPUTS_DIR, "evaluation_results.json")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
