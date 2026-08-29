"""Reproducible Flickr8k evaluation for baseline and attention captioners."""
from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

import config
from dataset import parse_captions, split_dataset
from inference import generate_captions, load_image, load_model
from metrics import caption_statistics, corpus_bleu, corpus_meteor_lite
from vocabulary import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained captioning checkpoint")
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument("--beam_size", type=int, default=config.BEAM_SIZE)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    vocab = Vocabulary.load()
    model, architecture = load_model(args.model, vocab, config.DEVICE)
    image_captions = parse_captions()
    _, _, test_keys = split_dataset(image_captions)
    if args.max_images is not None:
        test_keys = test_keys[: max(0, args.max_images)]

    hypotheses: list[str] = []
    references_list: list[list[str]] = []
    mean_token_confidences: list[float] = []
    failures: list[str] = []

    for image_name in tqdm(test_keys, desc="Evaluating", unit="image"):
        image_path = os.path.join(config.IMAGES_DIR, image_name)
        references = image_captions.get(image_name, [])
        if not os.path.exists(image_path) or not references:
            failures.append(image_name)
            continue
        try:
            image_tensor = load_image(image_path, config.DEVICE)
            candidates = generate_captions(
                model,
                architecture,
                image_tensor,
                vocab,
                beam_size=args.beam_size,
                temperature=args.temperature,
            )
            if not candidates:
                failures.append(image_name)
                continue
            best = candidates[0]
            hypotheses.append(best.caption)
            references_list.append(references)
            if best.tokens:
                mean_token_confidences.append(
                    sum(token.confidence for token in best.tokens) / len(best.tokens)
                )
        except Exception as exc:
            failures.append(f"{image_name}: {exc}")

    if not hypotheses:
        raise RuntimeError("Evaluation generated no captions")

    bleu = corpus_bleu(hypotheses, references_list)
    diversity = caption_statistics(hypotheses)
    results = {
        "architecture": architecture,
        "checkpoint": args.model,
        "beam_size": args.beam_size,
        "images_evaluated": len(hypotheses),
        "images_failed": len(failures),
        "bleu_1": bleu["bleu1_cumulative"],
        "bleu_2": bleu["bleu2_cumulative"],
        "bleu_3": bleu["bleu3_cumulative"],
        "bleu_4": bleu["bleu4_cumulative"],
        "meteor_lite": corpus_meteor_lite(hypotheses, references_list),
        **diversity,
        "mean_token_confidence": (
            sum(mean_token_confidences) / len(mean_token_confidences)
            if mean_token_confidences else None
        ),
        "metric_notes": {
            "meteor_lite": "Exact/stem matching approximation; not the full WordNet METEOR implementation.",
            "distinct_1_2": "Unique generated n-gram ratios; diversity diagnostics, not semantic-quality metrics.",
        },
    }

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUTS_DIR, "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    if failures:
        print(f"Skipped {len(failures)} image(s).")
    return results


if __name__ == "__main__":
    evaluate(parse_args())
