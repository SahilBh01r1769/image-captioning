"""Evaluate all pre-registered CaptionLab ablations on one frozen test set."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from tqdm import tqdm

import config
from coco_metrics import PYCOCOEVALCAP_VERSION, compute_coco_metrics
from dataset import parse_captions, resolve_dataset_split
from feature_cache import load_feature_cache
from inference import _build_model_for_checkpoint, generate_caption_from_features
from metrics import caption_statistics
from run_artifacts import atomic_write_json, stable_hash
from vocabulary import Vocabulary


RUN_NAMES = ("baseline_seed42", "attention_seed42", "attention_coverage_seed42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled CaptionLab evaluation")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--feature_cache", default=config.FEATURE_CACHE_PATH)
    parser.add_argument("--vocabulary", default=config.VOCAB_PATH)
    parser.add_argument("--output_dir", default="evaluation")
    parser.add_argument("--max_images", type=int, default=None, help="Smoke-only prefix; never report as a full result")
    return parser.parse_args()


def load_verified_inputs(args: argparse.Namespace):
    image_captions = parse_captions()
    split = resolve_dataset_split(image_captions)
    vocab = Vocabulary.load(args.vocabulary)
    if getattr(vocab, "metadata", {}).get("split_fingerprint") != split["split_fingerprint"]:
        raise RuntimeError("Vocabulary and split fingerprints differ")
    features, feature_index, cache_metadata = load_feature_cache(args.feature_cache, image_captions)
    identities = {
        "split_fingerprint": split["split_fingerprint"],
        "vocabulary_fingerprint": stable_hash(vocab.word2idx),
        "feature_cache_fingerprint": stable_hash(cache_metadata),
    }
    test_keys = list(split["splits"]["test"])
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max_images must be positive")
        test_keys = test_keys[: args.max_images]
    return image_captions, split, vocab, features, feature_index, identities, test_keys


def load_verified_checkpoint(path: Path, vocab: Vocabulary, identities: dict[str, str]):
    if not path.is_file():
        raise FileNotFoundError(f"Required best checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)
    if checkpoint.get("identities") != identities:
        raise RuntimeError(f"Checkpoint identity mismatch: {path}")
    model, architecture = _build_model_for_checkpoint(checkpoint, vocab, config.DEVICE)
    return checkpoint, model, architecture


def prediction_record(image_name: str, references: list[str], result) -> dict:
    return {
        "image_name": image_name,
        "prediction": result.caption,
        "references": list(references),
        "sequence_log_probability": result.score,
        "tokens": [asdict(token) for token in result.tokens],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(path)


def evaluate_run(
    run_name: str,
    run_dir: Path,
    output_dir: Path,
    image_captions: dict[str, list[str]],
    vocab: Vocabulary,
    features: torch.Tensor,
    feature_index: dict[str, int],
    identities: dict[str, str],
    test_keys: list[str],
) -> dict:
    checkpoint, model, architecture = load_verified_checkpoint(
        run_dir / "checkpoints" / "best.pt", vocab, identities
    )
    if checkpoint["experiment"]["run_name"] != run_name:
        raise RuntimeError(f"Run/checkpoint name mismatch for {run_name}")
    records: list[dict] = []
    for image_name in tqdm(test_keys, desc=run_name, unit="image"):
        spatial = features[feature_index[image_name]].float().unsqueeze(0).to(config.DEVICE)
        result = generate_caption_from_features(model, architecture, spatial, vocab)
        records.append(prediction_record(image_name, image_captions[image_name], result))
    write_jsonl(output_dir / run_name / "predictions.jsonl", records)
    metrics = {
        **compute_coco_metrics(records),
        **caption_statistics([record["prediction"] for record in records]),
    }
    summary = {
        "run_name": run_name,
        "architecture": architecture,
        "coverage_lambda": checkpoint["experiment"]["coverage_lambda"],
        "checkpoint_epoch": checkpoint["epoch"],
        "images_evaluated": len(records),
        "decoding": {"method": "greedy", "temperature": 1.0},
        "metrics": metrics,
        "identities": identities,
    }
    atomic_write_json(output_dir / run_name / "summary.json", summary)
    return summary


def write_failure_gallery_scaffold(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "image_name,model,failure_category,observation,likely_cause,evidence,follow_up\n",
        encoding="utf-8",
    )


def evaluate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    image_captions, split, vocab, features, feature_index, identities, test_keys = load_verified_inputs(args)
    summaries = [
        evaluate_run(
            name, Path(args.runs_dir) / name, output_dir, image_captions, vocab,
            features, feature_index, identities, test_keys,
        )
        for name in RUN_NAMES
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if args.max_images is None else "partial_smoke_only",
        "test_images_in_frozen_split": len(split["splits"]["test"]),
        "images_evaluated_per_run": len(test_keys),
        "run_order": list(RUN_NAMES),
        "primary_decoding": "greedy",
        "metric_implementation": f"pycocoevalcap=={PYCOCOEVALCAP_VERSION}",
        "metrics": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"],
        "identities": identities,
        "runs": summaries,
    }
    atomic_write_json(output_dir / "comparison.json", manifest)
    write_failure_gallery_scaffold(output_dir / "failure_gallery.csv")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    evaluate(parse_args())
