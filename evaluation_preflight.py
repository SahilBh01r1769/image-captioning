"""Fast, metric-free validation of CaptionLab evaluation artifacts."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

import config
from evaluate import RUN_NAMES, load_verified_checkpoint, load_verified_inputs
from inference import generate_caption_from_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight CaptionLab evaluation artifacts")
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def _stage(name: str, started: float) -> None:
    print(json.dumps({"stage": name, "seconds": round(time.monotonic() - started, 2)}), flush=True)


def preflight(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    config.DEVICE = device
    inputs = argparse.Namespace(
        feature_cache=args.feature_cache,
        vocabulary=args.vocabulary,
        max_images=1,
    )

    print(json.dumps({"stage": "start", "device": str(device)}), flush=True)
    started = time.monotonic()
    image_captions, split, vocab, features, feature_index, identities, test_keys = load_verified_inputs(inputs)
    _stage("protocol_and_cache_validated", started)

    image_name = test_keys[0]
    run_checks = []
    for run_name in RUN_NAMES:
        started = time.monotonic()
        run_dir = Path(args.runs_dir) / run_name
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing run status: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            raise RuntimeError(f"Run is not complete: {run_name}")
        checkpoint, model, architecture = load_verified_checkpoint(
            run_dir / "checkpoints" / "best.pt", vocab, identities
        )
        spatial = features[feature_index[image_name]].float().unsqueeze(0).to(device)
        result = generate_caption_from_features(model, architecture, spatial, vocab)
        if not result.caption.strip():
            raise RuntimeError(f"{run_name} produced an empty preflight caption")
        check = {
            "run_name": run_name,
            "architecture": architecture,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "generated_tokens": len(result.tokens),
            "seconds": round(time.monotonic() - started, 2),
        }
        run_checks.append(check)
        print(json.dumps({"stage": "run_decoded", **check}), flush=True)
        del spatial, result, model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "status": "preflight_passed",
        "device": str(device),
        "test_images_in_split": len(split["splits"]["test"]),
        "sample_image": image_name,
        "identities": identities,
        "runs": run_checks,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    preflight(parse_args())
