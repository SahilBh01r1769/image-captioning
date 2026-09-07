"""Command-line entrypoint for the shared frozen ResNet50 feature cache."""
from __future__ import annotations

import argparse
import json

import config
from dataset import parse_captions
from feature_cache import extract_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen ResNet50 spatial features")
    parser.add_argument("--output", default=config.FEATURE_CACHE_PATH)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = extract_feature_cache(
        parse_captions(),
        args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
