"""Pure, auditable helpers for freezing the CaptionLab data protocol."""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any


PROTOCOL_VERSION = 1


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_fingerprint(image_captions: dict[str, list[str]]) -> str:
    """Identify the exact image/caption collection used to create a split."""
    canonical = {
        image_name: list(image_captions[image_name])
        for image_name in sorted(image_captions)
    }
    return _stable_hash(canonical)


def create_split_manifest(
    image_captions: dict[str, list[str]],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Create a deterministic image-level split and its provenance record."""
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train and validation fractions must leave a non-empty test split")
    if not image_captions:
        raise ValueError("cannot split an empty caption dataset")

    image_keys = sorted(image_captions)
    random.Random(seed).shuffle(image_keys)
    train_end = int(len(image_keys) * train_fraction)
    val_end = train_end + int(len(image_keys) * val_fraction)
    splits = {
        "train": image_keys[:train_end],
        "validation": image_keys[train_end:val_end],
        "test": image_keys[val_end:],
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_fingerprint": dataset_fingerprint(image_captions),
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": val_fraction,
            "test": 1.0 - train_fraction - val_fraction,
        },
        "split_fingerprint": _stable_hash(splits),
        "splits": splits,
    }


def validate_split_manifest(
    manifest: dict[str, Any],
    image_captions: dict[str, list[str]],
) -> None:
    """Reject stale, overlapping, or incomplete split manifests."""
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("split manifest protocol version is not supported")
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(image_captions):
        raise ValueError("split manifest does not match the current caption dataset")

    splits = manifest.get("splits", {})
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("split manifest must contain train, validation, and test lists")
    split_sets = {name: set(keys) for name, keys in splits.items()}
    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("train and validation images overlap")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("train and test images overlap")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("validation and test images overlap")
    if set().union(*split_sets.values()) != set(image_captions):
        raise ValueError("split manifest does not cover the current image set exactly")
    if manifest.get("split_fingerprint") != _stable_hash(splits):
        raise ValueError("split manifest contents do not match its fingerprint")


def load_or_create_split_manifest(
    image_captions: dict[str, list[str]],
    path: str,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Load the frozen split, or create it once if it does not exist."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        validate_split_manifest(manifest, image_captions)
        expected = (seed, train_fraction, val_fraction)
        recorded = (
            manifest.get("seed"),
            manifest.get("fractions", {}).get("train"),
            manifest.get("fractions", {}).get("validation"),
        )
        if recorded != expected:
            raise ValueError("split settings changed after the manifest was frozen")
        return manifest

    manifest = create_split_manifest(
        image_captions,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def captions_for_images(
    image_captions: dict[str, list[str]], image_keys: list[str]
) -> list[str]:
    return [caption for image_name in image_keys for caption in image_captions[image_name]]
