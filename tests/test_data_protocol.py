import json

import pytest

import config
from data_protocol import (
    create_split_manifest,
    load_or_create_split_manifest,
    validate_split_manifest,
)
from train import build_vocabulary, vocabulary_diagnostics


def _captions(count: int = 20) -> dict[str, list[str]]:
    return {
        f"image_{idx:02d}.jpg": [f"sharedword imageword{idx}"] * 5
        for idx in range(count)
    }


def test_frozen_manifest_round_trip_and_dataset_mismatch(tmp_path):
    image_captions = _captions()
    path = tmp_path / "split.json"
    first = load_or_create_split_manifest(
        image_captions,
        str(path),
        train_fraction=0.8,
        val_fraction=0.1,
        seed=42,
    )
    second = load_or_create_split_manifest(
        image_captions,
        str(path),
        train_fraction=0.8,
        val_fraction=0.1,
        seed=42,
    )
    assert first == second

    changed = dict(image_captions)
    changed["image_00.jpg"] = ["the caption collection changed"]
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_split_manifest(
            changed,
            str(path),
            train_fraction=0.8,
            val_fraction=0.1,
            seed=42,
        )


def test_manifest_rejects_manual_split_edits():
    image_captions = _captions()
    manifest = create_split_manifest(
        image_captions,
        train_fraction=0.8,
        val_fraction=0.1,
        seed=42,
    )
    manifest["splits"]["test"].append(manifest["splits"]["train"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_split_manifest(manifest, image_captions)


def test_vocabulary_uses_training_captions_only(tmp_path, monkeypatch):
    image_captions = _captions()
    manifest = create_split_manifest(
        image_captions,
        train_fraction=0.8,
        val_fraction=0.1,
        seed=42,
    )
    monkeypatch.setattr(config, "VOCAB_PATH", str(tmp_path / "vocabulary.pkl"))
    monkeypatch.setattr(config, "MIN_WORD_FREQ", 1)

    vocab = build_vocabulary(image_captions, manifest)

    train_key = manifest["splits"]["train"][0]
    test_key = manifest["splits"]["test"][0]
    train_word = image_captions[train_key][0].split()[-1]
    test_only_word = image_captions[test_key][0].split()[-1]
    assert train_word in vocab.word2idx
    assert test_only_word not in vocab.word2idx
    assert vocab.metadata["source"] == "training captions only"

    diagnostics = vocabulary_diagnostics(vocab, image_captions, manifest)
    assert diagnostics["test_unknown_token_rate"] > 0


def test_stale_vocabulary_is_rejected(tmp_path, monkeypatch):
    image_captions = _captions()
    manifest = create_split_manifest(
        image_captions,
        train_fraction=0.8,
        val_fraction=0.1,
        seed=42,
    )
    monkeypatch.setattr(config, "VOCAB_PATH", str(tmp_path / "vocabulary.pkl"))
    monkeypatch.setattr(config, "MIN_WORD_FREQ", 1)
    build_vocabulary(image_captions, manifest)

    altered = json.loads(json.dumps(manifest))
    altered["split_fingerprint"] = "different-experiment"
    with pytest.raises(RuntimeError, match="Cached vocabulary"):
        build_vocabulary(image_captions, altered)
