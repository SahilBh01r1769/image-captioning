from pathlib import Path

import pytest
import torch

import inference
from visual_features import frozen_backbone_identity
from vocabulary import Vocabulary


def small_vocabulary() -> Vocabulary:
    vocab = Vocabulary()
    vocab.build_from_captions(["a dog runs", "a cat sleeps"], min_freq=1)
    return vocab


def portable_checkpoint(vocab: Vocabulary) -> dict:
    return {
        "checkpoint_version": 1,
        "architecture": "baseline",
        "experiment": {"embedding_dim": 8, "hidden_dim": 10},
        "model_state": {},
        "vocab_size": len(vocab),
        "frozen_backbone_excluded": True,
        "frozen_backbone": frozen_backbone_identity(),
    }


class DummyModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def capture_baseline_construction(monkeypatch):
    captured = {}

    def build_model(**kwargs):
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(inference, "ImageCaptioningModel", build_model)
    monkeypatch.setattr(inference, "load_trainable_model_state", lambda model, state: None)
    return captured


def test_raw_image_inference_restores_the_pretrained_backbone(monkeypatch):
    vocab = small_vocabulary()
    captured = capture_baseline_construction(monkeypatch)

    inference._build_model_for_checkpoint(
        portable_checkpoint(vocab), vocab, torch.device("cpu")
    )

    assert captured["pretrained_encoder"] is True


def test_cached_feature_evaluation_does_not_load_the_image_encoder(monkeypatch):
    vocab = small_vocabulary()
    captured = capture_baseline_construction(monkeypatch)

    inference._build_model_for_checkpoint(
        portable_checkpoint(vocab),
        vocab,
        torch.device("cpu"),
        load_image_encoder=False,
    )

    assert captured["pretrained_encoder"] is False


def test_legacy_portable_checkpoint_uses_the_known_backbone_contract(monkeypatch):
    vocab = small_vocabulary()
    checkpoint = portable_checkpoint(vocab)
    checkpoint.pop("frozen_backbone")
    captured = capture_baseline_construction(monkeypatch)

    inference._build_model_for_checkpoint(
        checkpoint, vocab, torch.device("cpu")
    )

    assert captured["pretrained_encoder"] is True


def test_mismatched_backbone_is_rejected_before_model_construction(monkeypatch):
    vocab = small_vocabulary()
    checkpoint = portable_checkpoint(vocab)
    checkpoint["frozen_backbone"] = {
        "name": "torchvision.resnet50",
        "weights": "different-weights",
    }
    monkeypatch.setattr(
        inference,
        "ImageCaptioningModel",
        lambda **kwargs: pytest.fail("model should not be constructed"),
    )

    with pytest.raises(RuntimeError, match="Checkpoint backbone does not match"):
        inference._build_model_for_checkpoint(
            checkpoint, vocab, torch.device("cpu")
        )


def test_checkpoint_and_vocabulary_sizes_must_match():
    vocab = small_vocabulary()
    checkpoint = portable_checkpoint(vocab)
    checkpoint["vocab_size"] += 1

    with pytest.raises(RuntimeError, match="vocabulary size does not match"):
        inference._validate_checkpoint(checkpoint, vocab)


def test_missing_and_corrupt_artifacts_have_actionable_errors(tmp_path: Path):
    vocab = small_vocabulary()
    missing_checkpoint = tmp_path / "missing.pt"
    missing_vocabulary = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError, match="CaptionLab checkpoint not found"):
        inference.load_model(str(missing_checkpoint), vocab, torch.device("cpu"))
    with pytest.raises(FileNotFoundError, match="CaptionLab vocabulary not found"):
        inference.load_vocabulary(str(missing_vocabulary))

    corrupt_checkpoint = tmp_path / "corrupt.pt"
    corrupt_checkpoint.write_bytes(b"not a torch checkpoint")
    with pytest.raises(RuntimeError, match="may be corrupt or incompatible"):
        inference.load_model(str(corrupt_checkpoint), vocab, torch.device("cpu"))

    corrupt_vocabulary = tmp_path / "corrupt.pkl"
    corrupt_vocabulary.write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="may be corrupt or incompatible"):
        inference.load_vocabulary(str(corrupt_vocabulary))
