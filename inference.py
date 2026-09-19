"""Caption generation for baseline and explainable attention architectures."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import glob
import math
import os
import pickle
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import config
from attention_model import ExplainableCaptioningModel
from model import ImageCaptioningModel
from run_artifacts import load_trainable_model_state
from visual_features import frozen_backbone_identity
from vocabulary import Vocabulary


_INFER_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(config.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
])


@dataclass
class GeneratedToken:
    word: str
    probability: float
    attention: list[float] = field(default_factory=list)


@dataclass
class CaptionResult:
    caption: str
    score: float
    tokens: list[GeneratedToken] = field(default_factory=list)


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    return _INFER_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)


def load_image(path: str, device: torch.device) -> torch.Tensor:
    return preprocess_image(Image.open(path), device)


def load_vocabulary(path: str = config.VOCAB_PATH) -> Vocabulary:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CaptionLab vocabulary not found: {path}. "
            "Add the vocabulary.pkl saved with the checkpoint before running inference."
        )
    try:
        vocab = Vocabulary.load(path)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as exc:
        raise RuntimeError(
            f"Could not read CaptionLab vocabulary {path}; it may be corrupt "
            "or incompatible."
        ) from exc
    if not isinstance(vocab, Vocabulary):
        raise RuntimeError(f"Vocabulary artifact is not a CaptionLab Vocabulary: {path}")
    return vocab


def _validate_checkpoint(checkpoint: object, vocab: Vocabulary) -> dict:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a valid CaptionLab checkpoint dictionary")
    if checkpoint.get("checkpoint_version") != 1:
        raise RuntimeError(
            f"Unsupported checkpoint version: {checkpoint.get('checkpoint_version')!r}"
        )
    if checkpoint.get("architecture") not in {"baseline", "attention"}:
        raise RuntimeError(
            f"Unsupported checkpoint architecture: {checkpoint.get('architecture')!r}"
        )
    if not isinstance(checkpoint.get("experiment"), dict):
        raise RuntimeError("Checkpoint is missing its experiment configuration")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise RuntimeError("Checkpoint is missing its model state")
    if checkpoint.get("vocab_size") != len(vocab):
        raise RuntimeError(
            "Checkpoint vocabulary size does not match the loaded vocabulary "
            f"({checkpoint.get('vocab_size')!r} != {len(vocab)})"
        )
    if checkpoint.get("frozen_backbone_excluded") is not True:
        raise RuntimeError(
            "Checkpoint does not declare the portable frozen-backbone format"
        )

    expected_backbone = frozen_backbone_identity()
    declared_backbone = checkpoint.get("frozen_backbone")
    if declared_backbone is not None and declared_backbone != expected_backbone:
        raise RuntimeError(
            "Checkpoint backbone does not match this inference build: "
            f"expected {expected_backbone}, found {declared_backbone}"
        )
    return checkpoint


def _build_model_for_checkpoint(
    checkpoint: dict,
    vocab: Vocabulary,
    device: torch.device,
    *,
    load_image_encoder: bool = True,
):
    checkpoint = _validate_checkpoint(checkpoint, vocab)
    architecture = checkpoint["architecture"]
    experiment = checkpoint["experiment"]
    common = dict(
        vocab_size=len(vocab),
        embed_dim=experiment.get("embedding_dim", config.EMBED_DIM),
        hidden_dim=experiment.get("hidden_dim", config.HIDDEN_DIM),
        dropout=0.0,
        pad_idx=vocab[config.PAD_TOKEN],
        fine_tune_cnn=False,
        # Portable checkpoints omit the canonical frozen ResNet50. Raw-image
        # inference must restore the exact ImageNet weights used to create the
        # training feature cache. Cached-feature evaluation does not need it.
        pretrained_encoder=load_image_encoder,
    )
    if architecture == "attention":
        model = ExplainableCaptioningModel(
            **common,
            encoder_dim=experiment.get("attention_encoder_dim", config.ATTENTION_ENCODER_DIM),
            attention_dim=experiment.get("attention_dim", config.ATTENTION_DIM),
        )
    elif architecture == "baseline":
        model = ImageCaptioningModel(**common, num_layers=config.NUM_LAYERS)
    load_trainable_model_state(model, checkpoint["model_state"])
    model.to(device).eval()
    return model, architecture


def load_model(ckpt_path: str, vocab: Vocabulary, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"CaptionLab checkpoint not found: {ckpt_path}. "
            "Add a compatible best.pt checkpoint before running inference."
        )
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            f"Could not read CaptionLab checkpoint {ckpt_path}; it may be corrupt "
            "or incompatible."
        ) from exc
    return _build_model_for_checkpoint(
        checkpoint, vocab, device, load_image_encoder=True
    )


@torch.no_grad()
def baseline_greedy_decode(
    model: ImageCaptioningModel,
    img_tensor: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = config.MAX_GEN_LEN,
) -> CaptionResult:
    return baseline_greedy_decode_features(model, model.encoder.feature_extractor(img_tensor), vocab, max_len)


@torch.no_grad()
def baseline_greedy_decode_features(
    model: ImageCaptioningModel,
    spatial_features: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = config.MAX_GEN_LEN,
) -> CaptionResult:
    """Greedily decode one caption from the shared frozen spatial features."""
    device = spatial_features.device
    start_idx = vocab[config.START_TOKEN]
    end_idx = vocab[config.END_TOKEN]
    pad_idx = vocab[config.PAD_TOKEN]
    img_feat = model.encode_features(spatial_features)
    hidden, cell = model.decoder._init_lstm_state(img_feat)
    token = torch.tensor([[start_idx]], device=device)
    generated: list[GeneratedToken] = []
    score = 0.0

    for _ in range(max_len):
        embedding = model.decoder.embedding(token)
        output, (hidden, cell) = model.decoder.lstm(embedding, (hidden, cell))
        logits = model.decoder.classifier(output.squeeze(1))
        probs = F.softmax(logits, dim=-1)
        selected_probability, prediction = probs.max(dim=-1)
        idx = int(prediction.item())
        if idx == end_idx:
            break
        if idx != pad_idx:
            word = vocab.idx2word.get(idx, config.UNK_TOKEN)
            prob = float(selected_probability.item())
            generated.append(GeneratedToken(word=word, probability=prob))
            score += math.log(max(prob, 1e-12))
        token = prediction.unsqueeze(1)

    return CaptionResult(caption=" ".join(item.word for item in generated), score=score, tokens=generated)


@torch.no_grad()
def attention_greedy_decode(
    model: ExplainableCaptioningModel,
    img_tensor: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = config.MAX_GEN_LEN,
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> CaptionResult:
    return attention_greedy_decode_features(
        model, model.encoder.feature_extractor(img_tensor), vocab, max_len, temperature
    )


@torch.no_grad()
def attention_greedy_decode_features(
    model: ExplainableCaptioningModel,
    spatial_features: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = config.MAX_GEN_LEN,
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> CaptionResult:
    """Greedily decode and retain attention weights from cached features."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = spatial_features.device
    encoder_out = model.encode_features(spatial_features)
    state = model.decoder.init_state(encoder_out)
    token = torch.tensor([vocab[config.START_TOKEN]], device=device)
    end_idx = vocab[config.END_TOKEN]
    pad_idx = vocab[config.PAD_TOKEN]
    generated: list[GeneratedToken] = []
    score = 0.0

    for _ in range(max_len):
        logits, state, alpha = model.decoder.step(token, encoder_out, state)
        probs = F.softmax(logits / temperature, dim=-1)
        selected_probability, prediction = probs.max(dim=-1)
        idx = int(prediction.item())
        if idx == end_idx:
            break
        if idx != pad_idx:
            prob = float(selected_probability.item())
            generated.append(GeneratedToken(
                word=vocab.idx2word.get(idx, config.UNK_TOKEN),
                probability=prob,
                attention=alpha.squeeze(0).detach().cpu().tolist(),
            ))
            score += math.log(max(prob, 1e-12))
        token = prediction

    return CaptionResult(caption=" ".join(item.word for item in generated), score=score, tokens=generated)


def _length_penalized(log_prob: float, length: int, alpha: float = 0.7) -> float:
    return log_prob / (((5.0 + max(length, 1)) / 6.0) ** alpha)


@torch.no_grad()
def baseline_beam_search_decode(
    model: ImageCaptioningModel,
    img_tensor: torch.Tensor,
    vocab: Vocabulary,
    beam_size: int = config.BEAM_SIZE,
    max_len: int = config.MAX_GEN_LEN,
) -> list[CaptionResult]:
    device = img_tensor.device
    start_idx = vocab[config.START_TOKEN]
    end_idx = vocab[config.END_TOKEN]
    pad_idx = vocab[config.PAD_TOKEN]
    img_feat = model.encode(img_tensor)
    hidden, cell = model.decoder._init_lstm_state(img_feat)
    beams = [(0.0, [start_idx], hidden, cell, [])]
    completed = []

    for step in range(max_len):
        candidates = []
        for log_prob, ids, h_t, c_t, evidence in beams:
            last = torch.tensor([[ids[-1]]], device=device)
            embedding = model.decoder.embedding(last)
            output, (h_new, c_new) = model.decoder.lstm(embedding, (h_t, c_t))
            log_probs = F.log_softmax(model.decoder.classifier(output.squeeze(1)), dim=-1).squeeze(0)
            top_lp, top_idx = log_probs.topk(beam_size)
            for lp, idx in zip(top_lp.tolist(), top_idx.tolist()):
                new_log_prob = log_prob + lp
                new_ids = ids + [idx]
                new_evidence = evidence
                if idx not in (end_idx, pad_idx):
                    new_evidence = evidence + [GeneratedToken(
                        word=vocab.idx2word.get(idx, config.UNK_TOKEN),
                        probability=float(math.exp(lp)),
                    )]
                if idx == end_idx or step == max_len - 1:
                    completed.append((new_log_prob, new_ids, new_evidence))
                else:
                    candidates.append((new_log_prob, new_ids, h_new, c_new, new_evidence))
        candidates.sort(key=lambda item: _length_penalized(item[0], len(item[1])), reverse=True)
        beams = candidates[:beam_size]
        if not beams:
            break

    if not completed:
        completed = [(lp, ids, evidence) for lp, ids, _, _, evidence in beams]
    completed.sort(key=lambda item: _length_penalized(item[0], len(item[1])), reverse=True)
    return [
        CaptionResult(
            caption=" ".join(token.word for token in evidence),
            score=_length_penalized(log_prob, len(ids)),
            tokens=evidence,
        )
        for log_prob, ids, evidence in completed[:beam_size]
    ]


@torch.no_grad()
def attention_beam_search_decode(
    model: ExplainableCaptioningModel,
    img_tensor: torch.Tensor,
    vocab: Vocabulary,
    beam_size: int = config.BEAM_SIZE,
    max_len: int = config.MAX_GEN_LEN,
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> list[CaptionResult]:
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    device = img_tensor.device
    encoder_out = model.encode(img_tensor)
    initial_state = model.decoder.init_state(encoder_out)
    start_idx = vocab[config.START_TOKEN]
    end_idx = vocab[config.END_TOKEN]
    pad_idx = vocab[config.PAD_TOKEN]
    beams = [(0.0, [start_idx], initial_state, [])]
    completed = []

    for step in range(max_len):
        candidates = []
        for log_prob, ids, state, evidence in beams:
            token = torch.tensor([ids[-1]], device=device)
            logits, next_state, alpha = model.decoder.step(token, encoder_out, state)
            log_probs = F.log_softmax(logits / temperature, dim=-1).squeeze(0)
            top_lp, top_idx = log_probs.topk(beam_size)
            attention = alpha.squeeze(0).detach().cpu().tolist()
            for lp, idx in zip(top_lp.tolist(), top_idx.tolist()):
                new_log_prob = log_prob + lp
                new_ids = ids + [idx]
                new_evidence = evidence
                if idx not in (end_idx, pad_idx):
                    new_evidence = evidence + [GeneratedToken(
                        word=vocab.idx2word.get(idx, config.UNK_TOKEN),
                        probability=float(math.exp(lp)),
                        attention=attention,
                    )]
                if idx == end_idx or step == max_len - 1:
                    completed.append((new_log_prob, new_ids, new_evidence))
                else:
                    candidates.append((new_log_prob, new_ids, next_state, new_evidence))
        candidates.sort(key=lambda item: _length_penalized(item[0], len(item[1])), reverse=True)
        beams = candidates[:beam_size]
        if not beams:
            break

    if not completed:
        completed = [(lp, ids, evidence) for lp, ids, _, evidence in beams]
    completed.sort(key=lambda item: _length_penalized(item[0], len(item[1])), reverse=True)
    return [
        CaptionResult(
            caption=" ".join(token.word for token in evidence),
            score=_length_penalized(log_prob, len(ids)),
            tokens=evidence,
        )
        for log_prob, ids, evidence in completed[:beam_size]
    ]


def generate_captions(
    model,
    architecture: str,
    img_tensor: torch.Tensor,
    vocab: Vocabulary,
    beam_size: int = config.BEAM_SIZE,
    max_len: int = config.MAX_GEN_LEN,
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> list[CaptionResult]:
    if architecture == "attention":
        if beam_size == 1:
            return [attention_greedy_decode(model, img_tensor, vocab, max_len, temperature)]
        return attention_beam_search_decode(model, img_tensor, vocab, beam_size, max_len, temperature)
    if architecture == "baseline":
        if beam_size == 1:
            return [baseline_greedy_decode(model, img_tensor, vocab, max_len)]
        return baseline_beam_search_decode(model, img_tensor, vocab, beam_size, max_len)
    raise ValueError(f"Unsupported architecture: {architecture}")


def generate_caption_from_features(
    model,
    architecture: str,
    spatial_features: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = config.MAX_GEN_LEN,
) -> CaptionResult:
    """Official controlled-comparison decode: deterministic greedy search."""
    if architecture == "baseline":
        return baseline_greedy_decode_features(model, spatial_features, vocab, max_len)
    if architecture == "attention":
        return attention_greedy_decode_features(model, spatial_features, vocab, max_len, 1.0)
    raise ValueError(f"Unsupported architecture: {architecture}")


def attention_grid(token: GeneratedToken) -> Optional[np.ndarray]:
    if not token.attention:
        return None
    side = int(round(len(token.attention) ** 0.5))
    if side * side != len(token.attention):
        return None
    return np.asarray(token.attention, dtype=np.float32).reshape(side, side)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explainable image captions")
    parser.add_argument("--image", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument("--vocabulary", type=str, default=config.VOCAB_PATH)
    parser.add_argument("--beam_size", type=int, default=config.BEAM_SIZE)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    return parser.parse_args()


def caption_path(path: str, model, architecture: str, vocab: Vocabulary, args: argparse.Namespace) -> None:
    image_tensor = load_image(path, config.DEVICE)
    results = generate_captions(
        model,
        architecture,
        image_tensor,
        vocab,
        beam_size=args.beam_size,
        temperature=args.temperature,
    )
    print(f"\n{os.path.basename(path)} [{architecture}]")
    for rank, result in enumerate(results[:3], 1):
        print(f"  {rank}. {result.caption}  (score={result.score:.3f})")
        if result.tokens:
            mean_probability = sum(token.probability for token in result.tokens) / len(result.tokens)
            print(f"     mean selected-token probability={mean_probability:.1%}")


def main() -> None:
    args = parse_args()
    vocab = load_vocabulary(args.vocabulary)
    model, architecture = load_model(args.model, vocab, config.DEVICE)
    if args.image:
        caption_path(args.image, model, architecture, vocab, args)
        return
    if args.image_dir:
        paths: list[str] = []
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            paths.extend(glob.glob(os.path.join(args.image_dir, extension)))
        for path in sorted(paths):
            caption_path(path, model, architecture, vocab, args)
        return
    raise SystemExit("Provide --image or --image_dir")


if __name__ == "__main__":
    main()
