from __future__ import annotations

import itertools
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from inference import attention_grid, generate_captions, load_model, preprocess_image
from vocabulary import Vocabulary

REFERENCE_MODEL_ID = "microsoft/git-base-coco"
CUSTOM_MODEL_PATH = ROOT / "models" / "best_model.pth"
CUSTOM_VOCAB_PATH = ROOT / "models" / "vocabulary.pkl"

st.set_page_config(page_title="CaptionLab | Explainable Image Captioning", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border:1px solid rgba(139,92,246,.30); border-radius:20px;
       background:linear-gradient(135deg,rgba(139,92,246,.13),rgba(20,26,43,.62)); margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.15rem;}
.hero p {margin:.45rem 0 0; color:#b6bfd1; max-width:850px;}
.pill {display:inline-block; margin:.7rem .35rem 0 0; padding:.25rem .65rem; border-radius:999px;
       border:1px solid rgba(139,92,246,.34); color:#c4b5fd; font-size:.76rem;}
.step {padding:.85rem 1rem; border-radius:14px; background:#111827; border:1px solid rgba(148,163,184,.15); min-height:115px;}
.step b {color:#c4b5fd; font-size:.77rem; text-transform:uppercase; letter-spacing:.08em;}
.step p {color:#aeb9cb; margin:.35rem 0 0; font-size:.89rem;}
.caption-card {padding:1.2rem 1.35rem; background:#131a2b; border:1px solid rgba(139,92,246,.28); border-radius:16px;}
.caption-main {font-size:1.45rem; font-weight:650; line-height:1.4;}
.muted {color:#94a3b8; font-size:.84rem;}
.word-chip {display:inline-block; padding:.26rem .52rem; margin:.16rem; border-radius:8px; background:#202942; font-size:.83rem;}
</style>
<div class="hero">
  <h1>CaptionLab · Explainable Image Captioning</h1>
  <p>Turn an image into language, compare alternative captions, and—when the custom attention checkpoint is available—inspect which image region influenced each generated word.</p>
  <span class="pill">RESNET50 SPATIAL FEATURES</span>
  <span class="pill">ADDITIVE ATTENTION</span>
  <span class="pill">LSTM DECODER</span>
  <span class="pill">BEAM SEARCH</span>
</div>
""",
    unsafe_allow_html=True,
)

objective, method, result = st.columns(3)
with objective:
    st.markdown('<div class="step"><b>Objective</b><p>Generate a concise natural-language description of visual content instead of only predicting object labels.</p></div>', unsafe_allow_html=True)
with method:
    st.markdown('<div class="step"><b>Method</b><p>Encode visual features, decode one word at a time, and keep multiple likely sequences through beam search.</p></div>', unsafe_allow_html=True)
with result:
    st.markdown('<div class="step"><b>Result</b><p>A ranked caption plus confidence/diversity signals; the custom model also exposes word-level spatial attention.</p></div>', unsafe_allow_html=True)


def custom_model_available() -> bool:
    return CUSTOM_MODEL_PATH.exists() and CUSTOM_VOCAB_PATH.exists()


@st.cache_resource(show_spinner=False)
def load_reference_model():
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(REFERENCE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(REFERENCE_MODEL_ID)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_custom_model():
    vocab = Vocabulary.load(str(CUSTOM_VOCAB_PATH))
    model, architecture = load_model(str(CUSTOM_MODEL_PATH), vocab, torch.device("cpu"))
    return vocab, model, architecture


def reference_captions(image: Image.Image, beam_size: int, count: int) -> list[tuple[str, float | None]]:
    processor, model = load_reference_model()
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
    with torch.inference_mode():
        output = model.generate(
            pixel_values=pixel_values,
            max_length=40,
            num_beams=max(beam_size, count),
            num_return_sequences=count,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    captions = processor.batch_decode(output.sequences, skip_special_tokens=True)
    sequence_scores = getattr(output, "sequences_scores", None)
    scores = sequence_scores.detach().cpu().tolist() if sequence_scores is not None else [None] * len(captions)
    return [(caption.strip(), score) for caption, score in zip(captions, scores)]


def candidate_diversity(captions: list[str]) -> float:
    pairs = list(itertools.combinations(captions, 2))
    if not pairs:
        return 0.0
    values = []
    for first, second in pairs:
        a, b = set(first.lower().split()), set(second.lower().split())
        union = a | b
        similarity = len(a & b) / len(union) if union else 1.0
        values.append(1.0 - similarity)
    return float(sum(values) / len(values))


def center_crop_224(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)))
    left = (resized.width - config.IMAGE_SIZE) // 2
    top = (resized.height - config.IMAGE_SIZE) // 2
    return resized.crop((left, top, left + config.IMAGE_SIZE, top + config.IMAGE_SIZE))


def attention_overlay(image: Image.Image, grid: np.ndarray) -> Image.Image:
    base = np.asarray(center_crop_224(image), dtype=np.float32)
    values = grid.astype(np.float32)
    values = values - values.min()
    peak = float(values.max())
    if peak > 0:
        values /= peak
    mask_img = Image.fromarray(np.uint8(values * 255)).resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    heat = np.zeros_like(base)
    heat[..., 0] = 255
    heat[..., 1] = 80 + 120 * mask
    heat[..., 2] = 80
    alpha = (0.12 + 0.48 * mask)[..., None]
    blended = base * (1 - alpha) + heat * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


with st.sidebar:
    st.header("Generation")
    available = custom_model_available()
    if available:
        mode = st.radio("Model", ["Custom attention model", "Hosted reference model"])
    else:
        mode = "Hosted reference model"
        st.caption("No custom checkpoint is bundled on this hosted branch, so the interactive demo uses the documented reference model.")
    beam_size = st.slider("Beam width", 1, 8, 5, help="Higher values retain more candidate word sequences before selecting the best caption.")
    candidate_count = st.slider("Alternatives", 1, 3, 3)
    st.divider()
    st.caption("The research code in this repository trains its own ResNet50 + additive-attention + LSTM architecture on Flickr8k. The hosted fallback is Microsoft GIT-base-coco and is not presented as that custom checkpoint.")

st.markdown("### 1 · Give the model an image")
st.caption("The model receives pixels only. The caption is generated autoregressively—one token conditioned on the image and previously generated words at a time.")
uploaded = st.file_uploader("Upload JPG, PNG or WEBP", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload an image to start the captioning walkthrough.")
    with st.expander("What makes the custom model different?"):
        st.markdown(
            """
**Baseline:** ResNet50 compresses the whole image to one vector → stacked LSTM generates the caption.

**Research model:** ResNet50 keeps a spatial feature grid → additive attention selects a weighted region at every word → an LSTMCell generates the next token. The attention weights can then be projected back onto the image, making the decoding process inspectable rather than completely opaque.
"""
        )
    st.stop()

image = Image.open(uploaded).convert("RGB")
left_image, right_context = st.columns([1.15, 1])
with left_image:
    st.image(image, use_container_width=True)
with right_context:
    st.markdown("#### What happens next")
    st.markdown(
        """
1. **Encode** visual information into deep feature representations.  
2. **Decode** candidate words conditioned on the image and previous tokens.  
3. **Search** multiple sequences instead of committing to the first local choice.  
4. **Interpret** the winning caption using confidence/diversity signals—and spatial attention when the custom checkpoint is active.
"""
    )

if not st.button("Generate caption", type="primary", use_container_width=True):
    st.stop()

st.markdown("### 2 · Read the result")
if mode == "Custom attention model":
    vocab, model, architecture = load_custom_model()
    tensor = preprocess_image(image, torch.device("cpu"))
    with st.spinner("Decoding with the custom checkpoint..."):
        results = generate_captions(
            model,
            architecture,
            tensor,
            vocab,
            beam_size=beam_size,
        )[:candidate_count]
    best = results[0]
    mean_confidence = (
        sum(token.confidence for token in best.tokens) / len(best.tokens)
        if best.tokens else 0.0
    )
    st.markdown(
        f'<div class="caption-card"><div class="muted">CUSTOM {architecture.upper()} · BEST CAPTION</div><div class="caption-main">{best.caption}</div><div class="muted">Mean generated-token confidence: {mean_confidence:.1%}</div></div>',
        unsafe_allow_html=True,
    )

    if len(results) > 1:
        st.markdown("#### Alternative beams")
        for index, candidate in enumerate(results[1:], 2):
            st.write(f"**{index}.** {candidate.caption}")

    if architecture == "attention" and best.tokens:
        st.markdown("### 3 · Where did it look?")
        st.caption("Each panel overlays the attention weights used while producing that word. Bright regions indicate stronger decoder focus; they are explanatory model weights, not object-detection boxes.")
        visible = [token for token in best.tokens if attention_grid(token) is not None][:9]
        for offset in range(0, len(visible), 3):
            columns = st.columns(3)
            for column, token in zip(columns, visible[offset:offset + 3]):
                grid = attention_grid(token)
                with column:
                    st.image(attention_overlay(image, grid), use_container_width=True)
                    st.caption(f"**{token.word}** · token confidence {token.confidence:.0%}")
else:
    with st.spinner("Loading the hosted reference captioner and generating candidates..."):
        candidates = reference_captions(image, beam_size, candidate_count)
    captions = [caption for caption, _ in candidates]
    best_caption, best_score = candidates[0]
    diversity = candidate_diversity(captions)
    score_text = f" · beam score {best_score:.3f}" if best_score is not None else ""
    st.markdown(
        f'<div class="caption-card"><div class="muted">HOSTED REFERENCE · MICROSOFT GIT-BASE-COCO</div><div class="caption-main">{best_caption}</div><div class="muted">Candidate lexical diversity: {diversity:.0%}{score_text}</div></div>',
        unsafe_allow_html=True,
    )
    if len(candidates) > 1:
        st.markdown("#### Alternative beams")
        for index, (caption, score) in enumerate(candidates[1:], 2):
            suffix = f"  ·  score {score:.3f}" if score is not None else ""
            st.write(f"**{index}.** {caption}{suffix}")
    st.info("This hosted result uses the MIT-licensed Microsoft GIT-base-coco checkpoint so the public demo is immediately interactive. The repository's custom attention architecture is separately trainable and automatically becomes the demo model when its checkpoint + vocabulary are present.")

with st.expander("How to interpret the numbers"):
    st.markdown(
        """
- **Beam score** ranks alternative generated sequences; it is useful comparatively and should not be read as a calibrated probability.
- **Candidate diversity** is lexical disagreement between returned beams. Higher diversity means the decoder found more substantially different descriptions.
- **Token confidence** (custom checkpoint) is the softmax probability assigned to the selected word at that decoding step; it is not a guarantee that the word is visually correct.
- **Attention overlays** show where the custom decoder weighted spatial CNN features while generating a word. They help inspect reasoning behavior but do not prove causal explanation.
"""
    )
