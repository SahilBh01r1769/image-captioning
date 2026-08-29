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
:root{--pink:#ff4fd8;--cyan:#36e6ff;--orange:#ff9d3d;--lime:#b8ff5a;--ink:#f8f7ff;--muted:#b9b5d0;--panel:#15102b;--panel2:#1d1240;--line:rgba(255,255,255,.13);}
[data-testid="stAppViewContainer"]{background:
radial-gradient(circle at 8% 0%,rgba(255,79,216,.18),transparent 27%),
radial-gradient(circle at 92% 7%,rgba(54,230,255,.16),transparent 25%),
linear-gradient(180deg,#09051a 0%,#0b0720 55%,#080417 100%);color:var(--ink);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#12092c,#0b071d);border-right:1px solid rgba(255,79,216,.18);}
.block-container {max-width:1320px;padding-top:1.8rem;padding-bottom:3rem;}
h1,h2,h3{letter-spacing:-.02em;}
.hero{position:relative;overflow:hidden;padding:1.65rem 1.8rem;border:1px solid rgba(255,255,255,.16);border-radius:24px;
background:linear-gradient(120deg,rgba(255,79,216,.28),rgba(112,63,255,.20) 48%,rgba(54,230,255,.20));margin-bottom:1rem;box-shadow:0 20px 60px rgba(57,22,120,.22);}
.hero:after{content:"";position:absolute;width:220px;height:220px;border-radius:50%;right:-55px;top:-85px;background:radial-gradient(circle,rgba(255,157,61,.34),rgba(255,157,61,0) 68%);}
.hero h1{margin:0;font-size:2.25rem;background:linear-gradient(90deg,#fff,#ffb5ef 32%,#9ff4ff 68%,#ffe0aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{margin:.5rem 0 0;color:#d4d0e5;max-width:850px;line-height:1.55;}
.pill{display:inline-block;margin:.75rem .36rem 0 0;padding:.28rem .68rem;border-radius:999px;font-weight:700;letter-spacing:.03em;font-size:.73rem;border:1px solid rgba(255,255,255,.18);background:rgba(9,5,26,.35);}
.pill:nth-of-type(1){color:#ff93e7;border-color:rgba(255,79,216,.45)}
.pill:nth-of-type(2){color:#81efff;border-color:rgba(54,230,255,.45)}
.pill:nth-of-type(3){color:#ffd08c;border-color:rgba(255,157,61,.45)}
.pill:nth-of-type(4){color:#d2ff93;border-color:rgba(184,255,90,.40)}
.step{padding:.95rem 1.05rem;border-radius:16px;background:linear-gradient(145deg,rgba(29,18,64,.95),rgba(18,12,43,.92));border:1px solid var(--line);min-height:118px;box-shadow:inset 0 1px 0 rgba(255,255,255,.03);}
.step:nth-child(1){border-top:2px solid var(--pink)}
.step b{color:#ff8de4;font-size:.76rem;text-transform:uppercase;letter-spacing:.1em;}
.step p{color:#c4bfd8;margin:.38rem 0 0;font-size:.89rem;line-height:1.5;}
.caption-card{padding:1.3rem 1.45rem;background:linear-gradient(135deg,rgba(255,79,216,.12),rgba(54,230,255,.08),rgba(29,18,64,.88));border:1px solid rgba(255,79,216,.32);border-radius:18px;box-shadow:0 12px 35px rgba(43,15,90,.18);}
.caption-main{font-size:1.55rem;font-weight:720;line-height:1.4;color:#fff;}
.muted{color:#b9b5d0;font-size:.84rem;}
.word-chip{display:inline-block;padding:.28rem .55rem;margin:.16rem;border-radius:9px;background:linear-gradient(135deg,rgba(255,79,216,.16),rgba(54,230,255,.12));border:1px solid rgba(255,255,255,.10);font-size:.83rem;}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(29,18,64,.75),rgba(17,10,39,.78));border:1px solid rgba(255,255,255,.10);padding:.75rem;border-radius:14px;}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#ff4fd8,#8b5cf6 48%,#36e6ff);border:0;color:#fff;font-weight:800;box-shadow:0 8px 26px rgba(255,79,216,.22);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.08);transform:translateY(-1px);}
[data-testid="stFileUploader"]{border:1px dashed rgba(54,230,255,.34);border-radius:16px;background:rgba(54,230,255,.035);padding:.25rem;}
hr{border-color:rgba(255,255,255,.08)!important;}
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
