from __future__ import annotations

import itertools
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

st.set_page_config(page_title="CaptionLab | Explainable Image Captioning", layout="wide")

st.markdown(
    """
<style>
:root {--page:#f1eee8;--panel:#e6e0d6;--panel2:#ddd6cb;--ink:#29251f;--muted:#6c655c;--line:#bdb4a8;--rust:#82503a;--slate:#50616d;}
html, body, [data-testid="stAppViewContainer"] {background:var(--page);color:var(--ink);font-family:Arial,Helvetica,sans-serif;}
[data-testid="stSidebar"] {background:#e4ded4;border-right:1px solid var(--line);}
.block-container {max-width:1320px;padding-top:1.8rem;padding-bottom:3rem;}
h1,h2,h3 {color:var(--ink)!important;letter-spacing:-.02em;}
.hero {padding:1.25rem 0 1rem;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-bottom:1rem;}
.hero h1 {margin:0;font-size:2.15rem;}.hero p {margin:.45rem 0 0;color:var(--muted);max-width:850px;}
.context-line {margin-top:.7rem;color:var(--muted);font-size:.78rem;letter-spacing:.025em;}
.process {border:1px solid var(--line);background:var(--panel);margin:1rem 0 1.4rem;}
.process-row {display:grid;grid-template-columns:130px 1fr;gap:18px;padding:12px 14px;border-bottom:1px solid var(--line);}
.process-row:last-child {border-bottom:0;}.process-row b {color:var(--rust);font-size:.76rem;text-transform:uppercase;letter-spacing:.07em;}.process-row span {color:var(--muted);font-size:.9rem;}
.caption-card {padding:1.1rem 1.2rem;background:var(--panel);border:1px solid var(--line);border-radius:3px;}
.caption-main {font-size:1.45rem;font-weight:650;line-height:1.4;}.muted {color:var(--muted);font-size:.84rem;}
.stButton>button,[data-testid="stFileUploaderDropzone"] {border-radius:3px!important;box-shadow:none!important;transition:none!important;}
.stButton>button:hover {transform:none!important;}
.skeleton {height:130px;background:var(--panel2);border:1px solid var(--line);border-radius:3px;margin:10px 0 16px;animation:skeletonPulse 1.05s ease-in-out infinite;}
@keyframes skeletonPulse {0%,100%{opacity:.45}50%{opacity:.78}}
</style>
<div class="hero">
  <h1>CaptionLab | Explainable Image Captioning</h1>
  <p>Turn an image into language, compare alternative captions, and inspect spatial attention when the custom checkpoint is available.</p>
  <div class="context-line">RESNET50 SPATIAL FEATURES / ADDITIVE ATTENTION / LSTM DECODER / BEAM SEARCH</div>
</div>
<div class="process">
  <div class="process-row"><b>Objective</b><span>Generate a concise natural-language description of visual content instead of only predicting object labels.</span></div>
  <div class="process-row"><b>Method</b><span>Encode visual features, decode one word at a time, and retain multiple likely sequences through beam search.</span></div>
  <div class="process-row"><b>Output</b><span>Return a ranked caption with confidence and diversity signals, plus spatial attention when the custom model is active.</span></div>
</div>
""",
    unsafe_allow_html=True,
)


def loading_placeholder():
    slot = st.empty()
    slot.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
    return slot


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
        output = model.generate(pixel_values=pixel_values,max_length=40,num_beams=max(beam_size,count),num_return_sequences=count,early_stopping=True,return_dict_in_generate=True,output_scores=True)
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
        values.append(1.0 - (len(a & b) / len(union) if union else 1.0))
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
    values = grid.astype(np.float32) - grid.astype(np.float32).min()
    peak = float(values.max())
    if peak > 0:
        values /= peak
    mask_img = Image.fromarray(np.uint8(values * 255)).resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR)
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    heat = np.zeros_like(base)
    heat[..., 0] = 140
    heat[..., 1] = 78 + 45 * mask
    heat[..., 2] = 55
    alpha = (0.10 + 0.38 * mask)[..., None]
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
    st.caption("The research code trains a ResNet50, additive-attention and LSTM architecture on Flickr8k. The hosted fallback is Microsoft GIT-base-coco and is not presented as that custom checkpoint.")

st.markdown("### 1. Give the model an image")
st.caption("The model receives pixels only. The caption is generated autoregressively, one token conditioned on the image and previously generated words at a time.")
uploaded = st.file_uploader("Upload JPG, PNG or WEBP", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload an image to start the captioning walkthrough.")
    with st.expander("What makes the custom model different?"):
        st.markdown("**Baseline:** ResNet50 compresses the image to one vector, then a stacked LSTM generates the caption.\n\n**Research model:** ResNet50 keeps a spatial feature grid. Additive attention selects a weighted region for each word, then an LSTMCell generates the next token. The attention weights can be projected back onto the image for inspection.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
left_image, right_context = st.columns([1.15, 1])
with left_image:
    st.image(image, use_container_width=True)
with right_context:
    st.markdown("#### What happens next")
    st.markdown("1. **Encode** visual information into deep feature representations.  \n2. **Decode** candidate words using the image and previous tokens.  \n3. **Search** multiple sequences instead of committing to the first local choice.  \n4. **Interpret** the winning caption using confidence and diversity signals, plus spatial attention when the custom checkpoint is active.")

if not st.button("Generate caption", type="primary", use_container_width=True):
    st.stop()

st.markdown("### 2. Read the result")
loading = loading_placeholder()
if mode == "Custom attention model":
    vocab, model, architecture = load_custom_model()
    tensor = preprocess_image(image, torch.device("cpu"))
    results = generate_captions(model, architecture, tensor, vocab, beam_size=beam_size)[:candidate_count]
    loading.empty()
    best = results[0]
    mean_confidence = sum(token.confidence for token in best.tokens) / len(best.tokens) if best.tokens else 0.0
    st.markdown(f'<div class="caption-card"><div class="muted">CUSTOM {architecture.upper()} / BEST CAPTION</div><div class="caption-main">{best.caption}</div><div class="muted">Mean generated-token confidence: {mean_confidence:.1%}</div></div>', unsafe_allow_html=True)
    if len(results) > 1:
        st.markdown("#### Alternative beams")
        for index, candidate in enumerate(results[1:], 2):
            st.write(f"**{index}.** {candidate.caption}")
    if architecture == "attention" and best.tokens:
        st.markdown("### 3. Where did it look?")
        st.caption("Each panel overlays the attention weights used while producing that word. Stronger rust regions indicate greater decoder focus. These are model weights, not object-detection boxes.")
        visible = [token for token in best.tokens if attention_grid(token) is not None][:8]
        for offset in range(0, len(visible), 2):
            columns = st.columns(2)
            for column, token in zip(columns, visible[offset:offset + 2]):
                with column:
                    st.image(attention_overlay(image, attention_grid(token)), use_container_width=True)
                    st.caption(f"**{token.word}** | token confidence {token.confidence:.0%}")
else:
    candidates = reference_captions(image, beam_size, candidate_count)
    loading.empty()
    captions = [caption for caption, _ in candidates]
    best_caption, best_score = candidates[0]
    diversity = candidate_diversity(captions)
    score_text = f" | beam score {best_score:.3f}" if best_score is not None else ""
    st.markdown(f'<div class="caption-card"><div class="muted">HOSTED REFERENCE / MICROSOFT GIT-BASE-COCO</div><div class="caption-main">{best_caption}</div><div class="muted">Candidate lexical diversity: {diversity:.0%}{score_text}</div></div>', unsafe_allow_html=True)
    if len(candidates) > 1:
        st.markdown("#### Alternative beams")
        for index, (caption, score) in enumerate(candidates[1:], 2):
            suffix = f" | score {score:.3f}" if score is not None else ""
            st.write(f"**{index}.** {caption}{suffix}")
    st.info("The hosted result uses the MIT-licensed Microsoft GIT-base-coco checkpoint. The repository's custom attention architecture becomes the demo model when its checkpoint and vocabulary are present.")

with st.expander("How to interpret the numbers"):
    st.markdown("- **Beam score** ranks alternative generated sequences and should not be read as a calibrated probability.\n- **Candidate diversity** measures lexical disagreement between returned beams.\n- **Token confidence** is the softmax probability assigned to the selected word at a decoding step, not a guarantee that the word is visually correct.\n- **Attention overlays** show where the custom decoder weighted spatial CNN features while generating a word. They are diagnostic, not causal proof.")
