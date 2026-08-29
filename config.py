"""Central configuration for the image-captioning research project."""
from __future__ import annotations

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FLICKR8K_DIR = os.path.join(DATA_DIR, "Flickr8k")
IMAGES_DIR = os.path.join(FLICKR8K_DIR, "Images")
CAPTIONS_FILE = os.path.join(FLICKR8K_DIR, "captions.txt")
GLOVE_DIR = os.path.join(DATA_DIR, "glove")
GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.6B.200d.txt")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocabulary.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")

for directory in [MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Vocabulary
MIN_WORD_FREQ = 5
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

# Image preprocessing
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Architecture
# ``attention`` is the primary research model. ``baseline`` keeps the original
# global-image-vector encoder/decoder available for ablation and comparison.
DEFAULT_ARCHITECTURE = "attention"
CNN_BACKBONE = "resnet50"
CNN_FEAT_DIM = 2048
EMBED_DIM = 512
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5

# Explainable spatial-attention model
ATTENTION_ENCODER_DIM = 512
ATTENTION_DIM = 512
ATTENTION_REGULARIZATION = 0.7

# GloVe
USE_GLOVE = True
GLOVE_DIM = 200

# Training
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 5.0
FREEZE_CNN_EPOCHS = 5
CNN_LR_FACTOR = 0.1
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
MAX_CAPTION_LENGTH = 35
CAPTIONS_PER_IMAGE = 5
SAVE_EVERY_N_EPOCHS = 2

# Inference
BEAM_SIZE = 5
MAX_GEN_LEN = 35
DEFAULT_TEMPERATURE = 1.0

# Evaluation
BLEU_WEIGHTS = {
    "bleu1": (1, 0, 0, 0),
    "bleu2": (0.5, 0.5, 0, 0),
    "bleu3": (0.33, 0.33, 0.33, 0),
    "bleu4": (0.25, 0.25, 0.25, 0.25),
}
