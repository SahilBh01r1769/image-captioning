# config.py
"""
Central configuration for the Image Captioning project.
All hyperparameters and paths live here — edit this file to customize.
"""

import os
import torch

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
FLICKR8K_DIR     = os.path.join(DATA_DIR, "Flickr8k")
IMAGES_DIR       = os.path.join(FLICKR8K_DIR, "Images")
CAPTIONS_FILE    = os.path.join(FLICKR8K_DIR, "captions.txt")
GLOVE_DIR        = os.path.join(DATA_DIR, "glove")
GLOVE_FILE       = os.path.join(GLOVE_DIR, "glove.6B.200d.txt")

MODELS_DIR       = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR      = os.path.join(BASE_DIR, "outputs")
VOCAB_PATH       = os.path.join(MODELS_DIR, "vocabulary.pkl")
BEST_MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.pth")

# Create dirs if they don't exist
for d in [MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  Vocabulary
# ─────────────────────────────────────────────
MIN_WORD_FREQ = 5        # Words with freq < this are replaced by <UNK>
PAD_TOKEN     = "<PAD>"
START_TOKEN   = "<START>"
END_TOKEN     = "<END>"
UNK_TOKEN     = "<UNK>"

# ─────────────────────────────────────────────
#  Image
# ─────────────────────────────────────────────
IMAGE_SIZE    = 224      # ResNet50 expects 224×224
IMAGE_MEAN    = [0.485, 0.456, 0.406]   # ImageNet mean
IMAGE_STD     = [0.229, 0.224, 0.225]   # ImageNet std

# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────
CNN_BACKBONE   = "resnet50"   # Options: "resnet50", "resnet101"
CNN_FEAT_DIM   = 2048         # ResNet50 final feature dimension
EMBED_DIM      = 512          # Shared embedding dimension (image + word)
HIDDEN_DIM     = 512          # LSTM hidden state dimension
NUM_LAYERS     = 2            # Number of LSTM layers
DROPOUT        = 0.5          # Dropout probability

# GloVe
USE_GLOVE      = True
GLOVE_DIM      = 200          # Must match glove file (200 for glove.6B.200d)

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 15
LEARNING_RATE     = 3e-4
WEIGHT_DECAY      = 1e-4
CLIP_GRAD_NORM    = 5.0       # Gradient clipping max norm

# CNN fine-tuning: freeze encoder for first N epochs, then unfreeze
FREEZE_CNN_EPOCHS = 5         # Set to 0 to train CNN from epoch 1
CNN_LR_FACTOR     = 0.1       # LR for CNN when unfrozen = LR * factor

# Dataset split (Flickr8k doesn't have an official split file)
TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.10

MAX_CAPTION_LENGTH = 35       # Pad/truncate captions to this length
CAPTIONS_PER_IMAGE = 5        # Flickr8k has 5 captions per image

# Checkpoint saving
SAVE_EVERY_N_EPOCHS = 2       # Save checkpoint every N epochs

# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────
BEAM_SIZE     = 5             # Beam search width (1 = greedy)
MAX_GEN_LEN   = 35            # Max tokens to generate

# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────
BLEU_WEIGHTS = {
    "bleu1": (1, 0, 0, 0),
    "bleu2": (0.5, 0.5, 0, 0),
    "bleu3": (0.33, 0.33, 0.33, 0),
    "bleu4": (0.25, 0.25, 0.25, 0.25),
}
