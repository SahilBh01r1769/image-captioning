# 🖼️ Image Captioning with CNN + LSTM (Encoder-Decoder)

A complete deep learning project that generates natural language captions for images using a ResNet50 encoder and LSTM decoder, with GloVe embeddings and BLEU/METEOR evaluation.

---

## 📁 Project Structure

```
image_captioning/
│
├── data/                        # Dataset storage
│   ├── Flickr8k/
│   │   ├── Images/              # Place all 8,091 images here
│   │   └── captions.txt         # Flickr8k captions file
│   └── glove/
│       └── glove.6B.200d.txt    # GloVe embeddings (200-dim)
│
├── models/                      # Saved model checkpoints
│
├── outputs/                     # Generated captions, plots, logs
│
├── utils/
│   ├── dataset.py               # Dataset loading & preprocessing
│   ├── vocabulary.py            # Vocabulary builder
│   └── metrics.py               # BLEU & METEOR evaluation
│
├── model.py                     # CNN Encoder + LSTM Decoder architecture
├── train.py                     # Full training pipeline
├── inference.py                 # Caption generation (greedy + beam search)
├── evaluate.py                  # Corpus-level BLEU/METEOR evaluation
├── config.py                    # All hyperparameters in one place
└── requirements.txt             # Dependencies
```

---

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Flickr8k Dataset
- Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Download and extract so that:
  - Images → `data/Flickr8k/Images/`
  - Captions → `data/Flickr8k/captions.txt`

### 3. Download GloVe Embeddings
```bash
mkdir -p data/glove
# Download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip
# Extract glove.6B.200d.txt into data/glove/
```
Or use the helper script:
```bash
python -c "from utils.dataset import download_glove; download_glove()"
```

---

## 🚀 Training

```bash
# Train with default config (ResNet50 + LSTM + GloVe)
python train.py

# Resume from checkpoint
python train.py --resume models/checkpoint_epoch_5.pth

# Custom hyperparameters
python train.py --epochs 20 --batch_size 64 --lr 3e-4
```

Training logs and loss curves are saved to `outputs/`.

---

## 🔍 Inference

```bash
# Caption a single image (greedy decoding)
python inference.py --image path/to/image.jpg

# Beam search decoding (better quality)
python inference.py --image path/to/image.jpg --beam_size 5

# Caption multiple images
python inference.py --image_dir path/to/folder/
```

---

## 📊 Evaluation

```bash
# Evaluate on Flickr8k test split
python evaluate.py

# Output example:
# BLEU-1: 0.712  BLEU-2: 0.534  BLEU-3: 0.381  BLEU-4: 0.267
# METEOR:  0.241
```

---

## 🧠 Architecture

```
Image (224×224×3)
       │
  [ResNet50 Encoder]          ← Pre-trained on ImageNet, frozen initially
       │
  GlobalAvgPool → (2048,)
       │
  Linear → (512,)             ← Project to embedding dimension
       │
       ├──────────────────────────────────────┐
       │                                      │
  [h₀, c₀] = Linear(img_feat)           [Word Embeddings]
                                              │ GloVe 200-dim → 512-dim
       └──────────────────┬───────────────────┘
                          │
                    [LSTM Decoder]
                          │
                    [Linear → Softmax]
                          │
                   Next Word Prediction
```

**Training**: Teacher Forcing — ground-truth previous word is fed at each step.  
**Inference**: Greedy decoding or Beam Search (k=5 recommended).

---

## 📈 Expected Results (Flickr8k)

| Metric   | Baseline (this code) | With Attention |
|----------|---------------------|----------------|
| BLEU-1   | ~0.60–0.65          | ~0.70–0.75     |
| BLEU-4   | ~0.18–0.22          | ~0.25–0.30     |
| METEOR   | ~0.20–0.23          | ~0.25–0.28     |

---

## 🔧 Key Hyperparameters (`config.py`)

| Parameter        | Default | Notes                          |
|-----------------|---------|--------------------------------|
| `embed_dim`     | 512     | Word + image projection size   |
| `hidden_dim`    | 512     | LSTM hidden state size         |
| `num_layers`    | 2       | LSTM layers                    |
| `dropout`       | 0.5     | Dropout probability            |
| `learning_rate` | 3e-4    | Adam optimizer LR              |
| `batch_size`    | 32      | Reduce if OOM                  |
| `epochs`        | 15      | ~15 for good convergence       |
| `glove_dim`     | 200     | GloVe embedding dimension      |
| `beam_size`     | 5       | Beam search width              |
| `max_length`    | 35      | Max caption length (tokens)    |
| `min_freq`      | 5       | Min word frequency for vocab   |

---

## 🌱 Extensions & Next Steps

1. **Attention Mechanism** — Add Bahdanau attention to `model.py` for significant BLEU improvement
2. **Fine-tune CNN** — Unfreeze ResNet50 layers after a few epochs
3. **Transformer Decoder** — Replace LSTM with a Transformer for SOTA results
4. **Visual Question Answering** — Extend to answer questions about images
5. **Style Transfer** — Generate captions in different styles

---

## 📚 References

- Vinyals et al., "Show and Tell: A Neural Image Caption Generator" (2015)
- Xu et al., "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" (2015)
- Pennington et al., "GloVe: Global Vectors for Word Representation" (2014)
