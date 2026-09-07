# CaptionLab — Image Captioning with Visual Attention

CaptionLab is an image-captioning project built with PyTorch. It includes two captioning models:

- a ResNet50 + LSTM baseline
- a ResNet50 + additive-attention + LSTM model that keeps spatial image features and produces an attention map for each generated word

I built the attention version to make the model easier to inspect instead of treating caption generation as a complete black box.

The repository also contains training, inference, evaluation, tests, and a Streamlit demo.

## How it works

### Baseline

The baseline uses ResNet50 to encode an image into a single feature vector. That vector initializes an LSTM which generates the caption one token at a time.

```text
Image -> ResNet50 -> global image vector -> LSTM -> caption
```

### Attention model

The attention model keeps the spatial feature grid from ResNet50 instead of reducing the image immediately to one vector. At each decoding step, additive attention weights those visual regions before the LSTMCell predicts the next word.

```text
Image -> ResNet50 spatial features
                  |
                  v
          additive attention <--- decoder state
                  |
                  v
             LSTMCell -> next word
```

The returned attention weights can be projected back onto the image. These maps show where the decoder placed more weight while generating a token; they should not be interpreted as object-detection boxes or as proof of causal explanation.

## Main features

- ResNet50 encoder with ImageNet pretrained weights
- baseline and attention architectures under the same training pipeline
- teacher-forced caption training
- attention coverage regularization for the attention model
- staged CNN fine-tuning after an initial frozen period
- greedy and beam-search decoding
- token confidence and per-word attention weights for the custom attention model
- image-level train/validation/test splitting so captions from the same image stay in one split
- BLEU, METEOR-lite and caption-diversity diagnostics
- Streamlit demo for uploading an image and inspecting generated captions

## Repository structure

```text
attention_model.py          attention-based captioning model
model.py                    baseline ResNet50 + LSTM model
train.py                    training entry point
dataset.py                  Flickr8k loading and data splits
vocabulary.py               vocabulary and optional GloVe initialization
inference.py                greedy/beam decoding and attention output
evaluate.py                 evaluation script
metrics.py                  caption metrics
config.py                   training/model configuration

demo/app.py                 Streamlit interface
tests/                      unit tests
image_captioning_walkthrough.ipynb
```

## Dataset

The training code expects Flickr8k in this layout:

```text
data/
└── Flickr8k/
    ├── Images/
    └── captions.txt
```

The project creates a seeded 80/10/10 split by image rather than by individual caption rows. This matters because Flickr8k contains several captions for each image; splitting flattened caption rows could leak the same image into more than one split.

## Setup

Python 3.11 is the intended environment.

```bash
git clone https://github.com/SahilBh01r1769/image-captioning.git
cd image-captioning
python -m venv venv
```

Activate the environment and install the dependencies:

```bash
pip install -r requirements.txt
```

For the test dependencies:

```bash
pip install -r requirements-dev.txt
```

## Training

Train the attention model:

```bash
python train.py --architecture attention
```

Train the baseline:

```bash
python train.py --architecture baseline
```

A few training options are exposed through the CLI:

```bash
python train.py --architecture attention --epochs 20 --batch_size 32 --lr 3e-4
```

Resume from a matching checkpoint:

```bash
python train.py --architecture attention --resume models/attention_epoch_10.pth
```

The CNN starts frozen and is later fine-tuned with a lower learning rate. Checkpoints also store the architecture type so the wrong model definition is not silently used during resume/inference.

### Optional GloVe initialization

If you want to initialize word embeddings from GloVe, place the file at:

```text
data/glove/glove.6B.200d.txt
```

Training also works without it:

```bash
python train.py --no_glove
```

## Inference

Generate captions for one image:

```bash
python inference.py --image path/to/image.jpg --beam_size 5
```

Or for a directory:

```bash
python inference.py --image_dir path/to/images --beam_size 5
```

For the attention architecture, inference keeps the generated tokens, token softmax confidence, beam score and spatial attention weights in addition to the final caption.

## Evaluation

```bash
python evaluate.py --model models/best_model.pth --beam_size 5
```

For a smaller smoke run:

```bash
python evaluate.py --model models/best_model.pth --beam_size 3 --max_images 100
```

Results are written to `outputs/evaluation_results.json`.

The evaluation code reports BLEU-1 through BLEU-4, a lightweight METEOR approximation, caption length, Distinct-1/2 and mean token confidence when that information is available.

I have deliberately not put headline model scores in this README because they should only be reported against a specific saved checkpoint and test split.

## Streamlit demo

Run the demo locally with:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

If `models/best_model.pth` and `models/vocabulary.pkl` are present, the app can use the repository's custom model. For an attention checkpoint it also displays per-word attention overlays.

The hosted/demo fallback uses Microsoft's `git-base-coco` checkpoint when a custom checkpoint is not available. That model is only used to keep the interface usable without committing a large trained checkpoint; it is separate from the model implemented and trained in this repository.

## Tests

```bash
python -m pytest -q
```

The tests cover the attention module, tensor shapes, coverage regularization, dataset splitting, vocabulary behavior and metric sanity checks. They are designed to run without downloading Flickr8k.

## Notes

This is a learning/research project rather than a production captioning system. Flickr8k is small, automatic caption metrics are limited, and attention maps are useful diagnostics rather than definitive explanations of model behavior.

Third-party datasets, pretrained weights and external model checkpoints retain their own licenses and usage terms.
