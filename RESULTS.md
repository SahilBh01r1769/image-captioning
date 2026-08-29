# Reproducible Results

The current repository implements and tests the captioning architectures, training lifecycle, inference, metrics, and interactive demo. It does **not** commit a trained Flickr8k checkpoint, so this file intentionally does not publish an unverified BLEU/METEOR score.

## What is verified in CI

The clean Python 3.11 workflow verifies, without downloading Flickr8k or ImageNet weights:

- spatial encoder output retains multiple image locations,
- additive-attention weights normalize across those locations,
- the attention decoder returns correctly shaped token logits and spatial maps,
- attention coverage regularization accepts a non-padding mask,
- image-level train/validation/test splits are deterministic and disjoint,
- vocabulary encode/decode behavior,
- BLEU and METEOR-lite sanity checks,
- caption-diversity bounds.

A separate workflow installs the hosted-demo dependencies, compiles the captioning/demo modules, starts Streamlit, and checks the application health endpoint.

## How to create an actual model result

After obtaining Flickr8k, train a checkpoint:

```bash
python train.py --architecture attention
```

Then evaluate the saved checkpoint:

```bash
python evaluate.py --model models/best_model.pth --beam_size 5
```

The machine-readable output is written to:

```text
outputs/evaluation_results.json
```

A future verified result section should record at minimum:

- exact checkpoint / commit,
- architecture (`attention` or `baseline`),
- split seed and split policy,
- beam size,
- images evaluated / failed,
- BLEU-1 through BLEU-4,
- METEOR-lite (clearly labelled as the repository approximation),
- Distinct-1 / Distinct-2,
- mean caption length,
- training configuration.

## Recommended ablation table

Once both models have been trained under the same protocol, this is the comparison worth reporting:

| Architecture | Spatial attention | CNN fine-tuning | BLEU-4 | METEOR-lite | Distinct-2 |
|---|---|---|---:|---:|---:|
| Global-vector baseline | No | Same staged policy | — | — | — |
| Attention LSTM | Yes | Same staged policy | — | — | — |

The dashes are deliberate. They should be replaced only by values reproduced from the repository's evaluation pipeline.
