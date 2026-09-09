# CaptionLab

CaptionLab is a focused deep-learning experiment on image captioning. It asks a narrow question:

> Under the same Flickr8k split, vocabulary, frozen ResNet50 features, decoder capacity, optimization settings, and training budget, does spatial attention improve a global-vector CNN–LSTM baseline, and does coverage regularization change that result?

This repository is not a production captioning service. The Streamlit code is an optional checkpoint diagnostic; the project is the controlled comparison and the evidence it produces.

## Current verified status

- Data protocol: complete and covered by synthetic tests.
- Three seed-42 training runs: complete for 20 epochs on a Colab Tesla T4.
- Test-set caption metrics: pending the controlled evaluation notebook.
- Qualitative failure analysis: pending exported test predictions.
- Claims about attention improving caption quality: not yet justified.

The three runs share split fingerprint `f70fba…`, vocabulary fingerprint `c30c3c…`, feature-cache fingerprint `e3bb64…`, and training-code commit `95f78c8`. The supplied artifacts report a 2,662-token training-only vocabulary, 6,472/809/810 train/validation/test images, 3.09% validation OOV, and 3.17% test OOV.

## Controlled comparison

| Run | Visual representation | Decoder | Coverage | Everything else |
| --- | --- | --- | ---: | --- |
| `baseline_seed42` | Mean-pooled 7×7 ResNet50 grid | One-layer LSTM | 0 | Controlled |
| `attention_seed42` | Same 7×7 grid | Additive attention + LSTMCell | 0 | Controlled |
| `attention_coverage_seed42` | Same 7×7 grid | Additive attention + LSTMCell | 0.1 | Controlled |

The ResNet50 encoder is frozen. Its deterministic spatial features are extracted once and reused by all runs. GloVe, label smoothing, CNN fine-tuning, and extra architectures are excluded because there is no prior result in this project that justifies adding those variables.

## Data protocol

Flickr8k is split by image, never by caption, so captions of the same image cannot cross partitions. The first accepted split is serialized as an immutable manifest. Vocabulary construction sees training captions only. Split, vocabulary, cache, configuration, and checkpoints carry fingerprints; a stale or mismatched artifact causes an error instead of being silently reused.

The held-out OOV rates are expected consequences of the train-only vocabulary, not defects to remove with validation/test leakage.

## Training observations so far

| Run | Best validation caption loss | Best epoch | Final training loss | Final validation loss |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.8382 | 20 | 2.8318 | 2.8382 |
| Attention | 2.8434 | 19 | 2.5120 | 2.8465 |
| Attention + coverage | 2.8190 | 19 | 2.5282 | 2.8191 |

Attention fit the training captions more strongly without improving validation caption loss over the baseline. Coverage produced the lowest validation loss, but the difference is small and comes from one seed. These are training-dynamics observations, not test-quality conclusions. Full traceable status is in [RESULTS.md](RESULTS.md).

## Evaluation protocol

The primary comparison uses the best validation-loss checkpoint for each pre-registered run, greedy decoding at temperature 1.0, all 810 frozen test images, and all five references per image. BLEU-1…4, ROUGE-L, and CIDEr use exactly `pycocoevalcap==1.2`. METEOR is intentionally not reported because its Java/WordNet dependencies stalled in Colab. Beam search is secondary analysis only.

Every prediction, reference set, token probability, and attention vector is exported before aggregate reporting. Attention weights are decoder allocations, not causal explanations. Details are fixed in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## Reproduce the experiment

The intended low-cost path is Google Colab:

1. Run [`notebooks/CaptionLab_Colab.ipynb`](notebooks/CaptionLab_Colab.ipynb) to download Flickr8k directly, create the shared feature cache, and run or resume the three configurations.
2. Run [`notebooks/CaptionLab_Evaluation.ipynb`](notebooks/CaptionLab_Evaluation.ipynb) after all three `best.pt` checkpoints exist in Drive.
3. Return `CaptionLab_evaluation_bundle.zip` for analysis. Do not manually select attractive outputs.

For command-line use:

```bash
python train.py --experiment experiments/baseline.json --feature_cache path/to/features.pt
python train.py --experiment experiments/attention.json --feature_cache path/to/features.pt
python train.py --experiment experiments/attention_coverage.json --feature_cache path/to/features.pt

python evaluate.py \
  --runs_dir runs \
  --feature_cache path/to/features.pt \
  --vocabulary models/vocabulary.pkl \
  --output_dir evaluation_seed42_greedy
```

Evaluation dependencies are isolated in `requirements-eval.txt`. A non-empty output directory is never overwritten.

## Repository map

| Path | Role |
| --- | --- |
| `data_protocol.py`, `vocabulary.py` | Frozen split, training-only vocabulary, fingerprints and OOV diagnostics |
| `visual_features.py`, `feature_cache.py` | Shared frozen ResNet50 representation |
| `model.py`, `attention_model.py` | Baseline and controlled attention ablations |
| `experiment.py`, `train.py`, `run_artifacts.py` | Configured runs, provenance, checkpoints and resume safety |
| `evaluate.py`, `coco_metrics.py` | Greedy test decoding, complete exports and pinned metrics |
| `report_evaluation.py` | Tables, loss curves and fixed-sample attention overlays |
| `demo/` | Optional local checkpoint diagnostic, outside the evidence path |

## Limitations

- Flickr8k is small and its captions are stylistically narrow.
- Only one seed has completed, so small differences may be noise.
- A frozen ImageNet encoder limits domain adaptation.
- Teacher-forced validation loss and autoregressive caption metrics answer different questions.
- BLEU, METEOR, ROUGE-L, and CIDEr are imperfect proxies for semantic quality.
- Attention maps can be inspected, but they do not establish faithful causal explanation.
- Free Colab availability and runtime interruptions constrained the experimental budget.

The next defensible decision comes after test evaluation: keep the result as a one-seed scoped finding, or spend limited compute on another seed only if the observed difference is large enough to make variance estimation worthwhile.
