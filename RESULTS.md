# Results and evidence status

Last evidence review: 2026-09-09.

## What has actually run

Three controlled training runs completed for 20 epochs on a Colab Tesla T4 using PyTorch 2.11.0+cu128. The artifacts identify training-code commit `95f78c8310027c32c80a430bc867e98ffa974328` and the same split, vocabulary, and feature-cache fingerprints for every run. No run reports early stopping.

| Run | Best validation caption loss | Best epoch | Final train caption loss | Final validation caption loss |
| --- | ---: | ---: | ---: | ---: |
| `baseline_seed42` | 2.838195 | 20 | 2.831761 | 2.838195 |
| `attention_seed42` | 2.843421 | 19 | 2.512012 | 2.846521 |
| `attention_coverage_seed42` | 2.819040 | 19 | 2.528234 | 2.819065 |

Observed, without overclaiming:

- The baseline ended with almost no train/validation loss gap.
- Plain attention reduced training loss substantially but did not improve its best validation caption loss over the baseline. This is consistent with mild overfitting, though one run cannot establish a general pattern.
- Coverage reached a validation caption loss about 0.019 below the baseline. The difference is too small and the evidence too limited to claim better captioning before test evaluation.
- Coverage loss rose from roughly 0.58 to 0.65; with coefficient 0.1 its contribution to total loss was roughly 0.06. Total loss is therefore not comparable across coverage and non-coverage runs.

## Data diagnostics

| Item | Verified value |
| --- | ---: |
| Training images | 6,472 |
| Validation images | 809 |
| Test images | 810 |
| Vocabulary size | 2,662 |
| Validation unknown-token rate | 3.09% |
| Test unknown-token rate | 3.17% |

The vocabulary was built from training captions only. Held-out unknown-token rates are reported rather than eliminated through leakage.

## Test comparison — pending

These fields remain blank until the complete evaluation artifact is returned. Do not fill them from a smoke run, README example, third-party checkpoint, or a selected subset.

| Run | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | — | — | — | — | — | — |
| Attention | — | — | — | — | — | — | — |
| Attention + coverage | — | — | — | — | — | — | — |

Required artifact: `CaptionLab_evaluation_bundle.zip` produced by `notebooks/CaptionLab_Evaluation.ipynb`. It must say `status: complete` and `images_evaluated_per_run: 810`.

## Qualitative analysis — pending

No generated test captions or attention maps have been reviewed yet. The failure gallery is deliberately empty. After evaluation, it will include a fixed, traceable set of images and will discuss failures rather than only attractive examples.

## What cannot yet go on a resume

- Any claim that attention improved caption quality.
- Any BLEU, METEOR, ROUGE-L, or CIDEr value.
- Any claim of explainability beyond exposing and visualizing decoder attention weights.
- Any statement implying production readiness or real-time service deployment.

The defensible achievement today is the controlled, leakage-resistant experimental pipeline and the completed training runs—not an unmeasured quality improvement.
