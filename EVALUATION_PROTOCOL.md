# Evaluation protocol

This protocol was fixed before inspecting test captions or test metrics.

## Selection and decoding

- Evaluate `baseline_seed42`, `attention_seed42`, and `attention_coverage_seed42` in that order.
- Select each run's `best.pt` using validation caption loss only.
- Decode every image in the immutable 810-image test split.
- Use deterministic greedy decoding with temperature 1.0 and the same maximum length.
- Use all available Flickr8k references for each image.
- Do not drop a failed image silently. A missing artifact, identity mismatch, empty caption, or metric failure stops the run.

Beam search can be examined later, but it is not part of the primary architecture comparison because it adds a decoding hyperparameter and can obscure whether the model itself improved.

## Metrics

The evaluator checks for exactly `pycocoevalcap==1.2` and reports its outputs without multiplying by 100:

- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr

SPICE is excluded because its setup downloads an additional Stanford CoreNLP model and is disproportionate to this constrained experiment. Mean caption length and Distinct-1/2 are descriptive diagnostics, not quality scores.

## Traceability

Before decoding, the evaluator recomputes and compares split, vocabulary, and feature-cache identities against every checkpoint. It then writes:

- `comparison.json`: protocol, identities, run order, status and metrics;
- `comparison.md`: generated comparison table;
- one `predictions.jsonl` per run with image name, prediction, all references, sequence log probability, selected-token probabilities, and attention weights where applicable;
- one `summary.json` per run;
- `loss_curves.png` from stored training histories;
- `failure_gallery.csv`, initially header-only so no failure is invented;
- optional attention overlays for six evenly spaced test positions.

Using `--max_images` changes status to `partial_smoke_only`. Those numbers are pipeline checks and must never be copied into results.

## Qualitative review rule

After the complete export exists, review all three captions together for a fixed set of test images. The default attention examples are evenly spaced through the frozen test order rather than chosen for visual appeal. Add failures to the gallery with a concrete category, observation, likely cause, supporting output, and possible follow-up. Useful categories include hallucination, missed object, wrong relation, counting, generic caption, repetition, and premature end token.

Attention overlays describe where the decoder placed weight while producing a token. They are not object boxes, correctness probabilities, or proof that the highlighted region caused the word.
