# CaptionLab training runbook

The notebook at `notebooks/CaptionLab_Colab.ipynb` is the supported training path. It pins the reviewed Batch 3 commit, stores durable artifacts in Google Drive, and uses local Colab storage for the large feature cache while training.

## Before starting

In Google Drive, prepare a Flickr8k folder containing:

```text
Flickr8k/
├── Images/
└── captions.txt
```

The notebook defaults to `MyDrive/Flickr8k`. Change its one dataset-path variable if your folder is elsewhere.

## Run order

1. Select a T4 GPU runtime.
2. Mount Drive and run the pinned setup cell.
3. Confirm the Flickr8k path.
4. Extract or restore the shared ResNet50 cache.
5. Run the two-batch attention smoke check.
6. Train `baseline_seed42`.
7. Train `attention_seed42`.
8. Train `attention_coverage_seed42`.
9. Inspect the validation-loss plot.
10. Create `CaptionLab_training_bundle.zip`.

If Colab disconnects, repeat the setup and cache cells, then rerun the interrupted experiment cell. The helper adds `--resume` when `last.pt` exists. It never overwrites a completed full run.

## What to observe

For each full run, note:

- whether training caption loss decreases steadily;
- the epoch at which validation caption loss is lowest;
- whether the training/validation gap begins widening;
- whether early stopping activates;
- for coverage, the unweighted coverage loss and its contribution (`0.1 × coverage loss`) relative to caption loss.

Do not interpret the smoke run as model evidence, compare total loss between coverage and non-coverage runs, or inspect test captions to choose a checkpoint.

## Artifact to return

Return `MyDrive/CaptionLab/CaptionLab_training_bundle.zip`. It includes:

- the frozen split and training-only vocabulary;
- exact experiment configurations and provenance;
- per-epoch histories and completion status;
- portable `best.pt` and `last.pt` checkpoints.

The Flickr8k images and approximately 1.6 GB feature cache are deliberately excluded.
