# CaptionLab training runbook

The notebook at `notebooks/CaptionLab_Colab.ipynb` is the supported training path. It pins the reviewed Batch 3 commit, stores durable artifacts in Google Drive, and uses local Colab storage for the large feature cache while training.

## Before starting

You do not need to download Flickr8k on your computer or upload its archive to Drive. The notebook downloads and extracts the public `adityajn105/flickr8k` Kaggle dataset directly in the Colab runtime.

During the first session, it retains only `captions.txt` and the extracted ResNet50 feature cache in `MyDrive/CaptionLab`. Later sessions use those files without downloading the images again. An existing `MyDrive/Flickr8k` folder containing `Images/` and `captions.txt` remains an optional fallback.

## Run order

1. Select a T4 GPU runtime.
2. Mount Drive and run the pinned setup cell.
3. Download Flickr8k directly, or restore its small metadata after the first session.
4. Extract or restore the shared ResNet50 cache.
5. Run the two-batch attention smoke check.
6. Train `baseline_seed42`.
7. Train `attention_seed42`.
8. Train `attention_coverage_seed42`.
9. Inspect the validation-loss plot.
10. Create `CaptionLab_training_bundle.zip`.

If Colab disconnects after feature extraction, repeat the setup, dataset and cache cells, then rerun the interrupted experiment cell. The images are not downloaded again because training uses the cached features. The helper adds `--resume` when `last.pt` exists. It never overwrites a completed full run.

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
