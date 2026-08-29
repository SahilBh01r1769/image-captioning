# Third-Party Models and Assets

This repository contains original training, inference, evaluation, attention, and demo code around several third-party datasets/model assets. Those assets are not represented as models trained or owned by this repository.

## ImageNet-pretrained ResNet50

The custom baseline and attention architectures use the ResNet50 implementation and ImageNet pretrained weights provided through `torchvision` as their visual backbone.

The project adds its own captioning-specific projection, recurrent decoder, additive attention mechanism, training objective, fine-tuning schedule, decoding logic, and evaluation pipeline around that pretrained visual representation.

## Flickr8k

Flickr8k is the training/evaluation dataset expected by the custom captioning pipeline. The dataset is not committed to this repository. Users are responsible for obtaining it from an appropriate source and following the dataset's terms of use.

## GloVe

The optional caption-word embedding initialization uses Stanford GloVe (`glove.6B.200d.txt`). GloVe files are not redistributed by this repository.

## Microsoft GIT-base-coco — hosted-demo fallback only

The public Streamlit demo can fall back to:

```text
microsoft/git-base-coco
```

when the repository's own custom checkpoint is not present.

This checkpoint is Microsoft's GIT (Generative Image-to-text Transformer) model fine-tuned for image captioning on COCO and is distributed through Hugging Face under the model card's MIT license declaration.

It exists in the hosted demo for one reason: large custom training artifacts are deliberately not committed to Git, but a public visitor should still be able to upload an image and understand the captioning workflow.

The UI labels this mode **Hosted reference model**. It must not be described as:

- a model trained by this repository,
- the repository's ResNet50 + attention + LSTM architecture,
- evidence for the custom model's evaluation metrics,
- a replacement for reproducing a Flickr8k training run.

When `models/best_model.pth` and `models/vocabulary.pkl` are present, CaptionLab can instead load the repository's custom checkpoint and expose its token confidence and spatial attention maps.

## Performance claims

No performance number from a third-party checkpoint should be copied into this project's own results. Custom-model metrics belong in reproducible artifacts produced by `evaluate.py` from a clearly identified repository checkpoint and test split.
