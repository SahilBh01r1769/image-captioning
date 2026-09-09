# Third-party captioner decision

An earlier demo could fall back to Microsoft GIT-base-coco when no CaptionLab checkpoint was present. That path has been removed from the controlled-experiment branch.

The fallback made the interface immediately interactive, but it weakened project identity: a visitor could see a good caption without seeing evidence from the CNN–LSTM models trained here. It also added large Transformer dependencies unrelated to the research question.

CaptionLab still depends on a pretrained ImageNet ResNet50 as its frozen visual encoder. That transfer-learning choice is part of the declared experimental setup. No pretrained caption-generation model contributes outputs, metrics, or qualitative examples.

If a third-party captioner is ever used for an external comparison, its outputs must be labeled separately and cannot replace the baseline, attention, or coverage ablation.
