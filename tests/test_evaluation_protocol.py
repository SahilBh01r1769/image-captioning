import json
from argparse import Namespace

import pytest
import torch

import config
from attention_model import ExplainableCaptioningModel
from coco_metrics import coco_payload
from evaluate import evaluate, evaluate_run, prediction_record, write_failure_gallery_scaffold, write_jsonl
from inference import generate_caption_from_features
from report_evaluation import _fixed_sample, comparison_markdown
from vocabulary import Vocabulary


def small_vocabulary() -> Vocabulary:
    vocabulary = Vocabulary()
    vocabulary.build_from_captions(["a dog runs", "a child plays"], min_freq=1)
    return vocabulary


def test_coco_payload_preserves_all_references_and_rejects_duplicates():
    records = [
        {"image_name": "a.jpg", "prediction": "a dog", "references": ["a dog runs", "one dog"]},
        {"image_name": "b.jpg", "prediction": "a child", "references": ["a child plays"]},
    ]
    references, hypotheses = coco_payload(records)
    assert references["a.jpg"] == ["a dog runs", "one dog"]
    assert hypotheses == {"a.jpg": ["a dog"], "b.jpg": ["a child"]}
    with pytest.raises(ValueError, match="Duplicate"):
        coco_payload(records + [records[0]])


def test_cached_feature_greedy_decode_exports_probabilities_and_attention():
    vocabulary = small_vocabulary()
    model = ExplainableCaptioningModel(
        vocab_size=len(vocabulary), encoder_dim=8, embed_dim=8, hidden_dim=10,
        attention_dim=6, dropout=0.0, pad_idx=vocabulary[config.PAD_TOKEN],
        pretrained_encoder=False,
    ).eval()
    spatial = torch.randn(1, 4, config.CNN_FEAT_DIM)
    result = generate_caption_from_features(model, "attention", spatial, vocabulary, max_len=3)
    record = prediction_record("a.jpg", ["a dog runs"], result)
    assert record["image_name"] == "a.jpg"
    assert len(record["tokens"]) <= 3
    for token in record["tokens"]:
        assert 0.0 <= token["probability"] <= 1.0
        assert len(token["attention"]) == 4
        assert "confidence" not in token


def test_evaluation_exports_are_restart_safe_and_review_scaffold_is_not_fabricated(tmp_path):
    records = [{"image_name": "a.jpg", "prediction": "a dog", "references": ["a dog"]}]
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, records)
    assert json.loads(path.read_text()) == records[0]

    gallery = tmp_path / "failure_gallery.csv"
    write_failure_gallery_scaffold(gallery)
    expected = "image_name,model,failure_category,observation,likely_cause,evidence,follow_up\n"
    assert gallery.read_text() == expected
    gallery.write_text(expected + "a.jpg,baseline,object error,,,,\n")
    write_failure_gallery_scaffold(gallery)
    assert "a.jpg" in gallery.read_text()


def test_comparison_table_uses_manifest_values_and_fixed_sampling_avoids_cherry_picking():
    metrics = {
        "BLEU-1": 0.1, "BLEU-2": 0.2, "mean_caption_length": 4.0,
        "distinct_1": 0.3, "distinct_2": 0.4,
    }
    manifest = {
        "status": "partial_smoke_only", "images_evaluated_per_run": 2,
        "metric_implementation": "pycocoevalcap==1.2", "metrics": ["BLEU-1", "BLEU-2"],
        "runs": [{"run_name": "baseline_seed42", "checkpoint_epoch": 3, "metrics": metrics}],
    }
    rendered = comparison_markdown(manifest)
    assert "partial_smoke_only" in rendered
    assert "0.1000" in rendered

    records = [{"image_name": f"{index}.jpg"} for index in range(10)]
    assert [row["image_name"] for row in _fixed_sample(records, 3)] == ["0.jpg", "4.jpg", "9.jpg"]


def test_incomplete_training_run_cannot_be_evaluated(tmp_path):
    run_dir = tmp_path / "baseline_seed42"
    run_dir.mkdir()
    (run_dir / "status.json").write_text('{"state":"interrupted"}')
    with pytest.raises(RuntimeError, match="incomplete"):
        evaluate_run(
            "baseline_seed42", run_dir, tmp_path / "evaluation", {}, small_vocabulary(),
            torch.empty(0), {}, {}, [],
        )


def test_metrics_can_be_skipped_only_for_explicit_smoke_run(tmp_path):
    args = Namespace(skip_metrics=True, max_images=None, output_dir=str(tmp_path / "evaluation"))
    with pytest.raises(ValueError, match="only with --max_images"):
        evaluate(args)
