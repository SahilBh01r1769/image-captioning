import json

import pytest
import torch
import torch.nn as nn

from experiment import ExperimentConfig
from run_artifacts import (
    atomic_write_json,
    load_trainable_model_state,
    prepare_run_directory,
    trainable_model_state,
)


EXPERIMENT_PATHS = [
    "experiments/baseline.json",
    "experiments/attention.json",
    "experiments/attention_coverage.json",
]


def test_primary_configs_are_a_controlled_comparison():
    baseline, attention, coverage = [ExperimentConfig.load(path) for path in EXPERIMENT_PATHS]
    assert baseline.controlled_signature() == attention.controlled_signature()
    assert attention.controlled_signature() == coverage.controlled_signature()
    assert baseline.architecture == "baseline"
    assert attention.architecture == coverage.architecture == "attention"
    assert baseline.coverage_lambda == attention.coverage_lambda == 0
    assert coverage.coverage_lambda == 0.1
    assert all(config.label_smoothing == 0 for config in (baseline, attention, coverage))


def test_experiment_config_rejects_coverage_on_baseline():
    payload = ExperimentConfig.load("experiments/baseline.json").to_dict()
    payload["coverage_lambda"] = 0.1
    with pytest.raises(ValueError, match="only applies"):
        ExperimentConfig.from_dict(payload)


def test_run_directory_refuses_to_overwrite_evidence(tmp_path):
    run_dir = prepare_run_directory(tmp_path, "run_one", resume=False)
    (run_dir / "history.json").write_text("[]", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        prepare_run_directory(tmp_path, "run_one", resume=False)

    replaced = prepare_run_directory(tmp_path, "run_one", resume=False, overwrite=True)
    assert not (replaced / "history.json").exists()
    assert (replaced / "checkpoints").is_dir()


def test_resume_requires_last_checkpoint(tmp_path):
    prepare_run_directory(tmp_path, "run_one", resume=False)
    with pytest.raises(FileNotFoundError, match="No resumable checkpoint"):
        prepare_run_directory(tmp_path, "run_one", resume=True)


def test_atomic_json_replacement(tmp_path):
    path = tmp_path / "status.json"
    atomic_write_json(path, {"state": "running"})
    atomic_write_json(path, {"state": "completed", "epoch": 4})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "state": "completed",
        "epoch": 4,
    }
    assert not (tmp_path / "status.json.tmp").exists()


class _FakeCaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.feature_extractor = nn.Linear(3, 3)
        self.encoder.projection = nn.Linear(3, 2)
        self.decoder = nn.Linear(2, 4)


def test_portable_checkpoint_excludes_only_frozen_backbone():
    source = _FakeCaptionModel()
    state = trainable_model_state(source)
    assert state
    assert all(not key.startswith("encoder.feature_extractor.") for key in state)
    assert "encoder.projection.weight" in state
    assert "decoder.weight" in state

    destination = _FakeCaptionModel()
    load_trainable_model_state(destination, state)
    assert torch.equal(source.encoder.projection.weight, destination.encoder.projection.weight)
    assert torch.equal(source.decoder.weight, destination.decoder.weight)
