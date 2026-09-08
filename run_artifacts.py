"""Small, explicit helpers for durable experiment artifacts and provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import torch


def stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_run_directory(
    runs_dir: str | Path,
    run_name: str,
    *,
    resume: bool,
    overwrite: bool = False,
) -> Path:
    run_dir = Path(runs_dir) / run_name
    if resume:
        if not (run_dir / "checkpoints" / "last.pt").exists():
            raise FileNotFoundError(f"No resumable checkpoint found in {run_dir}")
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory already contains artifacts: {run_dir}. "
            "Use --resume instead of overwriting evidence."
        )
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def trainable_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Exclude the canonical frozen backbone from small, portable checkpoints."""
    prefix = "encoder.feature_extractor."
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    }


def load_trainable_model_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_prefix = "encoder.feature_extractor."
    if unexpected or any(not key.startswith(allowed_prefix) for key in missing):
        raise RuntimeError(
            f"Checkpoint/model mismatch; missing={missing}, unexpected={unexpected}"
        )
