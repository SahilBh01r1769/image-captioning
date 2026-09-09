import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "CaptionLab_Colab.ipynb"
EVALUATION_NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "CaptionLab_Evaluation.ipynb"
PINNED_RUNNER_COMMIT = "95f78c8310027c32c80a430bc867e98ffa974328"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_colab_notebook_is_valid_python_and_pins_reviewed_runner():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4

    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert PINNED_RUNNER_COMMIT in source
    assert "torch.cuda.is_available()" in source

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")


def test_colab_notebook_covers_all_controlled_runs_and_restart_paths():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _load_notebook()["cells"]
    )

    for experiment in ("baseline", "attention", "attention_coverage"):
        assert f"run_or_resume('{experiment}'" in source

    assert "--resume" in source
    assert "--smoke" in source
    assert "--overwrite_smoke" in source
    assert "CaptionLab_training_bundle.zip" in source


def test_colab_downloads_public_dataset_and_avoids_repeat_image_download():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _load_notebook()["cells"]
    )

    assert "kagglehub.dataset_download('adityajn105/flickr8k')" in source
    assert "CAPTIONS_BACKUP" in source
    assert "DRIVE_CACHE.exists() and CAPTIONS_BACKUP.is_file()" in source


def test_evaluation_notebook_is_valid_pinned_and_marks_smoke_outputs():
    notebook = json.loads(EVALUATION_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "ed94547cfa1d60baf91c1e76d7687952276541fc" in source
    assert "--max_images','3'" in source
    assert "partial_smoke_only" in source
    assert "--images_dir" in source
    assert "CaptionLab_evaluation_bundle" in source
    assert "num_beams" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"evaluation-cell-{index}")
