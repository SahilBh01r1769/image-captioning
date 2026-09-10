import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "evaluation_preflight.py"


def test_preflight_script_is_metric_free_and_stage_visible():
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "compute_coco_metrics" not in source
    assert '"protocol_and_cache_validated"' in source
    assert '"run_decoded"' in source
    assert '"preflight_passed"' in source
    assert 'default="cpu"' in source
