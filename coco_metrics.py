"""Pinned COCO-caption metrics for the controlled CaptionLab comparison."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


PYCOCOEVALCAP_VERSION = "1.2"


def verify_metric_dependency() -> None:
    try:
        installed = version("pycocoevalcap")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Install the pinned evaluation dependency with "
            "`pip install -r requirements-eval.txt`."
        ) from exc
    if installed != PYCOCOEVALCAP_VERSION:
        raise RuntimeError(
            f"Expected pycocoevalcap=={PYCOCOEVALCAP_VERSION}, found {installed}."
        )


def coco_payload(
    predictions: list[dict],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Convert exported records to the mapping expected by COCO scorers."""
    references: dict[str, list[str]] = {}
    hypotheses: dict[str, list[str]] = {}
    for record in predictions:
        image_name = record["image_name"]
        if image_name in references:
            raise ValueError(f"Duplicate prediction for {image_name}")
        refs = list(record["references"])
        prediction = str(record["prediction"])
        if not refs or not prediction.strip():
            raise ValueError(f"Empty prediction or references for {image_name}")
        references[image_name] = refs
        hypotheses[image_name] = [prediction]
    if not references:
        raise ValueError("No predictions were supplied")
    return references, hypotheses


def compute_coco_metrics(predictions: list[dict]) -> dict[str, float]:
    """Compute standard caption metrics with pinned established libraries.

    SPICE is intentionally excluded because its first run downloads external
    Stanford CoreNLP assets. BLEU-1..4, METEOR, ROUGE-L, and CIDEr are enough
    for this small controlled comparison and are reported without rescaling.
    """
    verify_metric_dependency()
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    references, hypotheses = coco_payload(predictions)
    metrics: dict[str, float] = {}
    scorers = [
        (Bleu(4), ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4")),
        (Rouge(), ("ROUGE-L",)),
        (Cider(), ("CIDEr",)),
    ]
    for scorer, names in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        values = score if isinstance(score, (list, tuple)) else [score]
        if len(values) != len(names):
            raise RuntimeError(f"Unexpected metric output for {names}")
        metrics.update({name: float(value) for name, value in zip(names, values)})
    return metrics
