"""Create traceable tables and diagnostics from an evaluation directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render CaptionLab evaluation artifacts")
    parser.add_argument("--evaluation_dir", default="evaluation")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--attention_examples", type=int, default=6)
    return parser.parse_args()


def comparison_markdown(manifest: dict) -> str:
    metric_names = manifest["metrics"]
    header = ["Run", "Best epoch", *metric_names, "Mean length", "Distinct-1", "Distinct-2"]
    rows = []
    for run in manifest["runs"]:
        metrics = run["metrics"]
        rows.append([
            run["run_name"],
            str(run["checkpoint_epoch"]),
            *(f"{metrics[name]:.4f}" for name in metric_names),
            f"{metrics['mean_caption_length']:.2f}",
            f"{metrics['distinct_1']:.4f}",
            f"{metrics['distinct_2']:.4f}",
        ])
    implementation = manifest["metric_implementation"]
    if isinstance(implementation, dict):
        implementation = "; ".join(
            f"{name}: {package}" for name, package in implementation.items()
        )
    lines = [
        "# Generated controlled-comparison table",
        "",
        f"Status: `{manifest['status']}`. Decoding: greedy. Images per run: {manifest['images_evaluated_per_run']}.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
        f"Metric implementation: `{implementation}`. Values are emitted without rescaling.",
        "",
    ]
    return "\n".join(lines)


def plot_loss_curves(runs_dir: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for run_name in ("baseline_seed42", "attention_seed42", "attention_coverage_seed42"):
        history = json.loads((runs_dir / run_name / "history.json").read_text())
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train"]["caption_loss"] for row in history], label=run_name)
        axes[1].plot(epochs, [row["validation"]["caption_loss"] for row in history], label=run_name)
    axes[0].set_title("Training caption loss")
    axes[1].set_title("Validation caption loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Token cross-entropy")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fixed_sample(records: list[dict], count: int) -> list[dict]:
    if count < 1 or not records:
        return []
    indices = np.linspace(0, len(records) - 1, min(count, len(records)), dtype=int)
    return [records[index] for index in indices]


def render_attention_examples(
    evaluation_dir: Path, images_dir: Path, count: int
) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    for run_name in ("attention_seed42", "attention_coverage_seed42"):
        records = _fixed_sample(
            _read_jsonl(evaluation_dir / run_name / "predictions.jsonl"), count
        )
        destination = evaluation_dir / "attention_examples" / run_name
        destination.mkdir(parents=True, exist_ok=True)
        for record in records:
            image = np.asarray(Image.open(images_dir / record["image_name"]).convert("RGB"))
            tokens = [token for token in record["tokens"] if token["attention"]][:6]
            if not tokens:
                continue
            figure, axes = plt.subplots(1, len(tokens), figsize=(3 * len(tokens), 3))
            axes = np.atleast_1d(axes)
            for axis, token in zip(axes, tokens):
                weights = np.asarray(token["attention"], dtype=np.float32)
                side = int(round(weights.size ** 0.5))
                axis.imshow(image)
                axis.imshow(weights.reshape(side, side), cmap="magma", alpha=0.5,
                            extent=(0, image.shape[1], image.shape[0], 0))
                axis.set_title(token["word"])
                axis.axis("off")
            figure.suptitle(record["prediction"], fontsize=10)
            figure.tight_layout()
            figure.savefig(destination / f"{Path(record['image_name']).stem}.png", dpi=140)
            plt.close(figure)


def main(args: argparse.Namespace) -> None:
    evaluation_dir = Path(args.evaluation_dir)
    manifest = json.loads((evaluation_dir / "comparison.json").read_text())
    (evaluation_dir / "comparison.md").write_text(comparison_markdown(manifest), encoding="utf-8")
    plot_loss_curves(Path(args.runs_dir), evaluation_dir / "loss_curves.png")
    if args.images_dir:
        render_attention_examples(evaluation_dir, Path(args.images_dir), args.attention_examples)


if __name__ == "__main__":
    main(parse_args())
