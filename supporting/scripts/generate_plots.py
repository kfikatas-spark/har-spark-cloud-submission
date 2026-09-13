"""Create report-ready figures from completed experiment directories.

This post-processing script accepts named run directories, reads their JSON
metrics/confusion counts, and produces comparison bars, confusion matrices, and
per-class F1 charts.  It consumes outputs from ``har_spark.model`` and does not
rerun preprocessing, training, or cloud jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


ACTIVITY_NAMES = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
    "F": "Typing",
    "G": "Brushing teeth",
    "H": "Eating soup",
    "I": "Eating chips",
    "J": "Eating pasta",
    "K": "Drinking",
    "L": "Eating sandwich",
    "M": "Kicking",
    "O": "Catching",
    "P": "Dribbling",
    "Q": "Writing",
    "R": "Clapping",
    "S": "Folding clothes",
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def parse_experiment(value: str) -> tuple[str, Path]:
    """Parse a ``NAME=RUN_DIRECTORY`` command-line experiment reference."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Experiment must use LABEL=RUN_DIRECTORY")
    label, path = value.split("=", 1)
    run = Path(path).expanduser().resolve()
    if not label or not run.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid experiment: {value}")
    return label, run


def load_json(path: Path):
    """Read one UTF-8 JSON artifact and return its decoded value."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bar_chart(rows: list[dict], metric: str, ylabel: str, output: Path) -> None:
    """Plot one comparison metric for all named experiments and both models."""
    labels = [row["condition"] for row in rows[::2]]
    logistic = [row[metric] for row in rows if row["model"] == "logistic_regression"]
    forest = [row[metric] for row in rows if row["model"] == "random_forest"]
    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar([position - width / 2 for position in positions], logistic, width, label="Logistic Regression")
    axis.bar([position + width / 2 for position in positions], forest, width, label="Random Forest")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Model comparison: {ylabel.lower()}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    if metric in {"accuracy", "macro_f1"}:
        axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def confusion_chart(run: Path, output: Path) -> None:
    """Render the Random Forest confusion matrix for one experiment."""
    metrics = load_json(run / "random_forest" / "metrics.json")
    confusion = load_json(run / "random_forest" / "confusion.json")
    labels = metrics["labels"]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in confusion:
        matrix[index[row["actual"]]][index[row["predicted"]]] = row["count"]
    normalized = []
    for row in matrix:
        total = sum(row)
        normalized.append([value / total if total else 0 for value in row])

    names = [ACTIVITY_NAMES.get(label, label) for label in labels]
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(range(len(names)), names, fontsize=8)
    axis.set_xlabel("Predicted activity")
    axis.set_ylabel("Actual activity")
    axis.set_title("Random Forest: row-normalized confusion matrix")
    figure.colorbar(image, ax=axis, label="Fraction of actual class")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def class_f1_chart(run: Path, output: Path) -> None:
    """Render per-activity Random Forest F1 values for one experiment."""
    metrics = load_json(run / "random_forest" / "metrics.json")
    per_class = sorted(metrics["per_class"], key=lambda row: row["f1"])
    names = [ACTIVITY_NAMES.get(row["activity"], row["activity"]) for row in per_class]
    values = [row["f1"] for row in per_class]
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(names, values)
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1 score")
    axis.set_title("Random Forest F1 score by activity")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    """Load requested runs and generate the complete figure set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", required=True, type=parse_experiment)
    parser.add_argument("--best", required=True, type=Path, help="Best run for detailed plots")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for condition, run in args.experiment:
        for model in ("logistic_regression", "random_forest"):
            metrics = load_json(run / model / "metrics.json")
            rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "training_seconds": metrics["training_seconds"],
                    "inference_seconds": metrics["inference_seconds"],
                    "test_windows": metrics["test_predictions"],
                }
            )

    with (output / "experiment_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bar_chart(rows, "accuracy", "Accuracy", output / "accuracy_comparison.png")
    bar_chart(rows, "macro_f1", "Macro-F1", output / "macro_f1_comparison.png")
    bar_chart(rows, "training_seconds", "Training time (seconds)", output / "training_time.png")
    best = args.best.expanduser().resolve()
    confusion_chart(best, output / "best_confusion_matrix.png")
    class_f1_chart(best, output / "best_per_class_f1.png")
    print(json.dumps({"output": str(output), "experiments": len(args.experiment)}, indent=2))


if __name__ == "__main__":
    main()
