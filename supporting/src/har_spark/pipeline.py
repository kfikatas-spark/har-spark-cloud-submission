"""Coordinate the reusable local end-to-end HAR workflow.

This is the integration layer between configuration, Spark construction, raw
ingestion, window features, participant-disjoint splitting, and MLlib training.
The CLI calls either :func:`run_all`, :func:`prepare_features`, or
:func:`run_from_processed`.  Timestamped output directories and error-on-exist
writes make experiments reproducible and prevent accidental result replacement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame

from .config import AppConfig
from .features import assign_subject_splits, build_window_features, fuse_sensor_features
from .ingest import parse_wisdm_raw, valid_records
from .model import train_and_evaluate
from .spark import build_spark


def prepare_features(config: AppConfig) -> tuple[DataFrame, dict[str, Any]]:
    """Run parsing through participant splitting and return cached features plus a summary.
    Creates Spark, reads sensor selection from AppConfig, parses raw files, and caches
    parsed rows because quality counting and filtering reuse them."""
    spark = build_spark(config)
    data = config.data
    sensors = list(data.get("sensors", [data.get("sensor")]))

    # Parsing is reused for quality counts and filtering, so materialise it once.
    parsed = parse_wisdm_raw(spark, str(data["raw_glob"])).cache()
    quality_rows = parsed.groupBy("_parse_ok").count().collect()
    quality = {str(bool(row["_parse_ok"])).lower(): int(row["count"]) for row in quality_rows}

    records = valid_records(parsed, str(data["device"]), sensors)
    features = build_window_features(
        records,
        window_seconds=int(data["window_seconds"]),
        minimum_samples=int(data["minimum_samples_per_window"]),
    )
    features = fuse_sensor_features(features, sensors)
    split = config.split
    features = assign_subject_splits(
        features,
        seed=int(config.project["seed"]),
        train_fraction=float(split["train_fraction"]),
        validation_fraction=float(split["validation_fraction"]),
    ).cache()
    # These actions create a compact, human-readable provenance record alongside
    # the model metrics; they do not collect feature rows themselves.
    summary = {
        "parse_counts": quality,
        "sensors": sensors,
        "feature_windows": features.count(),
        "subjects": features.select("subject_id").distinct().count(),
        "activities": features.select("activity").distinct().count(),
        "split_windows": {
            row["split"]: int(row["count"]) for row in features.groupBy("split").count().collect()
        },
        "split_subjects": {
            row["split"]: int(row["count"])
            for row in features.select("subject_id", "split")
            .distinct()
            .groupBy("split")
            .count()
            .collect()
        },
    }
    parsed.unpersist()
    return features, summary


def write_features(features: DataFrame, path: str) -> None:
    """Persist prepared features as Parquet while refusing an existing destination. Spark
    writes a Parquet directory containing part files rather than one monolithic file."""
    features.write.mode("errorifexists").parquet(path)


def _run_models(features: DataFrame, config: AppConfig, data_summary: dict[str, Any]) -> Path:
    """Create a timestamped run and evaluate both configured classifiers. Creates a UTC run
    ID and refuses timestamp collisions."""
    # UTC timestamps make local/cloud evidence comparable and naturally give each
    # experiment a new directory; the explicit existence check closes collisions.
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = Path(str(config.output["results_root"])) / run_id
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=False)
    with (run_directory / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(data_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    comparison: dict[str, Any] = {}
    for model_name in ("logistic_regression", "random_forest"):
        comparison[model_name] = train_and_evaluate(
            features,
            model_name=model_name,
            settings=config.models[model_name],
            seed=int(config.project["seed"]),
            output_directory=run_directory / model_name,
        )
    with (run_directory / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
    features.unpersist()
    return run_directory


def run_all(config: AppConfig, *, write_processed: bool = False) -> Path:
    """Prepare raw data and train both models in one process. Connects preparation and
    training without serialising between them."""
    features, data_summary = prepare_features(config)
    if write_processed:
        write_features(features, str(config.data["processed_path"]))
    return _run_models(features, config, data_summary)


def run_from_processed(config: AppConfig, path: str | None = None) -> Path:
    """Load previously prepared Parquet features and train both models. Creates Spark and
    lazily reads the saved Parquet schema/data."""
    spark = build_spark(config)
    input_path = path or str(config.data["processed_path"])
    features = spark.read.parquet(input_path).cache()
    data_summary = {
        "prepared_feature_path": input_path,
        "feature_windows": features.count(),
        "subjects": features.select("subject_id").distinct().count(),
        "activities": features.select("activity").distinct().count(),
        "split_windows": {
            row["split"]: int(row["count"])
            for row in features.groupBy("split").count().collect()
        },
        "split_subjects": {
            row["split"]: int(row["count"])
            for row in features.select("subject_id", "split")
            .distinct()
            .groupBy("split")
            .count()
            .collect()
        },
    }
    return _run_models(features, config, data_summary)
