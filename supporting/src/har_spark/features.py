"""Turn validated WISDM samples into leakage-safe model examples.

``ingest.valid_records`` supplies typed accelerometer/gyroscope samples.  This
module groups them into time windows, calculates 18 statistics per sensor,
optionally fuses matching sensors, defines the stable model-column order, and
assigns whole participants to train/validation/test.  ``pipeline`` uses these
functions locally, while ``cloud/cloud_pipeline.py`` reuses them in Google Cloud.
"""

from __future__ import annotations

import hashlib
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

FEATURE_COLUMNS = [
    "x_mean",
    "x_std",
    "x_min",
    "x_max",
    "y_mean",
    "y_std",
    "y_min",
    "y_max",
    "z_mean",
    "z_std",
    "z_min",
    "z_max",
    "magnitude_mean",
    "magnitude_std",
    "energy_mean",
    "xy_corr",
    "xz_corr",
    "yz_corr",
]

IDENTIFIER_COLUMNS = {
    "subject_id",
    "activity",
    "device",
    "sensor",
    "window_id",
    "sample_count",
    "subject_rank",
    "split",
}


def build_window_features(
    records: DataFrame,
    *,
    window_seconds: int,
    minimum_samples: int,
) -> DataFrame:
    """Aggregate typed samples into fixed-duration statistical feature rows. Adds a numeric
    time bucket, magnitude, and energy to every sample."""
    window_ns = int(window_seconds) * 1_000_000_000  # Δευτερόλεπτα → νανοδευτερόλεπτα.

    # ``window_id`` is an integer time bucket.  Activity remains in the grouping
    # key so samples on opposite sides of a labelled transition are never mixed.
    enriched = (
        records.withColumn("window_id", F.floor(F.col("timestamp_ns") / F.lit(window_ns)))
        .withColumn("magnitude", F.sqrt(F.col("x") ** 2 + F.col("y") ** 2 + F.col("z") ** 2))
        .withColumn("energy", F.col("x") ** 2 + F.col("y") ** 2 + F.col("z") ** 2)
    )

    grouped = enriched.groupBy("subject_id", "activity", "device", "sensor", "window_id").agg(
        F.count("*").alias("sample_count"),
        F.mean("x").alias("x_mean"),
        F.stddev_samp("x").alias("x_std"),
        F.min("x").alias("x_min"),
        F.max("x").alias("x_max"),
        F.mean("y").alias("y_mean"),
        F.stddev_samp("y").alias("y_std"),
        F.min("y").alias("y_min"),
        F.max("y").alias("y_max"),
        F.mean("z").alias("z_mean"),
        F.stddev_samp("z").alias("z_std"),
        F.min("z").alias("z_min"),
        F.max("z").alias("z_max"),
        F.mean("magnitude").alias("magnitude_mean"),
        F.stddev_samp("magnitude").alias("magnitude_std"),
        F.mean("energy").alias("energy_mean"),
        F.corr("x", "y").alias("xy_corr"),
        F.corr("x", "z").alias("xz_corr"),
        F.corr("y", "z").alias("yz_corr"),
    )

    # Reject undersampled windows first. Correlations/stddev can then still be
    # undefined for constant axes; zero is the documented deterministic fallback.
    cleaned = grouped.where(F.col("sample_count") >= F.lit(int(minimum_samples)))
    for column in FEATURE_COLUMNS:
        cleaned = cleaned.withColumn(
            column,
            F.when(F.isnan(F.col(column)) | F.col(column).isNull(), F.lit(0.0)).otherwise(
                F.col(column)
            ),
        )
    return cleaned


def fuse_sensor_features(features: DataFrame, sensors: list[str]) -> DataFrame:
    """Inner-join per-sensor feature windows using clear sensor prefixes. For one sensor it
    returns the existing DataFrame."""

    if len(sensors) < 2:
        return features

    # The inner join deliberately keeps only windows containing every requested
    # sensor.  A row given to ML therefore always has the same feature schema.
    keys = ["subject_id", "activity", "device", "window_id"]
    available_features = [column for column in FEATURE_COLUMNS if column in features.columns]
    if not available_features:
        raise ValueError("No supported feature columns are available for sensor fusion")
    frames = []
    for sensor in sensors:
        selected = features.where(F.col("sensor") == sensor).select(
            *keys,
            F.col("sample_count").alias(f"{sensor}_sample_count"),
            *(F.col(column).alias(f"{sensor}_{column}") for column in available_features),
        )
        frames.append(selected)
    return reduce(lambda left, right: left.join(right, on=keys, how="inner"), frames)


def model_feature_columns(features: DataFrame) -> list[str]:
    """Return numeric feature columns in a stable, documented order. Inspects only the
    DataFrame schema, not its rows."""

    single_sensor = [column for column in FEATURE_COLUMNS if column in features.columns]
    prefixed = [
        f"{sensor}_{column}"
        for sensor in ("accel", "gyro")
        for column in FEATURE_COLUMNS
        if f"{sensor}_{column}" in features.columns
    ]
    columns = single_sensor + prefixed
    if not columns:
        raise ValueError("No supported model feature columns were found")
    return columns


def assign_subject_splits(
    features: DataFrame,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> DataFrame:
    """Assign whole participants after a deterministic pseudo-random ordering. Collects
    only the small distinct-subject table to the driver, never the millions of samples
    or thousands of windows."""

    # Only the small subject table (51 people in the full dataset) reaches the
    # driver.  Sensor/window rows stay distributed.  Assigning whole subjects is
    # the key safeguard against identity leakage between train and test.
    subject_ids = [row["subject_id"] for row in features.select("subject_id").distinct().collect()]
    ordered = sorted(
        subject_ids,
        key=lambda subject_id: hashlib.sha256(f"{seed}:{subject_id}".encode()).digest(),
    )
    train_end = int(len(ordered) * float(train_fraction))
    validation_end = int(len(ordered) * float(train_fraction + validation_fraction))
    assignments = []
    for rank, subject_id in enumerate(ordered):
        if rank < train_end:
            split = "train"
        elif rank < validation_end:
            split = "validation"
        else:
            split = "test"
        assignments.append((int(subject_id), rank, split))
    subjects = features.sparkSession.createDataFrame(
        assignments, schema="subject_id int, subject_rank int, split string"
    )
    return features.join(
        subjects.select("subject_id", "subject_rank", "split"), on="subject_id", how="inner"
    )
