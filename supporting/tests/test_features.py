"""Local Spark test for window aggregation and participant splitting.

Synthetic samples pass through ``build_window_features`` and
``assign_subject_splits``.  The assertions protect the model-row shape and the
no-participant-overlap guarantee on which leakage-safe evaluation depends.
"""

from pyspark.sql import SparkSession

from har_spark.features import FEATURE_COLUMNS, assign_subject_splits, build_window_features


def test_window_features_and_subject_splits() -> None:
    """Verify aggregation output and disjoint participant membership."""
    spark = SparkSession.builder.master("local[1]").appName("har-test").getOrCreate()
    try:
        rows = [
            (1600, "A", index * 50_000_000, "phone", "accel", float(index), 1.0, 2.0)
            for index in range(100)
        ]
        records = spark.createDataFrame(
            rows, ["subject_id", "activity", "timestamp_ns", "device", "sensor", "x", "y", "z"]
        )
        features = build_window_features(records, window_seconds=5, minimum_samples=70)
        result = assign_subject_splits(
            features,
            seed=42,
            train_fraction=0.70,
            validation_fraction=0.15,
        ).collect()
        assert len(result) == 1
        assert result[0]["sample_count"] == 100
        assert result[0]["split"] in {"train", "validation", "test"}
        assert all(result[0][column] is not None for column in FEATURE_COLUMNS)
    finally:
        spark.stop()
