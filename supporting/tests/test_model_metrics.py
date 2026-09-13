"""Unit test for confusion counts and macro-averaged metrics.

The small prediction DataFrame exercises the same helper used after both local
and cloud model inference, without fitting an expensive classifier.
"""

import pytest
from pyspark.sql import SparkSession

from har_spark.model import _confusion_and_macro_metrics


def test_confusion_matrix_and_macro_metrics() -> None:
    """Verify confusion cells and macro-F1 for known predictions."""
    spark = SparkSession.builder.master("local[1]").appName("har-metrics-test").getOrCreate()
    try:
        predictions = spark.createDataFrame(
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            ["label", "prediction"],
        )
        confusion, metrics = _confusion_and_macro_metrics(predictions, ["A", "B"])
        counts = {(row["actual"], row["predicted"]): row["count"] for row in confusion}
        assert counts[("A", "A")] == 1
        assert counts[("A", "B")] == 1
        assert counts[("B", "B")] == 2
        assert metrics["macro_recall"] == pytest.approx(0.75)
    finally:
        spark.stop()

