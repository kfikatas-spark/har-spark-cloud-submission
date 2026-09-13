"""Local Spark test for accelerometer/gyroscope window fusion.

It verifies that ``fuse_sensor_features`` keeps only common window keys and
creates the prefixed columns later selected by ``model_feature_columns``.
"""

from pyspark.sql import SparkSession

from har_spark.features import fuse_sensor_features, model_feature_columns


def test_sensor_fusion_joins_common_windows() -> None:
    """Verify inner fusion and sensor-prefixed feature columns."""
    spark = SparkSession.builder.master("local[1]").appName("har-fusion-test").getOrCreate()
    try:
        rows = [
            (1600, "A", "phone", "accel", 1, 100, 1.0),
            (1600, "A", "phone", "gyro", 1, 99, 2.0),
        ]
        features = spark.createDataFrame(
            rows, ["subject_id", "activity", "device", "sensor", "window_id", "sample_count", "x_mean"]
        )
        fused = fuse_sensor_features(features, ["accel", "gyro"]).collect()
        assert len(fused) == 1
        assert fused[0]["accel_x_mean"] == 1.0
        assert fused[0]["gyro_x_mean"] == 2.0
        assert model_feature_columns(spark.createDataFrame(fused)) == ["accel_x_mean", "gyro_x_mean"]
    finally:
        spark.stop()

