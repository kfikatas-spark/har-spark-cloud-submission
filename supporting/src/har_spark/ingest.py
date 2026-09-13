"""Parse and validate the raw WISDM text format with Spark expressions.

``pipeline.prepare_features`` and the cloud entry point pass file globs here.
The parser derives device/sensor metadata from source paths, casts the six raw
fields, and retains an ``_parse_ok`` flag for quality reporting.  The filtered,
typed rows then flow to :mod:`har_spark.features`; no feature engineering occurs
in this module.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def parse_wisdm_raw(spark: SparkSession, path: str) -> DataFrame:
    """Parse `<subject,activity,timestamp,x,y,z;>` records. spark.read.text creates a
    distributed row for each input line."""

    # Device and sensor are encoded in the WISDM directory names, not in each
    # CSV-like record, so retain the source path while parsing the line itself.
    raw = spark.read.text(path).withColumn("source_file", F.input_file_name())
    cleaned = F.regexp_replace(F.trim(F.col("value")), r";\s*$", "")
    parts = F.split(cleaned, ",")
    source = F.lower(F.col("source_file"))

    parsed = raw.select(
        F.col("value").alias("raw_line"),
        F.col("source_file"),
        parts.alias("parts"),
        F.when(source.contains("/phone/"), F.lit("phone"))
        .when(source.contains("/watch/"), F.lit("watch"))
        .otherwise(F.lit("unknown"))
        .alias("device"),
        F.when(source.contains("/accel/"), F.lit("accel"))
        .when(source.contains("/gyro/"), F.lit("gyro"))
        .otherwise(F.lit("unknown"))
        .alias("sensor"),
    ).select(
        "raw_line",
        "source_file",
        "device",
        "sensor",
        F.size("parts").alias("field_count"),
        F.col("parts").getItem(0).cast("int").alias("subject_id"),
        F.trim(F.col("parts").getItem(1)).alias("activity"),
        F.col("parts").getItem(2).cast("long").alias("timestamp_ns"),
        F.col("parts").getItem(3).cast("double").alias("x"),
        F.col("parts").getItem(4).cast("double").alias("y"),
        F.col("parts").getItem(5).cast("double").alias("z"),
    )

    valid_activity = F.col("activity").rlike("^[A-S]$") & (F.col("activity") != "N")
    required_not_null = (
        F.col("subject_id").isNotNull()
        & F.col("timestamp_ns").isNotNull()
        & F.col("x").isNotNull()
        & F.col("y").isNotNull()
        & F.col("z").isNotNull()
    )
    # Keep invalid rows with an explicit flag until the caller has counted them.
    # This makes data-quality reporting possible without reparsing the input.
    return parsed.withColumn(
        "_parse_ok",
        (F.col("field_count") == 6)
        & valid_activity
        & required_not_null
        & (F.col("timestamp_ns") >= 0)
        & F.col("device").isin("phone", "watch")
        & F.col("sensor").isin("accel", "gyro"),
    )


def valid_records(parsed: DataFrame, device: str, sensors: list[str]) -> DataFrame:
    """Keep valid rows for one device and the requested sensors, dropping parser metadata.
    Applies a distributed where filter for _parse_ok, device, and requested sensors,
    then selects only analysis columns."""
    return parsed.where(
        F.col("_parse_ok")
        & (F.col("device") == F.lit(device))
        & F.col("sensor").isin(sensors)
    ).select("subject_id", "activity", "timestamp_ns", "device", "sensor", "x", "y", "z")
