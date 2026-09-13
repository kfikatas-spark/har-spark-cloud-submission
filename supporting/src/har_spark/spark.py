"""Construct the Spark session used by local package entry points.

Settings come from ``AppConfig.spark``.  ``pipeline.prepare_features`` and
``pipeline.run_from_processed`` call this helper so application name, shuffle
parallelism, driver memory, local/cluster master, and UTC handling are applied
consistently.  The managed cloud job creates its own session in its entry script.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from .config import AppConfig


def build_spark(config: AppConfig) -> SparkSession:
    """Create or reuse a Spark session configured from the validated YAML settings.
    Configures application name, shuffle partition count, driver memory, and UTC
    timestamps."""
    settings = config.spark
    builder = (
        SparkSession.builder.appName(str(settings["app_name"]))
        .config("spark.sql.shuffle.partitions", str(settings["shuffle_partitions"]))
        .config("spark.driver.memory", str(settings["driver_memory"]))
        .config("spark.sql.session.timeZone", "UTC")
    )
    # Local YAML files provide ``local[*]``.  Cluster launchers omit ``master``
    # so the environment, rather than application code, selects the cluster.
    master = settings.get("master")
    if master:
        builder = builder.master(str(master))
    return builder.getOrCreate()
