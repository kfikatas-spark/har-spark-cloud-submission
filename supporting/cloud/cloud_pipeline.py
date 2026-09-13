"""Serverless Spark entry point for the recorded Google Cloud experiment.

Managed Service for Apache Spark launches this script with GCS input/output
arguments.  It reuses the package's ingestion, feature, split, and metric logic,
but keeps cloud-safe fixed model settings and writes distributed JSON-text
outputs to GCS.  Its results are later copied into ``cloud/evidence`` and used by
the report, presentations, plots, and static explorer.
"""

from __future__ import annotations

import argparse
import json
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from har_spark.features import (
    assign_subject_splits,
    build_window_features,
    fuse_sensor_features,
    model_feature_columns,
)
from har_spark.ingest import parse_wisdm_raw, valid_records
from har_spark.model import _confusion_and_macro_metrics


def arguments() -> argparse.Namespace:
    """Parse GCS paths and fixed experiment controls supplied to the cloud batch. Requires
    unique GCS input/output paths and exposes only the controls recorded for the
    experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", choices=["phone", "watch"], default="watch")
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--minimum-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def model_pipeline(model_name: str, feature_columns: list[str], seed: int) -> Pipeline:
    """Build the cloud classifier pipeline using the recorded hyperparameters. Mirrors the
    local label/vector/scale/classifier stages but fixes the measured cloud
    hyperparameters in the job artifact."""
    # These fixed values match the checked-in experiment configuration, allowing
    # local and managed-cloud results to be compared without a tuning difference.
    stages = [
        StringIndexer(
            inputCol="activity",
            outputCol="label",
            handleInvalid="error",
            stringOrderType="alphabetAsc",
        ),
        VectorAssembler(
            inputCols=feature_columns,
            outputCol="unscaled_features",
            handleInvalid="error",
        ),
        StandardScaler(
            inputCol="unscaled_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        ),
    ]
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            family="multinomial",
            maxIter=80,
            regParam=0.05,
            elasticNetParam=0.0,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            seed=seed,
            numTrees=80,
            maxDepth=12,
            featureSubsetStrategy="sqrt",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(stages=[*stages, classifier])


def write_json(spark: SparkSession, value: dict, path: str) -> None:
    """Write one JSON object through Spark to an error-on-exist GCS text directory.
    Serialises only a small aggregate object in the driver, creates a one-row Spark
    DataFrame, coalesces it to one partition, and writes an error-on-exist text
    directory."""
    payload = json.dumps(value, sort_keys=True)

    # Spark writes directories of part files.  Coalescing this tiny aggregate to
    # one partition makes later evidence collection simple without moving raw data
    # or prediction rows through the driver.
    spark.createDataFrame([(payload,)], ["value"]).coalesce(1).write.mode("errorifexists").text(path)


def main() -> None:
    """Run cloud preprocessing, train both models, and write all measured artifacts.
    Creates the managed Spark session, reuses package ingestion/features/split
    semantics, writes prepared features to a unique GCS prefix, caches the common
    splits, fits both models, evaluates test partitions, and writes models plus compact
    JSON evidence."""
    args = arguments()
    spark = SparkSession.builder.appName("WISDM-HAR-Cloud").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    started = time.perf_counter()

    # Reuse the package's data semantics; only orchestration and storage paths are
    # cloud-specific.  This avoids maintaining a second feature definition.
    parsed = parse_wisdm_raw(spark, args.raw_path).cache()
    parse_counts = {
        str(bool(row["_parse_ok"])).lower(): int(row["count"])
        for row in parsed.groupBy("_parse_ok").count().collect()
    }
    records = valid_records(parsed, args.device, ["accel", "gyro"])
    features = build_window_features(
        records,
        window_seconds=args.window_seconds,
        minimum_samples=args.minimum_samples,
    )
    features = fuse_sensor_features(features, ["accel", "gyro"])
    features = assign_subject_splits(
        features,
        seed=args.seed,
        train_fraction=0.70,
        validation_fraction=0.15,
    ).cache()
    feature_count = features.count()
    preprocessing_seconds = time.perf_counter() - started
    features.write.mode("errorifexists").parquet(f"{args.output_path}/features")
    parsed.unpersist()

    feature_columns = model_feature_columns(features)
    train = features.where(F.col("split") == "train").cache()
    validation = features.where(F.col("split") == "validation").cache()
    test = features.where(F.col("split") == "test").cache()
    split_counts = {
        "train": train.count(),
        "validation": validation.count(),
        "test": test.count(),
    }
    summary = {
        "raw_path": args.raw_path,
        "device": args.device,
        "sensors": ["accel", "gyro"],
        "window_seconds": args.window_seconds,
        "minimum_samples": args.minimum_samples,
        "parse_counts": parse_counts,
        "feature_windows": feature_count,
        "feature_columns": feature_columns,
        "preprocessing_seconds": preprocessing_seconds,
        "split_counts": split_counts,
        "subjects": features.select("subject_id").distinct().count(),
        "activities": features.select("activity").distinct().count(),
    }
    write_json(spark, summary, f"{args.output_path}/data_summary_json")

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    comparison = {}
    # Fit both classifiers against the same cached splits for a fair comparison.
    for model_name in ("logistic_regression", "random_forest"):
        started = time.perf_counter()
        fitted = model_pipeline(model_name, feature_columns, args.seed).fit(train)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = fitted.transform(test).select("activity", "label", "prediction").cache()
        prediction_count = predictions.count()
        inference_seconds = time.perf_counter() - started
        labels = list(fitted.stages[0].labels)
        confusion, macro = _confusion_and_macro_metrics(predictions, labels)
        metrics = {
            "model": model_name,
            "labels": labels,
            "feature_columns": feature_columns,
            "split_counts": split_counts,
            "test_predictions": prediction_count,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "predictions_per_second": prediction_count / inference_seconds,
            "accuracy": evaluator.setMetricName("accuracy").evaluate(predictions),
            "weighted_f1": evaluator.setMetricName("f1").evaluate(predictions),
            **macro,
        }
        fitted.write().save(f"{args.output_path}/{model_name}/model")
        write_json(spark, metrics, f"{args.output_path}/{model_name}/metrics_json")
        write_json(
            spark,
            {"model": model_name, "rows": confusion},
            f"{args.output_path}/{model_name}/confusion_json",
        )
        comparison[model_name] = metrics
        predictions.unpersist()

    write_json(spark, comparison, f"{args.output_path}/comparison_json")
    train.unpersist()
    validation.unpersist()
    test.unpersist()
    features.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
