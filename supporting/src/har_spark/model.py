"""Build, fit, evaluate, and persist the two local MLlib classifiers.

Input is the participant-split window DataFrame produced by ``features``.
Each fitted Spark pipeline learns its label mapping and scaler from train rows,
then predicts only the test rows.  This module writes the model, aggregate
metrics, per-class metrics, and confusion counts used by plots, reports, slides,
and the results explorer.  ``pipeline._run_models`` calls the public entry point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .features import model_feature_columns


def _pipeline(
    model_name: str, settings: dict[str, Any], seed: int, feature_columns: list[str]
) -> Pipeline:
    """Assemble label indexing, vectorisation, scaling, and the selected classifier.
    StringIndexer learns the alphabetic label mapping, VectorAssembler packs scalar
    feature columns into one vector, StandardScaler learns mean/std, and the chosen
    classifier learns model parameters."""
    # Because the complete Pipeline is fitted on ``train`` below, both the label
    # order and scaling statistics are learned without seeing test participants.
    label_indexer = StringIndexer(
        inputCol="activity",
        outputCol="label",
        handleInvalid="error",
        stringOrderType="alphabetAsc",
    )
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="unscaled_features",
        handleInvalid="error",
    )
    scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            family="multinomial",
            maxIter=int(settings["max_iter"]),
            regParam=float(settings["reg_param"]),
            elasticNetParam=float(settings["elastic_net_param"]),
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            seed=int(seed),
            numTrees=int(settings["num_trees"]),
            maxDepth=int(settings["max_depth"]),
            featureSubsetStrategy=str(settings["feature_subset_strategy"]),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(stages=[label_indexer, assembler, scaler, classifier])


def _confusion_and_macro_metrics(predictions: DataFrame, labels: list[str]) -> tuple[list[dict], dict]:
    """Collect compact confusion counts and derive per-class and macro metrics. Spark first
    groups predictions into at most C×C count cells."""
    # Aggregate in Spark before collect: the driver receives at most C×C rows
    # (18×18 here), never the complete prediction DataFrame.
    rows = (
        predictions.select(
            F.col("label").cast("int").alias("actual_index"),
            F.col("prediction").cast("int").alias("predicted_index"),
        )
        .groupBy("actual_index", "predicted_index")
        .count()
        .collect()
    )
    counts = {
        (int(row["actual_index"]), int(row["predicted_index"])): int(row["count"])
        for row in rows
    }
    confusion: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []

    for actual_index, actual_name in enumerate(labels):
        for predicted_index, predicted_name in enumerate(labels):
            confusion.append(
                {
                    "actual": actual_name,
                    "predicted": predicted_name,
                    "count": counts.get((actual_index, predicted_index), 0),
                }
            )

        tp = counts.get((actual_index, actual_index), 0)
        fp = sum(counts.get((other, actual_index), 0) for other in range(len(labels)) if other != actual_index)
        fn = sum(counts.get((actual_index, other), 0) for other in range(len(labels)) if other != actual_index)
        support = sum(counts.get((actual_index, other), 0) for other in range(len(labels)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "activity": actual_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    macro = {
        "macro_precision": sum(item["precision"] for item in per_class) / len(per_class),
        "macro_recall": sum(item["recall"] for item in per_class) / len(per_class),
        "macro_f1": sum(item["f1"] for item in per_class) / len(per_class),
        "per_class": per_class,
    }
    return confusion, macro


def train_and_evaluate(
    features: DataFrame,
    *,
    model_name: str,
    settings: dict[str, Any],
    seed: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Fit one classifier on train rows, evaluate test rows, and persist its artifacts.
    Refuses an existing output directory, caches train/validation/test subsets, and
    counts each to materialise them and detect empty splits."""
    # A run directory is immutable evidence.  Refusing an existing path prevents
    # a later experiment from silently replacing models or measured metrics.
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=False)

    # Cache all three logical splits because counts and later actions reuse them.
    # Validation is recorded for experiment completeness; model selection is not
    # performed in this fixed two-model comparison.
    train = features.where(F.col("split") == "train").cache()
    validation = features.where(F.col("split") == "validation").cache()
    test = features.where(F.col("split") == "test").cache()
    split_counts = {
        "train": train.count(),
        "validation": validation.count(),
        "test": test.count(),
    }
    if min(split_counts.values()) == 0:
        raise ValueError(f"At least one data split is empty: {split_counts}")

    feature_columns = model_feature_columns(features)
    pipeline = _pipeline(model_name, settings, seed, feature_columns)
    started = time.perf_counter()
    fitted = pipeline.fit(train)
    training_seconds = time.perf_counter() - started

    # ``count`` materialises the lazy prediction plan, so inference timing covers
    # actual distributed work rather than only DataFrame construction.
    started = time.perf_counter()
    predictions = fitted.transform(test).select("activity", "label", "prediction").cache()
    prediction_count = predictions.count()
    inference_seconds = time.perf_counter() - started

    labels = list(fitted.stages[0].labels)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    confusion, macro = _confusion_and_macro_metrics(predictions, labels)
    metrics: dict[str, Any] = {
        "model": model_name,
        "feature_columns": feature_columns,
        "split_counts": split_counts,
        "labels": labels,
        "test_predictions": prediction_count,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "predictions_per_second": prediction_count / inference_seconds if inference_seconds else None,
        "accuracy": evaluator.setMetricName("accuracy").evaluate(predictions),
        "weighted_f1": evaluator.setMetricName("f1").evaluate(predictions),
        **macro,
    }

    fitted.write().save(str(output_directory / "model"))
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_directory / "confusion.json").open("w", encoding="utf-8") as handle:
        json.dump(confusion, handle, indent=2)
        handle.write("\n")

    predictions.unpersist()
    train.unpersist()
    validation.unpersist()
    test.unpersist()
    return metrics
