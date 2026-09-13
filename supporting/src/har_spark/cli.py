"""Command-line boundary for the local HAR Spark workflow.

The ``har-spark`` entry point declared in ``pyproject.toml`` loads one YAML file
through :mod:`har_spark.config` and dispatches to :mod:`har_spark.pipeline`.
``prepare`` writes reusable Parquet features, ``train`` consumes those features,
and ``run`` performs both stages.  This module contains argument routing only;
Spark transformations and ML logic remain in the package modules.
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import prepare_features, run_all, run_from_processed, write_features


def build_parser() -> argparse.ArgumentParser:
    """Define the global config option and the prepare/train/run subcommands. The parser
    has one shared --config option and three mutually exclusive subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml", help="YAML configuration path")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="Parse raw data and write window features")
    prepare.add_argument("--output", help="Override processed feature path")
    train = subcommands.add_parser("train", help="Train both models from prepared Parquet features")
    train.add_argument("--input", help="Override processed feature path")
    run = subcommands.add_parser("run", help="Prepare data, train both models, and evaluate")
    run.add_argument("--write-processed", action="store_true")
    return parser


def main() -> None:
    """Load the selected configuration and dispatch exactly one pipeline workflow. Loads
    and validates YAML first, then selects one orchestration path."""
    args = build_parser().parse_args()
    config = load_config(args.config)

    # Each branch owns the lifecycle it starts.  ``prepare`` stops its Spark
    # session here; the training workflows release cached data in ``pipeline``.
    if args.command == "prepare":
        features, summary = prepare_features(config)
        output = args.output or str(config.data["processed_path"])
        write_features(features, output)
        print(json.dumps({"output": output, **summary}, indent=2, sort_keys=True))
        features.unpersist()
        features.sparkSession.stop()
    elif args.command == "train":
        run_directory = run_from_processed(config, path=args.input)
        print(json.dumps({"run_directory": str(run_directory)}, indent=2))
    elif args.command == "run":
        run_directory = run_all(config, write_processed=bool(args.write_processed))
        print(json.dumps({"run_directory": str(run_directory)}, indent=2))


if __name__ == "__main__":
    main()
