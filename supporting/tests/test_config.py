"""Contract test for the main local YAML configuration.

This fast test exercises ``config.load_config`` without starting Spark.  It
guards the same sensor and split assumptions consumed by the CLI and pipeline.
"""

from pathlib import Path

from har_spark.config import load_config


def test_local_configuration_is_valid() -> None:
    """Verify the checked-in local config satisfies core project assumptions."""
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "local.yaml")
    assert config.data["window_seconds"] == 5
    assert config.models["logistic_regression"]["max_iter"] > 0

