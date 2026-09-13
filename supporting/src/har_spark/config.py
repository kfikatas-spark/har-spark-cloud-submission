"""Load and validate the YAML contract shared by the local pipeline.

Every file in ``configs/`` has the six sections exposed by :class:`AppConfig`.
The CLI calls :func:`load_config` before creating Spark, so malformed split ratios,
window settings, or sensor choices fail early.  Downstream modules receive the
same immutable wrapper rather than reparsing YAML or duplicating validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Immutable wrapper exposing the validated top-level YAML sections. The wrapper is
    frozen so code cannot accidentally replace its two attributes."""
    raw: dict[str, Any]
    source_path: Path

    @property
    def project(self) -> dict[str, Any]:
        """Return reproducibility settings such as the project name and seed."""
        return self.raw["project"]

    @property
    def spark(self) -> dict[str, Any]:
        """Return Spark session and shuffle settings."""
        return self.raw["spark"]

    @property
    def data(self) -> dict[str, Any]:
        """Return raw-input, sensor, window, and processed-output settings."""
        return self.raw["data"]

    @property
    def split(self) -> dict[str, Any]:
        """Return participant-level train, validation, and test fractions."""
        return self.raw["split"]

    @property
    def models(self) -> dict[str, Any]:
        """Return hyperparameters for both MLlib classifiers."""
        return self.raw["models"]

    @property
    def output(self) -> dict[str, Any]:
        """Return the root directory for timestamped experiment results."""
        return self.raw["output"]


def load_config(path: str | Path) -> AppConfig:
    """Parse one YAML file, validate the shared contract, and return ``AppConfig``. Expands
    ~, resolves an absolute source path, safely parses YAML, checks all required
    sections, requires positive split fractions summing to one, verifies window/sample
    limits, and accepts only unique accel/gyro sensor names."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    # Validate the stable top-level contract before any downstream module tries
    # to access a section by name.
    required = {"project", "spark", "data", "split", "models", "output"}
    missing = sorted(required.difference(raw or {}))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    # Participant fractions—not row fractions—must form a complete partition.
    fractions = [
        float(raw["split"]["train_fraction"]),
        float(raw["split"]["validation_fraction"]),
        float(raw["split"]["test_fraction"]),
    ]
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("All split fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    if int(raw["data"]["window_seconds"]) <= 0:
        raise ValueError("window_seconds must be positive")
    if int(raw["data"]["minimum_samples_per_window"]) <= 1:
        raise ValueError("minimum_samples_per_window must be greater than one")
    # ``sensor`` is retained as a backwards-compatible one-sensor spelling;
    # current fusion configurations use the explicit ``sensors`` list.
    sensors = raw["data"].get("sensors", [raw["data"].get("sensor")])
    if not sensors or any(sensor not in {"accel", "gyro"} for sensor in sensors):
        raise ValueError("data.sensors must contain accel, gyro, or both")
    if len(set(sensors)) != len(sensors):
        raise ValueError("data.sensors cannot contain duplicates")

    return AppConfig(raw=raw, source_path=source)
