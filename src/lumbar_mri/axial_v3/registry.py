"""Experiment registry schema for axial v3."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REGISTRY_COLUMNS = [
    "experimentId",
    "iteration",
    "runId",
    "createdAtUtc",
    "gitCommit",
    "seed",
    "configPath",
    "splitSha256",
    "trainingStatus",
    "selectedEpoch",
    "monitorMetric",
    "validationDiceMacroForeground",
    "validationRaw0Dice",
    "validationRaw0Precision",
    "validationRaw0Recall",
    "validationRaw0FalsePositivePixels",
    "validationRaw0PredictedInGtAbsentCases",
    "validationDiceMacroExcludingRaw0",
    "artifactPath",
    "artifactSha256",
    "notes",
]


@dataclass(frozen=True)
class ExperimentRegistryRow:
    experimentId: str
    iteration: str
    runId: str
    createdAtUtc: str
    gitCommit: str
    seed: int
    configPath: str
    splitSha256: str
    trainingStatus: str
    selectedEpoch: int | None
    monitorMetric: str
    validationDiceMacroForeground: float | None = None
    validationRaw0Dice: float | None = None
    validationRaw0Precision: float | None = None
    validationRaw0Recall: float | None = None
    validationRaw0FalsePositivePixels: int | None = None
    validationRaw0PredictedInGtAbsentCases: int | None = None
    validationDiceMacroExcludingRaw0: float | None = None
    artifactPath: str = ""
    artifactSha256: str = ""
    notes: str = ""


def registry_row_from_config(config: dict[str, Any]) -> ExperimentRegistryRow:
    missing = [name for name in REGISTRY_COLUMNS[:11] if name not in config]
    if missing:
        raise ValueError(f"missing registry fields: {missing}")
    payload = {column: config.get(column) for column in REGISTRY_COLUMNS}
    return ExperimentRegistryRow(**payload)


def append_registry_row(path: Path, row: ExperimentRegistryRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))
