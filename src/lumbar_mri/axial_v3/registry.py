"""Experiment registry schema for axial v3."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ALLOWED_REGISTRY_STATUSES = {
    "planned",
    "preflight_passed",
    "smoke_completed",
    "running",
    "completed",
    "failed",
    "discarded",
    "validation_candidate",
    "blocked",
}
REGISTRY_COLUMNS = [
    "experimentId",
    "iteration",
    "experimentType",
    "runId",
    "createdAtUtc",
    "updatedAtUtc",
    "gitCommit",
    "aiServiceCommit",
    "seed",
    "configPath",
    "configSha256",
    "splitSha256",
    "trainingStatus",
    "smokeOnly",
    "selectedEpoch",
    "monitorMetric",
    "validationDiceMacroForeground",
    "validationRaw0Dice",
    "validationRaw0Precision",
    "validationRaw0Recall",
    "validationRaw0FalsePositivePixels",
    "validationRaw0PredictedInGtAbsentCases",
    "validationDiceMacroExcludingRaw0",
    "guardrailPassed",
    "artifactPath",
    "artifactSha256",
    "checkpointPath",
    "checkpointSha256",
    "durationSeconds",
    "notes",
]


@dataclass(frozen=True)
class ExperimentRegistryRow:
    experimentId: str
    iteration: str
    experimentType: str
    runId: str
    createdAtUtc: str
    updatedAtUtc: str
    gitCommit: str
    aiServiceCommit: str
    seed: int
    configPath: str
    configSha256: str
    splitSha256: str
    trainingStatus: str
    smokeOnly: bool
    selectedEpoch: int | None
    monitorMetric: str
    validationDiceMacroForeground: float | None = None
    validationRaw0Dice: float | None = None
    validationRaw0Precision: float | None = None
    validationRaw0Recall: float | None = None
    validationRaw0FalsePositivePixels: int | None = None
    validationRaw0PredictedInGtAbsentCases: int | None = None
    validationDiceMacroExcludingRaw0: float | None = None
    guardrailPassed: bool | None = None
    artifactPath: str = ""
    artifactSha256: str = ""
    checkpointPath: str = ""
    checkpointSha256: str = ""
    durationSeconds: float | None = None
    notes: str = ""


def registry_row_from_config(config: dict[str, Any]) -> ExperimentRegistryRow:
    missing = [name for name in REGISTRY_COLUMNS[:16] if name not in config]
    if missing:
        raise ValueError(f"missing registry fields: {missing}")
    status = str(config["trainingStatus"])
    if status not in ALLOWED_REGISTRY_STATUSES:
        raise ValueError(f"invalid registry status: {status}")
    payload = {column: config.get(column) for column in REGISTRY_COLUMNS}
    return ExperimentRegistryRow(**payload)


def append_registry_row(path: Path, row: ExperimentRegistryRow) -> None:
    validate_registry_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def validate_registry_row(row: ExperimentRegistryRow) -> None:
    if row.trainingStatus not in ALLOWED_REGISTRY_STATUSES:
        raise ValueError(f"invalid registry status: {row.trainingStatus}")


def read_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_registry_atomic(path: Path, rows: list[ExperimentRegistryRow]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        validate_registry_row(row)
        key = (row.experimentId, row.runId)
        if key in seen:
            raise ValueError(f"duplicate registry key: {key}")
        seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    os.replace(temp_path, path)


def upsert_registry_row(path: Path, row: ExperimentRegistryRow) -> None:
    existing = read_registry(path)
    rows: list[ExperimentRegistryRow] = []
    replaced = False
    for payload in existing:
        if payload.get("experimentId") == row.experimentId and payload.get("runId") == row.runId:
            rows.append(row)
            replaced = True
        else:
            rows.append(registry_row_from_config(payload))
    if not replaced:
        rows.append(row)
    write_registry_atomic(path, rows)
