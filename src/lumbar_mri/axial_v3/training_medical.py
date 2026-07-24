"""Axial v3 training facade with versioned medical-image I/O installed."""

from __future__ import annotations

from . import training as _training
from .medical_io import open_2d_array


# Training datasets resolve ``open_2d_array`` from the training module globals.
# Installing the versioned function here keeps the runtime behavior tied to the
# checked-out git commit instead of defining an untracked notebook function.
_training.open_2d_array = open_2d_array

AxialV3TrainConfig = _training.AxialV3TrainConfig
run_calibration = _training.run_calibration
run_preflight = _training.run_preflight
run_training = _training.run_training
summarize_validation_runs = _training.summarize_validation_runs


def assert_medical_io_installed() -> None:
    if _training.open_2d_array is not open_2d_array:
        raise RuntimeError("versioned axial medical I/O was not installed")


__all__ = [
    "AxialV3TrainConfig",
    "assert_medical_io_installed",
    "open_2d_array",
    "run_calibration",
    "run_preflight",
    "run_training",
    "summarize_validation_runs",
]
