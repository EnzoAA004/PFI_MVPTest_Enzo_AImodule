from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import nbformat
import pytest

from pfi_ai_service.training.rsna_preflight import (
    CONDITIONS,
    EXCLUDED_FROM_RSNA_TRAINING,
    LEVELS,
    SEVERITY,
    assert_no_official_test_access,
    normalize_condition,
    normalize_level,
    normalize_series_description,
    normalize_severity,
    sha256_file,
    verify_report_has_no_full_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "53_P10_6_RSNA_dataset_preflight.ipynb"
SCOPE_PATH = REPO_ROOT / "configs" / "p10_6_rsna_scope_v1.yaml"
DOC_PATH = REPO_ROOT / "docs" / "P10_6_RSNA_FINDINGS_SCOPE.md"


def _extract_yaml_list(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == f"{key}:")
    values: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" ") and not line.startswith("-"):
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            values.append(stripped[2:])
    return values


def test_notebook_json_is_valid_clean_and_has_unique_ids() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)

    ids = [cell.get("id") for cell in notebook.cells]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None, index
        assert cell.get("outputs") == [], index
        ast.parse(cell.get("source", ""), filename=f"{NOTEBOOK_PATH}:cell{index}")


def test_notebook_has_no_tokens_windows_paths_or_forbidden_test_reads() -> None:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    forbidden = [
        "access_token",
        "client_secret",
        "BEGIN PRIVATE KEY",
        "kaggle.json",
        "C:\\\\",
        "C:/Users/",
        "sample_submission.csv",
    ]
    for item in forbidden:
        assert item not in text

    forbidden_test_read_patterns = [
        r"test_images[^\\n]*(glob|rglob|walk|listdir|dcmread|read_csv)",
        r"test_series_descriptions[^\\n]*read_csv",
        r"sample_submission[^\\n]*read_csv",
    ]
    for pattern in forbidden_test_read_patterns:
        assert not re.search(pattern, text, flags=re.IGNORECASE)


def test_scope_yaml_contains_exact_governance_and_label_space() -> None:
    text = SCOPE_PATH.read_text(encoding="utf-8")
    assert _extract_yaml_list(text, "conditions") == CONDITIONS
    assert _extract_yaml_list(text, "levels") == LEVELS
    assert _extract_yaml_list(text, "severity") == SEVERITY
    excluded = set(_extract_yaml_list(text, "excludedFromRsnaTraining"))
    assert {"disc_protrusion", "disc_extrusion"}.issubset(excluded)
    assert EXCLUDED_FROM_RSNA_TRAINING.issubset(excluded)
    assert "  humanReviewRequired: true" in text
    assert "  notClinicalDiagnosis: true" in text
    assert "  autonomousDiagnosis: false" in text
    assert "  unit: study_id" in text
    assert "  testPolicy: official_test_forbidden" in text
    assert "  seed: 2026" in text


def test_documentation_uses_candidate_review_language_and_excludes_rsna_disc_targets() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "hallazgo candidato" in text
    assert "clasificacion asistida" in text
    assert "requiere revision profesional" in text
    assert "disc_protrusion" in text
    assert "disc_extrusion" in text
    assert "no se entrenan con RSNA" in text
    for forbidden in [
        "diagnostico automatico",
        "patologia confirmada",
        "lesion detectada definitivamente",
    ]:
        assert forbidden not in text


def test_label_normalizers_are_explicit_and_reversible_enough_for_rsna_names() -> None:
    assert normalize_condition("Spinal Canal Stenosis") == "spinal_canal_stenosis"
    assert normalize_condition("Left Neural Foraminal Narrowing") == "neural_foraminal_narrowing_left"
    assert normalize_condition("Right Neural Foraminal Narrowing") == "neural_foraminal_narrowing_right"
    assert normalize_condition("Left Subarticular Stenosis") == "subarticular_stenosis_left"
    assert normalize_condition("Right Subarticular Stenosis") == "subarticular_stenosis_right"
    assert normalize_level("L4/L5") == "L4-L5"
    assert normalize_level("l5_s1") == "L5-S1"
    assert normalize_severity("Normal/Mild") == "normal_mild"
    assert normalize_severity("Moderate") == "moderate"
    assert normalize_severity("Severe") == "severe"


def test_series_description_normalizer_does_not_classify_from_single_letters() -> None:
    assert normalize_series_description("Sagittal T1")["sequenceCategory"] == "sagittal_t1"
    assert normalize_series_description("Sagittal T2/STIR")["sequenceCategory"] == "sagittal_t2_stir"
    assert normalize_series_description("AXIAL T2")["sequenceCategory"] == "axial_t2"
    assert normalize_series_description("localizer A T")["sequenceCategory"] == "unknown"


def test_hashes_are_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("study_id,x\n1,a\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected
    assert sha256_file(path) == expected


def test_report_sanitizer_rejects_full_numeric_ids() -> None:
    assert verify_report_has_no_full_ids("study count: 2\nseries: opaque_abcd")
    assert not verify_report_has_no_full_ids("study_id 123456789")


def test_official_test_access_guard_fails_closed() -> None:
    assert_no_official_test_access(["train_images/1/2/1.dcm"])
    with pytest.raises(RuntimeError):
        assert_no_official_test_access(["test_images/1/2/1.dcm"])
    with pytest.raises(RuntimeError):
        assert_no_official_test_access(["sample_submission.csv"])


def test_preliminary_contract_avoids_diagnostic_fields() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    contract_text = text[text.index('"findingId"') : text.index("```", text.index('"findingId"'))]
    contract = json.loads("{" + contract_text.split("{", 1)[1])
    assert contract["humanReviewRequired"] is True
    assert contract["notClinicalDiagnosis"] is True
    assert contract["status"] == "requires_professional_review"
    for forbidden in ["diagnosis", "confirmedPathology", "treatment", "medicalConclusion", "clinicalRecommendation"]:
        assert forbidden not in contract


def test_synthetic_preflight_optional_when_medical_dependencies_exist(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("pydicom")

    from pfi_ai_service.training.rsna_preflight import RsnaPreflightConfig, run_preflight

    config = RsnaPreflightConfig(
        pfi_root=tmp_path / "pfi",
        rsna_root=tmp_path / "placeholder",
        train_images=tmp_path / "placeholder" / "train_images",
        output_root=tmp_path / "outputs",
        model_root=tmp_path / "models",
        synthetic=True,
        dicom_hash_opt_in=False,
    )
    summary = run_preflight(config)
    assert summary["syntheticMode"] is True
    assert summary["officialTestPresent"] is True
    assert summary["officialTestAccessed"] is False
    assert summary["nStudies"] == 2
    assert summary["nSeries"] == 6
    assert summary["nDicom"] == 12
    assert Path(summary["outputs"]["rsna_preflight_report.md"]).exists()
