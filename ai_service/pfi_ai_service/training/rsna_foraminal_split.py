"""Reproducible study-level split for RSNA foraminal narrowing (P10.6-AI)."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SIDES = ("left", "right")
LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
SEVERITIES = ("normal_mild", "moderate", "severe")
SPLITS = ("train", "validation", "internal_test")
FRACTIONS = {"train": 0.70, "validation": 0.15, "internal_test": 0.15}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp"
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp"
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _primary_stratum(row: pd.Series) -> str:
    maximum = int(row.max_severity_code)
    severe = int(row.severe_count)
    abnormal = int(row.abnormal_count)
    severe_bucket = 0 if severe == 0 else 1 if severe == 1 else 2
    abnormal_bucket = 0 if abnormal == 0 else 1 if abnormal <= 2 else 2 if abnormal <= 5 else 3
    return f"m{maximum}_s{severe_bucket}_a{abnormal_bucket}"


def _support_rule(total: int) -> str:
    if total >= 6:
        return "all_three_splits"
    if total >= 3:
        return "train_and_at_least_two_splits"
    if total == 2:
        return "train_and_one_holdout"
    return "train_only"


def _support_passes(counts: np.ndarray, total: int) -> bool:
    train, validation, test = map(int, counts)
    nonzero = int(np.count_nonzero(counts))
    if total >= 6:
        return bool(np.all(counts > 0))
    if total >= 3:
        return train > 0 and nonzero >= 2
    if total == 2:
        return train > 0 and validation + test > 0
    return train > 0


def run_split(
    source_manifest: Path,
    source_summary: Path,
    common_study_splits: Path,
    output_root: Path,
    *,
    repo_ref: str,
    repo_sha: str,
    seed: int = 2026,
    candidate_count: int = 2000,
    fraction_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Create and export a leakage-free 70/15/15 foraminal split."""
    source_manifest = Path(source_manifest)
    source_summary = Path(source_summary)
    common_study_splits = Path(common_study_splits)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in (source_manifest, source_summary) if not path.is_file()]
    if missing:
        raise RuntimeError("Faltan artefactos del Notebook 57:\n- " + "\n- ".join(missing))

    summary57 = json.loads(source_summary.read_text(encoding="utf-8"))
    if summary57.get("approved") is not True or summary57.get("nextNotebook") != 58:
        raise RuntimeError("El Notebook 57 no habilita el Notebook 58.")
    if summary57.get("governance", {}).get("officialTestAccessed") is not False:
        raise RuntimeError("La fuente declara acceso al test oficial.")

    source_hash = sha256_file(source_manifest)
    expected_hash = summary57.get("outputSha256", {}).get("manifest")
    if expected_hash and source_hash != expected_hash:
        raise RuntimeError("El hash del manifiesto del Notebook 57 no coincide.")

    manifest = pd.read_csv(source_manifest, dtype={"study_id": str, "coordinate_series_id": str})
    required = {
        "study_id", "side", "level", "severity", "severity_code", "usable_for_split",
        "coordinate_series_id", "coordinate_instance_number", "coordinate_x", "coordinate_y",
    }
    absent = sorted(required - set(manifest.columns))
    if absent:
        raise RuntimeError(f"Faltan columnas requeridas: {absent}")

    manifest["study_id"] = manifest.study_id.astype(str)
    manifest["usable_for_split"] = _bool_series(manifest.usable_for_split)
    manifest["severity_code"] = pd.to_numeric(manifest.severity_code, errors="coerce")
    usable = manifest.loc[manifest.usable_for_split].copy()
    if usable.empty or usable.severity.isna().any() or usable.severity_code.isna().any():
        raise RuntimeError("El conjunto utilizable contiene etiquetas inválidas.")
    if usable[["study_id", "side", "level"]].duplicated().any():
        raise RuntimeError("Hay claves study_id-side-level duplicadas.")
    if not usable.side.isin(SIDES).all() or not usable.level.isin(LEVELS).all():
        raise RuntimeError("Lado o nivel inesperado.")
    if not usable.severity.isin(SEVERITIES).all():
        raise RuntimeError("Severidad inesperada.")

    usable["label_key"] = usable.side + "__" + usable.level + "__" + usable.severity
    label_keys = [f"{side}__{level}__{severity}" for side in SIDES for level in LEVELS for severity in SEVERITIES]
    matrix = (
        usable.assign(present=1)
        .pivot_table(index="study_id", columns="label_key", values="present", aggfunc="max", fill_value=0)
        .reindex(columns=label_keys, fill_value=0)
        .astype(np.int8)
        .sort_index()
    )
    if set(matrix.columns[matrix.sum(axis=0) > 0]) != set(label_keys):
        raise RuntimeError("No están presentes los 30 estratos esperados.")

    stats = usable.groupby("study_id").agg(
        max_severity_code=("severity_code", "max"),
        severe_count=("severity_code", lambda values: int((values == 2).sum())),
        moderate_count=("severity_code", lambda values: int((values == 1).sum())),
    ).reindex(matrix.index)
    stats["abnormal_count"] = stats.severe_count + stats.moderate_count
    stats["primary_stratum"] = stats.apply(_primary_stratum, axis=1)
    if int(stats.primary_stratum.value_counts().min()) < 6:
        raise RuntimeError("Un estrato primario no admite el split estratificado.")

    study_ids = matrix.index.to_numpy(dtype=str)
    y = matrix.to_numpy(dtype=np.int8)
    totals = y.sum(axis=0).astype(int)
    names = np.array(matrix.columns, dtype=object)
    positions = np.arange(len(study_ids))
    primary = stats.loc[study_ids, "primary_stratum"].to_numpy()
    target = np.array([FRACTIONS[name] for name in SPLITS])

    def count_matrix(assignment: pd.Series) -> np.ndarray:
        values = assignment.loc[study_ids].astype(str).to_numpy()
        return np.stack([y[values == name].sum(axis=0) for name in SPLITS], axis=1).astype(int)

    def evaluate(counts: np.ndarray) -> tuple[float, dict[str, Any], pd.DataFrame]:
        expected = totals[:, None] * target[None, :]
        weights = (1.0 / np.maximum(np.sqrt(totals), 1.0)) * np.where(
            np.char.endswith(names.astype(str), "__severe"), 2.0, 1.0
        )
        base = float(np.mean(
            np.abs(counts - expected) / np.maximum(np.sqrt(totals[:, None]), 1.0) * weights[:, None]
        ))
        rows = []
        for index, label in enumerate(names):
            total = int(totals[index])
            passed = _support_passes(counts[index], total)
            rows.append({
                "label_key": str(label), "total_support": total,
                "train": int(counts[index, 0]), "validation": int(counts[index, 1]),
                "internal_test": int(counts[index, 2]), "support_rule": _support_rule(total),
                "support_rule_passed": bool(passed),
            })
        report = pd.DataFrame(rows)
        failures = int((~report.support_rule_passed).sum())
        return base + failures * 1000.0, {
            "baseDistributionScore": base,
            "failedSupportRules": failures,
            "supportRulesPassed": failures == 0,
        }, report

    def gates(assignment: pd.Series, counts: np.ndarray, details: dict[str, Any]) -> dict[str, Any]:
        assignment = assignment.loc[study_ids].astype(str)
        fractions = assignment.value_counts() / len(assignment)
        class_coverage = {}
        for severity in SEVERITIES:
            columns = [i for i, name in enumerate(names) if str(name).endswith(f"__{severity}")]
            class_coverage[severity] = all(int(counts[columns, split_index].sum()) > 0 for split_index in range(3))
        return {
            "allEligibleStudiesAssigned": len(assignment) == len(study_ids) and not assignment.index.duplicated().any(),
            "validSplitNames": set(assignment.unique()) == set(SPLITS),
            "fractionTolerance": all(abs(float(fractions.get(name, 0.0)) - FRACTIONS[name]) <= fraction_tolerance for name in SPLITS),
            "supportRulesPassed": bool(details["supportRulesPassed"]),
            "allSeverityClassesInEverySplit": bool(all(class_coverage.values())),
            "globalSeverityCoverage": class_coverage,
        }

    def audit(assignment: pd.Series) -> dict[str, Any]:
        assignment = assignment.loc[assignment.index.intersection(study_ids)].copy()
        if set(assignment.index) != set(study_ids):
            return {"available": True, "complete": False, "approved": False}
        counts = count_matrix(assignment)
        score, details, report = evaluate(counts)
        result_gates = gates(assignment, counts, details)
        approved = all(value for key, value in result_gates.items() if key != "globalSeverityCoverage")
        return {
            "available": True, "complete": True, "approved": bool(approved),
            "score": float(score), "scoreDetails": details, "gates": result_gates,
            "countMatrix": counts, "rareReport": report,
        }

    common_assignment = None
    common_audit: dict[str, Any] = {"available": False, "approved": False, "reason": "not_found"}
    if common_study_splits.is_file():
        common = pd.read_csv(common_study_splits, dtype={"study_id": str})
        if {"study_id", "split"}.issubset(common.columns) and not common.study_id.duplicated().any():
            common_assignment = common.set_index("study_id").split.astype(str)
            common_audit = audit(common_assignment)
        else:
            common_audit = {"available": True, "approved": False, "reason": "invalid_common_split"}

    search = {"performed": False, "candidateCount": 0, "approvedCandidateCount": 0, "bestSeed": None, "bestScore": None}
    if common_audit.get("approved") is True:
        selected = common_assignment.loc[study_ids].copy()
        selected_audit = common_audit
        policy = "reused_notebook54_common_split"
    else:
        policy = "task_specific_multilabel_search"
        search["performed"] = True
        best = None
        for offset in range(candidate_count):
            candidate_seed = seed + offset
            train_pos, holdout_pos = train_test_split(
                positions, test_size=0.30, random_state=candidate_seed, stratify=primary
            )
            validation_pos, test_pos = train_test_split(
                holdout_pos, test_size=0.50, random_state=candidate_seed + 100_000,
                stratify=primary[holdout_pos],
            )
            values = np.empty(len(study_ids), dtype=object)
            values[train_pos], values[validation_pos], values[test_pos] = "train", "validation", "internal_test"
            assignment = pd.Series(values, index=study_ids, name="split")
            counts = count_matrix(assignment)
            score, details, report = evaluate(counts)
            result_gates = gates(assignment, counts, details)
            candidate_approved = all(value for key, value in result_gates.items() if key != "globalSeverityCoverage")
            search["candidateCount"] += 1
            if not candidate_approved:
                continue
            search["approvedCandidateCount"] += 1
            if best is None or score < best["score"]:
                best = {"seed": candidate_seed, "score": score, "assignment": assignment,
                        "countMatrix": counts, "scoreDetails": details, "rareReport": report,
                        "gates": result_gates}
        if best is None:
            raise RuntimeError("No se encontró un split que supere los gates.")
        selected = best["assignment"]
        selected_audit = {"available": True, "complete": True, "approved": True, **best}
        search["bestSeed"], search["bestScore"] = int(best["seed"]), float(best["score"])

    eligible = selected.rename("split").rename_axis("study_id").reset_index()
    eligible["eligible_for_model"], eligible["exclusion_reason"] = True, ""
    all_studies = pd.Index(sorted(manifest.study_id.unique()), name="study_id")
    excluded_ids = all_studies.difference(eligible.study_id)
    excluded_assignments = pd.DataFrame({
        "study_id": excluded_ids.astype(str), "split": "excluded_no_usable_rows",
        "eligible_for_model": False, "exclusion_reason": "no_usable_foraminal_targets",
    })
    assignments = pd.concat([eligible, excluded_assignments], ignore_index=True)
    assigned = usable.merge(eligible[["study_id", "split"]], on="study_id", how="left", validate="many_to_one")
    if assigned.split.isna().any():
        raise RuntimeError("Hay filas utilizables sin split.")
    assigned["internal_test_sealed"] = assigned.split.eq("internal_test")
    assigned["human_review_required"], assigned["not_clinical_diagnosis"] = True, True
    assigned["official_test_accessed"] = False
    split_frames = {name: assigned.loc[assigned.split.eq(name)].copy() for name in SPLITS}

    sets = {name: set(frame.study_id) for name, frame in split_frames.items()}
    leakage = any((sets["train"] & sets["validation"], sets["train"] & sets["internal_test"], sets["validation"] & sets["internal_test"]))
    duplicates = int(assigned[["study_id", "side", "level"]].duplicated().sum())
    rows_conserved = sum(len(frame) for frame in split_frames.values()) == len(usable)

    distribution = assigned.groupby(["side", "level", "severity", "split"]).size().reset_index(name="count")
    totals_frame = assigned.groupby(["side", "level", "severity"]).size().reset_index(name="total_support")
    distribution = distribution.merge(totals_frame, on=["side", "level", "severity"], validate="many_to_one")
    distribution["target_fraction"] = distribution.split.map(FRACTIONS)
    distribution["expected_count"] = distribution.total_support * distribution.target_fraction
    distribution["observed_fraction"] = distribution["count"] / distribution.total_support
    distribution["deviation_percentage_points"] = (distribution.observed_fraction - distribution.target_fraction) * 100

    rare = selected_audit["rareReport"].copy()
    rare[["side", "level", "severity"]] = rare.label_key.str.split("__", expand=True)
    coverage = pd.DataFrame([{
        "split": name, "studies": int(frame.study_id.nunique()), "rows": int(len(frame)),
        "study_fraction": frame.study_id.nunique() / len(study_ids), "row_fraction": len(frame) / len(assigned),
    } for name, frame in split_frames.items()])
    leakage_report = {
        "studyLeakageDetected": bool(leakage), "duplicateStudySideLevelRows": duplicates,
        "eligibleRowsConserved": bool(rows_conserved), "eligibleSourceRows": int(len(usable)),
        "assignedRows": int(len(assigned)), "internalTestSealed": True, "officialTestAccessed": False,
    }

    selected_gates = selected_audit["gates"]
    final_gates = {
        "sourceNotebook57Approved": True,
        "sourceManifestHashVerified": not expected_hash or source_hash == expected_hash,
        "allEligibleStudiesAssigned": bool(selected_gates["allEligibleStudiesAssigned"]),
        "validSplitNames": bool(selected_gates["validSplitNames"]),
        "fractionTolerance": bool(selected_gates["fractionTolerance"]),
        "supportRulesPassed": bool(selected_gates["supportRulesPassed"]),
        "allSeverityClassesInEverySplit": bool(selected_gates["allSeverityClassesInEverySplit"]),
        "noStudyLeakage": not leakage, "rowsConserved": bool(rows_conserved),
        "noDuplicateStudySideLevelRows": duplicates == 0, "internalTestSealed": True,
        "officialTestAccessed": False, "humanReviewRequired": True, "notClinicalDiagnosis": True,
    }
    approved = all(value if key != "officialTestAccessed" else value is False for key, value in final_gates.items())

    paths = {
        "trainManifest": output_root / "train_manifest.csv",
        "validationManifest": output_root / "validation_manifest.csv",
        "internalTestManifest": output_root / "internal_test_manifest.csv",
        "studyAssignments": output_root / "study_split_assignments.csv",
        "splitDistribution": output_root / "split_distribution.csv",
        "rareStrata": output_root / "rare_strata_report.csv",
        "splitCoverage": output_root / "split_coverage.csv",
        "leakageReport": output_root / "split_leakage_report.json",
        "summary": output_root / "split_summary.json",
        "report": output_root / "split_report.md",
    }
    for name, frame in split_frames.items():
        _write_csv(paths[{"train": "trainManifest", "validation": "validationManifest", "internal_test": "internalTestManifest"}[name]], frame)
    _write_csv(paths["studyAssignments"], assignments)
    _write_csv(paths["splitDistribution"], distribution)
    _write_csv(paths["rareStrata"], rare)
    _write_csv(paths["splitCoverage"], coverage)
    _write_json(paths["leakageReport"], leakage_report)

    common_export = {key: value for key, value in common_audit.items() if key not in {"countMatrix", "rareReport"}}
    result = {
        "schemaVersion": "pfi.rsna-foraminal-split.v1", "ticket": "P10.6-AI", "notebook": 58,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(), "repoRef": repo_ref, "repoSha": repo_sha,
        "dataset": "RSNA_LumbarDISC", "task": "neural_foraminal_narrowing", "sequence": "Sagittal T1",
        "sourceNotebook": 57, "sourceManifestSha256": source_hash, "splitPolicy": policy,
        "seed": seed, "fractions": FRACTIONS, "eligibleStudies": int(len(study_ids)),
        "excludedStudies": int(len(excluded_ids)), "eligibleRows": int(len(usable)),
        "splits": {name: {"studies": int(frame.study_id.nunique()), "rows": int(len(frame))} for name, frame in split_frames.items()},
        "commonSplitAudit": common_export, "candidateSearch": search,
        "selectedDistributionScore": float(selected_audit["score"]), "gateResults": final_gates,
        "approved": bool(approved), "nextNotebook": 59 if approved else None,
        "governance": {"commercialUse": False, "humanReviewRequired": True, "notClinicalDiagnosis": True,
                       "autonomousDiagnosis": False, "officialTestAccessed": False, "internalTestSealed": True},
    }
    report = [
        "# P10.6-AI — Split foraminal RSNA", "",
        f"- Estado: `{'APPROVED_FOR_NOTEBOOK_59' if approved else 'SPLIT_REVIEW_REQUIRED'}`",
        f"- Política: `{policy}`", f"- Estudios elegibles: {len(study_ids)}",
        f"- Filas elegibles: {len(usable)}", "", "## Cohortes", "",
    ]
    report += [f"- {name}: {frame.study_id.nunique()} estudios, {len(frame)} filas" for name, frame in split_frames.items()]
    report += ["", "## Gates", ""] + [f"- {name}: `{str(value).lower()}`" for name, value in final_gates.items()]
    report += ["", "El internal test queda sellado y no debe usarse durante el Notebook 59.", ""]
    _write_text(paths["report"], "\n".join(report))
    result["outputSha256"] = {name: sha256_file(path) for name, path in paths.items() if path.is_file() and name != "summary"}
    _write_json(paths["summary"], result)

    missing_outputs = [str(path) for path in paths.values() if not path.is_file()]
    if missing_outputs:
        raise RuntimeError("Faltan outputs:\n- " + "\n- ".join(missing_outputs))
    return result
