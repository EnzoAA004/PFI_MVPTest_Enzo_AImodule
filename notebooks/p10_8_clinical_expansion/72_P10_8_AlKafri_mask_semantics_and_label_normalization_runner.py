from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.getenv("PFI_ROOT", "/content/drive/MyDrive/PFI_MVP"))
PREF = Path(
    os.getenv(
        "PFI_P10_8_PREFLIGHT_ROOT",
        str(ROOT / "results" / "P10_8_clinical_expansion_preflight"),
    )
)
N71 = Path(
    os.getenv(
        "PFI_P10_8_NOTEBOOK71_ROOT",
        str(PREF / "sudirman_alkafri_audit_v2"),
    )
)
marker_71 = N71 / "NOTEBOOK_71_COMPLETE.json"
if not marker_71.is_file():
    raise FileNotFoundError(f"Falta marcador 71: {marker_71}")

notebook_71 = json.loads(marker_71.read_text(encoding="utf-8"))
if (
    notebook_71.get("status") != "NOTEBOOK_71_COMPLETE"
    or notebook_71.get("trainingExecuted") is not False
    or notebook_71.get("weightsDeserialized") is not False
):
    raise RuntimeError("Notebook 71 inválido")
if (
    notebook_71.get("inventoryTruncated") is not False
    or int(notebook_71.get("indexedFileCount", 0)) < 1000
):
    raise RuntimeError("Notebook 71 no auditó la fuente primaria completa")

ALK = Path(
    os.getenv(
        "PFI_ALKAFRI_ROOT",
        str(ROOT / "data" / "AXIAL_ALKAFRI" / "extracted" / "_nested"),
    )
)
if not ALK.is_dir():
    raise FileNotFoundError(ALK)

BASE = ALK.parent.parent if ALK.name == "_nested" else ALK
MANUAL_ROOT = ALK / "ground_truth__Manual_Label_Data" / "03_Manual_Label_Data"
PROCESSED_ROOT = (
    ALK
    / "ground_truth__Ground_Truth_Label"
    / "04_Intermediary_Ground_Truth_Data"
)
MRI_ROOT = ALK / "main_dataset__MRI_Data" / "01_MRI_Data"
E7_ROOT = ROOT / "results" / "E7_alkafri_axial_curated_subset"
for required in (MANUAL_ROOT, PROCESSED_ROOT, MRI_ROOT):
    if not required.is_dir():
        raise FileNotFoundError(required)

OUT = Path(
    os.getenv(
        "PFI_P10_8_NOTEBOOK72_ROOT",
        str(PREF / "alkafri_mask_semantics_and_label_normalization"),
    )
)
SAMPLE_PER_GROUP = int(os.getenv("PFI_P10_8_MASK_SAMPLE_PER_GROUP", "12"))
MAX_PAIR_COMPARISONS = int(
    os.getenv("PFI_P10_8_MAX_PAIR_COMPARISONS", "120")
)
MAX_XCF_SAMPLES = int(os.getenv("PFI_P10_8_MAX_XCF_SAMPLES", "60"))
MAX_DOCUMENT_FILES = int(os.getenv("PFI_P10_8_MAX_DOCUMENT_FILES", "600"))
MAX_DOCUMENT_BYTES = int(
    os.getenv("PFI_P10_8_MAX_DOCUMENT_BYTES", str(4 * 1024 * 1024))
)
MAX_TABULAR_ROWS = int(os.getenv("PFI_P10_8_MAX_TABULAR_ROWS", "5000"))


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


MASK_PATTERN = re.compile(
    r"^(T[12])_(\d{4})_(D\d+)(?:[^.]*)\.(png|xcf)$",
    re.IGNORECASE,
)


def parse_mask_name(path: Path) -> tuple[str | None, str | None, str | None]:
    match = MASK_PATTERN.match(path.name)
    if not match:
        return None, None, None
    return (
        match.group(1).upper(),
        sha256_text("alkafri|" + match.group(2)),
        match.group(3).upper(),
    )


def source_type(path: Path) -> str:
    value = str(path).lower()
    if "manual_label_data" in value:
        return "manual"
    if "ground_truth_label" in value:
        return "processed_intermediary"
    return "unknown"


rows: list[dict[str, object]] = []
for source_root in (MANUAL_ROOT, PROCESSED_ROOT):
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".png", ".xcf"}:
            continue
        modality, case_hash, d_token = parse_mask_name(path)
        rows.append(
            {
                "sourceType": source_type(path),
                "relativePathHash": sha256_text(path.relative_to(ALK)),
                "suffix": path.suffix.lower(),
                "sizeBytes": path.stat().st_size,
                "modality": modality,
                "caseKeyHash": case_hash,
                "dToken": d_token,
                "nameParsed": case_hash is not None,
                "_path": path,
            }
        )

private_inventory = pd.DataFrame(rows)
if private_inventory.empty:
    raise RuntimeError("No se encontraron PNG/XCF")
public_inventory = private_inventory.drop(columns=["_path"])
inventory_summary = (
    public_inventory.groupby(
        ["sourceType", "suffix", "modality", "dToken", "nameParsed"],
        dropna=False,
    )
    .agg(
        fileCount=("relativePathHash", "count"),
        totalBytes=("sizeBytes", "sum"),
        uniqueCaseCount=("caseKeyHash", "nunique"),
    )
    .reset_index()
)
print(
    "Máscaras/XCF:",
    len(private_inventory),
    "PNG:",
    int((private_inventory["suffix"] == ".png").sum()),
    "XCF:",
    int((private_inventory["suffix"] == ".xcf").sum()),
)
display(inventory_summary)

png_inventory = private_inventory[private_inventory["suffix"] == ".png"].copy()
sampled_parts = [
    group.sort_values("relativePathHash").head(SAMPLE_PER_GROUP)
    for _, group in png_inventory.groupby(
        ["sourceType", "modality", "dToken"],
        dropna=False,
    )
]
sampled_png = pd.concat(sampled_parts, ignore_index=True)


def color_hex(color: np.ndarray) -> str:
    return "#" + "".join(f"{int(value):02X}" for value in color.tolist()[:4])


profile_rows: list[dict[str, object]] = []
for _, row in sampled_png.iterrows():
    try:
        with Image.open(row["_path"]) as image:
            original_mode = image.mode
            width, height = image.size
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        colors, counts = np.unique(
            rgba.reshape(-1, 4), axis=0, return_counts=True
        )
        order = np.argsort(counts)[::-1]
        colors, counts = colors[order], counts[order]
        unique_count = len(colors)
        total = rgba.shape[0] * rgba.shape[1]
        profile_rows.append(
            {
                "sourceType": row["sourceType"],
                "relativePathHash": row["relativePathHash"],
                "modality": row["modality"],
                "caseKeyHash": row["caseKeyHash"],
                "dToken": row["dToken"],
                "width": width,
                "height": height,
                "originalMode": original_mode,
                "uniqueColorCount": unique_count,
                "maskProfile": (
                    "SINGLE_COLOR"
                    if unique_count == 1
                    else "BINARY_COLOR"
                    if unique_count == 2
                    else "MULTICOLOR"
                ),
                "dominantColorHex": color_hex(colors[0]),
                "dominantColorFraction": float(counts[0] / total),
                "secondColorHex": color_hex(colors[1]) if unique_count > 1 else None,
                "secondColorFraction": (
                    float(counts[1] / total) if unique_count > 1 else None
                ),
                "colorSignatureSha256": sha256_text(
                    "|".join(
                        f"{color_hex(color)}:{int(count)}"
                        for color, count in zip(colors, counts)
                    )
                ),
                "readSucceeded": True,
                "readErrorType": None,
            }
        )
    except Exception as exc:
        profile_rows.append(
            {
                "sourceType": row["sourceType"],
                "relativePathHash": row["relativePathHash"],
                "modality": row["modality"],
                "caseKeyHash": row["caseKeyHash"],
                "dToken": row["dToken"],
                "width": None,
                "height": None,
                "originalMode": None,
                "uniqueColorCount": None,
                "maskProfile": "UNREADABLE",
                "dominantColorHex": None,
                "dominantColorFraction": None,
                "secondColorHex": None,
                "secondColorFraction": None,
                "colorSignatureSha256": None,
                "readSucceeded": False,
                "readErrorType": type(exc).__name__,
            }
        )
mask_profile = pd.DataFrame(profile_rows)
print("PNG perfilados:", len(mask_profile))
display(
    mask_profile.groupby(
        ["sourceType", "modality", "dToken", "maskProfile"],
        dropna=False,
    )
    .size()
    .reset_index(name="sampleCount")
)

pairable = png_inventory[
    png_inventory["nameParsed"]
    & png_inventory["caseKeyHash"].notna()
    & png_inventory["modality"].notna()
    & png_inventory["dToken"].notna()
]
key_columns = ["caseKeyHash", "modality", "dToken"]
manual_groups = {
    key: group.sort_values("relativePathHash")
    for key, group in pairable[pairable["sourceType"] == "manual"].groupby(
        key_columns
    )
}
processed_groups = {
    key: group.sort_values("relativePathHash")
    for key, group in pairable[
        pairable["sourceType"] == "processed_intermediary"
    ].groupby(key_columns)
}
pair_rows = []
for key in sorted(set(manual_groups) | set(processed_groups)):
    manual_count = len(manual_groups.get(key, []))
    processed_count = len(processed_groups.get(key, []))
    if manual_count == processed_count == 1:
        status = "ONE_TO_ONE"
    elif manual_count and processed_count:
        status = "ONE_OR_MANY_TO_ONE_OR_MANY"
    elif manual_count:
        status = "MANUAL_ONLY"
    else:
        status = "PROCESSED_ONLY"
    pair_rows.append(
        {
            "caseKeyHash": key[0],
            "modality": key[1],
            "dToken": key[2],
            "manualFileCount": manual_count,
            "processedFileCount": processed_count,
            "pairStatus": status,
        }
    )
pairing_registry = pd.DataFrame(pair_rows)
comparison_candidates = pairing_registry[
    (pairing_registry["manualFileCount"] > 0)
    & (pairing_registry["processedFileCount"] > 0)
].copy()
comparison_candidates["sortKey"] = (
    comparison_candidates["caseKeyHash"]
    + "|"
    + comparison_candidates["modality"]
    + "|"
    + comparison_candidates["dToken"]
).map(sha256_text)
comparison_rows = []
for _, row in comparison_candidates.sort_values("sortKey").head(
    MAX_PAIR_COMPARISONS
).iterrows():
    key = (row["caseKeyHash"], row["modality"], row["dToken"])
    manual = manual_groups[key].iloc[0]
    processed = processed_groups[key].iloc[0]
    result = {
        "caseKeyHash": key[0],
        "modality": key[1],
        "dToken": key[2],
        "manualRelativePathHash": manual["relativePathHash"],
        "processedRelativePathHash": processed["relativePathHash"],
        "shapeMatch": False,
        "exactPixelMatch": False,
        "manualColorCount": None,
        "processedColorCount": None,
        "colorSetJaccard": None,
        "comparisonSucceeded": False,
        "comparisonErrorType": None,
    }
    try:
        with Image.open(manual["_path"]) as image:
            manual_array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        with Image.open(processed["_path"]) as image:
            processed_array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        result["shapeMatch"] = manual_array.shape == processed_array.shape
        result["exactPixelMatch"] = bool(
            result["shapeMatch"]
            and np.array_equal(manual_array, processed_array)
        )
        manual_colors = {
            tuple(value)
            for value in np.unique(
                manual_array.reshape(-1, 4), axis=0
            ).tolist()
        }
        processed_colors = {
            tuple(value)
            for value in np.unique(
                processed_array.reshape(-1, 4), axis=0
            ).tolist()
        }
        union = manual_colors | processed_colors
        result.update(
            manualColorCount=len(manual_colors),
            processedColorCount=len(processed_colors),
            colorSetJaccard=(
                float(len(manual_colors & processed_colors) / len(union))
                if union
                else 1.0
            ),
            comparisonSucceeded=True,
        )
    except Exception as exc:
        result["comparisonErrorType"] = type(exc).__name__
    comparison_rows.append(result)
pair_comparison = pd.DataFrame(comparison_rows)
print(
    "Claves pairing:",
    len(pairing_registry),
    "comparaciones:",
    len(pair_comparison),
)

xcf_sample = private_inventory[
    private_inventory["suffix"] == ".xcf"
].sort_values("relativePathHash").head(MAX_XCF_SAMPLES)
xcf_rows = []
for _, row in xcf_sample.iterrows():
    try:
        with row["_path"].open("rb") as handle:
            header = handle.read(32)
        valid = header.startswith(b"gimp xcf")
        xcf_rows.append(
            {
                "relativePathHash": row["relativePathHash"],
                "modality": row["modality"],
                "caseKeyHash": row["caseKeyHash"],
                "dToken": row["dToken"],
                "sizeBytes": row["sizeBytes"],
                "xcfHeaderValid": valid,
                "externalLayerParserAvailable": bool(
                    shutil.which("xcfinfo")
                    or shutil.which("gimp")
                    or shutil.which("gimp-console")
                ),
                "layerNamesExtracted": False,
                "layerCount": None,
                "auditStatus": (
                    "VALID_XCF_HEADER_LAYER_SEMANTICS_NOT_PARSED"
                    if valid
                    else "INVALID_OR_UNKNOWN_XCF_HEADER"
                ),
            }
        )
    except Exception as exc:
        xcf_rows.append(
            {
                "relativePathHash": row["relativePathHash"],
                "modality": row["modality"],
                "caseKeyHash": row["caseKeyHash"],
                "dToken": row["dToken"],
                "sizeBytes": row["sizeBytes"],
                "xcfHeaderValid": False,
                "externalLayerParserAvailable": False,
                "layerNamesExtracted": False,
                "layerCount": None,
                "auditStatus": "XCF_READ_FAILED_" + type(exc).__name__,
            }
        )
xcf_audit = pd.DataFrame(xcf_rows)

DOCUMENT_SUFFIXES = {
    ".m",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
}
term_patterns = {
    "D3": r"\bD3\b",
    "D4": r"\bD4\b",
    "D5": r"\bD5\b",
    "disc": r"\b(disc|disk|intervertebral)\b",
    "vertebra": r"\bvertebr",
    "thecal_sac": r"thecal\s+sac|dural\s+sac",
    "facet": r"facet|zygapophy|facetar",
    "ligamentum_flavum": r"ligamentum\s+flavum|ligamento\s+amarillo|flavum",
    "nerve_root": r"nerve\s+root|ra[ií]z\s+nerv|radicular",
    "epidural_fat": r"epidural\s+fat|grasa\s+epidural",
    "annular_tear": r"annular\s+(tear|fissure)|(desgarro|fisura)\s+anular",
    "herniation": r"herniation|hernia\s+discal",
    "bulging": r"bulging|disc\s+bulge|abombamiento",
    "disc_height": r"(disc|disk)\s+height|altura\s+discal",
    "spondylolisthesis": r"spondylolisthesis|anterolisthesis|retrolisthesis|listesis",
}
term_patterns = {
    key: re.compile(value, re.IGNORECASE)
    for key, value in term_patterns.items()
}
document_candidates = []
for alias, search_root in (("dataset_base", BASE), ("e7_results", E7_ROOT)):
    if not search_root.exists():
        continue
    for path in search_root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in DOCUMENT_SUFFIXES
            and path.stat().st_size <= MAX_DOCUMENT_BYTES
        ):
            document_candidates.append((sha256_text(path), alias, path))
document_candidates = sorted(document_candidates)[:MAX_DOCUMENT_FILES]
document_rows = []
source_terms: dict[str, set[str]] = defaultdict(set)
for _, alias, path in document_candidates:
    source_hash = sha256_text(f"{alias}|{path.name}|{path.stat().st_size}")
    text = ""
    read_status = "READ_OK"
    try:
        if path.suffix.lower() in {".csv", ".tsv"}:
            frame = pd.read_csv(
                path,
                sep="\t" if path.suffix.lower() == ".tsv" else ",",
                nrows=MAX_TABULAR_ROWS,
                dtype=str,
                low_memory=False,
            )
            text = " ".join(map(str, frame.columns)) + " "
            text += " ".join(
                frame.fillna("").astype(str).to_numpy().ravel().tolist()
            )
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        read_status = "READ_FAILED_" + type(exc).__name__
    matched_terms = sorted(
        key for key, pattern in term_patterns.items() if text and pattern.search(text)
    )
    source_terms[source_hash].update(matched_terms)
    document_rows.append(
        {
            "rootAlias": alias,
            "relativePathHash": source_hash,
            "suffix": path.suffix.lower(),
            "sizeBytes": path.stat().st_size,
            "readStatus": read_status,
            "matchedTerms": "|".join(matched_terms),
            "matchedTermCount": len(matched_terms),
        }
    )
document_audit = pd.DataFrame(document_rows)

semantics_rows = []
for d_token in ("D3", "D4", "D5"):
    cooccurring_terms: set[str] = set()
    for terms in source_terms.values():
        if d_token in terms:
            cooccurring_terms.update(
                term for term in terms if term not in {"D3", "D4", "D5"}
            )
    token_inventory = public_inventory[public_inventory["dToken"] == d_token]
    semantics_rows.append(
        {
            "dToken": d_token,
            "manualPngCount": int(
                (
                    (token_inventory["sourceType"] == "manual")
                    & (token_inventory["suffix"] == ".png")
                ).sum()
            ),
            "processedPngCount": int(
                (
                    (token_inventory["sourceType"] == "processed_intermediary")
                    & (token_inventory["suffix"] == ".png")
                ).sum()
            ),
            "xcfCount": int((token_inventory["suffix"] == ".xcf").sum()),
            "cooccurringDocumentationTerms": "|".join(
                sorted(cooccurring_terms)
            ),
            "documentedExactMappingFound": False,
            "semanticStatus": (
                "SEMANTICS_INFERRED_REQUIRES_REVIEW"
                if cooccurring_terms
                else "UNKNOWN_NOT_USABLE"
            ),
            "anatomicalMeaningValidated": False,
            "clinicalFindingMeaningValidated": False,
            "trainingAuthorized": False,
        }
    )
semantics_registry = pd.DataFrame(semantics_rows)

finding_terms = {
    "facet_hypertrophy": "facet",
    "ligamentum_flavum_hypertrophy": "ligamentum_flavum",
    "annular_tear": "annular_tear",
    "nerve_root_compression": "nerve_root",
    "epidural_fat": "epidural_fat",
    "disc_height": "disc_height",
    "spondylolisthesis": "spondylolisthesis",
    "disc_herniation": "herniation",
    "disc_bulging": "bulging",
}
all_terms = {term for values in source_terms.values() for term in values}
candidate_support = pd.DataFrame(
    [
        {
            "findingType": finding,
            "documentationTermPresent": term in all_terms,
            "maskClassMappingValidated": False,
            "caseSeriesLevelSliceAlignmentValidated": False,
            "clinicalTaxonomyFrozen": False,
            "supportStatus": (
                "DOCUMENTATION_TERM_PRESENT_MASK_MAPPING_UNVALIDATED"
                if term in all_terms
                else "ANNOTATION_NOT_DEMONSTRATED"
            ),
            "trainingAuthorized": False,
        }
        for finding, term in finding_terms.items()
    ]
)
display(semantics_registry)
display(candidate_support)

OUT.mkdir(parents=True, exist_ok=True)
paths = {
    "maskInventory": OUT / "mask_file_inventory_v1.csv",
    "inventorySummary": OUT / "mask_inventory_summary_v1.csv",
    "maskProfile": OUT / "sampled_png_mask_profile_v1.csv",
    "pairingRegistry": OUT / "manual_processed_pairing_registry_v1.csv",
    "pairComparison": OUT / "manual_processed_pixel_comparison_v1.csv",
    "xcfAudit": OUT / "xcf_header_and_layer_audit_v1.csv",
    "documentAudit": OUT / "documentation_and_code_term_audit_v1.csv",
    "semanticsRegistry": OUT / "d_token_semantics_registry_v1.csv",
    "candidateSupport": OUT / "candidate_finding_support_matrix_v1.csv",
    "summaryJson": OUT / "NOTEBOOK_72_SUMMARY.json",
}
for frame, key in (
    (public_inventory, "maskInventory"),
    (inventory_summary, "inventorySummary"),
    (mask_profile, "maskProfile"),
    (pairing_registry, "pairingRegistry"),
    (pair_comparison, "pairComparison"),
    (xcf_audit, "xcfAudit"),
    (document_audit, "documentAudit"),
    (semantics_registry, "semanticsRegistry"),
    (candidate_support, "candidateSupport"),
):
    frame.to_csv(paths[key], index=False)

exact_mapping_count = int(
    semantics_registry["documentedExactMappingFound"].sum()
)
summary_payload = {
    "schemaVersion": "pfi.p10-8.alkafri-mask-semantics-summary.v1",
    "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
    "sourceDatasetAudited": True,
    "manualAndProcessedMasksPresent": True,
    "trainingExecuted": False,
    "weightsDeserialized": False,
    "internalTestAccessed": False,
    "officialHiddenTestAccessed": False,
    "patientIdentifiersExported": False,
    "clinicalGroundTruthCreated": False,
    "clinicalThresholdsFrozen": False,
    "trainingAuthorized": False,
    "maskFileCount": int(len(private_inventory)),
    "pngFileCount": int((private_inventory["suffix"] == ".png").sum()),
    "xcfFileCount": int((private_inventory["suffix"] == ".xcf").sum()),
    "sampledPngCount": int(len(mask_profile)),
    "pairingKeyCount": int(len(pairing_registry)),
    "pixelComparisonCount": int(len(pair_comparison)),
    "sampledXcfCount": int(len(xcf_audit)),
    "documentSourceCount": int(len(document_audit)),
    "exactDocumentedTokenMappings": exact_mapping_count,
    "unresolvedTokenCount": int(
        (semantics_registry["semanticStatus"] != "SEMANTICS_DOCUMENTED").sum()
    ),
    "semanticMappingValidated": exact_mapping_count == 3,
    "candidateFindingTrainingAuthorizedCount": 0,
    "nextRequiredGate": "NOTEBOOK_73_VIABILITY_GATE",
}
atomic_write_json(paths["summaryJson"], summary_payload)
marker = {
    "schemaVersion": "pfi.p10-8.notebook-72-complete.v1",
    "status": "NOTEBOOK_72_COMPLETE",
    **summary_payload,
    "outputs": {key: str(value) for key, value in paths.items()},
}
atomic_write_json(OUT / "NOTEBOOK_72_COMPLETE.json", marker)
print(json.dumps(marker, indent=2, ensure_ascii=False))
print("NOTEBOOK_72_COMPLETE")
