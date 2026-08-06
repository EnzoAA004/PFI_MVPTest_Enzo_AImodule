"""Whole-study DICOM zip ingestion.

A real lumbar MRI study is a single archive holding several *series* (sagittal T1,
sagittal T2, axial T2, ...). This module extracts such a zip, classifies each series
by plane (from ``ImageOrientationPatient``) and weighting (from ``SeriesDescription``
/ echo time), and selects one series per plane for inference:

- sagittal: prefer T2, fall back to T1, then to the largest sagittal series.
- axial: require T2 (the axial model was trained on axial T2); otherwise flagged.

The selected series are registered as ordinary per-plane inputs so the existing
``/multiplanar/run`` contract (sagittalInputId + axialInputId) is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .deidentify import UidRemapper
from .input_registry import (
    InputRegistryError,
    max_upload_bytes,
    register_series_files,
    safe_extract_zip,
    upload_root,
    write_limited_upload,
)

_T2_TOKENS = ("t2", "t2w", "t2_", "t2-", "t2 ")
_T1_TOKENS = ("t1", "t1w", "t1_", "t1-", "t1 ")


def _plane_from_orientation(iop: Any) -> str | None:
    if iop is None or len(iop) < 6:
        return None
    try:
        row = [float(iop[0]), float(iop[1]), float(iop[2])]
        col = [float(iop[3]), float(iop[4]), float(iop[5])]
    except (TypeError, ValueError):
        return None
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    axis = max(range(3), key=lambda i: abs(normal[i]))
    return {0: "sagittal", 1: "coronal", 2: "axial"}[axis]


def _series_plane(orientations: list[Any]) -> tuple[str | None, bool]:
    """Plane of a whole series, and whether its slices disagree on it.

    The plane has to be decided over every slice, not over the first one. A localizer
    is multi-plane *by definition* — it carries sagittal, coronal and axial slices in
    one series — so reading a single file gives it whichever plane happened to be
    sorted first, and it gets listed as a normal series of that plane. The same
    misread turns the secondary-capture screenshots into whatever they were captured
    from.

    A series where the slices disagree is reported as such rather than being forced
    into one plane: it can be shown as images, but it is not a volume any plane slot
    can take.
    """
    planes = [plane for plane in (_plane_from_orientation(iop) for iop in orientations) if plane]
    if not planes:
        return None, False
    # Una serie de un solo plano tiene acuerdo total: el plano sale del eje dominante
    # de la normal, y el redondeo del coseno no alcanza para cambiarlo salvo en un
    # oblicuo a 45 grados, que no es ninguna de estas series. Asi que cualquier
    # desacuerdo real ya es multiplano, sin umbral de mayoria que ajustar.
    #
    # El empate se rompe por nombre y no por el orden del `set`, que varia entre
    # corridas: en un localizer con la misma cantidad de cortes por plano el plano
    # informado es de todos modos secundario -queda marcado multiplano-, pero que
    # cambie solo entre dos ingestas del mismo estudio se lee como un error.
    dominant = min(sorted(set(planes)), key=lambda item: (-planes.count(item), item))
    return dominant, len(set(planes)) > 1


def _is_derived(ds: Any) -> bool:
    """Whether the series is a screenshot/reformat rather than acquired image data.

    The ``PosDisp:`` two-slice series a Siemens console exports are captures of what
    was on screen, with the geometry of whatever they were captured from. They are
    legitimate to list -the doctor may want to see what the technician marked- but
    they are not a series to segment, and picking one as a plane's input would run the
    model over a screenshot.
    """
    image_type = getattr(ds, "ImageType", None) or []
    try:
        tokens = {str(value).strip().upper() for value in image_type}
    except TypeError:
        return False
    return "DERIVED" in tokens or "SECONDARY" in tokens


def _weighting(description: str, echo_time: float | None) -> str:
    text = f" {description.lower()} "
    if any(token in text for token in _T2_TOKENS):
        return "t2"
    if any(token in text for token in _T1_TOKENS):
        return "t1"
    if echo_time is not None:
        if echo_time >= 80:
            return "t2"
        if echo_time <= 30:
            return "t1"
    return "unknown"


def classify_study_series(root: Path) -> list[dict[str, Any]]:
    """Return one descriptor per DICOM series found under ``root``."""
    try:
        import pydicom
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - dependency guard
        raise InputRegistryError("SimpleITK/pydicom requeridos para leer el estudio", status_code=500) from exc

    reader = sitk.ImageSeriesReader()
    series: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dirpath, _dirs, filenames in os.walk(root):
        # No extension gate: GDCM identifies DICOM by content, so this also handles
        # Siemens .ima and extension-less PACS exports, not only *.dcm files.
        if not filenames:
            continue
        for series_id in reader.GetGDCMSeriesIDs(dirpath):
            if series_id in seen:
                continue
            file_names = list(reader.GetGDCMSeriesFileNames(dirpath, series_id))
            if len(file_names) < 1:
                continue
            seen.add(series_id)
            # El encabezado de cada corte, no solo el del primero: el plano de la serie
            # es una propiedad del conjunto y un localizer no la cumple.
            headers = []
            for name in file_names:
                try:
                    headers.append(pydicom.dcmread(name, stop_before_pixels=True, force=True))
                except Exception:
                    continue
            if not headers:
                continue
            ds = headers[0]
            description = str(getattr(ds, "SeriesDescription", "") or "")
            echo_time = getattr(ds, "EchoTime", None)
            echo_time = float(echo_time) if echo_time is not None else None
            plane, multiplanar = _series_plane([getattr(h, "ImageOrientationPatient", None) for h in headers])
            series.append({
                "seriesInstanceUid": series_id,
                "description": description,
                "plane": plane,
                "multiplanar": multiplanar,
                "derived": _is_derived(ds),
                "weighting": _weighting(description, echo_time),
                "sliceCount": len(file_names),
                "files": file_names,
            })
    return series


def _is_analyzable(item: dict[str, Any]) -> bool:
    """Whether a series can be the input of an inference run.

    Un localizer y una captura de consola tienen plano y ponderacion, asi que entran
    en el ranking como cualquier otra y pueden ganarlo por tener mas cortes. Correr el
    modelo sobre eso devuelve una segmentacion con la misma pinta que una buena.
    """
    return not item.get("multiplanar") and not item.get("derived") and item["sliceCount"] >= 2


def _select_series(series: list[dict[str, Any]], plane: str, prefer: str) -> dict[str, Any] | None:
    candidates = [s for s in series if s["plane"] == plane and _is_analyzable(s)]
    if not candidates:
        return None
    order = [prefer, "t1" if prefer == "t2" else "t2", "unknown"]

    def rank(item: dict[str, Any]) -> tuple[int, int]:
        weight_rank = order.index(item["weighting"]) if item["weighting"] in order else len(order)
        return (weight_rank, -item["sliceCount"])

    return sorted(candidates, key=rank)[0]


def _summarize(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "seriesInstanceUid": item["seriesInstanceUid"],
        "description": item["description"],
        "plane": item["plane"],
        "multiplanar": bool(item.get("multiplanar")),
        "derived": bool(item.get("derived")),
        "analyzable": _is_analyzable(item),
        "weighting": item["weighting"],
        "sliceCount": item["sliceCount"],
    }


def register_study_zip(*, case_id: str, stream: Any) -> dict[str, Any]:
    study_id = uuid4().hex
    study_root = upload_root() / "studies" / study_id
    raw_dir = study_root / "raw"
    zip_path = study_root / "study.zip"
    study_root.mkdir(parents=True, exist_ok=True)

    try:
        write_limited_upload(stream, zip_path, max_upload_bytes())
        try:
            _total, count = safe_extract_zip(zip_path, raw_dir)
        finally:
            if zip_path.exists():
                zip_path.unlink()
        if count == 0:
            raise InputRegistryError("zip vacio", status_code=400)

        series = classify_study_series(raw_dir)
        if not series:
            raise InputRegistryError("no se encontraron series DICOM en el zip", status_code=400)

        warnings: list[str] = []
        sagittal = _select_series(series, "sagittal", prefer="t2")
        axial = _select_series(series, "axial", prefer="t2")

        # Se registran las siete series, no las dos que analiza la IA.
        #
        # Un estudio de RM lumbar trae sagital T1 y T2, axial T1 y T2, a veces
        # coronales, el localizer y las capturas de la consola. La IA corre sobre dos,
        # pero el medico lee el estudio entero: la T1 es la que muestra la grasa y la
        # medula osea, y sin ella no se distingue un hemangioma de una metastasis.
        # Antes las otras cinco se borraban junto con el directorio del zip a los
        # segundos de subirlas, asi que ni siquiera se podian pedir despues.
        #
        # Las no elegidas quedan marcadas como no analizables: se muestran, y el
        # registro las rechaza si alguien las manda como entrada de una corrida.
        registered: dict[str, dict[str, Any]] = {}
        summaries: list[dict[str, Any]] = []
        # Un solo remapeador para todo el estudio: es lo que hace que, despues de
        # reasignar los UIDs, las series sigan siendo series y los planos sigan
        # compartiendo marco de referencia. Ver deidentify.UidRemapper.
        remap = UidRemapper()
        for item in series:
            # Identidad, no igualdad: dos series distintas pueden tener el mismo
            # resumen -misma descripcion, mismo plano, misma cantidad de cortes- y con
            # `==` la segunda se llevaria la ranura de la primera.
            plane_slot = next((name for name, chosen in (("sagittal", sagittal), ("axial", axial))
                               if chosen is not None and chosen is item), None)
            meta = register_series_files(
                case_id=case_id,
                plane=plane_slot or ("unknown" if item["multiplanar"] else item["plane"]),
                file_paths=item["files"],
                analyzable=plane_slot is not None,
                remap=remap,
            )
            summary = {**_summarize(item), "inputId": meta["inputId"]}
            summaries.append(summary)
            if plane_slot:
                registered[plane_slot] = {**meta, **summary}

        result: dict[str, Any] = {
            "caseId": case_id,
            "studyId": study_id,
            "seriesFound": summaries,
            "warnings": warnings,
        }
        if sagittal is None:
            warnings.append("no se encontro serie sagital en el estudio")
        else:
            if sagittal["weighting"] != "t2":
                warnings.append(f"sagital sin T2 disponible; se usa {sagittal['weighting']}")
            result["sagittal"] = registered["sagittal"]
        if axial is None:
            warnings.append("no se encontro serie axial en el estudio")
        else:
            if axial["weighting"] != "t2":
                warnings.append(f"axial sin T2 real; el modelo axial espera T2 (encontrado {axial['weighting']})")
            result["axial"] = registered["axial"]

        if "sagittal" not in result and "axial" not in result:
            raise InputRegistryError("no se pudo identificar serie sagital ni axial en el estudio", status_code=422)
        return result
    finally:
        import shutil

        shutil.rmtree(study_root, ignore_errors=True)
