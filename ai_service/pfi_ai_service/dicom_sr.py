"""Las mediciones como DICOM SR, para que el informe viaje entre sistemas.

La contraparte de `dicom_seg`: ese exporta *dónde* está cada estructura, este exporta
*cuánto* mide. Hoy las 47 mediciones de una corrida salen en JSON, CSV o un HTML para
imprimir —todo formato propio—, así que un informe estructurado no puede entrar a otro
sistema sin que alguien lo transcriba.

Un SR TID 1500 (Measurement Report) es la forma estándar de decir "sobre esta imagen, esta
estructura mide esto". Referencia la instancia de origen, agrupa por nivel discal y declara
cada valor con su unidad.

## Qué es estándar acá y qué no

**Las unidades sí lo son.** ``mm``, ``mm2`` y ``deg`` son códigos UCUM, que es el
vocabulario que DICOM usa para unidades de medida. Cualquier sistema los interpreta igual.

**Los nombres de las mediciones no.** ``disc area``, ``raw_150 height`` son etiquetas del
modelo, no términos codificados. Se declaran con el esquema privado ``99PFI``, por el mismo
motivo que en `dicom_seg`: traducirlos de memoria a SNOMED o RadLex sería inventar
terminología clínica dentro de un objeto que otro sistema va a leer como autoritativo.

El resultado es un SR que otro sistema puede **cargar, recorrer y mostrar**, aunque no pueda
mapear automáticamente cada medición a su concepto clínico. Es exactamente el estado real
del proyecto, y decirlo así es más útil que aparentar un mapeo que no está validado.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CODING_SCHEME = "99PFI"

SERIES_DESCRIPTION = "Mediciones asistidas por IA - no apto para diagnostico clinico"

#: Unidades del proyecto a UCUM. Son las tres que produce el pipeline hoy; una unidad que
#: no esté acá no se exporta, en vez de mandarse con un código inventado.
UCUM_UNITS = {
    "mm": ("mm", "millimeter"),
    "mm2": ("mm2", "square millimeter"),
    "deg": ("deg", "degree"),
}


class ReportExportError(Exception):
    """No se pudo construir el SR."""


def _code(value: str, meaning: str | None = None):
    from highdicom.sr.coding import Code

    return Code(value, CODING_SCHEME, meaning or value)


def _unit_code(unit: str):
    from highdicom.sr.coding import Code

    mapped = UCUM_UNITS.get(unit)
    if mapped is None:
        return None
    value, meaning = mapped
    # UCUM es el vocabulario de unidades que usa DICOM: esto si es estandar.
    return Code(value, "UCUM", meaning)


def _observation_context():
    """Quién observó: un dispositivo, no una persona.

    Importa que quede declarado así en el objeto. Un SR cuyo observador es una persona
    afirma que alguien midió; acá midió un modelo y la revisión profesional viene después.
    """
    from highdicom.sr import (
        DeviceObserverIdentifyingAttributes,
        ObservationContext,
        ObserverContext,
    )
    from highdicom.sr.coding import CodedConcept
    from pydicom.uid import generate_uid

    # 121007 "Device" es el codigo DCM del tipo de observador. Terminologia publicada.
    device = CodedConcept(value="121007", scheme_designator="DCM", meaning="Device")
    return ObservationContext(
        observer_device_context=ObserverContext(
            observer_type=device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=generate_uid(),
                name="PFI lumbar MRI assistant",
                manufacturer_name="PFI",
            ),
        ),
    )


def _measurement_groups(measurements: list[dict[str, Any]], source_image):
    """Un grupo por nivel discal, que es como se lee el informe.

    Las mediciones sin nivel van juntas en su propio grupo en vez de repartirse: "sin nivel
    asignado" es información, y mezclarlas con las de un nivel diría que pertenecen ahí.
    """
    from highdicom.sr import (
        MeasurementsAndQualitativeEvaluations,
        Measurement,
        TrackingIdentifier,
    )
    from pydicom.uid import generate_uid

    by_level: dict[str, list[dict[str, Any]]] = {}
    for item in measurements:
        level = str(item.get("level") or "").strip() or "sin nivel asignado"
        by_level.setdefault(level, []).append(item)

    groups = []
    for level in sorted(by_level):
        values = []
        for item in by_level[level]:
            unit = _unit_code(str(item.get("unit") or ""))
            value = item.get("value")
            if unit is None or not isinstance(value, (int, float)):
                # Una medicion sin unidad conocida no se exporta con una inventada.
                continue
            values.append(Measurement(
                name=_code(str(item.get("labelKey") or "medicion")),
                value=float(value),
                unit=unit,
            ))
        if not values:
            continue
        groups.append(MeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(uid=generate_uid(), identifier=level),
            measurements=values,
            source_images=[_source_reference(source_image)],
        ))
    return groups


def _source_reference(source_image):
    from highdicom.sr import SourceImageForMeasurementGroup

    return SourceImageForMeasurementGroup(
        referenced_sop_class_uid=source_image.SOPClassUID,
        referenced_sop_instance_uid=source_image.SOPInstanceUID,
    )


def build_measurement_report(
    *,
    series_dir: Path,
    slice_index: int,
    measurements: list[dict[str, Any]],
    model_version: str,
):
    """Arma el SR de las mediciones de una corrida.

    :param series_dir: serie de origen, ya de-identificada.
    :param slice_index: corte sobre el que se midió.
    :param measurements: las mediciones tal como las publica la corrida.
    """
    import pydicom
    from highdicom.sr import ComprehensiveSR, MeasurementReport
    from pydicom.uid import generate_uid

    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise ReportExportError(f"la serie no tiene archivos DICOM: {series_dir}")
    if not 0 <= slice_index < len(files):
        raise ReportExportError(f"corte {slice_index} fuera de la serie, que tiene {len(files)}")
    source_image = pydicom.dcmread(str(files[slice_index]))

    groups = _measurement_groups(measurements or [], source_image)
    if not groups:
        raise ReportExportError("la corrida no tiene mediciones exportables")

    try:
        content = MeasurementReport(
            observation_context=_observation_context(),
            procedure_reported=_code("lumbar_mri_assisted_measurement", "Medicion asistida de RM lumbar"),
            imaging_measurements=groups,
        )
        # `CompletionFlag` queda en PARTIAL, que es el default de highdicom y ademas lo
        # correcto: el informe no esta completo hasta que un profesional lo revise.
        # Marcarlo COMPLETE seria afirmar que la lectura termino.
        return ComprehensiveSR(
            evidence=[source_image],
            content=content,
            series_instance_uid=generate_uid(),
            series_number=101,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer="PFI",
            series_description=SERIES_DESCRIPTION,
        )
    except Exception as exc:
        log.exception("event=dicom_sr_build_failed series_dir=%s slice=%s", series_dir.name, slice_index)
        raise ReportExportError(f"no se pudo construir el SR: {type(exc).__name__}") from exc


def write_report(report, destination: Path) -> Path:
    """Escribe el SR. Si falla, no deja un archivo a medias."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        report.save_as(str(destination), enforce_file_format=True)
        return destination
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise ReportExportError(f"no se pudo escribir el SR: {type(exc).__name__}") from exc


def report_for_plane_run(plane_run_id: str, plane: str):
    """Arma el SR de una corrida ya ejecutada, buscando sus piezas en disco."""
    from .dicom_seg import SegmentationExportError, _plane_result_for
    from .input_registry import resolve_registered_input
    from .settings import get_settings

    settings = get_settings()
    plane_result = _plane_result_for(plane_run_id, plane, settings.output_dir)
    if plane_result is None:
        raise ReportExportError("no se encontro el reporte de la corrida")

    source = plane_result.get("input") or {}
    input_id = str(source.get("inputId") or "")
    slice_index = source.get("selectedSliceIndex")
    if not input_id or not isinstance(slice_index, int):
        raise ReportExportError("la corrida no declara input o corte analizado")

    record = resolve_registered_input(input_id)
    if not record.path.is_dir():
        raise ReportExportError("la serie de origen no es DICOM: no se puede exportar SR")

    model = plane_result.get("model") or {}
    return build_measurement_report(
        series_dir=record.path,
        slice_index=int(slice_index),
        measurements=plane_result.get("measurements") or [],
        model_version=str(model.get("version") or ""),
    )
