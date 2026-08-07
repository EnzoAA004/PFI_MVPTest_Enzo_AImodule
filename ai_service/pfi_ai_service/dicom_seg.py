"""La segmentación como objeto DICOM SEG, para que otro sistema pueda abrirla.

Hoy una corrida produce `mask.npy`, `overlay.png` y un informe en JSON o CSV. Todo eso solo
lo entiende este producto: para mirar la segmentación en 3D Slicer, en OHIF o en cualquier
estación de trabajo hay que exportar a mano y perder la correspondencia con la imagen.

Un DICOM SEG es la respuesta estándar a eso. Referencia la serie de origen por sus UIDs,
lleva la geometría del corte y describe cada segmento con un término codificado, así que
cualquier visor lo carga alineado sobre el estudio.

## Dos decisiones que conviene entender

**Los segmentos se codifican con un esquema propio, no con SNOMED CT.** Las clases del
modelo axial se llaman ``raw_0``, ``raw_50``, ``raw_100``: son valores de gris de la máscara
de Al-Kafri, no nombres clínicos, y el registro los conserva así a propósito porque el
manifest del artefacto declara esa misma lista. Traducirlos a SNOMED de memoria sería
inventar terminología clínica dentro de un objeto que otros sistemas van a leer como
autoritativo, y un código equivocado ahí se propaga en silencio. Se usa ``99PFI`` —el
prefijo ``99`` es la convención DICOM para esquemas privados— con el nombre del registro.
Mapear a SNOMED es el paso siguiente y necesita revisión clínica, no una tabla escrita acá.

**El SEG es de un corte, no del volumen.** El modelo segmenta el corte que eligió, no la
serie entera, y el objeto refleja eso: un frame por segmento presente, referenciando la
instancia exacta. Publicar un volumen completo con el resto en cero diría que se analizó
todo, que es justo lo contrario de lo que pasó.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Esquema de codificación privado. El prefijo ``99`` lo reserva DICOM para esquemas
#: locales, así que ningún sistema va a confundirlo con terminología estándar.
CODING_SCHEME = "99PFI"

SERIES_DESCRIPTION = "Segmentacion asistida por IA - no apto para diagnostico clinico"


class SegmentationExportError(Exception):
    """No se pudo construir el objeto SEG."""


def _source_instances(series_dir: Path, slice_index: int) -> list:
    """La instancia DICOM del corte segmentado, leída de la serie de origen.

    Se lee del disco y no de la metadata de la corrida porque el SEG tiene que referenciar
    los UIDs reales de esa instancia: son los que le permiten a un visor externo ponerlo
    encima de la imagen correcta.
    """
    import pydicom

    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise SegmentationExportError(f"la serie no tiene archivos DICOM: {series_dir}")
    if not 0 <= slice_index < len(files):
        raise SegmentationExportError(
            f"corte {slice_index} fuera de la serie, que tiene {len(files)}"
        )
    return [pydicom.dcmread(str(files[slice_index]))]


def _segment_descriptions(labels_present: list[int], class_names: dict[int, str]) -> list:
    """Un descriptor por clase presente en la máscara.

    Las clases ausentes no se declaran: un segmento vacío en un SEG se lee como "se buscó
    y no había", y acá muchas veces significa "este corte no incluye esa estructura".
    """
    from highdicom.sr.coding import Code
    from highdicom.seg import SegmentDescription
    from highdicom.seg.enum import SegmentAlgorithmTypeValues

    descriptions = []
    for position, label in enumerate(labels_present, start=1):
        name = class_names.get(label, f"class_{label}")
        code = Code(name, CODING_SCHEME, name)
        descriptions.append(SegmentDescription(
            segment_number=position,
            segment_label=name,
            # La misma categoría y tipo: el modelo no distingue tejido de órgano, y
            # afirmar una categoría anatómica que no esta validada seria inventarla.
            segmented_property_category=code,
            segmented_property_type=code,
            # AUTOMATIC seria decir que no hubo intervencion humana en la localizacion.
            # SEMIAUTOMATIC es lo que corresponde: el corte lo eligio el pipeline y la
            # revision profesional es obligatoria.
            algorithm_type=SegmentAlgorithmTypeValues.SEMIAUTOMATIC,
            algorithm_identification=_algorithm_identification(),
        ))
    return descriptions


def _algorithm_identification():
    from highdicom.content import AlgorithmIdentificationSequence

    return AlgorithmIdentificationSequence(
        name="PFI lumbar MRI assistant",
        version="1.0",
        family=_algorithm_family(),
    )


def _algorithm_family():
    from highdicom.sr.coding import Code

    # Codigo estandar de DCM para segmentacion por red neuronal. Este si es terminologia
    # publicada y verificable, a diferencia de los nombres de clase del modelo.
    return Code("123105", "DCM", "Artificial Intelligence")


def build_segmentation(
    *,
    series_dir: Path,
    slice_index: int,
    mask,
    class_names: dict[int, str],
    model_version: str,
    instance_number: int = 1,
):
    """Arma el objeto SEG de un corte segmentado.

    :param series_dir: directorio de la serie de origen, ya de-identificada.
    :param slice_index: corte que analizó el modelo, en el orden de la serie.
    :param mask: máscara de etiquetas 2D, un entero por píxel.
    :param class_names: nombres de las clases del modelo, del registro.
    :param model_version: versión del artefacto, para que el objeto diga qué lo produjo.
    """
    import numpy as np
    from highdicom.seg import Segmentation
    from highdicom.seg.enum import SegmentationTypeValues
    from pydicom.uid import generate_uid

    labels = np.asarray(mask)
    if labels.ndim != 2:
        raise SegmentationExportError(f"la mascara tiene que ser 2D, llego {labels.ndim}D")

    present = sorted(int(value) for value in np.unique(labels) if int(value) != 0)
    if not present:
        raise SegmentationExportError("la mascara no tiene ninguna estructura segmentada")

    source_images = _source_instances(series_dir, slice_index)

    # Un plano binario por segmento, en el orden en que se declaran los descriptores.
    # highdicom espera (frames, filas, columnas, segmentos) para multiples segmentos.
    planes = np.stack([(labels == label).astype(np.uint8) for label in present], axis=-1)
    pixel_array = planes[np.newaxis, ...]

    try:
        return Segmentation(
            source_images=source_images,
            pixel_array=pixel_array,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=_segment_descriptions(present, class_names),
            series_instance_uid=generate_uid(),
            series_number=100,
            sop_instance_uid=generate_uid(),
            instance_number=instance_number,
            manufacturer="PFI",
            manufacturer_model_name="PFI lumbar MRI assistant",
            software_versions=model_version or "unknown",
            device_serial_number="pfi-ai-module",
            series_description=SERIES_DESCRIPTION,
        )
    except Exception as exc:
        log.exception("event=dicom_seg_build_failed series_dir=%s slice=%s", series_dir.name, slice_index)
        raise SegmentationExportError(f"no se pudo construir el SEG: {type(exc).__name__}") from exc


def write_segmentation(segmentation, destination: Path) -> Path:
    """Escribe el SEG. Si falla, no deja un archivo a medias."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        segmentation.save_as(str(destination), enforce_file_format=True)
        return destination
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise SegmentationExportError(f"no se pudo escribir el SEG: {type(exc).__name__}") from exc


def segmentation_summary(segmentation) -> dict[str, Any]:
    """Lo que se puede publicar del objeto sin exponer rutas ni UIDs de origen."""
    segments = getattr(segmentation, "SegmentSequence", []) or []
    return {
        "sopInstanceUid": str(getattr(segmentation, "SOPInstanceUID", "")),
        "segmentCount": len(segments),
        "segments": [str(getattr(item, "SegmentLabel", "")) for item in segments],
        "codingScheme": CODING_SCHEME,
        # Lo que el objeto no afirma, dicho explicitamente: los nombres de segmento son del
        # registro del modelo, no terminologia clinica estandar.
        "standardTerminology": False,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
