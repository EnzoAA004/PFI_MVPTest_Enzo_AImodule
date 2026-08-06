"""De-identificacion de los DICOM que entran al sistema.

El sistema ya trataba los identificadores con cuidado en la superficie de la API: el
contrato rechaza payloads que traigan UIDs, el `SeriesInstanceUID` se traduce a un indice
posicional porque ese UID lleva de vuelta al estudio en el PACS de origen, y `patientId`
es un placeholder fijo.

**Pero el archivo en disco era una copia byte a byte del original.** `PatientName`,
`PatientID`, `AccessionNumber` e `InstitutionName` sobrevivian intactos a la ingesta. La
proteccion cubria lo que el sistema *dice*, no lo que el sistema *guarda*.

Este modulo cierra eso. Aplica el perfil basico de de-identificacion de DICOM PS3.15
Anexo E: se sigue el estandar en vez de inventar una lista propia, porque la lista propia
siempre se olvida de un tag.

## Las dos cosas que hay que hacer bien

**Los UIDs se reasignan, no se borran.** Un DICOM sin `SeriesInstanceUID` no se puede
agrupar en una serie, y sin `FrameOfReferenceUID` los dos planos dejan de compartir marco
espacial: se rompen la linea de referencia y la asignacion de nivel axial, que son
justamente las dos cosas que cruzan sagital con axial. Por eso hay un mapa: dentro de una
misma ingesta, un UID original siempre recibe el mismo UID nuevo, y la estructura del
estudio se conserva sin que quede rastro del origen.

El mapa es nuevo en cada ingesta y los UIDs se generan al azar. Derivarlos del original
-por hash, digamos- haria el mapeo reproducible para cualquiera que tenga el estudio
fuente, que es exactamente el vinculo que se quiere cortar.

**La geometria clinica no se toca.** `ImageOrientationPatient`, `ImagePositionPatient`,
`PixelSpacing`, `EchoTime`, `ImageType`, `InstanceNumber` y las escalas de intensidad son
lo que hace que una medicion valga milimetros y que un corte sepa donde esta. Borrarlas
por prolijidad convertiria el estudio en imagenes sin escala.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class DeidentificationError(Exception):
    """No se pudo de-identificar un archivo, asi que no se puede guardar."""


#: Identificadores directos. Se vacian en vez de borrarse cuando el tag es de tipo 2
#: -obligatorio, puede estar vacio-, porque borrarlo produce un DICOM invalido que algunos
#: visores rechazan.
_BLANKED_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyID",
    "AccessionNumber",
    "ReferringPhysicianName",
    "StudyDate",
    "StudyTime",
)

#: Identificadores que no son obligatorios: se eliminan enteros.
_REMOVED_TAGS = (
    "OtherPatientIDs",
    "OtherPatientIDsSequence",
    "OtherPatientNames",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "PatientBirthName",
    "PatientBirthTime",
    "PatientInsurancePlanCodeSequence",
    "MilitaryRank",
    "BranchOfService",
    "MedicalRecordLocator",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "InstanceCreatorUID",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "PhysiciansOfRecord",
    "RequestingPhysician",
    "AdmittingDiagnosesDescription",
    "PatientComments",
    "StudyComments",
    "AdditionalPatientHistory",
    "DeviceSerialNumber",
    "StationName",
    "ContentDate",
    "ContentTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "AcquisitionDateTime",
    "SeriesDate",
    "SeriesTime",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "RequestAttributesSequence",
    "ScheduledProcedureStepDescription",
    "PerformedProcedureStepDescription",
    "PerformedProcedureStepID",
    "RequestedProcedureDescription",
    "RequestedProcedureID",
)

#: UIDs que se reasignan de forma consistente dentro de una ingesta. Ver el docstring:
#: borrarlos rompe el agrupamiento de series y el marco espacial compartido.
_REMAPPED_UID_TAGS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
    "SynchronizationFrameOfReferenceUID",
)

#: Lo que nunca se toca, y por que. Esta tupla no se usa en el codigo: existe para que la
#: proxima persona que agregue un tag a las listas de arriba vea de inmediato cual es el
#: costo de equivocarse. Los tests la verifican.
PRESERVED_CLINICAL_TAGS = (
    "ImageOrientationPatient",   # el plano de la serie, y de que lado esta cada cosa
    "ImagePositionPatient",      # donde esta cada corte: nivel axial y linea de referencia
    "PixelSpacing",              # sin esto una medicion no vale milimetros
    "SliceThickness",
    "SpacingBetweenSlices",
    "SeriesDescription",         # como el equipo nombro la serie: t2_tse_sag_384
    "EchoTime",                  # T1 vs T2 cuando la descripcion no alcanza
    "ImageType",                 # DERIVED/SECONDARY: capturas de consola no analizables
    "InstanceNumber",            # orden de los cortes
    "RescaleSlope",
    "RescaleIntercept",
    "Modality",
    "Rows",
    "Columns",
    "BitsAllocated",
    "PixelRepresentation",
)


class UidRemapper:
    """Asigna un UID nuevo a cada UID original, y siempre el mismo dentro de una ingesta.

    Una instancia por estudio. Sin esto, dos cortes de la misma serie recibirian
    `SeriesInstanceUID` distintos y dejarian de ser una serie.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def __call__(self, original: str) -> str:
        from pydicom.uid import generate_uid

        key = str(original)
        if key not in self._map:
            self._map[key] = generate_uid()
        return self._map[key]

    def __len__(self) -> int:
        return len(self._map)


def deidentify_dataset(dataset, remap: UidRemapper) -> None:
    """De-identifica un dataset en memoria, in place."""
    for name in _BLANKED_TAGS:
        if name in dataset:
            dataset.data_element(name).value = ""

    for name in _REMOVED_TAGS:
        if name in dataset:
            delattr(dataset, name)

    for name in _REMAPPED_UID_TAGS:
        current = getattr(dataset, name, None)
        if current:
            setattr(dataset, name, remap(str(current)))

    # Los privados de fabricante son el escondite habitual de identificadores: Siemens y
    # GE guardan ahi numero de paciente y datos del protocolo. No hay forma de auditarlos
    # tag por tag, asi que se van todos.
    dataset.remove_private_tags()

    # Marca de que paso por aca, como pide PS3.15. Sirve para distinguir un estudio
    # ingerido despues de este cambio de uno anterior.
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = "PFI basic profile (DICOM PS3.15 Annex E)"

    # El encabezado del archivo tambien lleva el SOP Instance UID, y tiene que coincidir
    # con el del dataset o el archivo queda internamente inconsistente.
    meta = getattr(dataset, "file_meta", None)
    if meta is not None and getattr(meta, "MediaStorageSOPInstanceUID", None):
        meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID


def copy_deidentified(source: Path, destination: Path, remap: UidRemapper) -> None:
    """Copia un DICOM de-identificandolo. Si no se puede, no escribe nada.

    Fallar es deliberado y es lo unico seguro: un archivo con el nombre del paciente,
    dentro de un sistema que promete lo contrario, es peor que un estudio que no se pudo
    cargar. El llamador aborta la ingesta.
    """
    try:
        import pydicom

        # `force=True` hace falta porque el estudio trae archivos .ima sin preambulo, que
        # son DICOM validos. Pero tambien hace que un archivo que no es DICOM se lea como
        # un dataset vacio en vez de fallar: sin el chequeo de abajo, esa basura se
        # escribiria como si fuera una imagen de-identificada.
        dataset = pydicom.dcmread(str(source), force=True)
        if not getattr(dataset, "SOPInstanceUID", None):
            raise ValueError("el archivo no tiene SOPInstanceUID: no es una instancia DICOM")
        deidentify_dataset(dataset, remap)
        dataset.save_as(str(destination), enforce_file_format=False)
    except Exception as exc:
        # No se deja a medias: un archivo escrito parcialmente se lee como valido.
        destination.unlink(missing_ok=True)
        log.exception("event=deidentification_failed source=%s", source.name)
        raise DeidentificationError(
            f"no se pudo de-identificar el archivo DICOM: {type(exc).__name__}"
        ) from exc
