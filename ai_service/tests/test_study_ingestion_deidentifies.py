"""Un estudio ingerido no deja datos del paciente en disco.

test_deidentify prueba el modulo aislado. Este prueba lo que realmente importa: que el
camino de ingesta completo -zip, clasificacion de series, registro- lo use, y que despues
de subir un estudio no quede un solo archivo con el nombre del paciente.

Se verifica **releyendo los archivos del disco**, no el valor de retorno. La API ya
ocultaba los identificadores antes de este cambio; lo que estaba mal era el dato guardado.
"""
import io
import zipfile

import numpy as np
import pytest
from pydicom import Dataset, dcmread
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from pfi_ai_service import input_registry
from pfi_ai_service.study_ingestion import register_study_zip

SAGITTAL = [0, 1, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]

PATIENT_NAME = "PEREZ^JUAN CARLOS"
PATIENT_ID = "DNI-30123456"
ACCESSION = "ACC-2026-00841"
INSTITUTION = "Hospital Central"
STUDY_UID = generate_uid()
FRAME_UID = generate_uid()


def _slice(series_uid, description, orientation, index):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = series_uid
    # Los dos planos comparten marco: de eso dependen la linea de referencia y el nivel.
    ds.FrameOfReferenceUID = FRAME_UID

    ds.PatientName = PATIENT_NAME
    ds.PatientID = PATIENT_ID
    ds.AccessionNumber = ACCESSION
    ds.InstitutionName = INSTITUTION
    ds.PatientBirthDate = "19750314"
    ds.StudyDate = "20260314"

    ds.Modality = "MR"
    ds.SeriesDescription = description
    ds.EchoTime = 95.0 if "t2" in description else 12.0
    ds.InstanceNumber = index + 1
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M"]
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = [0.0, 0.0, float(index) * 3.0]
    ds.PixelSpacing = [0.729, 0.729]
    ds.SliceThickness = 3.0
    ds.Rows = ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (np.random.rand(8, 8) * 1000).astype(np.uint16).tobytes()
    return ds


STUDY = [
    ("t2_tse_sag_384", [SAGITTAL] * 4),
    ("t2_tse_tra_384", [AXIAL] * 4),
]


@pytest.fixture
def ingested(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for description, orientations in STUDY:
            series_uid = generate_uid()
            for index, orientation in enumerate(orientations):
                slice_buffer = io.BytesIO()
                _slice(series_uid, description, orientation, index).save_as(
                    slice_buffer, enforce_file_format=True
                )
                archive.writestr(f"{description}/{index:03d}.dcm", slice_buffer.getvalue())
    buffer.seek(0)

    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    input_registry._INPUT_REGISTRY.clear()
    input_registry._REGISTRY_LOADED = False
    result = register_study_zip(case_id="CASE-DEIDENT", stream=buffer)
    yield result
    input_registry._INPUT_REGISTRY.clear()
    input_registry._REGISTRY_LOADED = False


def _stored_datasets():
    """Todo lo que quedo en disco, releido tal cual."""
    datasets = []
    for record in input_registry._INPUT_REGISTRY.values():
        for path in sorted(record.path.glob("*.dcm")):
            datasets.append(dcmread(str(path)))
    return datasets


def test_ninguna_serie_guardada_conserva_el_nombre_ni_el_documento(ingested):
    datasets = _stored_datasets()
    assert datasets, "la ingesta no dejo archivos"

    for ds in datasets:
        assert str(ds.PatientName) == ""
        assert str(ds.PatientID) == ""
        assert str(ds.AccessionNumber) == ""
        assert "InstitutionName" not in ds


def test_ningun_archivo_en_disco_contiene_los_identificadores_como_texto(ingested):
    """Barrido en crudo: si un tag se nos escapo, el string sigue estando en el archivo."""
    for record in input_registry._INPUT_REGISTRY.values():
        for path in record.path.glob("*.dcm"):
            blob = path.read_bytes()
            for secret in (PATIENT_NAME, PATIENT_ID, ACCESSION, INSTITUTION):
                assert secret.encode("ascii") not in blob, f"{secret} sigue en {path.name}"


def test_los_uids_del_estudio_original_no_sobreviven(ingested):
    for ds in _stored_datasets():
        assert ds.StudyInstanceUID != STUDY_UID
        assert ds.FrameOfReferenceUID != FRAME_UID


def test_las_dos_series_siguen_compartiendo_marco_de_referencia(ingested):
    """Si esto se rompe, se rompen la linea de referencia y el nivel del corte axial."""
    frames = {ds.FrameOfReferenceUID for ds in _stored_datasets()}
    assert len(frames) == 1


def test_cada_serie_sigue_siendo_una_sola_serie(ingested):
    for record in input_registry._INPUT_REGISTRY.values():
        uids = {dcmread(str(p)).SeriesInstanceUID for p in record.path.glob("*.dcm")}
        assert len(uids) == 1, f"{record.input_id} quedo partida en {len(uids)} series"

    # Y las dos series siguen siendo distintas entre si.
    por_serie = {
        next(iter({dcmread(str(p)).SeriesInstanceUID for p in record.path.glob("*.dcm")}))
        for record in input_registry._INPUT_REGISTRY.values()
    }
    assert len(por_serie) == len(input_registry._INPUT_REGISTRY)


def test_la_geometria_que_usa_la_clinica_sobrevive(ingested):
    for ds in _stored_datasets():
        assert list(ds.PixelSpacing) == [0.729, 0.729]
        assert len(ds.ImageOrientationPatient) == 6
        assert len(ds.ImagePositionPatient) == 3
        assert ds.SeriesDescription in {"t2_tse_sag_384", "t2_tse_tra_384"}
        assert ds.Modality == "MR"


def test_las_posiciones_de_los_cortes_no_se_alteran(ingested):
    """El nivel axial se calcula proyectando estas posiciones: no pueden moverse."""
    for record in input_registry._INPUT_REGISTRY.values():
        z = sorted(float(dcmread(str(p)).ImagePositionPatient[2]) for p in record.path.glob("*.dcm"))
        assert z == [0.0, 3.0, 6.0, 9.0]


def test_la_clasificacion_de_planos_sigue_funcionando_sobre_lo_de_identificado(ingested):
    """La de-identificacion corre antes de que el pipeline lea la serie: tiene que seguir leyendose."""
    planos = {r.plane for r in input_registry._INPUT_REGISTRY.values()}
    assert planos == {"sagittal", "axial"}


def test_las_series_registradas_quedan_marcadas_como_de_identificadas(ingested):
    for record in input_registry._INPUT_REGISTRY.values():
        assert record.deidentified is True
