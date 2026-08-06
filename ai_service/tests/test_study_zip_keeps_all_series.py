"""La ingesta de un estudio conserva todas sus series, no solo las dos que analiza.

Reproduce la forma del estudio de referencia: localizer multiplano, sagital T1 y T2,
axial T1 y T2, y una captura de consola de dos cortes. Antes de este cambio sobrevivian
dos de seis y el resto se borraba con el directorio del zip.
"""
import io
import zipfile

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from pfi_ai_service import input_registry
from pfi_ai_service.input_registry import InputRegistryError, resolve_input_id, resolve_viewable_input
from pfi_ai_service.real_inference_runtime import render_series_previews, series_preview_dir
from pfi_ai_service.study_ingestion import register_study_zip

SAGITTAL = [0, 1, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]
CORONAL = [1, 0, 0, 0, 0, -1]
STUDY_UID = generate_uid()


def _slice(series_uid, description, orientation, index, derived=False, rows=16, cols=16):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesDescription = description
    ds.InstanceNumber = index + 1
    ds.ImageType = ["DERIVED", "SECONDARY"] if derived else ["ORIGINAL", "PRIMARY", "M"]
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = [float(index), 0.0, 0.0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (np.random.rand(rows, cols) * 1000).astype(np.uint16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


# (descripcion, orientaciones por corte, derivada)
STUDY = [
    ("localizer", [SAGITTAL, SAGITTAL, CORONAL, CORONAL, AXIAL, AXIAL], False),
    ("t2_tse_sag_384", [SAGITTAL] * 6, False),
    ("t1_tse_sag_320", [SAGITTAL] * 6, False),
    ("t2_tse_tra_384", [AXIAL] * 6, False),
    ("t1_tse_tra", [AXIAL] * 6, False),
    ("PosDisp: [5] t1_tse_tra", [AXIAL] * 2, True),
]


@pytest.fixture
def study_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for description, orientations, derived in STUDY:
            series_uid = generate_uid()
            for index, orientation in enumerate(orientations):
                ds = _slice(series_uid, description, orientation, index, derived=derived)
                slice_buffer = io.BytesIO()
                ds.save_as(slice_buffer, write_like_original=False)
                archive.writestr(f"{description}/{index:03d}.dcm", slice_buffer.getvalue())
    buffer.seek(0)
    return buffer


@pytest.fixture
def ingested(tmp_path, monkeypatch, study_zip):
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    input_registry._INPUT_REGISTRY.clear()
    return register_study_zip(case_id="CASE-TEST", stream=study_zip)


def _by_description(result, description):
    return next(s for s in result["seriesFound"] if s["description"] == description)


def test_se_listan_todas_las_series(ingested):
    assert len(ingested["seriesFound"]) == len(STUDY)


def test_toda_serie_queda_registrada_y_recuperable(ingested):
    for series in ingested["seriesFound"]:
        assert series["inputId"], f"{series['description']} sin inputId"
        record = input_registry._INPUT_REGISTRY[series["inputId"]]
        assert record.path.exists()
        assert len(list(record.path.iterdir())) == series["sliceCount"]


def test_la_ia_sigue_eligiendo_los_t2(ingested):
    assert ingested["sagittal"]["description"] == "t2_tse_sag_384"
    assert ingested["axial"]["description"] == "t2_tse_tra_384"


def test_el_localizer_queda_marcado_multiplano_y_no_analizable(ingested):
    localizer = _by_description(ingested, "localizer")
    assert localizer["multiplanar"] is True
    assert localizer["analyzable"] is False


def test_la_captura_de_consola_no_es_analizable(ingested):
    captura = _by_description(ingested, "PosDisp: [5] t1_tse_tra")
    assert captura["derived"] is True
    assert captura["analyzable"] is False


def test_el_axial_t1_se_conserva_aunque_no_haya_modelo(ingested):
    # No hay modelo axial T1 (Al-Kafri es T2 puro), pero el medico lo necesita ver.
    axial_t1 = _by_description(ingested, "t1_tse_tra")
    assert axial_t1["inputId"]
    assert input_registry._INPUT_REGISTRY[axial_t1["inputId"]].path.exists()


def test_una_serie_de_solo_vista_no_puede_entrar_a_inferencia(ingested):
    # La T1 sagital tiene plano `sagittal`, asi que el chequeo de plano la dejaria
    # pasar: lo que la frena es la marca de no analizable.
    sagital_t1 = _by_description(ingested, "t1_tse_sag_320")
    with pytest.raises(InputRegistryError) as error:
        resolve_input_id(sagital_t1["inputId"], case_id="CASE-TEST", plane="sagittal")
    assert error.value.status_code == 409


def test_la_serie_elegida_si_puede_entrar_a_inferencia(ingested):
    record = resolve_input_id(ingested["sagittal"]["inputId"], case_id="CASE-TEST", plane="sagittal")
    assert record.analyzable is True


def test_una_serie_sin_corrida_igual_se_puede_mostrar(ingested):
    # El axial T1: no hay modelo que lo segmente, pero sus cortes se rinden igual.
    axial_t1 = _by_description(ingested, "t1_tse_tra")
    record = resolve_viewable_input(axial_t1["inputId"])
    count = render_series_previews(record.input_id, str(record.path))
    assert count == axial_t1["sliceCount"]
    rendered = series_preview_dir(record.input_id)
    assert (rendered / "slice-000.png").is_file()
    assert (rendered / f"slice-{count - 1:03d}.png").is_file()


def test_los_cortes_se_rinden_una_sola_vez(ingested):
    # Apilar el volumen es lo caro; el segundo pedido tiene que salir del disco.
    record = resolve_viewable_input(_by_description(ingested, "t1_tse_sag_320")["inputId"])
    first = render_series_previews(record.input_id, str(record.path))
    marker = series_preview_dir(record.input_id) / "count"
    stamped = marker.stat().st_mtime_ns
    assert render_series_previews(record.input_id, str(record.path)) == first
    assert marker.stat().st_mtime_ns == stamped
