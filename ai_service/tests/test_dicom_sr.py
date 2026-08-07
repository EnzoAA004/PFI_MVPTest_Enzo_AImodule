"""Las mediciones exportadas como DICOM SR.

Igual que con el SEG, lo que se protege es que el objeto sea **legible por otro sistema**.
Y una cosa más, propia del SR: que ninguna medición viaje con una unidad inventada. Un
número sin unidad correcta en un informe estructurado es peor que no exportarlo, porque el
sistema que lo lee le va a dar la unidad que suponga.
"""
import numpy as np
import pytest
from pydicom import Dataset, dcmread
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from pfi_ai_service.dicom_sr import (
    CODING_SCHEME,
    UCUM_UNITS,
    ReportExportError,
    build_measurement_report,
    write_report,
)

ROWS = COLS = 16
STUDY_UID = generate_uid()


def _slice(series_uid, index):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = generate_uid()
    # Tal como los deja deidentify.py: presentes y vacios.
    for tag in ("PatientID", "PatientName", "PatientBirthDate", "PatientSex",
                "StudyID", "AccessionNumber", "StudyDate", "StudyTime"):
        setattr(ds, tag, "")
    ds.PatientIdentityRemoved = "YES"
    ds.Modality = "MR"
    ds.SeriesDescription = "t2_tse_sag_384"
    ds.InstanceNumber = index + 1
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    ds.ImagePositionPatient = [float(index) * 3.0, 0.0, 0.0]
    ds.PixelSpacing = [0.729, 0.729]
    ds.SliceThickness = 3.0
    ds.Rows, ds.Columns = ROWS, COLS
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (np.random.rand(ROWS, COLS) * 1000).astype(np.uint16).tobytes()
    return ds


@pytest.fixture
def series(tmp_path):
    directory = tmp_path / "serie"
    directory.mkdir()
    series_uid = generate_uid()
    for index in range(4):
        _slice(series_uid, index).save_as(str(directory / f"{index:05d}.dcm"), enforce_file_format=True)
    return directory


def _texts(dataset):
    """Todos los valores de texto del árbol de contenido del SR.

    `str(dataset)` trunca las secuencias anidadas, así que buscar ahí da falsos negativos:
    una medición que está tres niveles adentro no aparece. Se recorre el árbol de verdad.
    """
    found = []

    def walk(item):
        for element in item:
            if element.VR == "SQ":
                for sub in element.value:
                    walk(sub)
            elif element.value is not None:
                found.append(str(element.value))

    walk(dataset)
    return found


def _has(dataset, needle):
    return any(needle in text for text in _texts(dataset))


def _measurement(label, value, unit, level):
    """Con la forma que publica la corrida."""
    return {
        "id": f"{level}-{label}".replace(" ", "-"),
        "labelKey": label, "value": value, "unit": unit,
        "level": level, "plane": "sagittal", "source": "ai",
    }


MEASUREMENTS = [
    _measurement("disc area", 156.71, "mm2", "L4-L5"),
    _measurement("disc width", 28.89, "mm", "L4-L5"),
    _measurement("disc height", 7.44, "mm", "L4-L5"),
    _measurement("segmental angle", 12.3, "deg", "L5-S1"),
]


def _build(series, measurements=None, slice_index=1):
    return build_measurement_report(
        series_dir=series,
        slice_index=slice_index,
        measurements=MEASUREMENTS if measurements is None else measurements,
        model_version="sagittal-spider-final-v1",
    )


# --- que otro sistema lo pueda abrir ------------------------------------------------

def test_el_archivo_escrito_se_relee_como_un_sr(series, tmp_path):
    destino = tmp_path / "measurements.dcm"
    write_report(_build(series), destino)

    releido = dcmread(str(destino))
    # 1.2.840.10008.5.1.4.1.1.88.33 es Comprehensive SR Storage.
    assert releido.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.33"
    assert releido.Modality == "SR"


def test_referencia_la_imagen_sobre_la_que_se_midio(series):
    fuente = dcmread(str(sorted(series.glob("*.dcm"))[1]))
    sr = _build(series, slice_index=1)

    referenced = sr.CurrentRequestedProcedureEvidenceSequence[0]
    sop_uids = {
        instance.ReferencedSOPInstanceUID
        for serie in referenced.ReferencedSeriesSequence
        for instance in serie.ReferencedSOPSequence
    }
    assert fuente.SOPInstanceUID in sop_uids


def test_pertenece_al_mismo_estudio(series):
    fuente = dcmread(str(sorted(series.glob("*.dcm"))[0]))
    sr = _build(series)

    assert sr.StudyInstanceUID == fuente.StudyInstanceUID
    assert sr.SeriesInstanceUID != fuente.SeriesInstanceUID


def test_hereda_el_contexto_de_paciente_de_identificado(series):
    sr = _build(series)
    assert str(sr.PatientName) == ""
    assert str(sr.PatientID) == ""


# --- lo que el informe declara ------------------------------------------------------

def test_el_informe_queda_marcado_como_incompleto(series):
    """PARTIAL, no COMPLETE: la lectura no termina hasta la revisión profesional."""
    assert _build(series).CompletionFlag == "PARTIAL"


def test_el_observador_es_un_dispositivo_y_no_una_persona(series):
    """Un SR con observador persona afirmaría que alguien midió. Acá midió un modelo."""
    assert _has(_build(series), "PFI lumbar MRI assistant")


def test_la_serie_avisa_que_no_es_apta_para_diagnostico(series):
    assert "no apto para diagnostico" in _build(series).SeriesDescription.lower()


# --- las unidades --------------------------------------------------------------------

def test_las_unidades_se_declaran_en_ucum(series, tmp_path):
    """UCUM sí es estándar: cualquier sistema interpreta mm y mm2 igual."""
    destino = tmp_path / "measurements.dcm"
    write_report(_build(series), destino)
    releido = dcmread(str(destino))

    assert _has(releido, "UCUM")
    for unit in ("mm", "mm2", "deg"):
        assert _has(releido, unit)


def test_una_medicion_con_unidad_desconocida_no_se_exporta(series):
    """No se le inventa un código: el sistema que lo lea le daría la unidad que suponga."""
    sr = _build(series, measurements=[
        _measurement("disc width", 28.89, "mm", "L4-L5"),
        _measurement("algo raro", 5.0, "parsecs", "L4-L5"),
    ])

    assert _has(sr, "disc width")
    assert not _has(sr, "algo raro")


def test_las_tres_unidades_del_pipeline_estan_mapeadas(series):
    # Si el pipeline agrega una unidad nueva y nadie la mapea, sus mediciones desaparecen
    # del informe en silencio. Este test es el recordatorio.
    assert set(UCUM_UNITS) == {"mm", "mm2", "deg"}


def test_un_valor_que_no_es_numero_no_se_exporta(series):
    sr = _build(series, measurements=[
        _measurement("disc width", 28.89, "mm", "L4-L5"),
        _measurement("rota", None, "mm", "L4-L5"),
    ])
    assert not _has(sr, "rota")


# --- agrupación ----------------------------------------------------------------------

def test_las_mediciones_se_agrupan_por_nivel(series):
    sr = _build(series)
    assert _has(sr, "L4-L5")
    assert _has(sr, "L5-S1")


def test_las_mediciones_sin_nivel_van_en_su_propio_grupo(series):
    """"Sin nivel asignado" es información: mezclarlas diría que pertenecen a un nivel."""
    sr = _build(series, measurements=[
        _measurement("disc width", 28.89, "mm", "L4-L5"),
        _measurement("medicion general", 3.0, "mm", ""),
    ])

    assert _has(sr, "sin nivel asignado")
    assert _has(sr, "L4-L5")


# --- los nombres de medicion ----------------------------------------------------------

def test_los_nombres_usan_el_esquema_privado_y_no_terminologia_clinica(series):
    """No se inventan códigos SNOMED ni RadLex para "disc area"."""
    assert _has(_build(series), CODING_SCHEME)
    assert CODING_SCHEME.startswith("99"), "el prefijo 99 marca el esquema como privado"


# --- lo que no se debe exportar --------------------------------------------------------

def test_una_corrida_sin_mediciones_no_produce_un_sr(series):
    with pytest.raises(ReportExportError):
        _build(series, measurements=[])


def test_una_corrida_donde_ninguna_medicion_es_exportable_tampoco(series):
    with pytest.raises(ReportExportError):
        _build(series, measurements=[_measurement("x", 1.0, "parsecs", "L4-L5")])


def test_un_corte_fuera_de_la_serie_se_rechaza(series):
    with pytest.raises(ReportExportError) as exc:
        _build(series, slice_index=99)
    assert "fuera de la serie" in str(exc.value)


def test_una_serie_sin_dicom_se_rechaza(tmp_path):
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    with pytest.raises(ReportExportError):
        build_measurement_report(
            series_dir=vacia, slice_index=0, measurements=MEASUREMENTS, model_version="v1",
        )
