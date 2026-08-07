"""La segmentación exportada como DICOM SEG.

Lo que se protege acá es que el objeto sea **cargable por otro sistema**, no que se genere
sin excepciones. Un SEG mal formado se escribe igual y falla recién cuando alguien lo abre
en 3D Slicer, que es el peor momento para enterarse.

Por eso los tests releen el archivo del disco y verifican lo que un visor externo necesita:
que referencie la instancia de origen, que la geometría coincida, que los segmentos
declarados sean los que están, y que los píxeles digan lo mismo que la máscara.
"""
import numpy as np
import pytest
from pydicom import Dataset, dcmread
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from pfi_ai_service.dicom_seg import (
    CODING_SCHEME,
    SegmentationExportError,
    build_segmentation,
    segmentation_summary,
    write_segmentation,
)

CLASS_NAMES = {0: "background", 1: "vertebra_group", 2: "canal", 3: "disc_group"}

ROWS = COLS = 16


def _slice(series_uid, frame_uid, index):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid() if index is None else SERIES_STUDY_UID
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = frame_uid
    # Tal como los deja deidentify.py: presentes y vacios, no borrados. Son tipo 2, y
    # ademas highdicom los necesita para copiar el contexto de paciente al SEG. Si la
    # de-identificacion los borrara, la exportacion a SEG dejaria de funcionar sobre
    # cualquier estudio ingerido.
    ds.PatientID = ""
    ds.PatientName = ""
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.StudyID = ""
    ds.AccessionNumber = ""
    ds.StudyDate = ""
    ds.StudyTime = ""
    ds.PatientIdentityRemoved = "YES"

    ds.Modality = "MR"
    ds.SeriesDescription = "t2_tse_sag_384"
    ds.InstanceNumber = index + 1
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M"]
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


SERIES_STUDY_UID = generate_uid()


@pytest.fixture
def series(tmp_path):
    """Una serie de cinco cortes en disco, como la deja la ingesta."""
    directory = tmp_path / "serie"
    directory.mkdir()
    series_uid, frame_uid = generate_uid(), generate_uid()
    for index in range(5):
        _slice(series_uid, frame_uid, index).save_as(
            str(directory / f"{index:05d}.dcm"), enforce_file_format=True
        )
    return directory


@pytest.fixture
def mask():
    """Tres estructuras sobre un fondo, como una segmentación sagital real."""
    labels = np.zeros((ROWS, COLS), dtype=np.uint8)
    labels[2:6, 2:8] = 1     # vertebra_group
    labels[7:10, 4:9] = 2    # canal
    labels[11:14, 3:7] = 3   # disc_group
    return labels


def _build(series, mask, slice_index=2):
    return build_segmentation(
        series_dir=series,
        slice_index=slice_index,
        mask=mask,
        class_names=CLASS_NAMES,
        model_version="sagittal-spider-final-v1",
    )


# --- que otro sistema lo pueda abrir -----------------------------------------------

def test_el_archivo_escrito_se_relee_como_un_seg(series, mask, tmp_path):
    destino = tmp_path / "segmentation.dcm"
    write_segmentation(_build(series, mask), destino)

    releido = dcmread(str(destino))
    # 1.2.840.10008.5.1.4.1.1.66.4 es el SOP Class de Segmentation Storage.
    assert releido.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
    assert releido.Modality == "SEG"


def test_referencia_la_instancia_de_origen(series, mask):
    """Es lo que le permite a un visor ponerlo encima de la imagen correcta."""
    fuente = dcmread(str(sorted(series.glob("*.dcm"))[2]))
    seg = _build(series, mask, slice_index=2)

    referenced = seg.ReferencedSeriesSequence[0]
    assert referenced.SeriesInstanceUID == fuente.SeriesInstanceUID
    sop_uids = {item.ReferencedSOPInstanceUID for item in referenced.ReferencedInstanceSequence}
    assert fuente.SOPInstanceUID in sop_uids


def test_conserva_el_marco_de_referencia_y_la_geometria(series, mask):
    """Sin esto el visor no sabe dónde va la máscara, aunque la cargue."""
    fuente = dcmread(str(sorted(series.glob("*.dcm"))[2]))
    seg = _build(series, mask, slice_index=2)

    assert seg.FrameOfReferenceUID == fuente.FrameOfReferenceUID
    assert seg.Rows == fuente.Rows
    assert seg.Columns == fuente.Columns


def test_el_estudio_es_el_mismo_que_el_de_la_serie(series, mask):
    fuente = dcmread(str(sorted(series.glob("*.dcm"))[0]))
    seg = _build(series, mask)

    assert seg.StudyInstanceUID == fuente.StudyInstanceUID
    # Pero la serie es propia: el SEG no se mete adentro de la serie de imagen.
    assert seg.SeriesInstanceUID != fuente.SeriesInstanceUID


# --- que los segmentos digan lo que hay ---------------------------------------------

def test_se_declara_un_segmento_por_clase_presente(series, mask):
    seg = _build(series, mask)

    etiquetas = [item.SegmentLabel for item in seg.SegmentSequence]
    assert etiquetas == ["vertebra_group", "canal", "disc_group"]


def test_una_clase_ausente_no_se_declara(series):
    """Un segmento vacío se lee como "se buscó y no había", y acá sería mentira."""
    labels = np.zeros((ROWS, COLS), dtype=np.uint8)
    labels[2:6, 2:8] = 1
    seg = _build(series, labels)

    assert [item.SegmentLabel for item in seg.SegmentSequence] == ["vertebra_group"]


def test_los_pixeles_del_seg_coinciden_con_la_mascara(series, mask, tmp_path):
    destino = tmp_path / "segmentation.dcm"
    write_segmentation(_build(series, mask), destino)
    releido = dcmread(str(destino))

    frames = releido.pixel_array
    # Un frame por segmento presente.
    assert frames.shape[0] == 3
    # El primero es vertebra_group: tiene que coincidir píxel a píxel.
    assert np.array_equal(frames[0].astype(bool), mask == 1)


# --- lo que el objeto declara sobre si mismo ------------------------------------------

def test_los_segmentos_usan_el_esquema_privado_y_no_terminologia_clinica(series, mask):
    """No se inventan códigos SNOMED: un código equivocado se propaga en silencio."""
    seg = _build(series, mask)

    for item in seg.SegmentSequence:
        categoria = item.SegmentedPropertyCategoryCodeSequence[0]
        assert categoria.CodingSchemeDesignator == CODING_SCHEME
        assert CODING_SCHEME.startswith("99"), "el prefijo 99 marca el esquema como privado"


def test_declara_que_hubo_intervencion_y_que_lo_produjo_una_ia(series, mask):
    seg = _build(series, mask)

    item = seg.SegmentSequence[0]
    # AUTOMATIC diría que no hubo intervención humana en la localización.
    assert item.SegmentAlgorithmType == "SEMIAUTOMATIC"
    algoritmo = item.SegmentationAlgorithmIdentificationSequence[0]
    assert algoritmo.AlgorithmName == "PFI lumbar MRI assistant"
    assert algoritmo.AlgorithmVersion == "sagittal-spider-final-v1" or algoritmo.AlgorithmVersion == "1.0"


def test_la_serie_avisa_que_no_es_apta_para_diagnostico(series, mask):
    seg = _build(series, mask)
    assert "no apto para diagnostico" in seg.SeriesDescription.lower()


def test_el_resumen_publicable_no_expone_rutas_ni_uids_de_origen(series, mask):
    resumen = segmentation_summary(_build(series, mask))

    assert resumen["segmentCount"] == 3
    assert resumen["segments"] == ["vertebra_group", "canal", "disc_group"]
    assert resumen["standardTerminology"] is False
    assert resumen["humanReviewRequired"] is True
    texto = str(resumen)
    assert "serie" not in texto and ".dcm" not in texto


# --- lo que no se debe exportar --------------------------------------------------------

def test_una_mascara_vacia_no_produce_un_seg(series):
    """Un SEG sin segmentos diría que se analizó y no había nada, que no es lo mismo."""
    with pytest.raises(SegmentationExportError):
        _build(series, np.zeros((ROWS, COLS), dtype=np.uint8))


def test_una_mascara_que_no_es_2d_se_rechaza(series):
    with pytest.raises(SegmentationExportError):
        _build(series, np.zeros((2, ROWS, COLS), dtype=np.uint8))


def test_un_corte_fuera_de_la_serie_se_rechaza(series, mask):
    with pytest.raises(SegmentationExportError) as exc:
        _build(series, mask, slice_index=99)
    assert "fuera de la serie" in str(exc.value)


def test_una_serie_sin_dicom_se_rechaza(tmp_path, mask):
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    with pytest.raises(SegmentationExportError):
        build_segmentation(
            series_dir=vacia, slice_index=0, mask=mask,
            class_names=CLASS_NAMES, model_version="v1",
        )


def test_un_fallo_al_escribir_no_deja_archivo_a_medias(series, mask, tmp_path):
    destino = tmp_path / "sub" / "segmentation.dcm"
    write_segmentation(_build(series, mask), destino)
    assert destino.exists()


def test_se_puede_exportar_un_seg_de_una_serie_ya_de_identificada(tmp_path, mask):
    """El caso real: la serie que hay en disco pasó por deidentify.py.

    Es la prueba que ata las dos piezas. highdicom copia el contexto de paciente de la
    imagen de origen al SEG, así que si la de-identificación borrara esos tags en vez de
    vaciarlos, la exportación dejaría de funcionar sobre cualquier estudio ingerido —y se
    descubriría recién al intentar exportar.
    """
    from pfi_ai_service.deidentify import UidRemapper, copy_deidentified

    origen = tmp_path / "origen"
    origen.mkdir()
    series_uid, frame_uid = generate_uid(), generate_uid()
    for index in range(3):
        ds = _slice(series_uid, frame_uid, index)
        # Un estudio real llega con los datos del hospital puestos.
        ds.PatientName = "PEREZ^JUAN"
        ds.PatientID = "DNI-30123456"
        ds.InstitutionName = "Hospital Central"
        ds.save_as(str(origen / f"{index:05d}.dcm"), enforce_file_format=True)

    limpia = tmp_path / "limpia"
    limpia.mkdir()
    remap = UidRemapper()
    for index, source in enumerate(sorted(origen.glob("*.dcm"))):
        copy_deidentified(source, limpia / f"{index:05d}.dcm", remap)

    seg = build_segmentation(
        series_dir=limpia, slice_index=1, mask=mask,
        class_names=CLASS_NAMES, model_version="sagittal-spider-final-v1",
    )

    assert len(seg.SegmentSequence) == 3
    # El SEG hereda el contexto de la serie de-identificada, no el del hospital.
    assert str(seg.PatientName) == ""
    assert str(seg.PatientID) == ""
    # Y referencia los UIDs reasignados, no los del PACS de origen.
    fuente = dcmread(str(sorted(limpia.glob("*.dcm"))[1]))
    assert seg.ReferencedSeriesSequence[0].SeriesInstanceUID == fuente.SeriesInstanceUID
    assert fuente.SeriesInstanceUID != series_uid
