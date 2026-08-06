"""Lo que sale del de-identificador no lleva al paciente de vuelta, y sigue sirviendo.

Dos mitades que se pueden romper por separado y las dos importan:

- Que los identificadores no esten. Es el motivo del modulo.
- Que la geometria clinica siga intacta y consistente. Un de-identificador que borra de
  mas deja imagenes sin escala y sin nivel, y eso se descubre mucho despues, cuando una
  medicion informa pixeles en vez de milimetros.
"""
import numpy as np
import pytest
from pydicom import Dataset, dcmread
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from pfi_ai_service.deidentify import (
    PRESERVED_CLINICAL_TAGS,
    DeidentificationError,
    UidRemapper,
    copy_deidentified,
    deidentify_dataset,
)


def _identified_slice(series_uid, study_uid, index=0, orientation=(1, 0, 0, 0, 1, 0)):
    """Un corte con todo lo que un DICOM real trae del hospital."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = study_uid  # los dos planos comparten marco

    # --- lo que identifica al paciente ---
    ds.PatientName = "PEREZ^JUAN CARLOS"
    ds.PatientID = "DNI-30123456"
    ds.PatientBirthDate = "19750314"
    ds.PatientSex = "M"
    ds.AccessionNumber = "ACC-2026-00841"
    ds.StudyID = "8841"
    ds.ReferringPhysicianName = "GOMEZ^ANA"
    ds.InstitutionName = "Hospital Central"
    ds.InstitutionAddress = "Av. Siempreviva 742"
    ds.OperatorsName = "TECNICO^LUIS"
    ds.StudyDate = "20260314"
    ds.StudyTime = "101500"
    ds.PatientComments = "control post quirurgico"
    ds.DeviceSerialNumber = "SN-99812"
    ds.StationName = "MR-01"
    # Un privado de fabricante, que es donde suelen esconderse mas identificadores.
    ds.add_new(0x00091001, "LO", "PACS-INTERNAL-ID-77213")

    # --- lo que la clinica necesita ---
    ds.Modality = "MR"
    ds.SeriesDescription = "t2_tse_sag_384"
    ds.EchoTime = 95.0
    ds.InstanceNumber = index + 1
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M"]
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = [float(index), 1.5, -2.5]
    ds.PixelSpacing = [0.729, 0.729]
    ds.SliceThickness = 3.0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    ds.Rows = ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (np.random.rand(8, 8) * 1000).astype(np.uint16).tobytes()
    return ds


# --- que no quede nada que identifique -------------------------------------------

@pytest.mark.parametrize("tag", [
    "PatientName", "PatientID", "PatientBirthDate", "AccessionNumber", "StudyID",
    "ReferringPhysicianName", "StudyDate", "StudyTime",
])
def test_los_identificadores_obligatorios_quedan_vacios(tag):
    ds = _identified_slice(generate_uid(), generate_uid())
    deidentify_dataset(ds, UidRemapper())

    # Se vacian y no se borran: son tipo 2, y borrarlos produce un DICOM que algunos
    # visores rechazan.
    assert tag in ds
    assert str(ds.data_element(tag).value) == ""


@pytest.mark.parametrize("tag", [
    "InstitutionName", "InstitutionAddress", "OperatorsName", "PatientComments",
    "DeviceSerialNumber", "StationName",
])
def test_los_identificadores_opcionales_se_borran(tag):
    ds = _identified_slice(generate_uid(), generate_uid())
    deidentify_dataset(ds, UidRemapper())

    assert tag not in ds


def test_los_privados_de_fabricante_se_van_enteros():
    ds = _identified_slice(generate_uid(), generate_uid())
    assert 0x00091001 in ds

    deidentify_dataset(ds, UidRemapper())

    assert 0x00091001 not in ds


def test_queda_declarado_que_el_estudio_fue_de_identificado():
    ds = _identified_slice(generate_uid(), generate_uid())
    deidentify_dataset(ds, UidRemapper())

    assert ds.PatientIdentityRemoved == "YES"
    assert "PS3.15" in ds.DeidentificationMethod


# --- que los UIDs corten el vinculo pero conserven la estructura -------------------

def test_los_uids_cambian():
    study, series = generate_uid(), generate_uid()
    ds = _identified_slice(series, study)
    original_sop = ds.SOPInstanceUID

    deidentify_dataset(ds, UidRemapper())

    assert ds.StudyInstanceUID != study
    assert ds.SeriesInstanceUID != series
    assert ds.SOPInstanceUID != original_sop


def test_los_cortes_de_una_serie_siguen_siendo_una_serie():
    """Sin mapa, cada corte recibiria un SeriesInstanceUID distinto y GDCM no los agruparia."""
    study, series = generate_uid(), generate_uid()
    remap = UidRemapper()
    cortes = [_identified_slice(series, study, index=i) for i in range(3)]
    for ds in cortes:
        deidentify_dataset(ds, remap)

    assert len({ds.SeriesInstanceUID for ds in cortes}) == 1
    assert len({ds.StudyInstanceUID for ds in cortes}) == 1
    # Cada corte sigue siendo una instancia distinta.
    assert len({ds.SOPInstanceUID for ds in cortes}) == 3


def test_los_dos_planos_siguen_compartiendo_marco_de_referencia():
    """De esto dependen la linea de referencia y la asignacion de nivel axial."""
    study = generate_uid()
    remap = UidRemapper()
    sagital = _identified_slice(generate_uid(), study, orientation=(0, 1, 0, 0, 0, -1))
    axial = _identified_slice(generate_uid(), study, orientation=(1, 0, 0, 0, 1, 0))
    deidentify_dataset(sagital, remap)
    deidentify_dataset(axial, remap)

    assert sagital.FrameOfReferenceUID == axial.FrameOfReferenceUID
    assert sagital.FrameOfReferenceUID != study
    # Y siguen siendo series distintas.
    assert sagital.SeriesInstanceUID != axial.SeriesInstanceUID


def test_dos_ingestas_del_mismo_estudio_no_producen_los_mismos_uids():
    """El mapeo no puede ser reproducible: seria el vinculo que se quiere cortar."""
    study, series = generate_uid(), generate_uid()
    primera = _identified_slice(series, study)
    segunda = _identified_slice(series, study)
    deidentify_dataset(primera, UidRemapper())
    deidentify_dataset(segunda, UidRemapper())

    assert primera.SeriesInstanceUID != segunda.SeriesInstanceUID


# --- que la clinica sobreviva ------------------------------------------------------

def test_la_geometria_y_la_escala_no_se_tocan():
    ds = _identified_slice(generate_uid(), generate_uid(), index=4)
    deidentify_dataset(ds, UidRemapper())

    assert list(ds.ImageOrientationPatient) == [1, 0, 0, 0, 1, 0]
    assert list(ds.ImagePositionPatient) == [4.0, 1.5, -2.5]
    assert list(ds.PixelSpacing) == [0.729, 0.729]
    assert ds.SliceThickness == 3.0
    assert ds.InstanceNumber == 5
    assert ds.RescaleSlope == 1.0


def test_lo_que_distingue_t1_de_t2_y_una_captura_de_consola_sobrevive():
    ds = _identified_slice(generate_uid(), generate_uid())
    deidentify_dataset(ds, UidRemapper())

    assert ds.SeriesDescription == "t2_tse_sag_384"
    assert ds.EchoTime == 95.0
    assert list(ds.ImageType) == ["ORIGINAL", "PRIMARY", "M"]
    assert ds.Modality == "MR"


def test_los_pixeles_no_se_alteran():
    ds = _identified_slice(generate_uid(), generate_uid())
    original = bytes(ds.PixelData)

    deidentify_dataset(ds, UidRemapper())

    assert bytes(ds.PixelData) == original


def test_ningun_tag_clinico_declarado_se_pierde():
    """PRESERVED_CLINICAL_TAGS documenta el contrato; este test lo hace cumplir."""
    ds = _identified_slice(generate_uid(), generate_uid())
    presentes_antes = {tag for tag in PRESERVED_CLINICAL_TAGS if tag in ds}

    deidentify_dataset(ds, UidRemapper())

    for tag in presentes_antes:
        assert tag in ds, f"{tag} desaparecio y esta declarado como preservado"


# --- el archivo en disco -----------------------------------------------------------

def test_el_archivo_escrito_no_trae_identificadores(tmp_path):
    source = tmp_path / "original.dcm"
    _identified_slice(generate_uid(), generate_uid()).save_as(str(source), enforce_file_format=True)
    destination = tmp_path / "limpio.dcm"

    copy_deidentified(source, destination, UidRemapper())

    releido = dcmread(str(destination))
    assert str(releido.PatientName) == ""
    assert str(releido.PatientID) == ""
    assert "InstitutionName" not in releido
    assert releido.PatientIdentityRemoved == "YES"
    # Y sigue siendo legible como imagen.
    assert releido.Rows == 8
    assert list(releido.PixelSpacing) == [0.729, 0.729]


def test_el_encabezado_del_archivo_queda_consistente_con_el_dataset(tmp_path):
    source = tmp_path / "original.dcm"
    _identified_slice(generate_uid(), generate_uid()).save_as(str(source), enforce_file_format=True)
    destination = tmp_path / "limpio.dcm"

    copy_deidentified(source, destination, UidRemapper())

    releido = dcmread(str(destination))
    assert releido.file_meta.MediaStorageSOPInstanceUID == releido.SOPInstanceUID


def test_un_archivo_que_no_se_puede_de_identificar_no_deja_nada_escrito(tmp_path):
    source = tmp_path / "no-es-dicom.dcm"
    source.write_bytes(b"esto no es un DICOM")
    destination = tmp_path / "salida.dcm"

    with pytest.raises(DeidentificationError):
        copy_deidentified(source, destination, UidRemapper())

    # Un archivo a medias se leeria como valido.
    assert not destination.exists()
