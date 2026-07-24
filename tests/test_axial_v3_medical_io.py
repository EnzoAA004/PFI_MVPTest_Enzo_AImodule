from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from lumbar_mri.axial_v3 import training
from lumbar_mri.axial_v3.medical_io import open_2d_array
from lumbar_mri.axial_v3.training_medical import assert_medical_io_installed


def _write_test_dicom(path: Path, pixels: np.ndarray) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MR"
    dataset.Rows = int(pixels.shape[0])
    dataset.Columns = int(pixels.shape[1])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.asarray(pixels, dtype=np.uint16).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


def test_open_2d_array_reads_ima_dicom(tmp_path: Path) -> None:
    expected = np.arange(20, dtype=np.uint16).reshape(4, 5)
    path = tmp_path / "slice.ima"
    _write_test_dicom(path, expected)

    actual = open_2d_array(path)

    assert actual.shape == (4, 5)
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)


def test_training_medical_facade_installs_versioned_loader() -> None:
    assert_medical_io_installed()
    assert training.open_2d_array is open_2d_array
