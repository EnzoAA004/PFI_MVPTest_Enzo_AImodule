"""Manual smoke test for POST /inputs/study.

Generates a synthetic multi-series DICOM study (sagittal T2, sagittal T1, axial T2),
zips it, posts it to the running AI module and prints the classification result.
Run inside the container: python ai_service/tests/_manual_study_zip_test.py
"""
import io
import zipfile

import httpx
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, MRImageStorage

STUDY_UID = generate_uid()


def make_slice(series_uid, description, orientation, index, rows=32, cols=32):
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
    ds.ImageOrientationPatient = list(orientation)
    # step the slice position along the series normal so GDCM can order slices
    ds.ImagePositionPatient = [float(index) if i == 0 else 0.0 for i in range(3)]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (np.random.rand(rows, cols) * 1000).astype(np.uint16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


SAGITTAL = [0, 1, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]
SERIES = [
    ("t2_tse_sag_320", SAGITTAL, 8),
    ("t1_tse_sag_320", SAGITTAL, 8),
    ("t2_tse_tra_384", AXIAL, 6),
]


def build_study_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for description, orientation, count in SERIES:
            series_uid = generate_uid()
            for index in range(count):
                ds = make_slice(series_uid, description, orientation, index)
                slice_buffer = io.BytesIO()
                pydicom.dcmwrite(slice_buffer, ds, write_like_original=False)
                archive.writestr(f"{description}/{index:03d}.dcm", slice_buffer.getvalue())
    buffer.seek(0)
    return buffer


def main():
    resp = httpx.post(
        "http://127.0.0.1:8000/inputs/study",
        files={"file": ("study.zip", build_study_zip(), "application/zip")},
        data={"caseId": "case-smoke-001"},
        timeout=120,
    )
    print("HTTP", resp.status_code)
    print(resp.text)


if __name__ == "__main__":
    main()
