"""Manual demo: build a 4-series multi-plane DICOM study (sag T1, sag T2, ax T2,
coronal STIR) and POST it to /inputs/study to show plane/weighting differentiation.
Run inside the container: python ai_service/tests/_manual_multiplane_demo.py
"""
import io
import zipfile

import httpx
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, MRImageStorage

STUDY_UID = generate_uid()

SAGITTAL = [0, 1, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]
CORONAL = [1, 0, 0, 0, 0, -1]

# (description, orientation, slice count, echo_time)
SERIES = [
    ("t2_tse_sag_320", SAGITTAL, 8, 110.0),
    ("t1_tse_sag_320", SAGITTAL, 8, 12.0),
    ("t2_tse_tra_384", AXIAL, 6, 105.0),
    ("stir_cor_384", CORONAL, 5, 60.0),
]


def make_slice(series_uid, description, orientation, index, echo, rows=32, cols=32):
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
    ds.EchoTime = echo
    ds.InstanceNumber = index + 1
    ds.ImageOrientationPatient = list(orientation)
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


def build_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for description, orientation, count, echo in SERIES:
            series_uid = generate_uid()
            for index in range(count):
                ds = make_slice(series_uid, description, orientation, index, echo)
                slice_buffer = io.BytesIO()
                pydicom.dcmwrite(slice_buffer, ds, write_like_original=False)
                archive.writestr(f"{description}/{index:03d}.dcm", slice_buffer.getvalue())
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    resp = httpx.post(
        "http://127.0.0.1:8000/inputs/study",
        files={"file": ("study.zip", build_zip(), "application/zip")},
        data={"caseId": "case-multiplane-demo"},
        timeout=120,
    )
    print("HTTP", resp.status_code)
    print(resp.text)
