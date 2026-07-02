from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .pipeline import PipelineRunRequest, run_pipeline


class Point(BaseModel):
    x: float
    y: float


class MaskContour(BaseModel):
    series_id: str = Field(..., alias="seriesId")
    slice_index: int = Field(..., alias="sliceIndex")
    points: List[Point]


class StudySeries(BaseModel):
    id: str
    name: str
    plane: Literal["sagittal", "axial"]
    sequence: str
    slice_count: int = Field(..., alias="sliceCount")
    selected_slice: int = Field(..., alias="selectedSlice")
    image_url: Optional[str] = Field(default=None, alias="imageUrl")
    overlay_url: Optional[str] = Field(default=None, alias="overlayUrl")
    overlay_opacity: float = Field(default=0.74, alias="overlayOpacity")
    status: str = "ai_output_pending"


class StudyMask(BaseModel):
    id: str
    label: str
    class_name: str = Field(..., alias="className")
    color: str
    confidence: float
    editable: bool = True
    enabled: bool = True
    contours: List[MaskContour] = Field(default_factory=list)


class StudyLandmark(BaseModel):
    id: str
    label: str
    series_id: str = Field(..., alias="seriesId")
    slice_index: int = Field(..., alias="sliceIndex")
    x: float
    y: float
    editable: bool = True
    linked_mask_id: Optional[str] = Field(default=None, alias="linkedMaskId")


class StudyMeasurement(BaseModel):
    id: str
    label: str
    level: str
    value: float
    ai_value: float = Field(..., alias="aiValue")
    reviewer_value: Optional[float] = Field(default=None, alias="reviewerValue")
    unit: str
    source: Literal["AI", "Reviewer"] = "AI"
    confidence: float
    status: str = "pendiente"
    outlier: bool = False
    linked_landmarks: List[str] = Field(default_factory=list, alias="linkedLandmarks")


class AiOutputState(BaseModel):
    status: str
    label: str
    description: str
    inference_mode: Optional[str] = Field(default=None, alias="inferenceMode")
    requested_inference_mode: Optional[str] = Field(default=None, alias="requestedInferenceMode")
    real_inference_available: bool = Field(default=False, alias="realInferenceAvailable")
    human_review_required: bool = Field(default=True, alias="humanReviewRequired")
    not_clinical_diagnosis: bool = Field(default=True, alias="notClinicalDiagnosis")
    agent_decision: Dict[str, Any] = Field(default_factory=dict, alias="agentDecision")


class StudyReviewResponse(BaseModel):
    study_id: str = Field(..., alias="studyId")
    case_id: str = Field(..., alias="caseId")
    patient_id: str = Field(..., alias="patientId")
    study_date: str = Field(..., alias="studyDate")
    modality: str
    body_region: str = Field(..., alias="bodyRegion")
    review_status: str = Field(..., alias="reviewStatus")
    model_key: str = Field(..., alias="modelKey")
    model_version: str = Field(..., alias="modelVersion")
    ai_output: AiOutputState = Field(..., alias="aiOutput")
    series: List[StudySeries]
    masks: List[StudyMask]
    landmarks: List[StudyLandmark]
    measurements: List[StudyMeasurement]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def demo_study_review_contract() -> Dict[str, Any]:
    """Devuelve el mismo contrato visual que /pipeline/run para evitar drift entre demo y pipeline."""
    pipeline_response = run_pipeline(PipelineRunRequest(
        caseId="CASE-DEMO-0142",
        plane="sagittal",
        modelKey="sagittal_spider",
        inputPath="demo/CASE-DEMO-0142",
        metadata={
            "patientId": "PAT-0087",
            "studyDate": "2026-07-01",
            "source": "ai-module-study-contract",
            "inferenceMode": "contract",
        },
    ))
    response = StudyReviewResponse(
        studyId=pipeline_response["studyId"],
        caseId=pipeline_response["caseId"],
        patientId=pipeline_response["patientId"],
        studyDate=pipeline_response["studyDate"],
        modality=pipeline_response["modality"],
        bodyRegion=pipeline_response["bodyRegion"],
        reviewStatus=pipeline_response["reviewStatus"],
        modelKey=pipeline_response["modelKey"],
        modelVersion=pipeline_response["modelVersion"],
        aiOutput=pipeline_response["aiOutput"],
        series=pipeline_response["series"],
        masks=pipeline_response["masks"],
        landmarks=pipeline_response["landmarks"],
        measurements=pipeline_response["measurementValues"],
        metadata={
            **pipeline_response["metadata"],
            "source": "ai-module-study-contract",
            "sameContractAsPipelineRun": True,
        },
    )
    return response.model_dump(by_alias=True)
