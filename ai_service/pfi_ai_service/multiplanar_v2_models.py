from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PlaneNameV2 = Literal["sagittal", "axial"]
InferenceModeV2 = Literal["contract", "mock", "real_baseline"]
EffectiveInferenceModeV2 = Literal["contract", "mock", "real_baseline", "mixed"]
WorkspaceModeV2 = Literal["sagittal_only", "axial_only", "dual_plane"]
PlaneStatusV2 = Literal["ready", "blocked", "error"]


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)


class PublicResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlaneExecutionV2Request(StrictRequestModel):
    inputId: str
    modelKey: str

    @field_validator("inputId", "modelKey")
    @classmethod
    def non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


class MultiplanarPlanesV2Request(StrictRequestModel):
    sagittal: PlaneExecutionV2Request | None = None
    axial: PlaneExecutionV2Request | None = None

    @model_validator(mode="after")
    def at_least_one_plane(self) -> "MultiplanarPlanesV2Request":
        if self.sagittal is None and self.axial is None:
            raise ValueError("NO_PLANE_REQUESTED")
        return self


class MultiplanarOptionsV2(StrictRequestModel):
    sliceIndex: int | None = None
    sliceAxis: int | None = None
    sliceWindowRadius: int = 3
    inputOrientationTransform: str | None = None


class MultiplanarRunV2Request(StrictRequestModel):
    caseId: str
    traceId: str | None = None
    inferenceMode: InferenceModeV2 = "real_baseline"
    allowContractFallback: bool = False
    planes: MultiplanarPlanesV2Request
    options: MultiplanarOptionsV2 = Field(default_factory=MultiplanarOptionsV2)

    @field_validator("caseId")
    @classmethod
    def case_id_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("caseId must not be empty")
        return normalized

    @field_validator("inferenceMode", mode="before")
    @classmethod
    def normalize_real_alias(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() == "real":
            return "real_baseline"
        return value


class PlaneModelV2(PublicResponseModel):
    key: str
    version: str | None
    readiness: str
    trainingStatus: str | None
    artifactHash: str | None
    baselineReady: bool
    availableForRealInference: bool
    manifestStatus: str | None
    manifestValid: bool


class PlaneInputV2(PublicResponseModel):
    inputId: str
    format: str | None
    sizeBytes: int | None
    nativeShape: list[int] | None
    canonicalShape: list[int] | None
    orientationTransform: str | None
    spacingXyzMm: list[float] | None
    canonicalAxisSpacingMm: list[float] | None
    selectedSliceIndex: int | None
    sliceCount: int | None
    selectedAxis: int | None
    inPlaneSpacingMm: list[float] | None


class CoordinateSpaceV2(PublicResponseModel):
    name: str
    width: int
    height: int
    units: Literal["pixel"]
    origin: Literal["top_left"]
    xDirection: Literal["right"]
    yDirection: Literal["down"]
    sourceSliceIndex: int | None
    sourceAxis: int | None


class PlaneSeriesV2(PublicResponseModel):
    id: str
    plane: PlaneNameV2
    sequence: str | None
    selectedSliceIndex: int | None
    sliceCount: int | None
    status: str


class PlaneAssetV2(PublicResponseModel):
    assetName: str
    role: Literal["input_preview", "overlay", "mask_preview", "mask_array", "confidence_array", "mesh_3d", "volume_3d"]
    contentType: str
    generated: bool
    relativePath: str


class PlaneMaskV2(PublicResponseModel):
    id: str
    classKey: str
    confidence: float | None
    enabled: bool
    editable: bool
    coordinateSpace: str
    geometry: dict[str, Any] | None = None


class PlaneLandmarkV2(PublicResponseModel):
    id: str
    labelKey: str
    x: float
    y: float
    confidence: float | None = None
    coordinateSpace: str
    source: Literal["ai", "derived_from_mask"]


class PlaneMeasurementV2(PublicResponseModel):
    id: str
    labelKey: str
    value: float
    unit: str
    confidence: float | None
    source: str
    status: Literal["pending_review"]
    plane: PlaneNameV2
    level: str | None = None
    measurementBasis: Literal["physical_spacing", "pixel_space"]
    linkedLandmarkIds: list[str] = Field(default_factory=list)


class PlaneQualityV2(PublicResponseModel):
    maskCount: int
    landmarkCount: int
    measurementCount: int
    meanConfidence: float | None = None
    meanForegroundConfidence: float | None = None
    foregroundRatio: float | None = None
    warnings: list[str] = Field(default_factory=list)


class PlaneRunV2Result(PublicResponseModel):
    status: PlaneStatusV2
    plane: PlaneNameV2
    runId: str
    effectiveInferenceMode: InferenceModeV2
    model: PlaneModelV2
    input: PlaneInputV2
    coordinateSpace: CoordinateSpaceV2
    series: list[PlaneSeriesV2]
    assets: list[PlaneAssetV2]
    masks: list[PlaneMaskV2]
    landmarks: list[PlaneLandmarkV2]
    measurements: list[PlaneMeasurementV2]
    quality: PlaneQualityV2
    synthetic: bool
    fallbackReason: str | None = None


class MultiplanarReadinessV2(PublicResponseModel):
    sagittal: bool
    axial: bool
    dual: bool


class ThreeDStatusV2(PublicResponseModel):
    enabled: bool
    status: Literal[
        "blocked_missing_axial",
        "blocked_missing_sagittal",
        "pending_registered_reconstruction",
        "experimental_ready",
        "experimental_blocked_insufficient_geometry",
    ]
    sourcePlaneRunIds: dict[PlaneNameV2, str | None]
    requiredInputs: list[str]
    assets: list[PlaneAssetV2] = Field(default_factory=list)
    reconstruction: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class WorkspaceQualityV2(PublicResponseModel):
    planeCount: int
    maskCount: int
    landmarkCount: int
    measurementCount: int
    byPlane: dict[PlaneNameV2, PlaneQualityV2 | None]


class ReviewPolicyV2(PublicResponseModel):
    status: Literal["pending"]
    required: bool
    approvalRequiresHumanConfirmation: bool


class GovernanceV2(PublicResponseModel):
    humanReviewRequired: bool
    notClinicalDiagnosis: bool
    deidentified: bool = True
    diagnosisGenerated: bool = False


class MultiplanarRunV2Response(PublicResponseModel):
    status: Literal["completed"]
    schemaVersion: Literal["pfi.multiplanar-run.v2"]
    runId: str
    traceId: str
    caseId: str
    workspaceMode: WorkspaceModeV2
    requestedInferenceMode: InferenceModeV2
    effectiveInferenceMode: EffectiveInferenceModeV2
    requestedPlanes: list[PlaneNameV2]
    completedPlanes: list[PlaneNameV2]
    readiness: MultiplanarReadinessV2
    planes: dict[PlaneNameV2, PlaneRunV2Result | None]
    threeD: ThreeDStatusV2
    quality: WorkspaceQualityV2
    review: ReviewPolicyV2
    governance: GovernanceV2
    synthetic: bool
    fallbackReason: str | None = None


class StructuredAiErrorV2(PublicResponseModel):
    status: Literal["error"]
    schemaVersion: Literal["pfi.error.v2"]
    code: Literal[
        "INVALID_MULTIPLANAR_REQUEST",
        "NO_PLANE_REQUESTED",
        "INPUT_NOT_FOUND",
        "MODEL_NOT_FOUND",
        "MODEL_PLANE_MISMATCH",
        "MODEL_NOT_READY",
        "REAL_INFERENCE_FAILED",
        "UNSUPPORTED_INFERENCE_MODE",
        "CONTRACT_FALLBACK_DISABLED",
        "INVALID_MULTIPLANAR_RESPONSE",
    ]
    message: str
    traceId: str
    caseId: str | None = None
    requestedPlanes: list[PlaneNameV2] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    governance: GovernanceV2
