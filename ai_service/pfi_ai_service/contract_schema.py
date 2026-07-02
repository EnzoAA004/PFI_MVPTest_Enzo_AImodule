from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS

REQUIRED_ROOT_FIELDS = [
    "runId",
    "caseId",
    "patientId",
    "studyDate",
    "plane",
    "modelKey",
    "series",
    "masks",
    "landmarks",
    "measurementValues",
    "aiOutput",
    "modelArtifact",
    "quality",
    "metadata",
    "review",
]


def _schema_hash(schema: Dict[str, Any]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hashable_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in schema.items() if key != "schemaHash"}


def pipeline_contract_schema() -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "schemaVersion": "visual-review-contract-v1",
        "status": "stable",
        "purpose": "Describe la estructura esperada de /pipeline/run para frontend, backend y defensa academica.",
        "generatedBy": "pfi-ai-module.contract_schema",
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        "rootFields": {
            "runId": "Identificador estable de la corrida tecnica.",
            "caseId": "Identificador de caso de-identificado.",
            "studyId": "Identificador academico del estudio.",
            "patientId": "Referencia de-identificada del sujeto.",
            "studyDate": "Fecha del estudio usada en la demo.",
            "plane": "Plano principal solicitado: sagittal o axial.",
            "modelKey": "Clave del modelo registrado.",
            "modelVersion": "Version del contrato/modelo reportado.",
            "series": "Lista de series disponibles para el viewer.",
            "masks": "Mascaras editables con contornos por serie y slice.",
            "landmarks": "Puntos de referencia derivados del contrato visual.",
            "measurementValues": "Mediciones normalizadas para IA vs Reviewer.",
            "aiOutput": "Estado explicable de la salida IA/contrato.",
            "agentDecision": "Decision operativa del agente: prioridad, flags y accion recomendada.",
            "modelArtifact": "Estado del artifact .pt esperado para inferencia real.",
            "quality": "Resumen cuantitativo del contrato visual.",
            "metadata": "Trazabilidad extendida de frontend/backend/AI Module.",
            "review": "Estado de revision profesional cuando backend lo adjunta.",
        },
        "seriesItem": {
            "id": "string",
            "name": "string",
            "plane": "sagittal|axial",
            "sequence": "T1|T2|other",
            "sliceCount": "number",
            "selectedSlice": "number",
            "imageUrl": "string|null",
            "overlayUrl": "string|null",
            "overlayOpacity": "number",
            "status": "contract_ready|reference_only|ai_output_pending",
        },
        "maskItem": {
            "id": "string",
            "label": "string",
            "className": "vertebral_body|disc|spinal_canal|nerve_root|foramen|other",
            "color": "hex color",
            "confidence": "0..1",
            "editable": "boolean",
            "enabled": "boolean",
            "contours": "Array<{ seriesId, sliceIndex, points: Array<{x,y}> }>",
        },
        "landmarkItem": {
            "id": "string",
            "label": "string",
            "seriesId": "string",
            "sliceIndex": "number",
            "x": "number",
            "y": "number",
            "editable": "boolean",
            "linkedMaskId": "string|null",
        },
        "measurementItem": {
            "id": "string",
            "label": "string",
            "level": "string",
            "aiValue": "number|string",
            "reviewerValue": "number|string|null",
            "unit": "string",
            "confidence": "0..1",
            "plane": "sagittal|axial",
            "source": "AI|Reviewer|Placeholder",
            "status": "pendiente|revisado|editado",
            "linkedLandmarks": "string[]",
        },
        "aiOutput": {
            "status": "contract_ready|real_inference|degraded",
            "inferenceMode": "contract|mock|real",
            "requestedInferenceMode": "contract|mock|real",
            "realInferenceAvailable": "boolean",
            "modelReadiness": "contract_only_missing_artifact|real_artifact_available",
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
        },
        "quality": {
            "maskCount": "number",
            "landmarkCount": "number",
            "measurementCount": "number",
            "meanMaskConfidence": "number",
            "pixelSpacingMm": "number",
            "measurementsDerivedFromContours": "boolean",
        },
        "guarantees": [
            "El contrato siempre declara humanReviewRequired=true.",
            "El contrato siempre declara notClinicalDiagnosis=true.",
            "Las mediciones son editables por Reviewer.",
            "La inferencia real no se declara disponible si falta el artifact .pt.",
            "Las salidas contract/mock/real conservan la misma forma para frontend y backend.",
        ],
    }
    schema["schemaHash"] = _schema_hash(schema)
    return schema


def contract_verification() -> Dict[str, Any]:
    schema = pipeline_contract_schema()
    recomputed_hash = _schema_hash(_hashable_schema(schema))
    root_fields = schema.get("rootFields", {})
    missing_root_fields = [field for field in REQUIRED_ROOT_FIELDS if field not in root_fields]
    governance_valid = (
        schema.get("humanReviewRequired") is True
        and schema.get("notClinicalDiagnosis") is True
        and schema.get("aiOutput", {}).get("humanReviewRequired") is True
        and schema.get("aiOutput", {}).get("notClinicalDiagnosis") is True
    )
    hash_valid = schema.get("schemaHash") == recomputed_hash
    valid = hash_valid and governance_valid and not missing_root_fields
    return {
        "schemaVersion": schema["schemaVersion"],
        "schemaHash": schema["schemaHash"],
        "recomputedHash": recomputed_hash,
        "hashValid": hash_valid,
        "governanceValid": governance_valid,
        "requiredRootFields": REQUIRED_ROOT_FIELDS,
        "missingRootFields": missing_root_fields,
        "valid": valid,
        "generatedBy": schema["generatedBy"],
        "humanReviewRequired": schema["humanReviewRequired"],
        "notClinicalDiagnosis": schema["notClinicalDiagnosis"],
    }
