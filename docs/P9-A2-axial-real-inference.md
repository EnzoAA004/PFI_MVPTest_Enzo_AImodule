# P9-A.2 - Inferencia axial real

## Alcance

El AI Module habilita `real_baseline` para el plano axial con el checkpoint congelado `axial_t2_alkafri_final_v2_candidate.pt`.

No se modifican pesos, notebooks, datasets ni checkpoints. La habilitacion se limita al runtime del servicio y conserva `humanReviewRequired=true` y `notClinicalDiagnosis=true`.

## Criterio de runtime axial

El modelo axial queda disponible para inferencia real solo si:

- el artefacto existe y su SHA-256 coincide con el manifest;
- el manifest es valido y corresponde a `axial_t2_alkafri`;
- `trainingStatus` es `candidate_below_quality_gate`;
- `qualityGate.runtimeVerification.finite=true`;
- `dice_macro_excluding_raw0 >= 0.80`.

La calificacion publicada es `runtimeQualification=axial_candidate_runtime_ready`. El manifest conserva la advertencia de calidad original porque la metrica macro incluyendo `raw_0` quedo debajo del umbral.

## Salida esperada

Un request axial estricto:

```json
{
  "caseId": "CASE-AI008-AXIAL-FIXTURE",
  "plane": "axial",
  "modelKey": "axial_t2_alkafri",
  "inputPath": "ai_service/tests/fixtures/real_baseline/axial_sample_input.npy",
  "metadata": {
    "inferenceMode": "real_baseline",
    "allowContractFallback": false
  }
}
```

devuelve `200` con:

- `aiOutput.inferenceMode=real_baseline`;
- `synthetic=false` y `fallbackReason=null`;
- `modelKey=axial_t2_alkafri`, `modelVersion=axial-final-v2`, `artifactHash` y `runtimeQualification`;
- mascaras derivadas de la prediccion;
- landmarks derivados de mascaras;
- mediciones axiales derivadas de la mascara predicha;
- assets `input.png`, `overlay.png`, `mask-preview.png`, `mask.npy`, `confidence.npy`;
- trazabilidad de `artifactHash`, version, serie, slice y spacing cuando esta disponible.

Las clases publicadas preservan el label-map congelado del checkpoint axial (`raw_0`, `raw_50`, `raw_100`, `raw_150`, `raw_200`) para no inventar una nomenclatura anatomica no contenida en el manifest. La semantica de producto sigue siendo medicion/segmentacion revisable por profesional, no diagnostico.

## Casos de error cubiertos

- `modelKey=axial_t2_alkafri` usado con `plane=sagittal` y `allowContractFallback=false` devuelve `409`.
- Un `inputId` registrado como axial no puede reutilizarse como sagital; devuelve `409`.
- Si una corrida real falla con fallback deshabilitado, no se declara `real_baseline` exitoso.
- Los raw assets (`mask.npy`, `confidence.npy`) quedan registrados pero no servidos al navegador.

## Integracion multiplanar v2

- `/v2/multiplanar/contract` declara axial disponible para inferencia real.
- `/multiplanar/run` legacy registra paths legacy como `inputId` interno antes de ejecutar el contrato canonico v2.
- Una corrida sagital+axial con fixtures reales devuelve `effectiveInferenceMode=real_baseline`, planos no sinteticos y `threeD.status=pending_registered_reconstruction`.

## Evidencia

Comando enfocado:

```powershell
$env:PYTHONPATH='ai_service'
.\.venv\Scripts\python.exe -m pytest ai_service\tests\test_strict_axial_real_baseline_fixture.py ai_service\tests\test_multiplanar_real_baseline_fixtures.py ai_service\tests\test_partial_real_readiness.py ai_service\tests\test_multiplanar_v2_contract.py -q --basetemp C:\tmp\pytest-p9a2-focused
```

Resultado observado: `20 passed, 1 warning`.

Suite del servicio:

```powershell
$env:PYTHONPATH='ai_service'
.\.venv\Scripts\python.exe -m pytest ai_service\tests -q --basetemp C:\tmp\pytest-p9a2-ai-service-review
```

Resultado observado: `164 passed, 4 skipped, 2 warnings`.
