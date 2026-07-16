# AI-012 - Multiplanar real_baseline evidence

Repo de trabajo: PFI_MVPTest_Enzo_AImodule
Ruta de trabajo: C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVPTest_Enzo_AImodule

Contrato de /multiplanar/run:
- Endpoint: POST /multiplanar/run.
- Request model: MultiplanarRunRequest en ai_service/pfi_ai_service/multiplanar_run.py.
- Inputs por plano: sagittalInputPath y axialInputPath.
- Model keys: sagittalModelKey y axialModelKey; defaults sagittal_spider y axial_t2_alkafri.
- metadata.inferenceMode=real_baseline y metadata.allowContractFallback=false se propagan a cada PipelineRunRequest hijo.
- Agrupacion: shared_run_id genera runId comun con prefijo multi-; metadata.multiplanarRunId conserva el mismo valor.
- effectiveInferenceMode del workspace es real_baseline solo si ambos planos devuelven aiOutput.inferenceMode=real_baseline; si difieren devuelve mixed.

Payload usado:
```json
{
  "caseId": "CASE-AI012-MULTI-FIXTURE",
  "sagittalInputPath": "ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy",
  "axialInputPath": "ai_service/tests/fixtures/real_baseline/axial_sample_input.npy",
  "sagittalModelKey": "sagittal_spider",
  "axialModelKey": "axial_t2_alkafri",
  "metadata": {
    "inferenceMode": "real_baseline",
    "allowContractFallback": false,
    "traceId": "trace-ai012-multiplanar-fixture"
  }
}
```

Resultado manual:
```text
status=200
multiplanarRunId=multi-9348e538a8dc7d1f
traceId=trace-ai012-multiplanar-fixture
requestedInferenceMode=real_baseline
effectiveInferenceMode=real_baseline
sagittal.runId=d05295dafe8db988
sagittal.aiOutput.inferenceMode=real_baseline
axial.runId=dcab06f1cc8b276b
axial.aiOutput.inferenceMode=real_baseline
outputs/multiplanar_reports/multi-9348e538a8dc7d1f.json: si
```

Outputs sagital:
```text
outputs/real_inference/d05295dafe8db988/sagittal/input.png: si
outputs/real_inference/d05295dafe8db988/sagittal/mask.npy: si
outputs/real_inference/d05295dafe8db988/sagittal/confidence.npy: si
outputs/real_inference/d05295dafe8db988/sagittal/overlay.png: si
```

Outputs axial:
```text
outputs/real_inference/dcab06f1cc8b276b/axial/input.png: si
outputs/real_inference/dcab06f1cc8b276b/axial/mask.npy: si
outputs/real_inference/dcab06f1cc8b276b/axial/confidence.npy: si
outputs/real_inference/dcab06f1cc8b276b/axial/overlay.png: si
```

Test enfocado:
```text
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe -m pytest ai_service\tests\test_multiplanar_real_baseline_fixtures.py -q

.                                                                        [100%]
1 passed, 1 warning in 3.92s
```

Git hygiene:
```text
outputs/: ignorado; no commitear outputs ni reporte JSON pesado.
models/final/*.pt: ignorados; no commitear checkpoints.
```

Bloqueos:
- Ninguno.

