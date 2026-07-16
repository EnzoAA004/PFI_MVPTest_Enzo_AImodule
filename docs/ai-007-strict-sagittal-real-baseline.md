# AI-007 - Strict real_baseline sagittal via /pipeline/run

Repo de trabajo: PFI_MVPTest_Enzo_AImodule
Ruta de trabajo: C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVPTest_Enzo_AImodule

Contrato relevado:
- Endpoint: POST /pipeline/run.
- Request model: PipelineRunRequest en ai_service/pfi_ai_service/pipeline.py.
- Campos requeridos: caseId, plane, modelKey, inputPath, metadata.
- Para sagital estricto: plane=sagittal, modelKey=sagittal_spider, metadata.inferenceMode=real_baseline, metadata.allowContractFallback=false, traceId.
- Formatos de input soportados por real_inference_runtime.py: .npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .mha, .mhd, .dcm.
- Shape aceptado: imagen 2D o volumen 3D; el fixture usado es 2D 256x256 float32 en rango 0..1.

Fixture usado:
```text
ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy
shape=(256, 256), dtype=float32, min=0.0, max=1.0
fixture_summary.json: procedencia SPIDER 101_t2.mha, slice 8; deidentificado, dataset publico
```

Corrida manual /pipeline/run:
```text
status 200
caseId=CASE-AI007-SAGITTAL-FIXTURE
traceId=trace-ai007-sagittal-fixture
runId=a1c9d70904e9f1bc
aiOutput.inferenceMode=real_baseline
metadata.inferenceMode=real_baseline
allowContractFallback=false
```

Outputs generados:
```text
outputs/real_inference/a1c9d70904e9f1bc/sagittal/input.png: si
outputs/real_inference/a1c9d70904e9f1bc/sagittal/mask.npy: si
outputs/real_inference/a1c9d70904e9f1bc/sagittal/confidence.npy: si
outputs/real_inference/a1c9d70904e9f1bc/sagittal/overlay.png: si
```

Test enfocado:
```text
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe -m pytest ai_service\tests\test_strict_sagittal_real_baseline_fixture.py -q

.                                                                        [100%]
1 passed, 1 warning in 3.85s
```

Git hygiene:
```text
Fixture sagital real_baseline: no esta gitignoreado; agregado al indice con git add normal (sin -f).
outputs/: ignorado, no commitear outputs generados.
models/final/*.pt: ignorados, no commitear checkpoints.
```

Resultado:
- Fixture usado: sagittal_sample_input.npy (2D). No fue necesario usar la variante CHW.
- effectiveInferenceMode observado: real_baseline via aiOutput.inferenceMode y metadata.inferenceMode.
- No hubo fallback ni contract mode.
- runId y traceId presentes.
- Outputs input.png, mask.npy, confidence.npy y overlay.png generados.

Bloqueos:
- Ninguno para el smoke real_baseline sagital estricto con fixture real.


