# AI-008 - Strict real_baseline axial via /pipeline/run

Repo de trabajo: PFI_MVPTest_Enzo_AImodule
Ruta de trabajo: C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVPTest_Enzo_AImodule

Contrato usado:
- Endpoint: POST /pipeline/run.
- Para axial estricto: plane=axial, modelKey=axial_t2_alkafri, metadata.inferenceMode=real_baseline, metadata.allowContractFallback=false, traceId.
- Formatos soportados por runtime: .npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .mha, .mhd, .dcm.
- Fixture usado: imagen 2D 256x256 float32 en rango 0..1.

Fixture usado:
```text
ai_service/tests/fixtures/real_baseline/axial_sample_input.npy
shape=(256, 256), dtype=float32, min=0.0, max=1.0
fixture_summary.json: procedencia ALKAFRI pairing_v1 pair_0001; deidentificado, dataset publico
```

Corrida manual /pipeline/run:
```text
status 200
caseId=CASE-AI008-AXIAL-FIXTURE
traceId=trace-ai008-axial-fixture
runId=0bef4a29445537ef
aiOutput.inferenceMode=real_baseline
metadata.inferenceMode=real_baseline
allowContractFallback=false
```

Outputs generados:
```text
outputs/real_inference/0bef4a29445537ef/axial/input.png: si
outputs/real_inference/0bef4a29445537ef/axial/mask.npy: si
outputs/real_inference/0bef4a29445537ef/axial/confidence.npy: si
outputs/real_inference/0bef4a29445537ef/axial/overlay.png: si
```

Test enfocado:
```text
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe -m pytest ai_service\tests\test_strict_axial_real_baseline_fixture.py -q

.                                                                        [100%]
1 passed, 1 warning in 4.31s
```

Git hygiene:
```text
Fixture axial real_baseline: no esta gitignoreado; agregado al indice con git add normal (sin -f).
outputs/: ignorado, no commitear outputs generados.
models/final/*.pt: ignorados, no commitear checkpoints.
```

Resultado:
- Fixture usado: axial_sample_input.npy (2D). No fue necesario usar la variante CHW.
- effectiveInferenceMode observado: real_baseline via aiOutput.inferenceMode y metadata.inferenceMode.
- No hubo fallback ni contract mode.
- runId y traceId presentes.
- Outputs input.png, mask.npy, confidence.npy y overlay.png generados.

Bloqueos:
- Ninguno para el smoke real_baseline axial estricto con fixture real.

