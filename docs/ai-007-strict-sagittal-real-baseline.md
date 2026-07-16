# AI-007 - Strict real_baseline sagittal via /pipeline/run

Repo de trabajo: PFI_MVPTest_Enzo_AImodule
Ruta de trabajo: C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVPTest_Enzo_AImodule

Contrato relevado:
- Endpoint: POST /pipeline/run.
- Request model: PipelineRunRequest en ai_service/pfi_ai_service/pipeline.py.
- Campos requeridos: caseId, plane, modelKey, inputPath, metadata.
- Para sagital estricto: plane=sagittal, modelKey=sagittal_spider, metadata.inferenceMode=real_baseline, metadata.allowContractFallback=false, traceId opcional pero requerido para evidencia.
- Formatos de input soportados por real_inference_runtime.py: .npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .mha, .mhd, .dcm.
- Shape esperado: imagen 2D o volumen 3D; para sagital 3D usa sagittal_axis del checkpoint si existe, y resize interno a targetSize.

Decision de modo:
- run_pipeline calcula requested_mode desde metadata.inferenceMode/mode.
- Intenta real solo si requested_mode es real o real_baseline, modelKey coincide con plane y model_status.availableForRealInference es true.
- Si run_real_inference falla y allowContractFallback=false, la excepcion se propaga; no debe caer a contrato.
- Si allowContractFallback=true, puede degradar a contract con metadata de fallo.

Busqueda de input:
```text
Comando: rg --files -uuu con extensiones .npy/.png/.jpg/.jpeg/.bmp/.tif/.tiff/.mha/.mhd/.dcm, excluyendo .venv/.git/models/final/*.pt
Resultado: sin archivos soportados encontrados en el repo real.
```

Resultado:
- Input sagital usado: FALTA.
- effectiveInferenceMode: NO EJECUTADO por falta de input valido.
- Outputs generados: no, porque no se ejecuto pipeline.
- runId / traceId presentes: no, porque no se ejecuto pipeline.
- Test enfocado: no creado/ejecutado; crear uno con ruido sintetico violaria la restriccion de no inventar input valido.

Input requerido para desbloquear:
- Archivo local sagital valido en formato .npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .mha, .mhd o .dcm.
- Debe representar una RM lumbar sagital valida/desidentificada o un fixture autorizado del proyecto, no ruido aleatorio.
- inputPath debe apuntar al archivo o a un directorio con archivos soportados.

Bloqueo:
- No hay fixture/sample/input sagital valido disponible en el repo real.
- Se frena AI-007 sin declarar real_baseline y sin generar outputs.

