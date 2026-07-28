# P9-A.3.1 - Proxy geometrico sagital-axial experimental

## Alcance implementado

P9-A.3.1 alinea el trabajo 3D con P9-A.2.1. El AI Module puede transportar un artefacto `lumbar-3d-mesh.json`, pero ese artefacto es un proxy geometrico experimental basado en bounding boxes 2D por plano.

No es todavia reconstruccion anatomica 3D final, no es fusion volumetrica y no debe presentarse como mesh paciente-especifico validado.

## Reglas de habilitacion

Un plano se considera real para el proxy si cumple:

- `availableForRealInference=true`;
- `effectiveInferenceMode=real_baseline`;
- `synthetic=false`;
- `fallbackReason=null`;
- `artifactHash` presente;
- `manifestValid=true`.

El axial candidato conserva:

- `baselineReady=false`;
- `readiness=real_candidate_ready`;
- `runtimeQualification=axial_candidate_runtime_ready`;
- `qualityGatePassed=false`.

Esto permite ejecutar inferencia real con checkpoint axial candidato sin promoverlo a baseline aprobado.

## Mapping anatomico

P9-A.3.1 no cruza IDs numericos entre modelos. Los IDs no son equivalentes:

- sagital: `vertebra_group`, `canal`, `disc_group`;
- axial: `raw_0`, `raw_50`, `raw_100`, `raw_150`, `raw_200`.

El proxy solo se genera si existe un mapping anatomico explicito, por ejemplo mediante `PFI_MULTIPLANAR_3D_ANATOMICAL_MAPPING_JSON`. Si no existe mapping validado:

- `threeD.enabled=false`;
- `threeD.status=experimental_blocked_missing_anatomical_mapping`;
- `threeD.assets=[]`.

No se inventan equivalencias para `raw_*` y no se infieren por coincidencia numerica.

## Contrato del proxy

Cuando el proxy se genera, `threeD.reconstruction` declara:

- `kind=experimental_geometric_proxy`;
- `method=dual_plane_bbox_proxy`;
- `anatomicalReconstruction=false`;
- `volumetricReconstruction=false`;
- `coordinateSystem=local_proxy_space`.

El asset se sirve como:

```text
GET /assets/{multiplanarRunId}/workspace/lumbar-3d-mesh.json
```

El registro puede rehidratar ese asset desde `outputs/multiplanar_3d/{runId}/lumbar-3d-mesh.json` si el registro en memoria fue limpiado. No se exponen paths internos en el contrato publico.

## Lo que falta para 3D final

La reconstruccion final requiere:

- stack completo;
- orden DICOM;
- `ImagePositionPatient`;
- `ImageOrientationPatient`;
- `FrameOfReferenceUID`;
- spacing entre cortes;
- registracion sagital-axial;
- generacion volumetrica y validacion E2E reproducible.

Hasta contar con eso, P9-A.3.1 debe describirse solo como proxy geometrico experimental.

## Evidencia

Suite enfocada:

```powershell
$env:PYTHONPATH='ai_service'
.\.venv\Scripts\python.exe -m pytest ai_service\tests\test_multiplanar_v2_contract.py ai_service\tests\test_multiplanar_real_baseline_fixtures.py ai_service\tests\test_asset_serving.py ai_service\tests\test_asset_registry.py -q --basetemp .pytest-tmp\p9a31-focused
```

Resultado observado:

```text
30 passed, 1 warning
```

Los tests distinguen:

- unitarios con mascaras fabricadas y mapping explicito;
- integracion con fixtures/checkpoints reales para validar flujo y transporte del proxy;
- bloqueo cuando falta mapping anatomico;
- no generacion de proxy si hay fallback/sintetico;
- rehidratacion del JSON tras limpiar el registro en memoria.
