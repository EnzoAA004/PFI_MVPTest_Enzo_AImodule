# P9-A.3 - Fusion sagital-axial y reconstruccion 3D lumbar experimental

## Alcance implementado

P9-A.3 conecta la respuesta multiplanar v2 con un artefacto 3D experimental generado a partir de segmentaciones reales sagitales y axiales ya producidas por `real_baseline`.

El AI Module no presenta este resultado como 3D clinico definitivo. El campo `threeD` queda habilitado solo cuando:

- la corrida es `dual_plane`;
- ambos planos terminaron en `real_baseline`;
- ningun plano es sintetico ni proviene de fallback;
- existen assets internos `mask.npy` registrados para sagital y axial;
- ambos planos reportan spacing fisico y mapping de slice;
- hay clases foreground compartidas entre las mascaras.

Si falta cualquiera de esos requisitos, `threeD.enabled=false` y el estado informa bloqueo por geometria insuficiente.

## Contrato de salida

En `/v2/multiplanar/run`, `threeD` puede devolver:

- `status=experimental_ready`;
- `assets[0].assetName=lumbar-3d-mesh.json`;
- `assets[0].relativePath=/assets/{multiplanarRunId}/workspace/lumbar-3d-mesh.json`;
- `reconstruction.method=dual_plane_mask_geometry`;
- `reconstruction.source=real_segmentation_masks`;
- trazabilidad de `modelKey`, `modelVersion`, `artifactHash`, `runId`, slice y spacing por plano.

El asset `lumbar-3d-mesh.json` se sirve como `application/json`. No expone paths internos.

## Naturaleza experimental

El mesh generado es un artefacto sparse derivado de mascaras reales de dos planos ortogonales. No es una extrusion 2D aislada, pero tampoco reemplaza una reconstruccion volumetrica validada con stack completo, registracion DICOM, spacing/orientacion completa y marching cubes sobre volumen segmentado.

Por seguridad academica y clinica:

- `experimental=true`;
- `humanReviewRequired=true`;
- `notClinicalDiagnosis=true`;
- se documentan limitaciones dentro del propio JSON del mesh.

## Evidencia reproducible

Comando enfocado:

```powershell
$env:PYTHONPATH='ai_service'
.\.venv\Scripts\python.exe -m pytest ai_service\tests\test_multiplanar_v2_contract.py -q --basetemp .pytest-tmp\p9a3-v2
```

Resultado registrado:

```text
16 passed, 1 warning
```

El test `test_v2_dual_real_baseline_generates_experimental_3d_mesh` valida que:

- una corrida dual real produce `threeD.enabled=true`;
- el estado es `experimental_ready`;
- se genera y sirve `lumbar-3d-mesh.json`;
- el mesh tiene vertices, caras, estructuras y trazabilidad de hashes/modelos;
- no se genera 3D cuando hay fallback o plano sintetico.
