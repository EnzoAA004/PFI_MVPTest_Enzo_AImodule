# P10.9 — auditoría de artefactos reales P10.6/P10.7

Fecha UTC: 2026-08-07

Esta evidencia verifica **los archivos congelados reales** utilizados para la etapa de integración. No ejecuta entrenamiento ni modifica checkpoints.

## P10.7 — SPIDER degenerative multitask

Archivo auditado: `frozen_p10_7_spider_degenerative_multitask.pt`

- tamaño: `22,182,638` bytes
- SHA-256: `16eccff327e6794b127fe372ecd03ea619a0f69d939b84ae1aa2e904191c6293`
- `torch.load(..., map_location="cpu", weights_only=False)`: OK
- payload: `dict`
- `schemaVersion`: `pfi.p10-7-research-export.v1`
- `taskOrder`: 8 tareas P10.7 esperadas
- `trainConfig.backbone`: `efficientnet_b0`
- `trainConfig.image_size`: `224`
- `modelStateDict`: 381 tensores/entradas
- `humanReviewRequired`: presente en el export
- `notClinicalDiagnosis`: presente en el export

Tareas verificadas en `taskOrder`:

1. `pfirrmann_grade`
2. `modic_change`
3. `upper_endplate_change`
4. `lower_endplate_change`
5. `spondylolisthesis`
6. `disc_herniation`
7. `disc_narrowing`
8. `disc_bulging`

Estado del gate:

- artefacto/hash/contrato de checkpoint: **VERIFICADO**
- forward real mediante runtime productivo: **PENDIENTE**
- E2E con localización derivada de segmentación: **PENDIENTE**

La ausencia del forward en esta auditoría no se convierte en un `PASS`: el runtime completo depende de `timm` y del entorno de servicio, y debe ejecutarse posteriormente en el smoke real de P10.9.

## P10.6 — subarticular axial T2 2.5D

Archivo auditado: `frozen_subarticular_checkpoint.pt`

- tamaño: `48,655,231` bytes
- SHA-256: `d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a`
- `torch.load(..., map_location="cpu", weights_only=False)`: OK
- payload: `dict`
- `schemaVersion`: `pfi.rsna-subarticular-training-checkpoint.v1`
- `epoch`: `6`
- `task`: `subarticular_stenosis_left_right`
- `sequence`: `Axial T2`
- `classNames`: `normal_mild`, `moderate`, `severe`
- `modelStateDict`: 364 tensores/entradas
- `config.model_name`: `efficientnet_b0`
- `config.image_size`: `224`

Estado del gate:

- artefacto/hash/contrato de checkpoint: **VERIFICADO**
- forward HTTP real sobre ROI registrado: **PENDIENTE para P10.9**

## Gobernanza

- no se entrenó ningún modelo;
- no se abrió test oculto;
- no se modificó ningún `.pt`;
- no se exportaron identificadores de paciente;
- esta auditoría no demuestra precisión clínica ni generalización;
- ambos flujos continúan requiriendo revisión profesional.
