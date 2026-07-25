# Model Card - sagittal_spider_multiclass_final_best.pt

## Nombre y version

- Modelo: sagittal_spider
- Version: sagittal-spider-final-v1
- Artifact runtime: `sagittal_spider_multiclass_final_best.pt`
- Release GCS: `gs://pfi-rm-lumbar-artifacts-2026-ef/models/releases/sagittal_spider_final_v1/`

## Objetivo

Segmentacion multiclase de RM lumbar sagital SPIDER para apoyo academico revisable. No produce diagnostico automatico ni recomendaciones terapeuticas.

## Arquitectura y formato

- Arquitectura: SagittalUNet2D
- Entrada: tensor 1x256x256
- Salida: 4 clases: background, vertebra_group, canal, disc_group
- base_channels: 16
- target_size: 256x256
- Runtime alias compatible: `sagittal_spider_multiclass_final_best.pt`

## Dataset y split

- Dataset: SPIDER sagittal lumbar MRI
- Split por paciente: train=152, val=33, test=33
- Slices: train=5271, val=1174, test=1218
- Validation selecciona modelo; test se evalua una sola vez con el mejor checkpoint.

## Entrenamiento

- Notebook fuente: notebooks/45_gcs_spider_final_training.ipynb
- Optimizer: AdamW
- Loss: CrossEntropy ponderada + Dice sin fondo
- Early stopping: patience 12
- Epochs completados: 75
- Mejor epoch: 63
- Reason finished: early_stopping

## Metricas finales

- Validation Dice macro no background: 0.8992978910787764
- Test Dice macro no background: 0.8934316063846954
- Test IoU macro no background: 0.8079981902423006
- Umbral Dice: 0.70
- Quality gate: aprobado

## Procedencia y hashes

- Model SHA-256: `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944`
- Training repository commit: `6013e160f45c9263fd4ae50e864ceb37245323e2`
- Architecture SHA-256: `d83f735cca9cbefc0e65dd8863466f4a528f205f3d674ebb73c49d68f8687c90`
- Dataset manifest SHA-256: `fa54c89a278d850021c0f91c0a27b3b5211c86301c9e4f125d75d517f39b793b`
- Training index SHA-256: `2720b7218c92870f6f0a000b57111ed36b5cf3b78c716f244f427ca7fee4a4ba`

## Formatos soportados por runtime

El runtime actual acepta entradas de imagen y volumen soportadas por el servicio existente. Esta release solo publica artifacts; la materializacion cloud desde gs:// requiere una tarea separada controlada o una estrategia HTTPS autenticada.

## Limitaciones

- No tiene validacion clinica.
- Puede sesgarse al dominio SPIDER evaluado.
- Requiere revision profesional humana.
- No debe usarse para diagnostico automatico, seguridad clinica, eficacia clinica ni generalizacion fuera del conjunto evaluado.
