# Variables de entorno cloud

| Variable | Ejemplo | Obligatoria | Descripcion |
| --- | --- | --- | --- |
| `PORT` | `8000` | Si | Puerto HTTP del servicio. Render, Railway y Cloud Run pueden inyectarlo dinamicamente. |
| `PFI_MODEL_DIR` | `/app/models/final` | Si | Directorio donde el AI Module busca los pesos finales copiados por Docker. |
| `PFI_SAGITTAL_MODEL_PATH` | `/app/models/final/sagittal_spider_multiclass_final_best.pt` | Si | Ruta completa del checkpoint sagital autorizado. |
| `PFI_AXIAL_MODEL_PATH` | `/app/models/final/axial_t2_alkafri_final_v2_candidate.pt` | Si | Ruta completa del checkpoint axial candidate. No renombrar como `best`. |
| `PFI_ROOT` | `/app` | No | Fallback Colab y raiz externa para resultados, figuras y reportes tecnicos. No debe apuntar a datos privados versionados en Git. |
| `PFI_MODEL_REGISTRY` | `config/model_registry_final.json` | Si | Ruta al registro tecnico de modelos. Puede ser ruta interna del contenedor o montada desde almacenamiento controlado. |
| `PFI_DATA_FREEZE_CONFIG` | `config/data_freeze_config.json` | Si | Ruta a la configuracion de congelamiento/criterios de datos para reproducibilidad academica. |
| `PFI_OUTPUT_DIR` | `/app/outputs` | Si | Directorio de salida para reportes, overlays y artefactos tecnicos generados por el AI Module. |
| `PFI_GCP_PROJECT_ID` | `pfi-asplanatti-fabrello-v1` | Si para release GCS | Proyecto usado por ADC para leer la release verificada. |
| `PFI_SAGITTAL_RELEASE_URI` | `gs://pfi-rm-lumbar-artifacts-2026-ef/models/releases/sagittal_spider_final_v1/` | Si para sync GCS sagital | Release sagital verificada. Tiene prioridad sobre `PFI_SAGITTAL_MODEL_URI`. |
| `PFI_SAGITTAL_RELEASE_CONTENT_SHA256` | `7420ad4271fe634c970b2a543d1ef8fb1437888c99ca8bd5733a06e5f63e3e7e` | Si para sync GCS sagital | Hash de contenido publicado esperado. |
| `PFI_SAGITTAL_RELEASE_MANIFEST_SHA256` | `d36d0c4fe183ba9a98f0a3471486be5dee1cf1fa820dc32b3a50177ce322be21` | Si para sync GCS sagital | Hash esperado de `release_manifest.json`. |
| `PFI_SAGITTAL_MODEL_SHA256` | `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944` | Si para sync GCS sagital | Hash esperado del checkpoint runtime. |

## Notas

- No configurar rutas que expongan datasets completos o imagenes medicas reales en servicios publicos.
- En despliegues Docker del repo, usar `/app/models/final` porque la imagen copia ahi `models/final`.
- Para el axial v2 candidate deben existir tambien `/app/models/final/axial_t2_alkafri_final_v2_candidate.pt.manifest.json` y su hash debe coincidir con el checkpoint.
- No guardar secretos en `.env.example`.
- En Cloud Run o servicios equivalentes, usar Secret Manager o variables protegidas para credenciales de almacenamiento.
- Para desarrollo local con GCS usar Application Default Credentials: `gcloud auth application-default login`.
- No copiar archivos JSON de service account ni `.config/gcloud` dentro de la imagen Docker o el repositorio.
- El AI Module debe responder con `human_review_required=true` o `humanReviewRequired=true` en resultados de inferencia, pipeline y agente.
