# Deploy del AI Module

El AI Module puede desplegarse como servicio Docker FastAPI. Los ejemplos asumen que los modelos, datasets y resultados pesados viven fuera del repositorio, en almacenamiento controlado.

## Variables necesarias

```text
PFI_ROOT=/content/drive/MyDrive/PFI_MVP
PFI_MODEL_REGISTRY=config/model_registry_final.json
PFI_DATA_FREEZE_CONFIG=config/data_freeze_config.json
PFI_OUTPUT_DIR=outputs
PORT=8000
```

En produccion, `PFI_ROOT` debe apuntar a un volumen, bucket montado o ruta interna donde esten disponibles modelos y artefactos tecnicos. No incluir datasets completos, imagenes medicas privadas ni checkpoints pesados dentro de la imagen Docker salvo decision explicita del proyecto.

## Docker local

```bash
cd ai_service
docker build -t pfi-ai-module .
docker run --rm -p 8000:8000 --env-file ../.env.example pfi-ai-module
```

Health check:

```bash
curl http://localhost:8000/health
```

## Render Docker Web Service

1. Crear un nuevo Web Service desde el repositorio.
2. Elegir entorno Docker.
3. Definir `ai_service/Dockerfile` como Dockerfile si Render no lo detecta automaticamente.
4. Configurar las variables de entorno necesarias.
5. Configurar almacenamiento externo para modelos y resultados, o descargar artefactos desde una ubicacion controlada durante el arranque si el proyecto lo autoriza.

Render inyecta `PORT`; el Dockerfile ejecuta Uvicorn usando `${PORT:-8000}`.

## Railway Docker

1. Crear un nuevo servicio desde el repositorio.
2. Usar despliegue Docker con contexto `ai_service` o configurar la ruta del Dockerfile.
3. Cargar variables de entorno.
4. Conectar almacenamiento externo o mecanismo de descarga controlada para modelos.

Railway tambien define `PORT` en ejecucion. El servicio debe exponer FastAPI por ese puerto.

## Google Cloud Run

1. Construir y subir la imagen a Artifact Registry.
2. Crear un servicio Cloud Run apuntando a la imagen.
3. Configurar variables de entorno.
4. Conectar acceso a Cloud Storage, Secret Manager o volumen compatible para modelos/resultados.
5. Restringir acceso de red si el servicio solo debe ser invocado por el backend.

Ejemplo conceptual:

```bash
gcloud run deploy pfi-ai-module \
  --image REGION-docker.pkg.dev/PROJECT/REPOSITORY/pfi-ai-module:TAG \
  --region REGION \
  --allow-unauthenticated
```

Para un entorno real, preferir invocacion autenticada desde el backend.

## Advertencias sobre modelos y datasets

- No subir datasets completos al repositorio.
- No incluir imagenes medicas privadas en Git.
- No borrar modelos, notebooks o resultados existentes sin confirmacion.
- Si los pesos son parte de la entrega y pesan demasiado, evaluar Git LFS o almacenamiento externo versionado.
- Registrar version de modelo, configuracion y limitaciones tecnicas junto con cada corrida.

## Seguridad metodologica

El servicio es asistivo. No debe emitir diagnostico clinico ni recomendacion terapeutica. Toda salida debe permanecer sujeta a revision profesional.
