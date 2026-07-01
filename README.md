# PFI AI Module

Modulo de IA independiente para el Proyecto Final de Ingenieria sobre analisis asistido de resonancias magneticas lumbares.

Este repositorio contiene el servicio Python/FastAPI, el agente IA, componentes reutilizables de inferencia, preprocesamiento, mediciones geometricas, overlays, configuracion tecnica de modelos y evidencia experimental/notebooks cuando corresponda. No corresponde a este repositorio implementar el frontend React ni el backend Java final.

## Arquitectura general

```text
Frontend React -> Spring Boot Backend -> Python FastAPI AI Module
```

El frontend consume el backend del producto. El backend Spring Boot orquesta usuarios, archivos, permisos, persistencia y flujos de negocio. Este modulo IA recibe solicitudes tecnicas desde el backend, ejecuta procesamiento/inferencia o consulta resultados existentes, y devuelve respuestas estructuradas para revision profesional.

## Responsabilidad de este repo

- Exponer una API FastAPI para el modulo de IA.
- Mantener configuracion de modelos y rutas tecnicas.
- Ejecutar o preparar inferencia sagital y axial cuando este implementada.
- Calcular mediciones geometricas derivadas de mascaras.
- Generar overlays, reportes tecnicos y trazabilidad.
- Mantener notebooks/evidencia tecnica sin incluir datasets completos ni archivos privados.

El sistema es asistivo. No emite diagnostico clinico, no recomienda tratamientos y no reemplaza el criterio profesional. Las salidas deben considerarse resultados tecnicos derivados de procesamiento de imagenes y requieren revision profesional. Cuando aplique, las respuestas deben mantener `human_review_required=true`.

## Estructura principal

```text
ai_service/              Servicio FastAPI del modulo IA
ai_service/pfi_ai_service/api.py
src/lumbar_mri/          Codigo reutilizable de procesamiento, metricas, mediciones y overlays
notebooks/               Evidencia tecnica y experimentacion
docs/                    Documentacion tecnica
tests/                   Tests con datos sinteticos
scripts/                 Scripts de ejecucion local
```

## Ejecucion local

Crear un entorno virtual e instalar dependencias del servicio:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ai_service/requirements-ai-service.txt
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai_service\requirements-ai-service.txt
```

Levantar FastAPI:

```bash
PORT=8000 scripts/run_local.sh
```

O manualmente:

```bash
cd ai_service
uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port ${PORT:-8000}
```

La documentacion interactiva queda disponible en:

```text
http://localhost:8000/docs
```

## Variables de entorno

Ver `.env.example`.

```text
PFI_ROOT=/content/drive/MyDrive/PFI_MVP
PFI_MODEL_REGISTRY=config/model_registry_final.json
PFI_DATA_FREEZE_CONFIG=config/data_freeze_config.json
PFI_OUTPUT_DIR=outputs
PORT=8000
```

`PFI_ROOT` apunta a la raiz externa donde pueden vivir modelos, resultados, figuras y reportes. Los datasets completos, imagenes medicas pesadas, checkpoints y resultados grandes no deben subirse a este repositorio.

## Endpoints FastAPI actuales

Implementados actualmente en `ai_service/pfi_ai_service/api.py`:

- `GET /health`: estado del servicio y bandera `human_review_required`.
- `GET /models`: registro tecnico de modelos configurado en codigo y rutas esperadas.
- `POST /inference/sagittal`: contrato tecnico para pipeline sagital.
- `POST /inference/axial`: contrato tecnico para pipeline axial complementario.
- `POST /pipeline/run`: contrato principal consumible por backend.
- `GET /agent/worklist`: lee la worklist tecnica del agente IA desde `PFI_ROOT/results/E14_ai_agent_orchestrator`.
- `GET /agent/report`: genera resumen tecnico del agente IA a partir de la worklist y metricas disponibles.
- `GET /agent/report/{run_id}`: recupera reporte tecnico materializado por corrida.
- `GET /agent/regression-test`: smoke tecnico de politica asistiva.

Ver `docs/AI_MODULE_API_CONTRACT.md` para el contrato detallado.

## Comunicacion con el backend

El backend Spring Boot debe llamar a este modulo mediante HTTP interno o privado. La integracion recomendada es:

1. El backend recibe/sube archivos y gestiona permisos.
2. El backend deja datos de trabajo en almacenamiento controlado o envia referencias tecnicas al AI Module.
3. El AI Module procesa la solicitud y devuelve JSON con resultados, rutas de artefactos, metricas tecnicas y `human_review_required=true`.
4. El backend persiste el resultado editable y lo presenta al frontend para revision profesional.

Este modulo no debe gestionar usuarios finales, autorizacion de producto, historia clinica ni decisiones medicas.

## Validacion

Cuando se modifique codigo funcional:

```bash
pytest
python -m compileall ai_service/pfi_ai_service
```

Para validar import del API:

```bash
cd ai_service
python -c "from pfi_ai_service.api import app; print(app.title)"
```

## Limitaciones metodologicas

- Prototipo academico reproducible, no dispositivo medico.
- No emitir diagnostico clinico.
- No recomendar tratamientos.
- No presentar resultados como decision medica.
- Toda medicion debe describirse como valor geometrico derivado de mascara.
- Toda salida debe ser revisable y editable por profesionales.
- No usar datos sensibles ni identificables de pacientes.
- Mantener datasets, checkpoints grandes y outputs pesados fuera del repositorio.
