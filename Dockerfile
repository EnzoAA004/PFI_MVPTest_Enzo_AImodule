FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/ai_service \
    PORT=8000 \
    PFI_MODEL_DIR=/app/models/final \
    PFI_OUTPUT_DIR=/app/outputs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY ai_service/requirements-ai-service.txt /tmp/requirements-ai-service.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir -r /tmp/requirements-ai-service.txt

COPY ai_service /app/ai_service
COPY config /app/config
COPY models/final /app/models/final
# Los scripts de evaluacion y los de infraestructura acompanan a los tests, que ya
# viajan dentro de `ai_service`. Sin ellos la suite no se puede correr contra la
# imagen: veintiuna pruebas fallaban siempre por un archivo ausente, y una tanda de
# fallas permanentes entrena a ignorar el resultado de la suite entera.
#
# Son texto y no se ejecutan en runtime: 220 KB de shell y Python que no agregan
# dependencias ni abren un camino de ejecucion nuevo.
COPY scripts /app/scripts
COPY infra /app/infra
# El contrato de hallazgos degenerativos vive en `docs/contracts` y sus pruebas lo
# leen de ahi: es el esquema congelado, no documentacion decorativa. Sin copiarlo, dos
# pruebas fallan contra la imagen aunque pasen desde el repo.
COPY docs /app/docs

RUN mkdir -p /app/models/final /app/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
