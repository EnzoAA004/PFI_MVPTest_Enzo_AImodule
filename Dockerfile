FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/ai_service \
    PORT=8000 \
    PFI_MODEL_DIR=/models/final \
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

RUN mkdir -p /models/final /app/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port ${PORT:-8000}"]