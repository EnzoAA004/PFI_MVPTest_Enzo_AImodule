FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

COPY ai_service/requirements-ai-service.txt .
RUN pip install --no-cache-dir -r requirements-ai-service.txt

COPY ai_service/pfi_ai_service ./pfi_ai_service
COPY config ./config
COPY models/final ./models/final

EXPOSE 8000

CMD ["sh", "-c", "uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
