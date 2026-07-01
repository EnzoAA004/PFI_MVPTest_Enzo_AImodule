#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker build -t pfi-ai-module -f Dockerfile .

cat <<'MSG'
Docker image built as pfi-ai-module.

The container will start in this terminal. In another terminal, test:

  curl http://localhost:8000/health
  curl http://localhost:8000/models
  curl -X POST http://localhost:8000/pipeline/run \
    -H "Content-Type: application/json" \
    -d '{"caseId":"case-demo-001","plane":"sagittal","modelKey":"sagittal_spider","inputPath":"demo/case-demo-001","metadata":{"source":"docker-smoke"}}'

This smoke does not require heavy models, datasets, or real medical images.
MSG

docker run --rm -p "${PORT:-8000}:8000" --env-file .env.example pfi-ai-module
