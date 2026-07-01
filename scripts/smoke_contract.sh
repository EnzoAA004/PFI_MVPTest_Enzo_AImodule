#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

curl --fail --silent --show-error "${BASE_URL}/health"
printf "\n"

curl --fail --silent --show-error "${BASE_URL}/models"
printf "\n"

curl --fail --silent --show-error \
  -X POST "${BASE_URL}/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "caseId": "case-001",
    "plane": "sagittal",
    "modelKey": "sagittal_spider",
    "inputPath": "studies/case-001",
    "metadata": {
      "source": "smoke-contract"
    }
  }'
printf "\n"
