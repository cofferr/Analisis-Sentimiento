#!/bin/sh
set -e

python /mlflow/ensure_bucket.py

exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000
