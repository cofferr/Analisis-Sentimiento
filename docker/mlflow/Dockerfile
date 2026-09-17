FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    boto3==1.35.36 \
    psycopg2-binary==2.9.9

EXPOSE 5000

ENTRYPOINT ["sh", "-c", "mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000"]
