FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.16.2

EXPOSE 5000

ENTRYPOINT ["sh", "-c", "mlflow server \
    --backend-store-uri sqlite:////mlflow/data/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000"]
