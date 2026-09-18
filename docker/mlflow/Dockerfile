FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    boto3==1.35.36 \
    psycopg2-binary==2.9.9

COPY ensure_bucket.py /mlflow/ensure_bucket.py
COPY entrypoint.sh /mlflow/entrypoint.sh
RUN chmod +x /mlflow/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/mlflow/entrypoint.sh"]
