"""Crea el bucket S3 de MLFLOW_ARTIFACT_ROOT si no existe. Se ejecuta antes de
arrancar el servidor MLflow (ver entrypoint.sh)."""

import os
import sys
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def main():
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT", "")
    if not artifact_root.startswith("s3://"):
        print(f"MLFLOW_ARTIFACT_ROOT no es una ruta s3:// ({artifact_root!r}); "
              "se omite la creacion de bucket.")
        return

    parsed = urlparse(artifact_root)
    bucket_name = parsed.netloc
    if not bucket_name:
        print(f"No se pudo extraer el nombre de bucket de {artifact_root!r}", file=sys.stderr)
        sys.exit(1)

    region = os.environ.get("AWS_DEFAULT_REGION")
    s3 = boto3.client("s3", region_name=region)

    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' ya existe.")
        return
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in ("404", "NoSuchBucket"):
            raise

    print(f"Bucket '{bucket_name}' no existe. Creando en region '{region}'...")
    create_kwargs = {"Bucket": bucket_name}
    if region and region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**create_kwargs)
    print(f"Bucket '{bucket_name}' creado.")


if __name__ == "__main__":
    main()
