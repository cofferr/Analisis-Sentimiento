# Lab 2 NLP — Análisis de Sentimientos (Sentiment140)

Universidad Sergio Arboleda. Clasificador binario de sentimientos (positive / negative)
sobre el dataset [Sentiment140](https://huggingface.co/datasets/adilbekovich/Sentiment140Twitter).
Todo el proceso experimental se registra en MLflow; el modelo final se expone vía FastAPI en AWS.

Ver [`CLAUDE.md`](./CLAUDE.md) para el detalle completo del flujo de trabajo, convenciones
de MLflow, esquema de configuración y checklist de entrega.

## Estructura

```
.
├── CLAUDE.md
├── docker/
│   ├── docker-compose.yml           # Levanta MLflow + FastAPI juntos
│   ├── mlflow/Dockerfile
│   └── api/
│       ├── Dockerfile
│       └── main.py                  # FastAPI
├── notebooks/
│   ├── experiment.ipynb             # Notebook principal — se ejecuta en SageMaker
│   └── experiment_audit.ipynb       # Entregable de sustentación
├── protocol/                        # partitions.csv y members.csv (generados una sola vez)
├── reports/
│   ├── error_analysis.csv
│   └── error_analysis.md
└── requirements.txt
```

## Cómo ejecutar

### Notebook de experimentación

`notebooks/experiment.ipynb` debe subirse a una **SageMaker Notebook Instance** para que
los runs queden con un `ResourceArn` válido; los runs generados fuera de SageMaker no son
válidos para la entrega. Antes de ejecutar cualquier run, confirmar el `member_id` de cada
integrante con el profesor.

### Infraestructura Docker (EC2)

```bash
cd docker
export MLFLOW_BACKEND_STORE_URI=...
export MLFLOW_ARTIFACT_ROOT=s3://...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
docker compose up -d
```

MLflow queda en el puerto `5000` y la API FastAPI en el puerto `8000`. Ambos deben ser
accesibles desde internet durante la ventana de evaluación.

### API

Documentación interactiva en `http://<host>:8000/docs`. Endpoints principales:
`POST /api/v1/predict`, `GET /audit/protocol`, `GET /audit/runs`,
`GET /audit/contributions`, `GET /audit/model`, `GET /health`.

## Herramientas de IA utilizadas

Este proyecto fue desarrollado con asistencia de **Claude Code** (Anthropic, modelo
Claude Sonnet 5), usado para:

- Generar la estructura inicial del repositorio (Docker, FastAPI, notebooks).
- Redactar el código base de preprocesamiento, representación, clasificadores y
  utilidades de logging a MLflow siguiendo el esquema definido en `CLAUDE.md`.
- Redactar la API FastAPI (`docker/api/main.py`) y su validación de entrada.

Todo el código generado fue revisado por el equipo antes de su ejecución. Las decisiones
experimentales (elección del pipeline candidato, interpretación del análisis de errores,
selección de hiperparámetros) fueron tomadas por los integrantes del equipo, no por la IA.
