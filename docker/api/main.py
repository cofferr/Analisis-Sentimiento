import os
import time
from typing import Union

import mlflow
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, field_validator

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "nlp-lab2-sentiment140")
REGISTERED_MODEL_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "sentiment140")
MODEL_ALIAS = os.environ.get("MLFLOW_MODEL_ALIAS", "champion")
MAX_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 1000
PREDICT_TIMEOUT_SECONDS = 10

# Sin esto, el cliente MLflow reintenta con backoff y puede tardar mucho mas de
# PREDICT_TIMEOUT_SECONDS antes de reportar que el tracking server no responde.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="nlp-lab2-sentiment140 API")

_model_cache: dict = {"pipeline": None, "run_id": None}


class PredictRequest(BaseModel):
    text: Union[str, list]

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v):
        if v is None:
            raise ValueError("text no puede ser null")

        if not isinstance(v, (str, list)):
            raise ValueError("text debe ser string o lista de strings")

        if isinstance(v, str):
            if v.strip() == "":
                raise ValueError("text no puede estar vacio o ser solo espacios")
            if len(v) > MAX_TEXT_LENGTH:
                raise ValueError(f"text no puede superar {MAX_TEXT_LENGTH} caracteres")
            return v

        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("la lista de text no puede estar vacia")
            if len(v) > MAX_BATCH_SIZE:
                raise ValueError(f"la lista de text no puede superar {MAX_BATCH_SIZE} elementos")
            for item in v:
                if not isinstance(item, str):
                    raise ValueError("todos los elementos de text deben ser string")
                if item.strip() == "":
                    raise ValueError("ningun elemento de text puede estar vacio o ser solo espacios")
                if len(item) > MAX_TEXT_LENGTH:
                    raise ValueError(f"ningun elemento de text puede superar {MAX_TEXT_LENGTH} caracteres")
            return v

        raise ValueError("text debe ser string o lista de strings")


def _get_mlflow_client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _load_champion_model():
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    client = _get_mlflow_client()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    pipeline = mlflow.sklearn.load_model(model_uri)
    return pipeline, mv.run_id


def _get_model():
    if _model_cache["pipeline"] is None:
        pipeline, run_id = _load_champion_model()
        _model_cache["pipeline"] = pipeline
        _model_cache["run_id"] = run_id
    return _model_cache["pipeline"], _model_cache["run_id"]


@app.post("/api/v1/predict")
def predict(request: PredictRequest):
    start = time.monotonic()

    texts = [request.text] if isinstance(request.text, str) else request.text

    try:
        pipeline, run_id = _get_model()
    except (MlflowException, OSError) as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc

    try:
        raw_predictions = pipeline.predict(texts)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if time.monotonic() - start > PREDICT_TIMEOUT_SECONDS:
        raise HTTPException(status_code=504, detail="prediction_timeout")

    predictions = [_normalize_label(p) for p in raw_predictions]

    return {"model_run_id": run_id, "predictions": predictions}


def _normalize_label(prediction) -> str:
    """El pipeline final (CLAUDE.md) ya devuelve 'positive'/'negative' directamente;
    se tolera tambien la codificacion binaria 0/1 por robustez ante otros pipelines."""
    if isinstance(prediction, str):
        return prediction
    return "positive" if int(prediction) == 1 else "negative"


@app.get("/health")
def health():
    try:
        _get_model()
        return {"status": "ok"}
    except (MlflowException, OSError):
        return {"status": "unavailable"}


@app.get("/audit/protocol")
def audit_protocol():
    try:
        client = _get_mlflow_client()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.lab_run_type = 'protocol'",
            max_results=1,
        )
        if not runs:
            raise HTTPException(status_code=404, detail="protocol_run_not_found")

        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "params": run.data.params,
            "tags": run.data.tags,
        }
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc


@app.get("/audit/runs")
def audit_runs():
    try:
        client = _get_mlflow_client()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.lab_run_type != ''",
            max_results=5000,
        )
        return {
            "runs": [
                {
                    "run_id": run.info.run_id,
                    "tags": run.data.tags,
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                }
                for run in runs
            ]
        }
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc


@app.get("/audit/contributions")
def audit_contributions():
    try:
        client = _get_mlflow_client()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.lab_run_type = 'experiment'",
            max_results=5000,
        )

        contributions: dict = {}
        for run in runs:
            member_id = run.data.tags.get("lab_member_id", "unknown")
            stage = run.data.tags.get("lab_stage", "unknown")
            entry = contributions.setdefault(
                member_id, {"run_count": 0, "stages": set()}
            )
            entry["run_count"] += 1
            entry["stages"].add(stage)

        return {
            member_id: {
                "run_count": data["run_count"],
                "stages": sorted(data["stages"]),
            }
            for member_id, data in contributions.items()
        }
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc


@app.get("/audit/model")
def audit_model():
    try:
        client = _get_mlflow_client()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        run = client.get_run(mv.run_id)
        return {
            "registered_model_name": REGISTERED_MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": mv.version,
            "run_id": mv.run_id,
            "tags": run.data.tags,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc
