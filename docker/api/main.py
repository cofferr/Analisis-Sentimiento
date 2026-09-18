import json
import os
import tempfile
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
        _, run_id = _get_model()
        return {"status": "ok", "model_run_id": run_id}
    except (MlflowException, OSError):
        return {"status": "unavailable", "model_run_id": None}


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
            max_results=2,
        )
        if len(runs) != 1:
            raise HTTPException(status_code=409, detail="protocol_not_unique")

        run = runs[0]
        params = run.data.params

        return {
            "protocol_run_id": run.info.run_id,
            "dataset_id": params.get("dataset_id"),
            "dataset_revision": params.get("dataset_revision"),
            "sampling_strategy": params.get("sampling_strategy"),
            "sample_size": int(params["sample_size"]) if "sample_size" in params else None,
            "random_seed": int(params["random_seed"]) if "random_seed" in params else None,
            "cv_strategy": params.get("cv_strategy"),
            "cv_folds": int(params["cv_folds"]) if "cv_folds" in params else None,
            "cv_shuffle": params.get("cv_shuffle", "").lower() == "true" if "cv_shuffle" in params else None,
            "partitions_artifact": "protocol/partitions.csv",
            "members_artifact": "protocol/members.csv",
        }
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc


def _list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = "") -> list:
    """Lista rutas de artefactos de un run, recorriendo subdirectorios, en orden lexicografico."""
    paths = []
    for artifact in client.list_artifacts(run_id, path or None):
        if artifact.is_dir:
            paths.extend(_list_artifacts_recursive(client, run_id, artifact.path))
        else:
            paths.append(artifact.path)
    return sorted(paths)


def _load_configuration_artifact(client: MlflowClient, run_id: str, artifact_paths: list):
    """Devuelve el contenido de run/configuration.json si existe y es JSON valido, o None."""
    if "run/configuration.json" not in artifact_paths:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, "run/configuration.json", tmp_dir)
            with open(local_path) as f:
                return json.load(f)
    except (OSError, ValueError, MlflowException):
        return None


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

        result = []
        for run in runs:
            artifact_paths = _list_artifacts_recursive(client, run.info.run_id)
            result.append({
                "run_id": run.info.run_id,
                "status": run.info.status,
                "run_type": run.data.tags.get("lab_run_type"),
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
                "artifacts": artifact_paths,
                "configuration": _load_configuration_artifact(client, run.info.run_id, artifact_paths),
            })

        result.sort(key=lambda r: r["run_id"])
        return result
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc


NON_COUNTED_EXPERIMENT_IDS = {"T0", "B0"}


def _load_members_artifact(client: MlflowClient, protocol_run_id: str) -> list:
    """Descarga protocol/members.csv (member_id,notebook_arn) desde el artifact
    store de MLflow para el run de protocolo. El archivo no existe en el
    filesystem local de la API; se obtiene por run_id via el tracking server."""
    import csv

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(protocol_run_id, "protocol/members.csv", tmp_dir)
        with open(local_path, newline="") as f:
            return list(csv.DictReader(f))


@app.get("/audit/contributions")
def audit_contributions():
    try:
        client = _get_mlflow_client()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")

        protocol_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.lab_run_type = 'protocol'",
            max_results=2,
        )
        if len(protocol_runs) != 1:
            raise HTTPException(status_code=409, detail="protocol_not_unique")

        members_rows = _load_members_artifact(client, protocol_runs[0].info.run_id)
        known_member_ids = {row["member_id"] for row in members_rows}
        notebook_arn_by_member = {row["member_id"]: row["notebook_arn"] for row in members_rows}

        experiment_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.lab_run_type = 'experiment'",
            max_results=5000,
        )

        members_data: dict = {
            member_id: {
                "member_id": member_id,
                "notebook_arn": notebook_arn_by_member[member_id],
                "run_ids": [],
                "counted_run_ids": [],
                "configuration_ids": set(),
                "stages": set(),
            }
            for member_id in known_member_ids
        }
        invalid_run_ids = []
        unattributed_run_ids = []

        for run in experiment_runs:
            run_id = run.info.run_id
            member_id = run.data.tags.get("lab_member_id")
            lab_experiment_id = run.data.tags.get("lab_experiment_id")
            configuration_id = run.data.tags.get("lab_configuration_id")
            stage = run.data.tags.get("lab_stage")

            if not member_id or not configuration_id or not lab_experiment_id:
                invalid_run_ids.append(run_id)
                continue

            if member_id not in known_member_ids:
                unattributed_run_ids.append(run_id)
                continue

            entry = members_data[member_id]
            entry["run_ids"].append(run_id)
            if lab_experiment_id not in NON_COUNTED_EXPERIMENT_IDS:
                entry["counted_run_ids"].append(run_id)
            entry["configuration_ids"].add(configuration_id)
            if stage:
                entry["stages"].add(stage)

        members = []
        for member_id in sorted(members_data.keys()):
            entry = members_data[member_id]
            members.append({
                "member_id": entry["member_id"],
                "notebook_arn": entry["notebook_arn"],
                "run_ids": sorted(entry["run_ids"]),
                "counted_run_ids": sorted(entry["counted_run_ids"]),
                "configuration_ids": sorted(entry["configuration_ids"]),
                "stages": sorted(entry["stages"]),
                "valid_configurations": len(entry["configuration_ids"]),
            })

        return {
            "members": members,
            "invalid_run_ids": sorted(invalid_run_ids),
            "unattributed_run_ids": sorted(unattributed_run_ids),
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
