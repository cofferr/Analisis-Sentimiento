# CLAUDE.md — Lab 2 NLP: Análisis de Sentimientos

## Contexto

Lab de la Universidad Sergio Arboleda. Se construye un clasificador binario de sentimientos
(positive / negative) sobre el dataset Sentiment140. Todo el proceso experimental se registra
en MLflow; el modelo final se expone via FastAPI en AWS.

---

## Estructura del repositorio

```
.
├── CLAUDE.md
├── README.md                        # Declara herramientas de IA usadas
├── docker/
│   ├── docker-compose.yml           # Levanta MLflow + FastAPI juntos
│   ├── mlflow/
│   │   └── Dockerfile
│   └── api/
│       ├── Dockerfile
│       └── main.py                  # FastAPI
├── notebooks/
│   ├── experiment.ipynb             # Notebook principal — se ejecuta en SageMaker
│   └── experiment_audit.ipynb       # Entregable de sustentación
├── reports/
│   ├── error_analysis.csv
│   └── error_analysis.md
└── requirements.txt
```

---

## Flujo de trabajo

```
Local (desarrollo)                SageMaker (ejecución oficial)
──────────────────                ──────────────────────────────
Escribir y probar el código  →    Subir experiment.ipynb al Notebook Instance
Probar con ~1000 tweets      →    Ejecutar celdas desde SageMaker
Verificar JSON de config     →    Runs quedan registrados con ResourceArn válido
```

**Los runs ejecutados fuera de SageMaker no son válidos.** El evaluador cruza el
`ResourceArn` de `/opt/ml/metadata/resource-metadata.json` con el tag `notebook_arn`
de cada run. Sin esa correspondencia el run se descarta.

---

## Infraestructura Docker (EC2)

El `docker-compose.yml` levanta dos servicios en la misma EC2:

- **MLflow Tracking Server** — con soporte para Model Registry y aliases.
  Usa S3 como artifact store.
- **FastAPI** — sirve el modelo desde `sentiment140@champion`, nunca desde archivo local.

```bash
# En la EC2, una vez configurada
docker compose up -d

# Variables de entorno requeridas en la EC2
MLFLOW_BACKEND_STORE_URI=...      # postgresql o sqlite
MLFLOW_ARTIFACT_ROOT=s3://...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Ambos servicios deben ser accesibles desde internet durante la ventana de evaluación.

---

## Dataset

```python
from datasets import load_dataset

ds = load_dataset(
    "adilbekovich/Sentiment140Twitter",
    revision="b6037e127257d95b9b23d31f78b264b9ebe697fd",
)
# train: 1_360_000 registros | test: 240_000 registros
# 0 → negative,  1 → positive
# test se usa UNA SOLA VEZ al final, nunca para seleccionar configuraciones
```

---

## Muestra y folds — generar UNA sola vez, nunca volver a cambiar

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold

train_df = ds["train"].to_pandas()
train_df["original_index"] = train_df.index  # índice original del split train (desde 0)

sample = train_df.sample(n=200_000, stratify=train_df["label"], random_state=42)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
folds = list(skf.split(sample, sample["label"]))
```

Generar `protocol/partitions.csv` con encabezado exacto `index,fold`:
- `index` = posición original en `train` (no el índice del sample)
- `fold` = fold k en que ese registro actúa como validación
- 200 000 filas, sin duplicados, ordenadas ascendentemente por `index`

Este archivo se registra como artefacto del run de protocolo y **no cambia más**.

---

## Secciones del notebook `experiment.ipynb`

El notebook tiene estas secciones en orden. Cada integrante ejecuta las secciones
que le corresponden desde su propio SageMaker.

```
0. Setup y conexión a MLflow
1. Cargar dataset y generar muestra/folds
2. Run de protocolo  ← ejecutar primero, una sola vez entre todos
3. T0 — baseline trivial
4. B0 — baseline experimental
5. Preprocesamiento  (P_STOPWORDS, P_STOPWORDS_NEGATION, P_LEMMA, P_ELONGATION, P_EMOJI)
6. Representación    (R_BOW, R_TFIDF_UNI, R_TFIDF_UNI_BI, R_SPACY)
7. Clasificador      (C_LOGREG, C_LINEAR_SVM, C_SGD)
8. Selección del pipeline candidato
9. Ablación
10. Reentrenamiento final + evaluación en test
11. Análisis de errores
```

---

## Convenciones MLflow

| Campo | Valor |
|-------|-------|
| Experiment name | `nlp-lab2-sentiment140` (exacto, sensible a mayúsculas) |
| `lab_run_type` | `protocol` / `experiment` / `final` |
| `lab_experiment_id` | `T0`, `B0`, `P_STOPWORDS`, `P_STOPWORDS_NEGATION`, `P_LEMMA`, `P_ELONGATION`, `P_EMOJI`, `R_BOW`, `R_TFIDF_UNI`, `R_TFIDF_UNI_BI`, `R_SPACY`, `C_LOGREG`, `C_LINEAR_SVM`, `C_SGD`, `ABLATION`, `EXTRA` |
| `lab_stage` | `reference`, `baseline`, `preprocessing`, `representation`, `classifier`, `ablation` |

Runs sin el tag `lab_run_type` son ignorados por el evaluador.
`lab_run_type`, `lab_stage` y `lab_experiment_id` son sensibles a mayúsculas/minúsculas.

### Tags obligatorios — run experimental

```
lab_run_type=experiment
lab_protocol_run_id=<run_id del run de protocolo>
lab_experiment_id=<código del experimento>
lab_stage=<etapa>
lab_member_id=<id asignado por el curso>
lab_configuration_id=<CFG_XXX>
notebook_arn=<ResourceArn del SageMaker Notebook Instance>
```

### Métricas obligatorias — run experimental

```
macro_f1_fold_0
macro_f1_fold_1
macro_f1_fold_2
macro_f1_mean   # promedio aritmético de los tres folds
macro_f1_std    # desviación estándar poblacional ddof=0
```

El evaluador verifica aritmética con tolerancia absoluta 1e-6.

### Artefactos obligatorios — run experimental

```
run/configuration.json
provenance/sagemaker-resource-metadata.json
```

### Tags / params / métricas / artefactos — run final

```
Tags:       lab_run_type=final, lab_protocol_run_id, lab_selected_experiment_run_id,
            lab_configuration_id, lab_member_id, notebook_arn
Param:      training_size=1360000
Métrica:    test_macro_f1
Artefactos: run/configuration.json, provenance/sagemaker-resource-metadata.json,
            reports/error_analysis.csv, reports/error_analysis.md
```

### Run de protocolo — params y artefactos

```
Params:
  dataset_id=adilbekovich/Sentiment140Twitter
  dataset_revision=b6037e127257d95b9b23d31f78b264b9ebe697fd
  sampling_strategy=stratified
  sample_size=200000
  random_seed=42
  cv_strategy=StratifiedKFold
  cv_folds=3
  cv_shuffle=true

Artefactos:
  protocol/partitions.csv
  protocol/members.csv      # encabezado: member_id,notebook_arn
```

---

## Procedencia en SageMaker (obligatorio en cada run)

```python
import shutil, mlflow

shutil.copy(
    "/opt/ml/metadata/resource-metadata.json",
    "provenance/sagemaker-resource-metadata.json"
)
mlflow.log_artifact(
    "provenance/sagemaker-resource-metadata.json",
    artifact_path="provenance"
)
```

El `ResourceArn` de ese archivo debe coincidir exactamente con el tag `notebook_arn`.
**No editar el archivo.**

---

## Esquema de `run/configuration.json`

Exactamente tres claves de primer nivel. No incluir nombres, timestamps ni run IDs.

```json
{
  "preprocessing": {
    "lowercase": true,
    "url": "token:url",
    "mention": "token:user",
    "whitespace": "normalize",
    "stopwords": "keep",
    "negators": [],
    "lemmatize": false,
    "elongation": "keep",
    "elongation_spec": null,
    "emoji": "keep",
    "emoji_spec": null,
    "resources": {},
    "additional": {}
  },
  "representation": {
    "type": "bow",
    "ngram_range": [1, 1],
    "library": "sklearn",
    "library_version": "X.Y.Z",
    "spacy_model": null,
    "spacy_model_version": null,
    "parameters": {}
  },
  "classifier": {
    "type": "logistic_regression",
    "library": "sklearn",
    "library_version": "X.Y.Z",
    "parameters": {}
  }
}
```

**Valores controlados:**
- `url` / `mention`: `keep` | `drop` | `token:<valor>`
- `whitespace`: `keep` | `normalize`
- `stopwords`: `keep` | `remove` | `remove_preserve_negation`
- `elongation`: `keep` | `normalize`
- `emoji`: `keep` | `text`
- `representation.type`: `bow` | `tfidf` | `spacy_embedding`
- `classifier.type`: `logistic_regression` | `linear_svm` | `sgd` | `most_frequent`

**Reglas de campos nulos:**
- `elongation_spec`: no nulo solo si `elongation=normalize`
- `emoji_spec`: no nulo solo si `emoji=text`
- `negators`: lista no vacía solo si `stopwords=remove_preserve_negation`
- `spacy_model` / `spacy_model_version`: no nulos solo si `type=spacy_embedding`
- `ngram_range`: `[]` si `type=spacy_embedding`; lista de dos enteros para `bow`/`tfidf`
- Si `type=spacy_embedding`: `parameters` debe incluir `document_vector_method` (string no vacío)
- Para T0: `preprocessing=null`, `representation=null`, `classifier.type=most_frequent`

---

## Experimentos obligatorios y su distribución sugerida

**PENDIENTE: confirmar `member_id` con el profesor antes de ejecutar cualquier run.**

### Todos ejecutan (uno lo hace y queda en el experimento para todos)
| Run | `lab_experiment_id` | `lab_stage` | Quién lo ejecuta |
|-----|--------------------|--------------|--------------------|
| Protocolo | — | — | Un integrante |
| T0 | `T0` | `reference` | Un integrante |
| B0 | `B0` | `baseline` | Un integrante |

T0 y B0 no cuentan para el mínimo individual.

### Distribución sugerida por integrante (≥3 runs contados, ≥2 etapas)

| Integrante | Runs | Etapas cubiertas |
|------------|------|-----------------|
| MEMBER_ID_AQUI | `P_STOPWORDS`, `P_STOPWORDS_NEGATION`, `P_LEMMA`, `P_ELONGATION`, `P_EMOJI` + 1 ablación | preprocessing + ablation |
| Integrante 2 | `R_BOW`, `R_TFIDF_UNI`, `R_TFIDF_UNI_BI`, `R_SPACY` + 1 ablación | representation + ablation |
| Integrante 3 | `C_LOGREG`, `C_LINEAR_SVM`, `C_SGD` + 1 ablación | classifier + ablation |

Cada integrante ejecuta sus runs desde **su propio** SageMaker Notebook Instance.

---

## Ablación

Después de seleccionar el pipeline candidato, revertir decisiones una a la vez.

Decisiones ablacionables (valores exactos para el param `ablation_reverted_decision`):
```
preprocessing.stopwords
preprocessing.lemmatize
preprocessing.elongation
preprocessing.emoji
representation
classifier
```

- Si el candidato difiere de B0 en 1 decisión → evaluar 1 ablación
- Si difiere en 2 o más → evaluar al menos 2

```python
# Métrica adicional obligatoria en runs de ablación
macro_f1_delta = macro_f1_mean(candidato) - macro_f1_mean(ablación)

# Tag adicional
lab_ablation_parent_run_id = <run_id del candidato>
```

---

## Modelo final

```python
import mlflow

# 1. Reentrenar con los 1_360_000 registros de train
# 2. Evaluar UNA SOLA VEZ en test — no ajustar nada a partir de este resultado
# 3. Registrar el pipeline completo (preprocesamiento + vectorizador + clasificador)

with mlflow.start_run() as run:
    mlflow.log_param("training_size", 1_360_000)
    mlflow.log_metric("test_macro_f1", score)
    mlflow.sklearn.log_model(pipeline, "model")

# Registrar en Model Registry
mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "sentiment140")
client = mlflow.MlflowClient()
client.set_registered_model_alias("sentiment140", "champion", mv.version)
```

El pipeline debe recibir texto crudo y devolver `negative` o `positive`.
La API carga el modelo **solo desde `sentiment140@champion`**, nunca desde archivo local.

---

## API FastAPI — endpoints obligatorios

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/v1/predict` | Inferencia |
| `GET` | `/audit/protocol` | Run de protocolo desde MLflow |
| `GET` | `/audit/runs` | Todos los runs presentados |
| `GET` | `/audit/contributions` | Contribución por integrante |
| `GET` | `/audit/model` | Modelo desplegado y trazabilidad |
| `GET` | `/health` | Estado del modelo |

### `/api/v1/predict`

```jsonc
// Request — un texto
{"text": "this movie was great"}

// Request — lote (máx 32 elementos, máx 1000 chars por texto)
{"text": ["i loved it", "worst day ever"]}

// Response 200
{"model_run_id": "FINAL_RUN_ID", "predictions": ["positive"]}
{"model_run_id": "FINAL_RUN_ID", "predictions": ["positive", "negative"]}
```

Inválidos → HTTP 4xx sin resultados parciales:
`text` ausente, `null`, vacío, solo espacios, lista vacía, >32 elementos,
texto >1000 chars, elemento no-string.
Tiempo máximo: **10 segundos**.

### Si MLflow no está disponible
`/audit/*` → HTTP 503 `{"detail": "mlflow_unavailable"}`. No reconstruir desde caché.

---

## Análisis de errores

```python
# Semilla 42, mínimo 20 errores, ambas clases si existen
```

`reports/error_analysis.csv` — encabezado exacto:
```
index,text,true_label,predicted_label,category
```

Categorías válidas: `negation`, `intensification`, `contrast`, `mixed`, `emoji`,
`elongation`, `informal`, `hashtag`, `sarcasm`, `other`

`reports/error_analysis.md` — frecuencia por categoría + interpretación de los
dos patrones más frecuentes.

Ambos archivos van también como artefactos del run final.

---

## Checklist antes de la entrega

- [ ] `member_id` de los tres integrantes confirmado con el profesor
- [ ] Run de protocolo único con `protocol/partitions.csv` y `protocol/members.csv`
- [ ] T0 y B0 ejecutados con los mismos tres folds del protocolo
- [ ] Todos los P_*, R_*, C_* registrados con los mismos tres folds
- [ ] Cada integrante: ≥3 configuraciones contadas, ≥2 etapas distintas
- [ ] Configuración final elegida sin ver `test`
- [ ] `test` evaluado una sola vez; `test_macro_f1` registrado en el run final
- [ ] `sentiment140@champion` apunta al run final válido
- [ ] API carga modelo desde `sentiment140@champion`, no desde archivo local
- [ ] `/api/v1/predict`, `/audit/model` y `/health` identifican el mismo run final
- [ ] MLflow Tracking Server accesible desde internet
- [ ] API accesible desde internet
- [ ] `reports/error_analysis.csv` y `.md` registrados como artefactos del run final
- [ ] README declara las herramientas de IA utilizadas
- [ ] Repositorio sin credenciales, claves ni tokens
