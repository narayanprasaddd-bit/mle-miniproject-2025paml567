# Evidence screenshots

## M3 — Experimentation & Reproducibility

| File | Shows |
|---|---|
| `mlflow_runs_table.png` | Three tracked runs with accuracy and f1_macro |
| `mlflow_run_detail.png` | run1 parameters, metrics, and reproducibility tags |
| `mlflow_model_registry.png` | sentiment-classifier v1 with the staging alias |

## M4 — Packaging & Deployment

| File | Shows |
|---|---|
| `01_api_docs.png` | OpenAPI page, four endpoints, generated schemas |
| `02_api_health.png` | 200 — both artefacts loaded, 20,000 features, three classes |
| `03_api_predict_request.png` | Request body submitted to /predict/sentiment |
| `04_api_predict_response.png` | 200 — label, probabilities, confidence, latency, model version |
| `05_api_422_request.png` | Empty string submitted as input |
| `06_api_422_response.png` | **422** — `string_too_short`, rejected before reaching model code |

The 422 pair is the direct evidence for the brief's "handle malformed/edge-case
inputs" requirement. Note the response body names the failing field and the
rule that failed, rather than returning an opaque error.

## Still to capture

| File | How |
|---|---|
| `ci_docker_build.png` | GitHub repo -> Actions tab -> green CI run |
