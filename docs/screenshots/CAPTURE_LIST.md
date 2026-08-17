# Remaining screenshots to capture

Six items. Each command goes in **Anaconda Prompt** with `(mle)` active, from
the repo root. Save every image into `docs/screenshots/`.

---

## M2 — the validation gate  (2 shots)

### 1. `validation_pass.png`

```
python -m src.validation.validate --config configs/sentiment.yaml
```

Capture the whole output. Must show `RESULT: PASS -- 27,481 rows validated`
and the three `[L4/BR-...]` findings.

### 2. `validation_fail.png`

```
python -m tests.inject_defects
python -m src.validation.validate --config configs/sentiment.yaml --input data/raw/_corrupt_test.csv
```

Capture the second command's output. Must show the `BLOCKING FAILURES (5)`
block and `RESULT: FAIL -- pipeline halted`.

**This is the more valuable of the pair** — it proves the gate actually stops
bad data rather than merely existing.

---

## M2 — the DVC pipeline  (2 shots)

### 3. `dvc_repro.png`

```
dvc repro
```

If everything is already up to date it will say so, which is itself fine
evidence — it demonstrates DVC skipping unchanged stages. To force a full run:

```
dvc repro --force
```

### 4. `dvc_dag.png`

```
dvc dag
```

Shows the stage dependency graph as ASCII art.

---

## M4 — the CI container build  (1 shot)

### 5. `ci_docker_build.png`

Browser, not terminal:

1. Open your repo on GitHub
2. **Actions** tab
3. Click the most recent **CI** run
4. Capture the two jobs — "Unit and contract tests" and "Build container image"
   — with their green ticks

If it is red, open the failing step and send me the log.

---

## Test suite  (1 shot)

### 6. `test_suite_all.png`

```
pytest tests/ -q
```

Must show `57 passed`. You already ran this — just re-run and snip.

---

## Then verify against the index

```bash
ls docs/screenshots/
```

Expected 16 files (15 images + README.md + CAPTURE_LIST.md):

```
01_api_docs.png                 mlflow_model_registry.png
02_api_health.png               mlflow_run_detail.png
03_api_predict_request.png      mlflow_runs_table.png
04_api_predict_response.png     validation_pass.png
05_api_422_request.png          validation_fail.png
06_api_422_response.png         dvc_repro.png
drift_detection_1.png           dvc_dag.png
drift_detection_2.png           ci_docker_build.png
drift_detection_3.png           test_suite_all.png
drift_tests.png
```

If a filename differs from the index, either rename it or tell me the actual
name and I will adjust `README.md` instead. The index and the folder must agree
— a README referencing files that do not exist is worse than no README.

Delete `CAPTURE_LIST.md` once everything is captured.
