# Sprint 4 — Paste-Ready Commands

**Sprint 4 = M5 Monitoring, Drift & Retraining → 20% of the grade.**

---

## 1. Unpack

Git Bash:

```bash
cd /c/projects/mle-miniproject-2025paml567
unzip -o "/c/NP Documents and Tools/BITS/Mini Project/sprint4_files.zip" -d .
```

---

## 2. Run the drift detection

Anaconda Prompt, `(mle)` active:

```
cd C:\projects\mle-miniproject-2025paml567
conda activate mle
python -m src.monitoring.drift --config configs/sentiment.yaml
```

You will see five windows. What to check:

- **W0 control** must show **0 of 4 breached** and `NO_ACTION`.
  This is the most important line in the whole sprint. A monitor that fires on
  clean traffic is worse than no monitor.
- **W1, W2, W3** should each show `RETRAIN`
- **W4** should show `HOLD` — drift detected, but too few samples to act

**Screenshot the full output.** Save as `docs/screenshots/drift_detection.png`.

---

## 3. Run the drift tests

```
pytest tests/test_drift.py -v
```

Expect **6 passed**.

Then the full suite:

```
pytest tests/ -q
```

Expect **57 passed** (17 feature + 34 API + 6 drift).

---

## 4. Score live traffic (optional but nice for the demo)

Start the service in one window:

```
uvicorn src.serving.app:app --port 8000
```

Send it some requests via http://127.0.0.1:8000/docs, then in a second
Anaconda Prompt:

```
conda activate mle
cd C:\projects\mle-miniproject-2025paml567
python -m src.monitoring.drift --config configs/sentiment.yaml --from-log
```

This scores your actual logged predictions rather than the simulation. Good
material for the demo video — it shows the monitor reading real traffic.

---

## 5. Commit

Git Bash:

```bash
git add -A
git commit -m "Sprint 4: drift detection, simulation, and retraining trigger" -m "- Four label-free signals: OOV rate, token chi-squared, confidence KS, length sigma" -m "- Control window fires 0 of 4: no false alarm on undrifted traffic" -m "- Vocabulary, topic, and format drift all detected; signals discriminate by drift type" -m "- HOLD decision separates detecting drift from being able to act on it" -m "- Effect-size floor added to the KS signal: p=0.0025 with delta -0.009 was significant but operationally meaningless" -m "- Retraining trigger requires 2 of 4 signals plus 500 samples; promotion gated on human approval" -m "- 6 drift tests, 57 total"
git push
```
