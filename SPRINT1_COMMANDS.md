# Sprint 1 — Paste-Ready Commands

**Sprint 1 = M2 Data Engineering & Versioning → 20% of the grade.**

Run the Python steps in **Anaconda Prompt** with `(mle)` active.
Run the `git` steps in **Git Bash**.

---

## 1. Unpack

Git Bash, from the repo root:

```bash
cd /c/projects/mle-miniproject-2025paml567
unzip -o "/c/NP Documents and Tools/BITS/ML_Engineering/sprint1_files.zip" -d .
ls -la
```

Adjust the path to wherever the zip landed.

---

## 2. Initialise DVC

Anaconda Prompt:

```
cd C:\projects\mle-miniproject-2025paml567
dvc init
dvc remote add -d localremote D:/dvcstore
```

If you have no D: drive, use a folder on C: instead:

```
dvc remote add -d localremote C:/dvcstore
```

Where the bytes physically live does not affect the mark. What is graded is
that `.dvc` pointer files are committed to Git while the real data is not.

---

## 3. Track the raw data with DVC

```
dvc add data/raw/train.csv
dvc add data/raw/test.csv
```

This creates `train.csv.dvc` and `test.csv.dvc` — small text pointer files —
and adds the real CSVs to `.gitignore`. **The pointers go into Git; the data
does not.**

Push the data to the DVC remote:

```
dvc push
```

---

## 4. Run the pipeline

```
dvc repro
```

DVC runs validate → ingest → build_features in dependency order.

Expected end state:
```
BASELINE OOV RATE on held-out test = 0.0914
RESULT: feature store built.
```

To see the graph:

```
dvc dag
```

---

## 5. Run the unit tests

```
pytest tests/test_features.py -v
```

Expected: **17 passed**.

---

## 6. Commit and tag

Git Bash:

```bash
git status --porcelain -uall | grep -i "\.csv$"
```

Must return **nothing** — only `.csv.dvc` pointers should be tracked.

```bash
git add -A
git commit -m "Sprint 1: DVC versioning, ingestion audit, shared feature module" -m "- dvc.yaml pipeline: validate -> ingest -> build_features" -m "- Ingestion audit trail: every dropped row accounted for in reports/" -m "- Shared feature module: fitted TF-IDF vectorizer as the training/serving contract" -m "- SQLite offline feature store, split before fit to prevent leakage" -m "- 17 unit tests on the feature pipeline, including the skew test" -m "- OOV drift signal restricted to unigrams: bigram OOV saturates at 0.32 and cannot indicate drift"
git tag -a v1.0-raw -m "Raw corpus: 27,481 rows validated, 27,480 after cleaning"
git push
git push --tags
```

Tags need their own push — a plain `git push` does not send them.

---

## 7. Verify on GitHub

- Two commits in the history
- Tag `v1.0-raw` visible under Releases/Tags
- `data/raw/train.csv.dvc` present, `data/raw/train.csv` **absent**
