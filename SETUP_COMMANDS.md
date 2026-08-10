# Sprint 0 — Paste-Ready Commands

Run these in **Git Bash** from the repo root.

## 1. Confirm you are in the repo

```bash
cd /c/projects/mle-miniproject-2025paml567
pwd
```

Expected: `/c/projects/mle-miniproject-2025paml567`

## 2. Unpack the scaffold

Assuming the zip landed in Downloads:

```bash
unzip -o ~/Downloads/sprint0_scaffold.zip -d .
ls -la
```

## 3. Copy the dataset in

Only `train.csv` and `test.csv` are needed. Skip the 145 MB Sentiment140 file.

```bash
cp /path/to/train.csv data/raw/
cp /path/to/test.csv  data/raw/
ls -la data/raw/
```

## 4. Create the conda environment

In **Anaconda Prompt** (not Git Bash):

```
cd C:\projects\mle-miniproject-2025paml567
conda env create -f environment.yml
conda activate mle
```

## 5. Run the validation gate

```
python -m src.validation.validate --config configs/sentiment.yaml
```

Expected: `RESULT: PASS -- 27,481 rows validated.` with 3 findings reported.

## 6. Run the negative test

```
python -m tests.inject_defects
python -m src.validation.validate --config configs/sentiment.yaml --input data/raw/_corrupt_test.csv
```

Expected: `RESULT: FAIL -- pipeline halted.` — this is the gate working.

## 7. Commit 1

Back in **Git Bash**:

```bash
git add -A
git status
git commit -m "Sprint 0: scaffold, environment, four-level validation framework

- Task-agnostic config pattern (sentiment primary, tickets secondary)
- M2 four-level validation: schema, range, statistical, business rules
- Business rules BR-01..BR-04 encoding real dataset findings
- Data quality incident report: Sentiment140 truncation, fabricated columns
- Negative test proving the validation gate fails on injected defects"
git push -u origin main
```

First push opens a browser for device authorization — enter the code Git Bash
prints, approve once, and it is remembered.

## 8. Verify on GitHub

Open your repo page. You should see the file tree and one commit.
