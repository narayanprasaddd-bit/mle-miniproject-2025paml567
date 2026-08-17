# Two fixes, one blocker

## 1. BLOCKER — `reproduce.py` crashed the `dvc repro` pipeline

Your screenshot 9 shows:

```
TypeError: Object of type bool_ is not JSON serializable
ERROR: failed to reproduce 'reproduce': ... exited with 1
```

**Cause.** `delta = abs(original - value)` produces a numpy float, so
`delta < TOLERANCE` yields `numpy.bool_` rather than a Python `bool`, and
`json.dump` cannot serialise it.

**Note what happened before the crash** — the metric comparison printed
correctly:

```
f1_macro   0.6821   0.6821   0.00e+00  OK
accuracy   0.6807   0.6807   0.00e+00  OK
```

So the audit itself succeeded and the run *is* reproducible. Only the write of
the report failed. **A correct console output is not a passing stage** — and
because `dvc repro` returned non-zero, the pipeline halted. Worth keeping in mind
generally.

**Fixed.** `bool()` cast on both `match` and `all_match`. Re-run:

```
dvc repro
```

The `reproduce` stage should now complete and write
`reports/reproducibility_audit.json`.

---

## 2. Your bootstrap disagrees with the documentation

Screenshot 7 shows:

```
vs run2_linearsvc     +0.0082  [-0.0004, +0.0161]  NOISE
vs run3_logreg_C0.5   +0.0048  [-0.0013, +0.0108]  NOISE
```

`docs/MODEL_SELECTION.md` claimed run2's difference was **MEANINGFUL** — that was
true on the reference environment, and is **not true on yours**. A report
contradicting its own evidence is a serious problem, so both documents are
corrected.

### What your result actually means

**All three runs are statistically indistinguishable.** No pairwise interval
excludes zero. The ordering is stable but the gaps are not real.

Look at how close run2 sits to the boundary: `[-0.0004, +0.0161]` misses zero by
0.0004. A different resampling seed would plausibly flip the verdict — and **a
conclusion that depends on the resampling seed is not a conclusion.**

### Why this makes your write-up stronger

The run2 rejection now rests entirely on the second argument, which was always
the better one:

> LinearSVC is, on the evidence, **as good a classifier as the baseline**. It was
> rejected anyway, because it has no `predict_proba` — which breaks the serving
> layer's low-confidence flag and blinds the M5 confidence drift signal.

**A modelling choice decided by a serving constraint.** That ordering is the
whole argument of the course: the best classifier that cannot be monitored is
worse than an equivalent classifier that can be.

Say this in the demo. It is the most defensible decision in the project, and it
is now supported by the numbers rather than merely alongside them.

---

## Unpack

```bash
cd /c/projects/mle-miniproject-2025paml567
unzip -o "/c/NP Documents and Tools/BITS/Mini Project/fixes.zip" -d .
```

Overwrites `src/training/reproduce.py`, `docs/MODEL_SELECTION.md`, and
`SUBMISSION.md`, and adds six renamed screenshots.

## Then re-run the pipeline to completion

```
dvc repro
```

Screenshot the final clean run as `docs/screenshots/dvc_repro_complete.png`.

## Screenshots included

| File | Shows |
|---|---|
| `dvc_repro_1_validate.png` | Stage 1: validation PASS with three BR findings |
| `dvc_repro_2_ingest.png` | Stage 2: ingestion audit trail, 27,481 -> 27,480 |
| `dvc_repro_3_features.png` | Stage 3: split before fit, baseline OOV 0.0909 |
| `dvc_repro_7_significance.png` | Bootstrap: all comparisons NOISE |
| `dvc_repro_8_reproduce.png` | Reproducibility audit: delta 0.00e+00 |
| `dvc_repro_6_train_clean.png` | Three runs with metrics and run ids |

Three of your nine captures were duplicates of adjacent scroll positions and are
not included.
