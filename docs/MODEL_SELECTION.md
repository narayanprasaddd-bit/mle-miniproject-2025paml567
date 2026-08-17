# Model Selection Record

**Project:** Mini-Project I — Flavor C, Text Classification
**Module reference:** M3 — Experimentation, Versioning & Reproducibility
**Primary metric:** `f1_macro` (3 classes, mild imbalance)
**Data version:** `v1.0-raw` — 27,481 rows validated, 27,480 after cleaning
**Code version:** git commit `cfb4809`, branch `main`
**Splits:** train 21,984 / test 5,496, stratified, `random_state=42`

---

## 1. Decision

**Selected: `run1_baseline_logreg`** — MLflow run, git commit `cfb4809` — TF-IDF (1–2 grams, 20,000 features) with
Logistic Regression, `C=1.0`.

The baseline won. That is a legitimate outcome and is recorded here as such
rather than being papered over with a more elaborate model that did not earn
its place.

---

## 2. Results

| Run | Varied from baseline | f1_macro | Accuracy |
|---|---|---|---|
| **run1_baseline_logreg** | — (baseline) | **0.6821** | 0.6807 |
| run2_linearsvc | algorithm → hinge loss | 0.6739 | 0.6705 |
| run3_logreg_C0.5 | regularisation `C` 1.0 → 0.5 | 0.6773 | 0.6776 |

Each run changed **exactly one variable** from the baseline. Where two things
change and the score moves, the experiment cannot attribute the movement — so
it tells you nothing.

Full metric set for the selected run:

| Metric | Value |
|---|---|
| f1_macro *(primary)* | 0.6821 |
| f1_weighted | 0.6815 |
| accuracy | 0.6807 |
| precision_macro | 0.7019 |
| recall_macro | 0.6725 |

Precision exceeding recall by ~3 points means the model is conservative: when it
commits to a class it is usually right, but it under-detects. For a triage
system that is the preferable direction — a missed routing is recoverable, a
confidently wrong one is not.

---

## 3. Is the difference real? — bootstrap testing

A test set is one sample. A different 20% split gives different numbers.
Declaring a winner on a 0.002 margin is choosing noise, so the comparison was
tested rather than assumed.

Method: 1,000 paired bootstrap resamples of the test set. Both models are
scored on the **same** resampled rows each time, so per-sample difficulty
cancels and only the model difference remains.

| Run | 95% CI on f1_macro |
|---|---|
| run1_baseline_logreg | 0.6808 [0.6689, 0.6925] |
| run2_linearsvc | 0.6714 [0.6589, 0.6841] |
| run3_logreg_C0.5 | 0.6787 [0.6660, 0.6903] |

Paired differences against the baseline:

| Comparison | Difference | 95% CI | Verdict |
|---|---|---|---|
| run1 − run2_linearsvc | +0.0094 | [+0.0010, +0.0176] | **Meaningful** — interval excludes zero |
| run1 − run3_logreg_C0.5 | +0.0020 | [−0.0040, +0.0078] | **Noise** — interval spans zero |

Reproduce with `python -m src.training.significance --config configs/sentiment.yaml`.
Full output in `reports/significance_test.json`.

---

## 4. Why not run 2 — LinearSVC

**Hypothesis before running.** LinearSVC optimises a hinge loss and a maximum
margin rather than likelihood. On high-dimensional sparse text this often edges
out logistic regression, so a small gain was expected.

**Result.** It lost, and the loss is real: the paired 95% CI is
[+0.0010, +0.0176] and excludes zero. This is not a coin-flip.

**Two independent reasons to reject it:**

1. **It scored worse**, and the gap survives resampling.
2. **It cannot produce probabilities.** `LinearSVC` has no `predict_proba`.
   That is disqualifying regardless of score, because two downstream
   components depend on calibrated confidence:
   - the serving layer flags low-confidence predictions below 0.50 for review
     (`configs/sentiment.yaml → serving.low_confidence_threshold`);
   - the M5 drift monitor runs a KS test on the `predict_proba` distribution
     to detect widening uncertainty.

   Recovering probabilities would require Platt scaling — an extra calibration
   layer, extra artefact, and extra failure surface — to reach a score that is
   already lower.

**A note on the second reason.** Had LinearSVC *won* on score, this would have
been the interesting decision: a better classifier that breaks two system
requirements. The system requirement would still have taken precedence, and
that reasoning would have been recorded here. It happened not to be tested,
because the score went the other way.

---

## 5. Why not run 3 — stronger regularisation

**Hypothesis before running.** 20,000 features against 21,984 training rows is
a wide problem, so halving `C` (doubling the penalty) might generalise better.

**Result.** −0.0020, with a 95% CI of [−0.0040, +0.0078] that spans zero. The
two models are **statistically indistinguishable** on this data.

**So the choice between them cannot be made on score.** Deciding by the fourth
decimal place would be selecting noise. Secondary criteria apply:

- **Simplicity.** `C=1.0` is scikit-learn's default. A non-default value must
  earn its place, and this one did not.
- **Interpretability of the record.** "We used the default" is a cleaner and
  more honest line in a governance document than "we used C=0.5", which invites
  a question the data cannot answer.

**The finding is more useful than the winner.** A flat response to a doubled
regularisation penalty says the baseline was **not overfitting**. If it had
been, tightening the penalty would have improved held-out performance
materially. It did not, which independently confirms that 20,000 TF-IDF
features are not too many for 21,984 rows — and rules out regularisation
tuning as a productive direction for further work.

---

## 6. Why no transformer fine-tune

A DistilBERT fine-tune was scoped and **deliberately not run**. Recorded as an
engineering judgement, not an omission:

- **Marginal accuracy did not justify the cost.** Published results put
  fine-tuned transformers around 5–8 points above TF-IDF baselines on short
  social text. Against a rubric where model accuracy carries **zero weight**,
  that buys nothing.
- **Serving latency.** A transformer raises per-request inference from
  sub-millisecond to tens of milliseconds on CPU, and multiplies the container
  image size.
- **Reproducibility cost.** GPU training introduces non-determinism from
  cuDNN kernel selection and atomic accumulation ordering. The reproducibility
  audit in §8 — exact agreement to four decimals — would not have been
  achievable.
- **Schedule risk.** A fine-tuning loop on a CPU-only Windows laptop inside a
  compressed timeline puts hours of compute on the critical path in exchange
  for no rubric marks.

The relevant precedent is in the coursework itself: on IMDB, TF-IDF with
logistic regression reaches roughly 88% — matching a from-scratch LSTM while
training in seconds. Deep learning wins when data is large, the signal is
compositional, or pre-training is available. Here the first two do not hold and
the third was rejected on the grounds above.

---

## 7. Is 0.68 a good score?

**In absolute terms it is moderate, and that is expected for this corpus.**

- Three classes, so random guessing scores ≈ 0.33. The model roughly doubles it.
- The `neutral` class is intrinsically ambiguous. Human annotators disagree
  substantially on whether a flat statement is neutral or mildly negative, so
  the label itself carries irreducible noise.
- Tweets are short — median 64 characters — giving few tokens of evidence per
  example.
- Published results on this dataset cluster in the 0.68–0.72 range for
  classical approaches, so this sits inside the expected band.

The ceiling here is set by the data, not the algorithm. The productive next
step would be better labels or more context, not a better classifier.

---

## 8. Reproducibility audit

The selected run was rebuilt **from its logged MLflow parameters alone** — no
access to the original script's in-memory objects — retrained, and re-scored.

| Metric | Logged | Recomputed | Delta |
|---|---|---|---|
| f1_macro | 0.6821 | 0.6821 | 0.00e+00 |
| accuracy | 0.6807 | 0.6807 | 0.00e+00 |

**Exact agreement.** The logged configuration is sufficient to rebuild the run.

Reproduce with `python -m src.training.reproduce --config configs/sentiment.yaml`.
Output in `reports/reproducibility_audit.json`.

Against the five sources of irreproducibility in M3 §3.2:

| Source | Control |
|---|---|
| Random seeds | `random_state=42`, logged as a parameter on every run |
| Environment drift | Pinned in `environment.yml`, re-exported after the DVC/pathspec incident |
| Data version drift | Git tag `v1.0-raw`, DVC pointer files, logged as a run tag |
| Code state | Git commit hash logged as a run tag |
| Hardware / parallelism | Not applicable — single-threaded CPU, no GPU |

---

## 9. What is registered

```
Model     : sentiment-classifier
Version   : 1
Stage     : Staging
Run ID    : see reports/model_comparison.json → selected_run_id
Artefacts : model + feature_bundle.joblib  (inseparable — see §10)
Approver  : Narayan Prasad (2025paml567)
```

Promotion to Production is deliberately **not** automatic. It requires the M4
contract tests to pass and the M5 monitoring baseline to be established.

---

## 10. The artefact is not the model

The registered artefact is **the model *and* the fitted TF-IDF vectorizer**,
frozen together.

A TF-IDF vector is meaningless without the vocabulary that produced it. Token
index 4,182 means one specific word only in the vocabulary that assigned it.
Re-fit the vectorizer on different text and index 4,182 becomes a different
word — while the model's learned coefficient for that position continues to be
applied, now to the wrong feature.

The failure is silent. No exception, no error, HTTP 200 throughout, and every
prediction subtly wrong.

Hence `src/features/text_features.py` exposes `transform()` and **no `fit()`**:
the serving path is structurally incapable of re-fitting. `tests/test_features.py`
asserts that property rather than trusting it.
