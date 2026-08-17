# Monitoring and Retraining Design

**Module reference:** M5 — Monitoring, Observability & Retraining
**Model under monitoring:** `sentiment-v1` (`run1_baseline_logreg`, id `ed72d76c`)
**Baseline reference:** 5,496 held-out texts, never trained on

---

## 1. The constraint that shapes everything

**In production the true labels are not available at inference time.** That is
the entire reason drift detection exists. If labels were available you would
compute accuracy directly and skip all of this.

Every signal below is therefore computed **without ground truth**. A monitor
that needs labels is not a monitor — it is an evaluation.

---

## 2. Why four signals, and why text is harder than tabular data

The course tutorial monitors a tabular feature by comparing its mean against a
training baseline: if `distance_km` shifts more than 1σ, investigate.

That has no direct analogue here — you cannot take the mean of a sentence. Four
proxies are used instead, each mapped to a technique M2 §2.5.3 prescribes:

| Signal | Detects | Test | Threshold |
|---|---|---|---|
| **OOV rate** | New words entering the traffic | Proportion | > 0.15 |
| **Token frequency** | The vocabulary *mix* shifting | Chi-squared (categorical) | p < 0.01 |
| **Confidence** | The model becoming less certain | KS (continuous) + effect floor | p < 0.01 **and** \|Δ\| ≥ 0.02 |
| **Document length** | Input *shape* changing | σ shift | > 1σ investigate, > 2σ halt |

**No single signal is sufficient**, and this is the reason the trigger requires
two:

- OOV rate rises for harmless reasons (a new product name) as well as harmful
  ones (a topic the model has never seen).
- Confidence can fall while accuracy holds, and hold while accuracy falls.
- Length can shift with no semantic change — a new client integration that pads
  its inputs.

---

## 3. Results — five windows

Reproduce with `python -m src.monitoring.drift --config configs/sentiment.yaml`.
Full output in `reports/drift_report.json`.

| Window | OOV | Chi² | Confidence | Length σ | Breached | Decision |
|---|---|---|---|---|---|---|
| **W0** control, no drift | 0.094 | 0.911 | 0.979 | 0.05 | **0 of 4** | NO_ACTION |
| **W1** vocabulary drift | **0.198** | **0.000** | **0.000** | 0.71 | 3 of 4 | **RETRAIN** |
| **W2** topic drift | **0.277** | **0.000** | **0.000** | 0.38 | 3 of 4 | **RETRAIN** |
| **W3** format drift | 0.090 | **0.004** | 0.003 † | **7.51** | 2 of 4 | **RETRAIN** |
| **W4** drift, small sample | **0.272** | **0.000** | **0.001** | 0.41 | 3 of 4 | **HOLD** |

† significant but below the effect floor — see §5.

### W0 is the primary result, not a footnote

**A monitor that always fires is worse than no monitor.** It trains the operator
to ignore it, and then the real alert is ignored too.

W0 is a random resample of the same held-out pool used as the baseline, so it is
undrifted **by construction**. Any signal firing on it would be a false positive
and a threshold defect. Zero of four fired. Demonstrating silence on clean
traffic is what makes the alerts on W1–W4 meaningful.

### W3 isolates one signal

Format drift concatenates five existing texts into one request — simulating a
client integration that batches messages. Vocabulary is **unchanged**, so OOV
correctly stays quiet at 0.090 while length explodes to 7.51σ. The signals are
discriminating between *kinds* of drift, not just detecting "something changed".

### W4 separates detection from action

Same drift as W2, but only 120 samples. Three signals breach — the drift is
real — yet the decision is **HOLD**, because retraining on 120 examples would
fit the noise in the drift rather than correct for it.

**Detecting drift and being able to act on it are different problems.** A
trigger that fires without checking data sufficiency produces a worse model than
the one it replaces.

---

## 4. The retraining trigger

```
IF   signals_breached >= 2
AND  new_samples >= 500
THEN RETRAIN
```

| Decision | Condition | Action |
|---|---|---|
| `NO_ACTION` | 0 signals | Continue |
| `MONITOR` | 1 signal | Observe. One signal is noise. |
| `HOLD` | ≥2 signals, <500 samples | Alert, accumulate data, do not retrain |
| `RETRAIN` | ≥2 signals, ≥500 samples | Retrain, evaluate, gate promotion |

### Why two signals rather than one

One signal is noise; two is corroboration. With four largely independent
signals at roughly a 1% individual false-positive rate, requiring two
simultaneous breaches drops the compound false-alarm rate by roughly two orders
of magnitude — at the cost of missing drift that expresses through a single
channel.

That trade is deliberate. **A false retrain is more expensive than a delayed
one:** it consumes compute, requires re-validation and re-approval, and risks
promoting a model fitted to a transient anomaly. A delayed retrain costs some
accuracy for a further monitoring window.

### Why 500 samples

Below roughly 500 examples the retrained model would be fitting the drift's
sampling noise. 500 is a judgement, not a derived constant — the honest
justification is that it is large enough for a stratified split to retain a
usable minority-class count, and it is recorded here as tunable rather than
optimal.

### Retraining is not automatic

The trigger **raises a decision, it does not deploy a model.** The path after
`RETRAIN` is:

1. Retrain on baseline + accumulated drifted data
2. Evaluate against the frozen held-out set **and** the drifted window
3. Bootstrap-compare against the incumbent (as in Sprint 2 — a higher score is
   not automatically a better model)
4. Register a new version to `Staging`
5. **Named human approves promotion to Production**

Step 5 is deliberate. M6 §governance: a model that can promote itself has no
accountable owner.

---

## 5. A finding — statistical vs practical significance

The confidence signal originally fired on `p < 0.01` alone. On W3 that produced:

```
p = 0.0025  (significant)      mean confidence 0.631 -> 0.622
```

A 0.009 shift in mean confidence is operationally meaningless. But with n=600
against a 5,496-row baseline, the KS test has ample power to call it
significant. In production, where windows are larger, the test is *more*
sensitive and this would generate steady false alarms — precisely the
alert-fatigue failure W0 is designed to guard against.

**The fix:** the signal now requires both `p < 0.01` **and**
`|Δ mean confidence| ≥ 0.02`. W3's confidence signal is now reported as
observed-but-not-actioned, and W3 correctly drops from 3 breaches to 2.

**The same lesson appeared in Sprint 2**, where a 0.002 f1_macro difference
between two runs was statistically detectable and operationally irrelevant.
There the remedy was a bootstrap confidence interval; here it is an effect-size
floor. Same underlying error — treating a p-value as a decision — caught twice
in two different modules.

---

## 6. Prediction logging

Every prediction is written to SQLite by the serving layer
(`src/serving/app.py`) with: timestamp, raw text, text length, predicted class,
confidence, low-confidence flag, model version, and latency.

**Raw text is stored, not the feature vector.** The monitor needs original
tokens to compute OOV rate and token-frequency drift, and a stored vector cannot
be un-vectorised.

Score live traffic instead of the simulation with:

```bash
python -m src.monitoring.drift --config configs/sentiment.yaml --from-log
```

`GET /metrics` gives the serving-side view: volume, mean latency, mean
confidence, and the **low-confidence rate** — the cheapest early signal, since
it needs no baseline comparison at all.

---

## 7. Limitations, stated plainly

- **The drift is synthetic.** Real drift needs production traffic collected over
  months. Constructed drift is cleaner and more abrupt than the real thing; real
  drift is usually gradual, and gradual shifts are harder to detect than these
  windows suggest. **These windows test the detector's wiring and thresholds —
  they are not a claim about real-world sensitivity.**
- **No label feedback loop.** Retraining assumes labels arrive for drifted data
  from somewhere. In practice that means human annotation, and the annotation
  budget — not the trigger — usually sets the retraining cadence.
- **Thresholds are not empirically tuned.** 0.15 OOV, 0.02 effect size, 2
  signals, 500 samples are reasoned choices, not values fitted against observed
  false-positive rates. Tuning them requires production traffic.
- **No temporal windowing.** The monitor compares one window against a fixed
  baseline. A production system would use rolling windows and track the
  trajectory, since the *rate* of drift matters as much as its magnitude.
- **The baseline is fixed.** After a retrain the baseline should be regenerated
  from the new training set. That is a manual step here.
