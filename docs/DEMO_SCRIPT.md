# Demo Script — 6 minutes

**Deliverable 5:** *"a 5–7 minute recorded/live demo or presentation."*

Target **6:00**, which leaves margin at both ends. Recording with OBS or the
Windows Game Bar (`Win+G`) is fine; audio matters more than video quality.

---

## Before you record

```
conda activate mle
cd C:\projects\mle-miniproject-2025paml567
```

Open and arrange in advance, so no time is lost switching:

1. **Terminal A** — Anaconda Prompt, in the project directory
2. **Terminal B** — Anaconda Prompt, ready to start uvicorn
3. **Browser tab 1** — the GitHub repository
4. **Browser tab 2** — http://127.0.0.1:8000/docs (start the service first)

Start the service in Terminal B **before recording**:
```
uvicorn src.serving.app:app --port 8000
```

---

## 0:00–0:40 · What this is

> "This is Flavor C — a sentiment classifier for short text, built as a
> production ML system rather than a notebook.
>
> The thing worth saying up front is that the model itself is the *smallest*
> part. It's TF-IDF and logistic regression, and it scores 0.68 macro-F1. Every
> interesting decision in the project is about what surrounds it — how the data
> is validated, how experiments are compared, how the model is served, and how
> I'd know if it stopped working."

*Show the README architecture diagram.*

---

## 0:40–1:50 · M2 — the data, and what validation found

> "The dataset archive I was given contained four files from two different
> Kaggle datasets. One of them I rejected."

*Terminal A:*
```
python -m src.validation.validate --config configs/sentiment.yaml
```

> "Four validation levels — schema, range, statistical, business rule. This run
> passes with three findings.
>
> The rejected corpus is the interesting one. A 1.6-million-row Sentiment140
> export that was truncated at 1,048,572 rows — that's the Excel worksheet limit,
> two to the twentieth. Someone had opened it in a spreadsheet and saved it. The
> positive class went from 800,000 rows to 248,000.
>
> The important thing is that **the file loads fine and passes schema
> validation**. Every column has the right type, every value is in range. Only
> the Level 3 distribution check caught it.
>
> Second finding: this `Country` column repeats on a fixed 195-row cycle — I
> verified it across 585 consecutive positions. It was appended by row position,
> not joined by key, so it carries zero information about the tweet next to it.
> Six columns excluded because of that."

*Then the negative test:*
```
python -m src.validation.validate --config configs/sentiment.yaml --input data/raw/_corrupt_test.csv
```

> "And this is the same gate with three defects injected. Five blocking failures,
> exit code 1, pipeline halted. That failure is the system working — a validation
> suite that's only ever run on clean data proves nothing was detected, not that
> detection works."

---

## 1:50–3:00 · M3 — experiments, and why the winner isn't the highest score

*Terminal A:*
```
python -m src.training.significance --config configs/sentiment.yaml
```

> "Three tracked runs in MLflow, one variable changed each time, and the
> hypothesis recorded before each run rather than after.
>
> The baseline won on raw score. But I didn't stop there — I ran a paired
> bootstrap, a thousand resamples of the test set. And the result is that **none
> of the three runs is statistically distinguishable from the others.** Every
> confidence interval spans zero.
>
> So the score can't make the decision. Look at LinearSVC in particular: on the
> evidence it's as good a classifier as the baseline. I rejected it anyway,
> because it has no `predict_proba`.
>
> That matters for two reasons downstream — the API flags low-confidence
> predictions for human review, and my drift monitor runs a KS test on the
> confidence distribution. Choosing LinearSVC would have blinded both.
>
> **So a serving constraint decided a modelling choice.** The best classifier
> that can't be monitored is worse than an equivalent one that can be."

*Optionally show `docs/MODEL_SELECTION.md` §4.*

---

## 3:00–4:15 · M4 — the service

*Browser tab 2, the `/docs` page.*

> "FastAPI, with the model and the vectorizer loaded once at startup."

*Expand `/health`, Execute.*

> "Health reports **both** artefacts separately, and that's deliberate. The
> deployed artefact isn't the model — it's the model *and* the fitted vectorizer,
> frozen together. A TF-IDF vector is meaningless without the vocabulary that
> produced it. Load a mismatched pair and every prediction is silently wrong,
> HTTP 200 throughout.
>
> So my feature module exposes `transform()` and deliberately no `fit()`. The
> serving path is structurally incapable of re-fitting the vectorizer."

*`/predict/sentiment` with `{"text": ""}`, Execute.*

> "Malformed input returns 422 with the failing field named — not 500. That
> distinction matters in production: 422 means the client sent something invalid
> and the service is fine; 500 means the service broke. Conflating them sends an
> engineer looking for a bug that doesn't exist."

*Terminal A:*
```
bash api_tests/curl_collection.sh
```
*(or just show the saved screenshot if timing is tight)*

> "Nineteen request/response calls, all passing. Measured latency: p50 is
> 0.56 milliseconds, about 1,400 requests a second per worker. Artefact load is
> 396 milliseconds — that's 700 times the inference time, which is exactly why
> it happens once at startup and not per request.
>
> The container image builds in GitHub Actions rather than locally, because this
> is a managed laptop without admin rights and Docker Desktop needs an elevated
> installer. The CI log is arguably better evidence anyway — it proves the
> Dockerfile works on a clean machine."

---

## 4:15–5:30 · M5 — monitoring and drift

*Terminal A:*
```
python -m src.monitoring.drift --config configs/sentiment.yaml
```

> "Four drift signals. All of them computed **without labels** — in production
> the true labels aren't available at inference time, which is the whole reason
> drift detection exists.
>
> Five windows. **Start with W0**, because it's the most important one. That's a
> resample of the baseline — undrifted by construction — and it raises zero
> signals. A monitor that fires on clean traffic is worse than no monitor,
> because it trains you to ignore the alerts, and then you ignore the real one.
> Demonstrating silence is what makes the other alerts mean anything.
>
> W1 is new slang, W2 is a different topic — both fire three of four and trigger
> a retrain.
>
> W3 is the one I'd point at. It concatenates existing texts, so the length
> signal goes to 7.5 sigma while **OOV stays quiet** — the vocabulary didn't
> change. The signals discriminate between *kinds* of drift, not just 'something
> changed.'
>
> Also on W3 — see this confidence line? Statistically significant at p equals
> 0.0012, but the mean only moved 0.009. That's meaningless operationally, so I
> added an effect-size floor and it's reported but not actioned. Same mistake I'd
> caught in M3 with the bootstrap: treating a p-value as a decision. Two
> different modules, one recurring failure mode.
>
> And W4 — three signals breached, but only 120 samples, so the decision is HOLD,
> not retrain. Detecting drift and being able to act on it are different
> problems. Retraining on 120 examples would fit the drift's noise instead of
> correcting for it."

---

## 5:30–6:00 · Close

> "Fifty-seven tests. Seven-stage DVC pipeline, reproducible with one command.
> The selected run rebuilds from its logged configuration to zero difference in
> the fourth decimal.
>
> The honest limitations are in the README: the drift is synthetic, the model
> over-predicts the majority class at 49% against a true rate of 40%, and the
> thresholds are reasoned rather than tuned against real production data.
>
> Retraining raises a decision — it doesn't deploy. Promotion to production needs
> a named human approval. A model that can promote itself has no accountable
> owner."

---

## Timing discipline

| Section | Budget | Cut first if over |
|---|---|---|
| Intro | 0:40 | Trim to 0:25 |
| M2 validation | 1:10 | Drop the negative test, show the screenshot |
| M3 experiments | 1:10 | **Do not cut** — this is the strongest section |
| M4 serving | 1:15 | Skip the curl run, show the screenshot |
| M5 drift | 1:15 | **Do not cut W0 or W4** |
| Close | 0:30 | Trim to 0:15 |

If you overrun, cut demonstrations rather than reasoning. A grader can see that
the code runs from the repository; what they can't get anywhere else is *why*
each decision was made.

## The three sentences that matter most

If everything else goes wrong, land these:

1. **"A serving constraint decided a modelling choice."** (M3/M4)
2. **"A monitor that fires on clean traffic is worse than no monitor."** (M5)
3. **"The file loads fine and passes schema validation — only the distribution
   check caught it."** (M2)

## Practical notes

- Record in one take if you can. Editing costs more time than a small stumble.
- Say the numbers out loud — 0.68, 0.56 milliseconds, 0 of 4. Reading them off
  the screen silently wastes the audio channel.
- If a command fails live, say what it should have shown and move on. Do not
  debug on camera.
- Save as `demo.mp4`. If it exceeds GitHub's 100 MB limit, upload unlisted to
  YouTube or Drive and put the link in the README.
