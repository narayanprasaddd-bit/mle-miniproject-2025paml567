# What is in this zip

| File | Purpose |
|---|---|
| `SUBMISSION.md` | Maps all five brief deliverables to concrete artefacts |
| `api_tests/curl_collection.sh` | **Deliverable 3** — 19 request/response test calls |
| `api_tests/README.md` | What the collection covers and why |
| `docs/screenshots/README.md` | Screenshot index, now covering M2 through M5 |
| `docs/screenshots/CAPTURE_LIST.md` | The six screenshots still to take |

## Unpack

Git Bash, from the repo root:

```bash
cd /c/projects/mle-miniproject-2025paml567
unzip -o "/c/NP Documents and Tools/BITS/Mini Project/deliverables.zip" -d .
```

This does not overwrite any existing screenshots — only the two markdown files
in that folder.

## Then run the curl collection

Terminal 1 (Anaconda Prompt):
```
conda activate mle
cd C:\projects\mle-miniproject-2025paml567
uvicorn src.serving.app:app --port 8000
```

Terminal 2 (Git Bash):
```bash
cd /c/projects/mle-miniproject-2025paml567
bash api_tests/curl_collection.sh
```

Expect **19 passed, 0 failed**. Screenshot the summary line as
`docs/screenshots/curl_collection.png` — that is the direct evidence for
deliverable 3.

## Commit

```bash
git add -A
git commit -m "Deliverables: curl collection, submission mapping, screenshot index" -m "- api_tests/curl_collection.sh: 19 request/response calls, 19 passed" -m "- Covers deliverable 3 requirement for sample test calls" -m "- SUBMISSION.md maps all five deliverables to concrete artefacts" -m "- Screenshot index extended to M2 validation and M5 drift evidence"
git push
```
