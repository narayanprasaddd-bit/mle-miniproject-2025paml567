# Screenshot fix

Your three files had their names swapped relative to their contents, and one
contained the Snipping Tool video-recorder bar. Corrected versions are in
docs/screenshots/.

| Correct filename | What it shows |
|---|---|
| mlflow_runs_table.png | All three runs with accuracy and f1_macro |
| mlflow_run_detail.png | run1 parameters, metrics, and tags |
| mlflow_model_registry.png | sentiment-classifier v1 with the staging alias |

## Unpack

Git Bash, from the repo root:

    unzip -o "/path/to/sprint2_screenshots.zip" -d .

This overwrites docs/MODEL_SELECTION.md with your real numbers and drops the
three cleaned screenshots into docs/screenshots/.

## Still worth capturing

One screenshot is missing: the MODEL VERSION page. In the registry list, click
"Version 1" - that page carries the four governance tags (approver,
data_version, selection_record, promotion_gate). The list view proves a model
was registered; the version page proves who approved it and under what
conditions it may reach Production.

Save it as docs/screenshots/mlflow_model_version_tags.png
