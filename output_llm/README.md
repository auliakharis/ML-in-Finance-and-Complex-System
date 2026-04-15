# output_llm

LLM evaluation results produced by `run_llm_eval.py`. Each run creates a timestamped pair of CSV and JSON files.

## File Naming

```
eval_results_YYYYMMDD_HHMMSS.csv
eval_results_YYYYMMDD_HHMMSS.json
```

`eval_results.csv` / `eval_results.json` are symlinks or copies of the most recent run.

## Result Format

Each record contains:
- `question_id`, `question`, `ground_truth`
- `model` — model name (e.g., `Qwen3.5-4B`)
- `predicted` — model's extracted numeric answer
- `correct` — whether the prediction is within tolerance of ground truth
- `dataset` — `10q` or `90q`

## Evaluation Runs

| Timestamp | Notes |
|---|---|
| 2026-03-31 19:53 | First eval run |
| 2026-04-01 08:44 | Updated dataset |
| 2026-04-13 09:29 | Augmented dataset eval |

Models are loaded from `/cluster/scratch/$USER/models/`. Accuracy is computed as the fraction of answers within a relative tolerance (default ±1%) of the ground-truth numeric value.
