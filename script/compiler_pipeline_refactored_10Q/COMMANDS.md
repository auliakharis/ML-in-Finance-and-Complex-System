# 10Q Eval Commands

Question-generation commands run from `compiler_pipeline_refactored_10Q/`.
Eval commands run from the repo root.

Note: per-depth goes only to depth 4. Depth 5+ has too few samples in the
generated dataset to reach 100 — the generator fails when constrained to a
single deep depth with only 50 synthetic reports.

---

## Step 1 — Generate questions

```bash
cd compiler_pipeline_refactored_10Q

python3 make_random_questions_10q.py \
  --depth-min 1 --depth-max 5 \
  --n-questions 2000 --seed 42 \
  --output output/random_questions_10q_d1-5.csv
```

Then split into per-depth files of 100:

```bash
python3 - <<'EOF'
import csv
from pathlib import Path
src = Path("output/random_questions_10q_d1-5.csv")
rows = list(csv.DictReader(open(src)))
fieldnames = list(rows[0].keys())
for d in [1, 2, 3, 4]:
    subset = [r for r in rows if int(r["depth"]) == d][:100]
    out = src.parent / f"random_questions_10q_d{d}_100.csv"
    w = csv.DictWriter(open(out, "w", newline=""), fieldnames=fieldnames)
    w.writeheader(); w.writerows(subset)
    print(f"d{d}: {len(subset)} -> {out.name}")
EOF
```

---

## Step 2 — Generate adversarial corruptions

```bash
python3 adversarial_10q.py \
  --atoms output/atoms_10q.json \
  --questions output/random_questions_10q_d1-5.csv \
  --output-dir output/adversarial \
  --seed 42
```

---

## Step 3 — Eval commands

Replace `<model>` with e.g. `Qwen/Qwen3.5-27B`.

### 10Q single-turn mixed depth 1-5
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```

### 10Q multi-turn mixed depth 1-5
```bash
python3 run_llm_eval_api_call.py \
  --datasets mt_10q \
  --models <model>
```

### Depth 1 only (100 samples)
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1_100.csv \
  --models <model>
```

### Depth 2 only (100 samples)
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d2_100.csv \
  --models <model>
```

### Depth 3 only (100 samples)
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d3_100.csv \
  --models <model>
```

### Depth 4 only (100 samples)
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d4_100.csv \
  --models <model>
```

### Missing values
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q_missing \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```

### Garbage values
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q_garbage \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```

### Look-alike substitution
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q_lookalike \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```

### Cross-sheet contamination
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q_cross \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```

### Scaling
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q_scaling \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```

### Useless info
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q_useless_info \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```

### Combined (atom corruptions + scaling + useless info all at once)
```bash
python3 run_llm_eval_api_call.py \
  --datasets 10q_combined_all \
  --questions-10q compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
  --models <model>
```
