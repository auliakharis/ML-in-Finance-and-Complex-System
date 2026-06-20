# LoRA Fine-tuning with Chain-of-Thought

## Overview

`lora2_cot.py` fine-tunes **Qwen3.5-4B** on the financial QA dataset using LoRA.
It first generates verified Chain-of-Thought (CoT) traces (Phase 1), then trains on them (Phase 4).

Adapters are saved to `/cluster/scratch/$USER/lora-cot-checkpoints/adapters`.

---

## Running on the Cluster (SLURM)

Edit `batch.sh` in the repo root to point to `lora2_cot.py`, then submit:

```bash
sbatch batch.sh
```

Logs land in `logs/<jobid>.out`. Monitor progress:

```bash
tail -f logs/<jobid>.out
```

---

## Running Locally

Activate your venv first, then:

```bash
# Full run — 3 epochs, all ~1000 examples
python lora/lora2_cot.py

# Smoke test — 20 examples, 1 epoch
python lora/lora2_cot.py --limit 20 --epochs 1

# Adjust CoT generation budget or tolerance
python lora/lora2_cot.py --epochs 3 --gen-tokens 512 --tol 0.01
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `3` | Number of training epochs |
| `--limit` | all | Cap total training examples (smoke-test mode) |
| `--gen-tokens` | `512` | Max new tokens for CoT generation per example |
| `--tol` | `0.01` | Relative tolerance for accepting a CoT trace as correct (1%) |

---

## Output

After training completes, adapters are saved to:

```
/cluster/scratch/$USER/lora-cot-checkpoints/adapters/
```

---

## Running Benchmarks After Training

Use `run_llm_eval.py` from the repo root with `--finetune` pointing at the saved adapters.

```bash
# Evaluate fine-tuned model on all datasets (10q, 90q, multi-turn)
python run_llm_eval.py \
  --models Qwen3.5-4B \
  --finetune /cluster/scratch/$USER/lora-cot-checkpoints/adapters

# 90q only, limit to 90 questions
python run_llm_eval.py \
  --models Qwen3.5-4B \
  --datasets 90q \
  --finetune /cluster/scratch/$USER/lora-cot-checkpoints/adapters \
  --limit 90

# Compare base vs fine-tuned: run twice, once without and once with --finetune
python run_llm_eval.py --models Qwen3.5-4B --datasets 90q --output output_llm/base.json
python run_llm_eval.py --models Qwen3.5-4B --datasets 90q \
  --finetune /cluster/scratch/$USER/lora-cot-checkpoints/adapters \
  --output output_llm/finetuned.json
```

### Benchmark Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | all | Model name(s) under `$SCRATCH/models/` |
| `--finetune` | none | Path to LoRA adapter directory |
| `--datasets` | all | Which datasets: `10q`, `90q`, `mt` (multi-turn) |
| `--limit` | all | Max questions per dataset per model |
| `--tol` | `0.1` | Relative tolerance for numeric correctness |
| `--output` | `output_llm/eval_results.json` | Path to save per-question results JSON |
| `--csv` | `output_llm/eval_results.csv` | Path to save per-question results CSV |
| `--thinking-mode` | off | Enable extended thinking (saves `<think>` traces) |
| `--thinking-budget` | `1024` | Max tokens inside `<think>` (only with `--thinking-mode`) |

### Visualising Results

Open `benchmark_visualization.ipynb` in the repo root to plot accuracy comparisons across models and datasets.
