# ML-in-Finance-and-Complex-System

A research project for benchmarking LLMs on compositional financial reasoning. Synthetic financial spreadsheets are generated, expression trees are sampled and bound to them to produce multi-step Q&A pairs, and language models are evaluated on the resulting dataset — including under adversarial conditions (corrupted data, tricky phrasing, prompt injection).

## Project Structure

```
.
├── script/
│   ├── compiler_pipeline_adversarial/  # ACTIVE pipeline: synthetic data → expression trees → questions → obstacles
│   ├── benchmark/                      # LLM evaluation (local models, API models, adversarial eval)
│   ├── lora/                           # LoRA fine-tuning (SFT and chain-of-thought variants)
│   └── Multi-turn/                     # Decomposes compound questions into multi-turn chains
├── output/
│   ├── 10q/                            # 10-Q report generator + small Q&A dataset
│   ├── 90q/                            # 90-question benchmark dataset (mixed operators/depths)
│   ├── dataset_output/                 # Final merged Q&A dataset (used for eval)
│   └── output_llm/                     # LLM evaluation results (CSV + JSON)
├── parser/                             # Standalone CSV/PDF <-> JSON conversion utilities
├── src/                                # LoRA adapter checkpoints (qwen-lora-adapters/, qwen-lora-out/)
├── logs/                               # SLURM job stdout/stderr logs
├── temp/                               # Deprecated/superseded pipeline versions, scratch notebooks
├── batch.sh                            # SLURM submission script
└── requirements.txt
```

**What each piece does:**

- **`script/compiler_pipeline_adversarial/`** — the current, active pipeline. It generates a synthetic financial spreadsheet, samples expression trees over it (e.g. `growth(revenue, 2021, 2023) / avg(net_income, 2020, 2022)`), renders them into English questions with known ground-truth answers, and then applies two independent kinds of "obstacles": *data obstacles* (scaled/injected spreadsheet values) and *query obstacles* (tricky question phrasing). This is where new datasets are built.
- **`script/benchmark/`** — takes a generated dataset and runs it against LLMs, either loaded locally from disk or called through an API (Anthropic/OpenAI), scoring each answer against the numeric ground truth within a tolerance. Also has adversarial-specific evaluation and result-comparison/plotting tools.
- **`script/lora/`** — fine-tunes a local model with LoRA on the generated Q&A data, optionally generating chain-of-thought reasoning traces during training.
- **`script/Multi-turn/`** — takes a single compound question (built from a multi-node expression tree) and splits it into a sequence of simpler sub-questions, so a model can be evaluated turn-by-turn instead of solving the whole composition at once.
- **`output/`** — where generated datasets and evaluation artifacts land: `10q/` and `90q/` are two dataset "flavors" (quarterly-report-style vs. a fixed 90-question benchmark), `dataset_output/` holds the merged dataset actually used for evaluation, and `output_llm/` holds the raw eval results.
- **`parser/`** — standalone helpers to convert financial data between CSV/PDF/JSON; not wired into the active pipeline, useful for inspecting or reformatting a dataset by hand.
- **`src/`** — saved LoRA adapter weights from past fine-tuning runs.
- **`temp/`** — earlier versions of the pipeline (`compiler_pipeline/`, `compiler_pipeline_refactored/`) that were superseded by `compiler_pipeline_adversarial/`. Kept for reference only — don't build new work on them.

## Quick Start

### 0. Setup
```bash
pip install -r requirements.txt
```
Local models are loaded from `/cluster/scratch/$USER/models/` (e.g. `Qwen3.5-4B`, `Qwen3.5-9B`, `gemma-4-E4B-it`); LoRA checkpoints are written to `/cluster/scratch/$USER/lora-checkpoints/`.

### 1. Generate synthetic data + question variants
This is a two-step process: first build the underlying financial spreadsheet data (and any "data obstacle" variants of it), then sample expression trees against that data to produce English questions (with optional "query obstacles" applied to the wording). Both steps run from `script/compiler_pipeline_adversarial/` — see its [README_adversarial.md](script/compiler_pipeline_adversarial/README_adversarial.md) for the full obstacle catalog and combination examples.

```bash
cd script/compiler_pipeline_adversarial

# Step 1 — generate the spreadsheet data, in each data-obstacle variant:
#   baseline (clean), big_numbers (scaled up), multi_factor (extra multiplier),
#   prompt_injection (adversarial instructions embedded in cell values)
python data_obstacles.py

# Step 2 — sample expression trees over that data and render them into questions,
# in each query-obstacle variant: unit_scale_change, useless_info, conditional, negation.
# Also generates cell-level corruptions (missing_values, garbage, lookalike, cross_contaminated)
# for cells that are irrelevant to any question, to test robustness to noisy context.
python query_obstacles.py

# Data and query obstacles are independent axes and can be combined:
# generate a data variant first, then point query_obstacles.py at it.
python data_obstacles.py --obstacles big_numbers
python query_obstacles.py --data-dir output/data/big_numbers --seed 42
```

### 2. Build multi-turn chains
Turns one compound question into a sequence of simpler questions — one per node in its expression tree — so a model's ability to chain intermediate results can be evaluated step by step instead of all at once.
```bash
python script/Multi-turn/multi_turn_parser.py \
  --input script/compiler_pipeline_adversarial/output/questions/baseline/questions.json \
  --output output_multiturn.json \
  --sheet script/compiler_pipeline_adversarial/output/data/baseline/financial_spreadsheet.csv
```

### 3. Run LLM evaluation
Feed a generated dataset to one or more models and score their answers against the numeric ground truth (within `--tol` relative tolerance). Use local weights for open models, the API runner for hosted models, and the adversarial runner when testing against corrupted/injected data.
```bash
# Local models (loaded from /cluster/scratch/$USER/models/)
python script/benchmark/run_llm_eval.py --models Qwen3.5-4B Qwen3.5-9B --datasets 10q 90q mt --limit 20 --tol 0.1

# Same, but evaluating a fine-tuned LoRA adapter on top of the base model
python script/benchmark/run_llm_eval.py --models Qwen3.5-4B --finetune ./qwen-lora-adapters --limit 90

# API models (Anthropic/OpenAI) instead of local weights
python script/benchmark/run_llm_eval_api_call.py --limit 10

# Evaluate against adversarial data variants (e.g. prompt injection) via the API
python script/benchmark/run_llm_eval_api_call_adversarial.py --datasets baseline prompt_injection --limit 90
```
Results are written as JSON/CSV to `output/output_llm/`. Compare and visualize them with `script/benchmark/compare_results.py` or `script/benchmark/benchmark_visualization.ipynb`.

### 4. Fine-tune with LoRA
Trains a LoRA adapter on the generated Q&A pairs. `lora2.py` does plain supervised fine-tuning on question→answer pairs; `lora2_cot.py` additionally generates and trains on chain-of-thought reasoning traces (each capped at `--gen-tokens`, kept only if within `--tol` of the correct answer).
```bash
python script/lora/lora2.py --limit 200 --epochs 3
# Chain-of-thought variant:
python script/lora/lora2_cot.py --limit 200 --epochs 3 --gen-tokens 512 --tol 0.01
```
Adapters are saved to `/cluster/scratch/$USER/lora-checkpoints/`.

### On the cluster
`batch.sh` is a SLURM job template (1 GPU, 16GB gpumem, 12h wall time) that loads the required modules, activates the project venv, and runs one of the commands above. Edit the `MODEL` variable and the command at the bottom of the script to pick what to run, then submit with `sbatch batch.sh`. stdout/stderr for each job land in `logs/<job_id>.out` / `.err`.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `peft`/`trl` (LoRA fine-tuning), `anthropic`/`openai` (API eval), `pandas`/`numpy`/`scipy`, `pdfplumber`/`reportlab` (PDF parsing/generation in `parser/`), `matplotlib`/`plotly` (results visualization).
