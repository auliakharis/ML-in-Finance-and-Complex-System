# ML-in-Finance-and-Complex-System

A research project for benchmarking LLMs on compositional financial reasoning. The pipeline generates synthetic financial data, constructs expression trees to form multi-step Q&A pairs, and evaluates language models on the resulting dataset.

## Project Structure

```
.
├── compiler_pipeline/   # Core pipeline: synthetic data → expression trees → questions
├── form10q/                 # 10-Q report generator and small Q&A dataset
├── 90q/                 # 90-question benchmark dataset (mixed operators/depths)
├── Multi-turn/          # Multi-turn conversation builder from expression trees
├── dataset_output/      # Final merged Q&A dataset (used for eval)
├── output/              # Output from the form10q pipeline run
├── output_llm/          # LLM evaluation results (CSV + JSON)
├── logs/                # SLURM job stdout/stderr logs
└── TEMP/                # Scratch/experimental files
```

## Quick Start

### 1. Generate synthetic data and questions
```bash
cd compiler_pipeline
bash run_pipeline.sh
```

### 2. Run LLM evaluation
```bash
python run_llm_eval.py
# Options:
#   --limit 20        evaluate on first N questions
#   --tol 0.1         relative tolerance for numeric answers (default 0.01)
#   --models Qwen3.5-4B
#   --output results.json
```

### 3. Build multi-turn chains
```bash
cd Multi-turn
python multi_turn_parser.py --input ../annual/random_questions_90.json --output output.json
```

## Requirements

```bash
pip install -r requirements.txt
```

Models are loaded from `/cluster/scratch/$USER/models/`.
