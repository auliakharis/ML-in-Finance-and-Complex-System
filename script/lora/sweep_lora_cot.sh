#!/usr/bin/env bash
#SBATCH --job-name=lora_cot_sweep
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
SCRIPT="script/lora/lora2_cot.py"
BENCHMARK="script/benchmark/run_llm_eval.py"
COT_CACHE="/cluster/scratch/${USER}/lora-cot-checkpoints/cot_traces_cache.json"
CHECKPOINTS_DIR="/cluster/scratch/${USER}/lora-cot-checkpoints"

# Optional: pass --limit N and --epochs N as env vars or args
LIMIT="${LIMIT:-}"
EPOCHS="${EPOCHS:-3}"
BENCH_LIMIT="${BENCH_LIMIT:-90}"

# Load modules (adjust to your cluster)
module load python/3.13.0
module load cuda/13.0.2

# Install deps
# pip install torch accelerate bitsandbytes
# pip install git+https://github.com/huggingface/transformers.git

source /cluster/home/arakhmasari/LLM-as-a-Judge-in-Finance/venv/bin/activate


LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

for r in 8 16; do
    for alpha in 4 8 16; do
        for dropout in 0.10 0.15; do
            echo ">>> r=$r alpha=$alpha dropout=$dropout"
            python "$SCRIPT" \
                --r "$r" \
                --lora-alpha "$alpha" \
                --lora-dropout "$dropout" \
                --cot-cache "$COT_CACHE" \
                --epochs "$EPOCHS" \
                $LIMIT_ARG

            # RUN_TAG="r${r}_alpha${alpha}_dropout${dropout}"
            # ADAPTER_PATH="${CHECKPOINTS_DIR}/${RUN_TAG}/adapters"
            # echo ">>> Benchmarking $RUN_TAG ..."
            # python -u "$BENCHMARK" \
            #     --models Qwen3.5-4B \
            #     --datasets 90q \
            #     --finetune "$ADAPTER_PATH" \
            #     --limit "$BENCH_LIMIT" \
            #     --output "output_llm/${RUN_TAG}_eval_results.json" \
            #     --csv    "output_llm/${RUN_TAG}_eval_results.csv"
        done
    done
done