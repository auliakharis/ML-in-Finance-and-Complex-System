#!/bin/bash
#SBATCH --job-name=ml_finance_eval
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ─── EDIT THESE ───

MODEL="gemma-4-E4B-it"
# ──────────────────

mkdir -p logs

# Load modules (adjust to your cluster)
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load cuda/13.0.2

source /cluster/home/lturgut/ML-in-Finance-and-Complex-System/.venv/bin/activate


MODEL_PATH="$SCRATCH/models/$MODEL"
echo "Checking model path..."
ls "$MODEL_PATH" || echo "ERROR: Model path not found: $MODEL_PATH"

echo "Starting pipeline: $(date)"
echo "Model: $MODEL"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi


# python run_llm_eval.py --datasets mt  --models gemma-4-E4B-it --limit 5
python run_llm_eval.py --datasets mt annual sec10Q  --models gemma-4-E4B-it Qwen3.5-4B Qwen3.5-9B --limit 90


echo "Done: $(date)"