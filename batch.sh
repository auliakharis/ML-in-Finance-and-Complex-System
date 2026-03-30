#!/bin/bash
#SBATCH --job-name=eu_reg_pipeline
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ─── EDIT THESE ───

MODEL="Qwen3.5-9B"
# ──────────────────

mkdir -p logs

# Load modules (adjust to your cluster)
module load python/3.13.0
module load cuda/13.0.2

# Install deps
# pip install torch accelerate bitsandbytes transformers

source /cluster/home/arakhmasari/LLM-as-a-Judge-in-Finance/venv/bin/activate


MODEL_PATH="$SCRATCH/models/$MODEL"
echo "Checking model path..."
ls "$MODEL_PATH" || echo "ERROR: Model path not found: $MODEL_PATH"

echo "Starting pipeline: $(date)"
echo "Model: $MODEL"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi


python run_llm_eval.py


echo "Done: $(date)"