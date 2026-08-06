#!/bin/bash
#SBATCH --job-name=lora_retrain
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

module load python/3.13.0
module load cuda/13.0.2

source /cluster/home/arakhmasari/LLM-as-a-Judge-in-Finance/venv/bin/activate

echo "Starting LoRA retraining: $(date)"
nvidia-smi

# ── LoRA configurations ──────────────────────────────────────────────
NAMES=(
  r8_alpha16_dropout0.1
  r8_alpha4_dropout0.1
  r8_alpha8_dropout0.1
  r4_alpha4_dropout0.1
  r8_alpha16_dropout0.15
  r8_alpha4_dropout0.15
  r8_alpha8_dropout0.15
)
RS=(    8   8   8   4   8   8   8)
ALPHAS=(16  4   8   4  16   4   8)
DROPS=( 0.1 0.1 0.1 0.1 0.15 0.15 0.15)

COT_CACHE="/cluster/scratch/arakhmasari/lora-cot-checkpoints/cot_cache.json"

# ── Phase 1: Train ───────────────────────────────────────────────────
for i in "${!NAMES[@]}"; do
  NAME="${NAMES[$i]}"
  R="${RS[$i]}"
  ALPHA="${ALPHAS[$i]}"
  DROP="${DROPS[$i]}"
  echo ""
  echo "══════════════════════════════════════════════════"
  echo " TRAIN [$((i+1))/${#NAMES[@]}]: $NAME"
  echo "══════════════════════════════════════════════════"
  python -u script/lora/lora2_cot.py \
    --r "$R" --lora-alpha "$ALPHA" --lora-dropout "$DROP" \
    --cot-cache "$COT_CACHE"
done

# ── Phase 2: Benchmark ───────────────────────────────────────────────
for i in "${!NAMES[@]}"; do
  NAME="${NAMES[$i]}"
  echo ""
  echo "══════════════════════════════════════════════════"
  echo " BENCHMARK [$((i+1))/${#NAMES[@]}]: $NAME"
  echo "══════════════════════════════════════════════════"
  python -u script/benchmark/run_llm_eval.py \
    --models Qwen3.5-4B \
    --datasets 90q \
    --finetune "/cluster/scratch/arakhmasari/lora-cot-checkpoints/${NAME}/adapters" \
    --limit 90
done

echo ""
echo "All done: $(date)"
