# ============================================================
# Benchmark matrix runner
# - Mixed depth = depths 1..5, 250 generated questions
# - Fixed depths = 1..8, 300 generated questions each
# - Adversarial = mixed depths 1..5, preserving existing 200/100 setup
# - Every requested multiturn run is placed immediately after a 90q run
#   of the same model. For models that were only listed under multiturn,
#   a prerequisite 90q run is added so the pairing rule is always true.
# ============================================================

PYTHON="${PYTHON:-python}"
PREP="script/compiler_pipeline_adversarial/data_prep_adversarial.py"
MAKE="script/compiler_pipeline_adversarial/make_random_questions_adversarial.py"
ADVERSARIAL="script/compiler_pipeline_adversarial/adversarial"
EVAL="script.benchmark.run_llm_eval_api_call"
FINANCIAL_JSON="script/compiler_pipeline_adversarial/output/financial_spreadsheet.json"
ADV_DIR="script/compiler_pipeline_adversarial/output/adversarial"

MIXED_N=250
DEPTH_N=100
ADV_N=200
ADV_LIMIT=100

# -------------------------
# Provider mapping (from your request)
# -------------------------
PROVIDER_DEEPSEEK="Together"
PROVIDER_GEMMA="Together"
PROVIDER_APERTUS="Apertus"
PROVIDER_LIQUID="Togehter"
PROVIDER_QWEN="Fireworks"
PROVIDER_LLAMA="Together"
PROVIDER_GPT_OSS="Together"

# -------------------------
# Model IDs
# IDs directly evidenced by your pasted commands are prefilled.
# Fill the remaining REPLACE_* values with the exact IDs accepted by your API.
# -------------------------
DEEPSEEK_V4_FLASH="deepseek-ai/DeepSeek-V4-Flash-0731"
APERTUS_70B="CSCS-Inference/swiss-ai/Apertus-v1.5-70B-thinking"

QWEN_3_7_PLUS="accounts/fireworks/models/qwen3p7-plus"
QWEN_3_5_9B="Qwen/Qwen3.5-9B"

GPT_OSS_120B="openai/gpt-oss-120b"
APERTUS_8B="CSCS-Inference/swiss-ai/Apertus-8B-Instruct-2509"
GEMMA_31B="google/gemma-4-31B-it"

LLAMA_70B="meta-llama/Llama-3.3-70B-Instruct-Turbo"
LIQUID_LFM_2_5_8B="LiquidAI/LFM2.5-8B-A1B"

mkdir -p \
  logs/last_runs_9th_aug_after_21/setup logs/last_runs_9th_aug_after_21/mixed_depth logs/last_runs_9th_aug_after_21/depths logs/last_runs_9th_aug_after_21/adversarial \
  output_llm/last_runs_9th_aug_after_21/mixed_depth output_llm/last_runs_9th_aug_after_21/depths output_llm/last_runs_9th_aug_after_21/adversarial

MASTER_LOG="output_llm/last_runs_9th_aug_after_21/benchmark_matrix.log"

stamp() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$MASTER_LOG"
}

slug() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's#[^a-z0-9]+#_#g; s#^_+##; s#_+$##'
}

check_model_ids() {
  local bad=0
  local vars=(GPT_OSS_120B APERTUS_8B GEMMA_31B LLAMA_70B LIQUID_LFM_2_5_8B)
  local v value
  for v in "${vars[@]}"; do
    value="${!v}"
    if [[ "$value" == REPLACE_* ]]; then
      echo "ERROR: set $v (currently: $value)" >&2
      bad=1
    fi
  done
  if (( bad )); then
    echo "Edit the model-ID block at the top of this script, then rerun." >&2
    exit 2
  fi
}

run_eval() {
  local section="$1"
  local tag="$2"
  local provider="$3"
  local model="$4"
  local dataset="$5"
  local limit="$6"

  local safe_tag
  safe_tag="$(slug "$tag")"
  local log="logs/last_runs_9th_aug_after_21/${section}/${safe_tag}.log"
  local out="output_llm/last_runs_9th_aug_after_21/${section}/${safe_tag}.json"

  stamp "START section=$section tag=$tag provider=$provider model=$model dataset=$dataset limit=$limit"
  echo "<===== START $tag | provider=$provider | model=$model | dataset=$dataset | limit=$limit =====>" | tee -a "$log" "$MASTER_LOG"

  "$PYTHON" -m "$EVAL" \
    --provider "$provider" \
    --models "$model" \
    --datasets "$dataset" \
    --limit "$limit" \
    --output "$out" \
    2>&1 | tee -a "$log" "$MASTER_LOG"

  echo "<===== END $tag =====>" | tee -a "$log" "$MASTER_LOG"
  stamp "DONE section=$section tag=$tag output=$out log=$log"
}

prepare_data() {
  local tag="$1"
  shift
  local log="logs/last_runs_9th_aug_after_21/setup/$(slug "$tag").log"
  stamp "PREP $tag: $MAKE $*"
  "$PYTHON" -u "$MAKE" "$@" 2>&1 | tee -a "$log" "$MASTER_LOG"
}

run_pair() {
  # Always runs 90q first, then multiturn-90q immediately after it.
  local section="$1"
  local base_tag="$2"
  local provider="$3"
  local model="$4"
  local limit="$5"

  # run_eval "$section" "${base_tag}_90q" "$provider" "$model" "90q" "$limit"
  run_eval "$section" "${base_tag}_multiturn_90q" "$provider" "$model" "multiturn-90q" "$limit"
}

run_setup() {
  stamp "Running data prep"
  "$PYTHON" -u "$PREP" 2>&1 | tee -a logs/last_runs_9th_aug_after_21/setup/last_runs_9th_aug_after_21/data_prep.log "$MASTER_LOG"
}

run_mixed_depth() {
  stamp "=== MIXED DEPTH 1-5 / n=$MIXED_N ==="
  prepare_data "mixed_depth_1_5_n${MIXED_N}" --n "$MIXED_N" --depth-min 1 --depth-max 5

  # Requested 90q + multiturn models: paired immediately.
  # run_pair mixed_depth "deepseek_v4_flash_mixed_depth_1_5" "$PROVIDER_DEEPSEEK" "$DEEPSEEK_V4_FLASH" "$MIXED_N"
  # run_pair mixed_depth "gpt_oss_120b_mixed_depth_1_5" "$PROVIDER_GPT_OSS" "$GPT_OSS_120B" "$MIXED_N"
  # run_pair mixed_depth "apertus_8b_mixed_depth_1_5" "$PROVIDER_APERTUS" "$APERTUS_8B" "$MIXED_N"
  # run_pair mixed_depth "gemma_31b_mixed_depth_1_5" "$PROVIDER_GEMMA" "$GEMMA_31B" "$MIXED_N"

  # run_pair mixed_depth "llama_70b_mixed_depth_1_5" "$PROVIDER_LLAMA" "$LLAMA_70B" "$MIXED_N"
  run_pair mixed_depth "qwen_3_7_plus_mixed_depth_1_5" "$PROVIDER_QWEN" "$QWEN_3_7_PLUS" "$MIXED_N"
  # run_pair mixed_depth "liquid_lfm_2_5_8b_mixed_depth_1_5" "$PROVIDER_LIQUID" "$LIQUID_LFM_2_5_8B" "$MIXED_N"
  # run_pair mixed_depth "qwen_3_5_9b_mixed_depth_1_5" "$PROVIDER_QWEN" "$QWEN_3_5_9B" "$MIXED_N"
}

run_fixed_depths() {
  local depth
  for depth in {8..8}; do
    stamp "=== DEPTH $depth / n=$DEPTH_N ==="
    if [ "$depth" -lt 3 ]; then
      prepare_data "depth_${depth}_n${DEPTH_N}" --n "$DEPTH_N" --depth-min "$depth" --depth-max "$depth" --derived-prob-min 0 --derived-prob-max 0
    else 
      prepare_data "depth_${depth}_n${DEPTH_N}" --n "$DEPTH_N" --depth-min "$depth" --depth-max "$depth" 
    fi
    # run_eval depths "depth_${depth}_apertus_70b_90q" "$PROVIDER_APERTUS" "$APERTUS_70B" "90q" "$DEPTH_N"
    run_eval depths "depth_${depth}_gemma_31b_90q" "$PROVIDER_GEMMA" "$GEMMA_31B" "90q" "$DEPTH_N"
    # run_eval depths "depth_${depth}_deepseek_v4_flash_90q" "$PROVIDER_DEEPSEEK" "$DEEPSEEK_V4_FLASH" "90q" "$DEPTH_N"
  done
}

adversarial_source() {
  case "$1" in
    missing_values) echo "$ADV_DIR/missing_values.json" ;;
    garbage_values) echo "$ADV_DIR/garbage.json" ;;
    look_alike) echo "$ADV_DIR/lookalike.json" ;;
    cross_sheet) echo "$ADV_DIR/cross_contaminated.json" ;;
    scaling) echo "$ADV_DIR/scaling.json" ;;
    useless_info) echo "$ADV_DIR/useless_info.json" ;;
    combined) echo "$ADV_DIR/combined_adversarial.json" ;;
    *) echo "Unknown adversarial variant: $1" >&2; return 2 ;;
  esac
}

run_adversarial() {
  stamp "=== ADVERSARIAL / base depth 1-5 / n=$ADV_N / eval limit=$ADV_LIMIT ==="
  prepare_data "adversarial_base_depth_1_5_n${ADV_N}" --n "$ADV_N" --depth-min 1 --depth-max 5 --obstacle useless_info
  "$PYTHON" "$ADVERSARIAL" 2>&1 | tee -a logs/last_runs_9th_aug_after_21/setup/adversarial_generation.log "$MASTER_LOG"
# useless_info combined
  local variants=(combined)
  local variant src
  for variant in "${variants[@]}"; do
    src="$(adversarial_source "$variant")"
    if [[ ! -f "$src" ]]; then
      echo "ERROR: adversarial file not found for '$variant': $src" | tee -a "$MASTER_LOG" >&2
      echo "If your generator uses a different filename, edit adversarial_source() in this script." >&2
      exit 3
    fi

    stamp "Loading adversarial variant=$variant from $src"
    cp "$src" "$FINANCIAL_JSON"

    run_eval adversarial "${variant}_gemma_31b_90q" "$PROVIDER_GEMMA" "$GEMMA_31B" "90q" "$ADV_LIMIT"
    run_eval adversarial "${variant}_deepseek_v4_flash_90q" "$PROVIDER_DEEPSEEK" "$DEEPSEEK_V4_FLASH" "90q" "$ADV_LIMIT"
  done
}

usage() {
  cat <<'USAGE'
Usage:
  ./run_benchmark_matrix.sh all
  ./run_benchmark_matrix.sh mixed
  ./run_benchmark_matrix.sh depths
  ./run_benchmark_matrix.sh adversarial
  ./run_benchmark_matrix.sh setup

Logs:
  logs/benchmark_matrix.log        master chronological log
  logs/mixed_depth/*.log           one log per mixed-depth run
  logs/depths/*.log                one log per fixed-depth/model run
  logs/adversarial/*.log           one log per adversarial/model run
  logs/setup/*.log                 generation/prep logs

JSON outputs mirror the same structure under output_llm/.
USAGE
}

main() {
  check_model_ids
  local mode="${1:-all}"
  case "$mode" in
    all)
      run_setup
      run_mixed_depth
      run_fixed_depths
      run_adversarial
      ;;
    mixed)
      run_setup
      run_mixed_depth
      ;;
    depths)
      run_setup
      run_fixed_depths
      ;;
    adversarial)
      run_setup
      run_adversarial
      ;;
    setup)
      run_setup
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
