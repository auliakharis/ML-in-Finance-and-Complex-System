import argparse
import json
import os
import ssl
import sys
import certifi
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Cap total training examples (e.g. --limit 20 for a smoke test)")
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════
# 0. SSL FIX
# ══════════════════════════════════════════════════════════════════
ca_bundle = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
os.environ["SSL_CERT_FILE"]      = ca_bundle
ssl._create_default_https_context = ssl.create_default_context

# ══════════════════════════════════════════════════════════════════
# 1. PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR    = Path(__file__).parent.parent          # repo root
MODEL_PATH  = Path(f"/cluster/scratch/{os.environ.get('USER', 'arakhmasari')}/models")
SCRATCH_DIR = Path(f"/cluster/scratch/{os.environ.get('USER', 'arakhmasari')}/lora-checkpoints")
MODEL       = "Qwen3.5-4B"
MODEL_ID    = str(MODEL_PATH / MODEL)

SHEET_PATH  = BASE_DIR / "dataset_output" / "synthetic_company_data_refactored.json"
CSV_PATH    = BASE_DIR / "lora" / "random_1000.csv"

# Import prompt builders from the eval script so training format == eval format
sys.path.insert(0, str(BASE_DIR))
from run_llm_eval import build_prompt, sheet_to_text_90q

# ══════════════════════════════════════════════════════════════════
# 2. DEVICE DETECTION
# ══════════════════════════════════════════════════════════════════
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE  = torch.backends.mps.is_available()

if CUDA_AVAILABLE:
    DEVICE = "cuda"
    print("[INFO]  CUDA GPU detected - using 4-bit quantisation.")
elif MPS_AVAILABLE:
    DEVICE = "mps"
    print("[INFO]  Apple Silicon MPS detected - skipping 4-bit (not supported on MPS).")
else:
    DEVICE = "cpu"
    print("[WARN]  No GPU detected - loading in float32 on CPU. This will be slow.")

# ══════════════════════════════════════════════════════════════════
# 3. MODEL + TOKENIZER
# ══════════════════════════════════════════════════════════════════
if CUDA_AVAILABLE:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ══════════════════════════════════════════════════════════════════
# 4. LORA CONFIG
# ══════════════════════════════════════════════════════════════════
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ══════════════════════════════════════════════════════════════════
# 5. DATASET
# ══════════════════════════════════════════════════════════════════
# Build entity → yearly rows lookup (same as eval script)
with open(SHEET_PATH) as f:
    sheet_raw = json.load(f)

sheet_lookup: dict = {}
for row in sheet_raw:
    ticker = row["ticker"]
    sheet_lookup.setdefault(ticker, []).append(row)
for ticker in sheet_lookup:
    sheet_lookup[ticker].sort(key=lambda r: r.get("year", "0"))

df = pd.read_csv(CSV_PATH)

# Depth rebalancing: depth-0 is already easy (77% accuracy), cap it so
# the model trains more on the hard cases (depth-1/2) that need improvement.
depth0  = df[df["depth"] == 0].sample(n=150, random_state=42)
depth12 = df[df["depth"].isin([1, 2])]           # keep all ~286 examples
depth3p = df[df["depth"] >= 3].sample(          # sample harder depths
    n=min(200, len(df[df["depth"] >= 3])), random_state=42
)
df = pd.concat([depth0, depth12, depth3p]).sample(frac=1, random_state=42).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)
    print(f"[INFO]  --limit {args.limit}: using {len(df)} examples (smoke test mode)")
print(f"[INFO]  Training set: {len(df)} examples | depth dist: {df['depth'].value_counts().sort_index().to_dict()}")


def format_row(row):
    """CSV row → (prompt in eval format, concise Answer: response)."""
    entity = None
    for i in range(1, 15):
        e = row.get(f"leaf_{i}_entity")
        if pd.notna(e) and e:
            entity = str(e)
            break

    sheet_rows = sheet_lookup.get(entity, [])
    sheet_text = sheet_to_text_90q(sheet_rows)

    question = row["question"]
    answer   = float(row["answer"])
    # Use same format as eval parser expects: clean number, no trailing zeros
    answer_str = f"{answer:.6g}"

    # Build the prompt exactly as run_llm_eval.py does during evaluation
    prompt = build_prompt(sheet_text, question, thinking_mode=False)

    # Target response: direct answer, no deliberation — teaches brevity
    assistant_msg = f"Answer: {answer_str}"

    text = tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": assistant_msg},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


df_formatted = df.apply(format_row, axis=1, result_type="expand")
dataset = Dataset.from_pandas(df_formatted)

# ══════════════════════════════════════════════════════════════════
# 6. TRAINING ARGS
# ══════════════════════════════════════════════════════════════════
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

training_args = SFTConfig(
    output_dir=str(SCRATCH_DIR / "checkpoints"),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1 if not CUDA_AVAILABLE else 2,
    gradient_accumulation_steps=16 if not CUDA_AVAILABLE else 8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=CUDA_AVAILABLE,
    fp16=False,
    logging_steps=10,
    save_strategy="no",                     # skip mid-training checkpoints (Python 3.13/torch pickle bug)
    optim="paged_adamw_8bit" if CUDA_AVAILABLE else "adamw_torch",
    report_to="none",
    dataset_text_field="text",
    max_length=2048,                        # new prompt format (full spreadsheet) is 1667–1893 tokens
)

# ══════════════════════════════════════════════════════════════════
# 7. TRAIN
# ══════════════════════════════════════════════════════════════════
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

# ══════════════════════════════════════════════════════════════════
# 8. SAVE FINAL ADAPTERS TO SCRATCH
# ══════════════════════════════════════════════════════════════════
adapter_save_path = str(SCRATCH_DIR / "adapters")
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)
print(f"[INFO]  Adapters saved to {adapter_save_path}")
