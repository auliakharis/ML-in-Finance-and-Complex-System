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
parser.add_argument("--gen-tokens", type=int, default=512,
                    help="Max new tokens for CoT generation per example")
parser.add_argument("--tol", type=float, default=0.01,
                    help="Relative tolerance for accepting a CoT trace as correct")
# LoRA hyperparameters
parser.add_argument("--r", type=int, default=8, help="LoRA rank")
parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
parser.add_argument("--lora-dropout", type=float, default=0.10, help="LoRA dropout")
# CoT trace cache — generate once, reuse across hyperparameter sweeps
parser.add_argument("--cot-cache", type=str, default=None,
                    help="Path to a JSON file for saving/loading Phase 1 CoT traces. "
                         "If the file exists, Phase 1 is skipped and traces are loaded from it.")
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
BASE_DIR    = Path(__file__).parent.parent.parent
MODEL_PATH  = Path(f"/cluster/scratch/{os.environ.get('USER', 'arakhmasari')}/models")
_run_tag   = f"r{args.r}_alpha{args.lora_alpha}_dropout{args.lora_dropout}"
SCRATCH_DIR = Path(f"/cluster/scratch/{os.environ.get('USER', 'arakhmasari')}/lora-cot-checkpoints") / _run_tag
MODEL       = "Qwen3.5-4B"
MODEL_ID    = str(MODEL_PATH / MODEL)

SHEET_PATH  = BASE_DIR / "output"/ "dataset_output" / "synthetic_company_data_refactored.json"
CSV_PATH    = BASE_DIR / "script" / "lora" / "random_1000.csv"

sys.path.insert(0, str(BASE_DIR / "script" / "benchmark"))
from run_llm_eval import build_prompt, sheet_to_text_90q, parse_answer_line, is_correct

# ══════════════════════════════════════════════════════════════════
# 2. DEVICE
# ══════════════════════════════════════════════════════════════════
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE  = torch.backends.mps.is_available()

if CUDA_AVAILABLE:
    DEVICE = "cuda"
    print("[INFO]  CUDA GPU detected - using 4-bit quantisation.")
elif MPS_AVAILABLE:
    DEVICE = "mps"
    print("[INFO]  Apple Silicon MPS detected - skipping 4-bit.")
else:
    DEVICE = "cpu"
    print("[WARN]  No GPU detected - loading in float32 on CPU.")

# ══════════════════════════════════════════════════════════════════
# 3. LOAD BASE MODEL (no LoRA yet — needed for CoT generation)
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
        torch_dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ══════════════════════════════════════════════════════════════════
# 4. DATASET — same depth rebalancing as lora2.py
# ══════════════════════════════════════════════════════════════════
with open(SHEET_PATH) as f:
    sheet_raw = json.load(f)

sheet_lookup: dict = {}
for row in sheet_raw:
    sheet_lookup.setdefault(row["company_name"], []).append(row)
for company in sheet_lookup:
    sheet_lookup[company].sort(key=lambda r: r.get("year", "0"))

df = pd.read_csv(CSV_PATH)
depth0  = df[df["depth"] == 0].sample(n=150, random_state=42)
depth12 = df[df["depth"].isin([1, 2])]
depth3p = df[df["depth"] >= 3].sample(n=min(200, len(df[df["depth"] >= 3])), random_state=42)
df = pd.concat([depth0, depth12, depth3p]).sample(frac=1, random_state=42).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)
    print(f"[INFO]  --limit {args.limit}: using {len(df)} examples (smoke test mode)")
print(f"[INFO]  Pool: {len(df)} examples | depth dist: {df['depth'].value_counts().sort_index().to_dict()}")


def get_entity(row):
    for i in range(1, 15):
        e = row.get(f"leaf_{i}_entity")
        if pd.notna(e) and e:
            return str(e)
    return None


@torch.no_grad()
def generate_response(prompt: str) -> str:
    """Run greedy inference with the base model and return the decoded response."""
    try:
        raw = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,   # suppress Qwen3 thinking block so Answer: fits in budget
        )
    except TypeError:
        raw = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    # apply_chat_template may return a BatchEncoding (dict-like) or a plain tensor
    input_ids = (raw["input_ids"] if hasattr(raw, "__getitem__") and not isinstance(raw, torch.Tensor)
                 else raw).to(DEVICE)
    out = model.generate(
        input_ids,
        max_new_tokens=args.gen_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 1 — Generate CoT traces from the base model (or load cache)
#
# For each training example we run the base model (no LoRA) and only
# keep the trace if it produces the CORRECT final answer. These traces
# become the assistant targets for LoRA training, so the model learns
# to reason through the computation AND land on the right answer.
#
# Pass --cot-cache <path> to save traces on the first run and skip
# generation entirely on subsequent hyperparameter sweep runs.
# ══════════════════════════════════════════════════════════════════
cot_cache_path = Path(args.cot_cache) if args.cot_cache else None

if cot_cache_path and cot_cache_path.exists():
    print(f"\n[PHASE 1] Loading cached CoT traces from {cot_cache_path} ...")
    with open(cot_cache_path) as f:
        cot_records = json.load(f)
    print(f"[PHASE 1] Loaded {len(cot_records)} traces — skipping generation.")
else:
    print("\n[PHASE 1] Generating CoT traces from base model ...")
    model.eval()

    cot_records = []
    n_skipped   = 0

    for i, (_, row) in enumerate(df.iterrows()):
        entity     = get_entity(row)
        sheet_rows = sheet_lookup.get(entity, [])
        sheet_text = sheet_to_text_90q(sheet_rows)
        question   = row["question"]
        truth      = float(row["answer"])

        prompt   = build_prompt(sheet_text, question, thinking_mode=False)
        response = generate_response(prompt)
        pred     = parse_answer_line(response)
        correct  = is_correct(pred, truth, tol=args.tol)

        status = "KEEP" if correct else "SKIP"
        print(f"  [{i+1}/{len(df)}] depth={row.get('depth',0)} {status} | truth={truth} pred={pred}")
        print(f"         Q: {question[:80]}")
        print(f"         R: {response[:300].strip()}")
        print()

        if correct:
            cot_records.append({
                "prompt":   prompt,
                "response": response,
                "depth":    row.get("depth", 0),
            })
        else:
            n_skipped += 1

        done = i + 1
        if done % 50 == 0 or done == len(df):
            print(f"  [{done}/{len(df)}]  kept={len(cot_records)}  skipped={n_skipped}  "
                  f"({len(cot_records)/done*100:.0f}% pass rate so far)")

    if cot_cache_path:
        cot_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cot_cache_path, "w") as f:
            json.dump(cot_records, f)
        print(f"[PHASE 1] CoT traces saved to {cot_cache_path}")

kept_pct   = len(cot_records) / len(df) * 100
depth_dist = {}
for rec in cot_records:
    d = rec["depth"]
    depth_dist[d] = depth_dist.get(d, 0) + 1

print(f"\n[PHASE 1] Done: {len(cot_records)}/{len(df)} kept ({kept_pct:.0f}%)")
print(f"[PHASE 1] Depth dist of kept CoT examples: {dict(sorted(depth_dist.items()))}")

if len(cot_records) < 10:
    raise RuntimeError(
        f"Only {len(cot_records)} correct CoT traces — too few to train. "
        "Try increasing --gen-tokens or --tol."
    )

# ══════════════════════════════════════════════════════════════════
# PHASE 2 — Wrap base model with LoRA adapters
# ══════════════════════════════════════════════════════════════════
print(f"\n[PHASE 2] Applying LoRA adapters (r={args.r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}) ...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ══════════════════════════════════════════════════════════════════
# PHASE 3 — Build HF Dataset from CoT traces
# ══════════════════════════════════════════════════════════════════
def format_record(record):
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    try:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": record["prompt"]},
                {"role": "assistant", "content": record["response"]},
            ],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": record["prompt"]},
                {"role": "assistant", "content": record["response"]},
            ],
            **kwargs,
        )
    return {"text": text}

dataset = Dataset.from_list([format_record(r) for r in cot_records])

# ══════════════════════════════════════════════════════════════════
# PHASE 4 — Train
#
# max_length is larger than lora2.py because CoT responses add ~200-500
# tokens on top of the ~1800-token spreadsheet prompt.
# Batch size is 1 (vs 2) to compensate for longer sequences.
# ══════════════════════════════════════════════════════════════════
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

training_args = SFTConfig(
    output_dir=str(SCRATCH_DIR / "checkpoints"),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,          # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=CUDA_AVAILABLE,
    fp16=False,
    logging_steps=10,
    save_strategy="no",                      # Python 3.13/torch pickle bug
    optim="paged_adamw_8bit" if CUDA_AVAILABLE else "adamw_torch",
    report_to="none",
    dataset_text_field="text",
    max_length=2560,                         # prompt (~1800) + CoT (~512) + buffer
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("\n[PHASE 4] Training ...")
trainer.train()

# ══════════════════════════════════════════════════════════════════
# SAVE ADAPTERS
# ══════════════════════════════════════════════════════════════════
adapter_save_path = str(SCRATCH_DIR / "adapters")
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)
print(f"[INFO]  CoT adapters saved to {adapter_save_path}")
