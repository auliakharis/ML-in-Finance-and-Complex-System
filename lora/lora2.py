import os
import ssl
import certifi
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ══════════════════════════════════════════════════════════════════
# 0. SSL FIX — must happen before any network call (model download)
#    Needed when the venv path contains spaces or special characters
#    that confuse certifi's bundle lookup on macOS.
# ══════════════════════════════════════════════════════════════════
ca_bundle = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
os.environ["SSL_CERT_FILE"]      = ca_bundle
# Patch the default SSL context used by urllib / requests
ssl._create_default_https_context = ssl.create_default_context

# ══════════════════════════════════════════════════════════════════
# 1. DEVICE DETECTION
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
# 2. MODEL + TOKENIZER
# ══════════════════════════════════════════════════════════════════
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

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
    # MPS or CPU: use `dtype` (not the deprecated `torch_dtype`)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ══════════════════════════════════════════════════════════════════
# 3. LORA CONFIG
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
# 4. DATASET
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv("random_1000.csv")

SYSTEM_PROMPT = """You are a financial reasoning assistant.
Given a set of financial facts and a question, compute the answer step by step.
Always end your response with: ANSWER: <numeric_value>"""


def extract_leaf_context(row):
    """Collapse the leaf_N_* columns into a readable fact list."""
    facts = []
    for i in range(1, 15):
        key    = row.get(f"leaf_{i}_key")
        label  = row.get(f"leaf_{i}_label")
        entity = row.get(f"leaf_{i}_entity")
        period = row.get(f"leaf_{i}_period")
        unit   = row.get(f"leaf_{i}_unit")
        value  = row.get(f"leaf_{i}_value")
        if pd.notna(key) and pd.notna(value):
            facts.append(
                f"- {entity} | {label} | {period} | {unit} {float(value):,.2f}"
            )
    return "\n".join(facts)


def format_row(row):
    """One CSV row → final 'text' string in Qwen chat format."""
    context      = extract_leaf_context(row)
    question     = row["question"]
    expr         = row["expression"]
    answer       = row["answer"]

    user_msg = f"""Financial facts:
{context}

Question: {question}

Express the calculation formula and compute the final answer."""

    assistant_msg = f"""Expression: {expr}

Calculating step by step based on the given values.

ANSWER: {float(answer):.4f}"""

    text = tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


df_formatted = df.apply(format_row, axis=1, result_type="expand")
dataset = Dataset.from_pandas(df_formatted)

# ══════════════════════════════════════════════════════════════════
# 5. TRAINING ARGS
# ══════════════════════════════════════════════════════════════════
training_args = TrainingArguments(
    output_dir="./qwen-lora-out",
    num_train_epochs=3,
    per_device_train_batch_size=1 if not CUDA_AVAILABLE else 2,
    gradient_accumulation_steps=16 if not CUDA_AVAILABLE else 8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=CUDA_AVAILABLE,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit" if CUDA_AVAILABLE else "adamw_torch",
    report_to="none",
)

# ══════════════════════════════════════════════════════════════════
# 6. TRAIN
# ══════════════════════════════════════════════════════════════════
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
)

trainer.train()

# ══════════════════════════════════════════════════════════════════
# 7. SAVE
# ══════════════════════════════════════════════════════════════════
model.save_pretrained("./qwen-lora-adapters")
tokenizer.save_pretrained("./qwen-lora-adapters")
print("[INFO]  Adapters saved to ./qwen-lora-adapters")