# requirements: transformers, peft, bitsandbytes, trl, datasets, accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset


# ── Load your CSV ──────────────────────────────────────────────────────────
df = pd.read_csv("compiler_pipeline/output/random_questions_90_alicia.csv")

# ── Build the prompt for each row ─────────────────────────────────────────
def extract_leaf_context(row):
    """Pull all non-null leaf values into a readable context block."""
    facts = []
    for i in range(1, 15):
        key    = row.get(f"leaf_{i}_key")
        label  = row.get(f"leaf_{i}_label")
        entity = row.get(f"leaf_{i}_entity")
        period = row.get(f"leaf_{i}_period")
        unit   = row.get(f"leaf_{i}_unit")
        value  = row.get(f"leaf_{i}_value")
        if pd.notna(key) and pd.notna(value):
            facts.append(f"- {entity} | {label} | {period} | {unit} {value:,.2f}")
    return "\n".join(facts)


SYSTEM_PROMPT = """You are a financial reasoning assistant.
Given a set of financial facts and a question, compute the answer step by step.
Always end your response with: ANSWER: <numeric_value>"""


def format_row(row):
    context  = extract_leaf_context(row)
    question = row["question"]
    expr     = row["expression"]       # optional: teach the model the formula too
    answer   = row["answer"]

    user_msg = f"""Financial facts:
{context}

Question: {question}

Express the calculation formula and compute the final answer."""

    # Chain-of-thought assistant turn: show the expression then the answer
    assistant_msg = f"""Expression: {expr}

Calculating step by step based on the given values.

ANSWER: {answer:.4f}"""

    return {"question": user_msg, "answer": assistant_msg}


# ── Apply formatting ───────────────────────────────────────────────────────
df_formatted = df.apply(format_row, axis=1, result_type="expand")
dataset = Dataset.from_pandas(df_formatted)


# ── Tokenise into Qwen chat template ──────────────────────────────────────
def to_chat_format(example):
    return {
        "text": tokenizer.apply_chat_template(
            [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(to_chat_format)
# dataset now has a "text" column ready for SFTTrainer


# ── 1. Load base model in 4-bit ────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # adjust to your target checkpoint

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 – best quality
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # nested quantisation saves ~0.4 bpp
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token   # Qwen uses <|endoftext|>

# ── 2. LoRA configuration ──────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # rank – higher = more capacity, more memory
    lora_alpha=32,          # scaling factor (rule of thumb: 2 × r)
    lora_dropout=0.05,
    bias="none",
    # Target the attention projections (and optionally the MLP gate)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # remove if VRAM is tight
    ],
)

# ── 3. Wrap model with PEFT ────────────────────────────────────────────────
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# e.g. → trainable params: 39,976,960 || all params: 7,282,847,744 (0.55 %)

# ── 4. Prepare dataset & train ────────────────────────────────────────────
dataset = load_dataset("your_org/your_dataset", split="train")  # swap in yours

def format_prompt(example):
    """Convert rows to Qwen chat format."""
    return {
        "text": tokenizer.apply_chat_template(
            [{"role": "user",    "content": example["instruction"]},
             {"role": "assistant","content": example["output"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(format_prompt)

training_args = TrainingArguments(
    output_dir="./qwen-lora-out",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=False,
    bf16=True,                          # bfloat16 is stable on Ampere+
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",           # memory-efficient optimiser
    report_to="none",                   # swap for "wandb" if tracking
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
)

trainer.train()

# ── 5. Save LoRA adapters (~50–200 MB) ────────────────────────────────────
model.save_pretrained("./qwen-lora-adapters")
tokenizer.save_pretrained("./qwen-lora-adapters")

# ── 6a. Load adapters at inference time (no merge) ────────────────────────
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto",
                                             trust_remote_code=True)
model_inf = PeftModel.from_pretrained(base, "./qwen-lora-adapters")

# ── 6b. Or merge into base weights for maximum inference speed ────────────
merged = model_inf.merge_and_unload()          # fuses A·B into W; no overhead
merged.save_pretrained("./qwen-lora-merged")   # deploy as a normal model