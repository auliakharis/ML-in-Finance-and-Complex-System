"""
run_llm_eval.py
====================
Runs LLM evaluation on two Q&A datasets:

  10q  — 10q/final_qa_dataset.json
         context: one spreadsheet row per company from 10q/financial_spreadsheet.json

  90q  — 90q/random_questions_90.json
         context: all yearly rows for the entity from 90q/financial_spreadsheet.json

For each question, a prompt is built:
  "You are a financial data assistant. Given the following financial sheets statements below
   [company financial sheet]
   Answer the following question using only the given instruction.
   [question]
   Answer only the final numeric value and give your reasoning."

Models evaluated (loaded from /cluster/scratch/$USER/models/):
  - Qwen3.5-4B
  - Qwen3.5-9B

Accuracy is computed as the fraction of answers within a relative tolerance
of the ground-truth numeric answer (default ±1%).

Usage:
  python run_llm_eval.py
  python run_llm_eval.py --limit 20 --tol 0.1
  python run_llm_eval.py --models Qwen3.5-4B --limit 10
  python run_llm_eval.py --output results.json
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODELS_DIR = Path(f"/cluster/scratch/{os.environ.get('USER', 'user')}/models")

DATASET_10Q = BASE_DIR / "10q" / "final_qa_dataset.json"
SHEET_10Q   = BASE_DIR / "10q" / "financial_spreadsheet.json"

DATASET_90Q = BASE_DIR / "90q" / "random_questions_90.json"
SHEET_90Q   = BASE_DIR / "90q" / "financial_spreadsheet.json"

DEFAULT_MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "gemma-4-E4B-it"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def build_10q_sheet_lookup(sheet: list) -> dict:
    """company_name -> dict of financial fields."""
    return {row["company_name"]: row for row in sheet}


def build_90q_sheet_lookup(sheet: list) -> dict:
    """company_name -> list of rows (one per year), sorted by year."""
    lookup: dict = {}
    for row in sheet:
        name = row["company_name"]
        lookup.setdefault(name, []).append(row)
    for name in lookup:
        lookup[name].sort(key=lambda r: r.get("year", "0"))
    return lookup


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def sheet_to_text_10q(row: dict) -> str:
    """Format a 10-Q spreadsheet row as readable key-value text."""
    lines = []
    for k, v in row.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def sheet_to_text_90q(rows: list) -> str:
    """Format multiple annual rows for a company as a table-like text."""
    if not rows:
        return "(no data)"
    lines = []
    for row in rows:
        year = row.get("year", "?")
        lines.append(f"  Year {year}:")
        for k, v in row.items():
            if k not in ("company_name", "year"):
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def build_prompt(sheet_text: str, question: str) -> str:
    return (
    "ROLE: Financial analyst. Answer using ONLY the data below. No commentary.\n\n"
    "=== DATA ===\n"
    f"{sheet_text}\n"
    "=== END ===\n\n"
    f"QUESTION: {question}\n\n"
    "RULES:\n"
    "- Extract only the numbers you need.\n"
    "- Show calculations in 1–3 lines max if needed.\n"
    "- Last line MUST be: Answer: <value>\n"
    "- <value> is either a number, True, or False. Nothing else.\n"
    "- Do NOT explain, summarize, or add anything after the Answer line.\n\n"
    "- Do NOT show your thinking process.\n\n" 
    "SOLVE NOW:"
    )

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_number(text: str) -> float | bool | None:
    """
    Extract the value from the 'Answer: <value>' line.
    Handles numbers, True/False. Falls back to last number if no Answer line found.
    """
    # 1. Look for explicit "Answer: <value>" line (case-insensitive)
    match = re.search(r'Answer\s*:\s*(.+)', text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip().rstrip(".,;")

        # Check for True/False first
        if raw.lower() == "true":
            return True
        if raw.lower() == "false":
            return False

        # Try to parse as number
        cleaned = re.sub(r'(\d),(\d)', r'\1\2', raw)
        num_match = re.search(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned)
        if num_match:
            return float(num_match.group())

    # 2. Fallback: last number in text (less reliable)
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
    matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned)
    return float(matches[-1]) if matches else None


def is_correct(predicted, ground_truth, tol: float) -> bool:
    if predicted is None or ground_truth is None:
        return False

    # Boolean comparison
    if isinstance(predicted, bool) or isinstance(ground_truth, bool):
        return predicted == ground_truth

    # Coerce string ground truth to float; skip if not numeric
    if isinstance(ground_truth, str):
        try:
            ground_truth = float(ground_truth.replace(",", ""))
        except ValueError:
            return False

    # Numeric comparison
    if ground_truth == 0:
        return abs(predicted) < 1e-6
    return abs(predicted - ground_truth) / abs(ground_truth) <= tol


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load a HuggingFace model and tokenizer from the models directory."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    model_path = str(MODELS_DIR / model_name)
    print(f"  Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"  Loading model from {model_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        trust_remote_code=True,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Model loaded on {device}.")
    return tokenizer, model


def run_inference(tokenizer, model, prompt: str, max_new_tokens: int = 512) -> str:
    """Run a single forward pass and return the decoded response."""
    import torch

    # Use chat template if available, otherwise plain tokenization
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def evaluate_10q(
    questions: list,
    sheet_lookup: dict,
    tokenizer,
    model,
    limit: int | None,
    tol: float,
) -> list:
    """Evaluate on 10-Q questions. Returns per-question result dicts."""
    results = []
    subset = questions[:limit] if limit else questions

    for i, q in enumerate(subset, 1):
        company = q.get("company", "")
        question_text = q.get("question", "")
        ground_truth = q.get("answer")

        sheet_row = sheet_lookup.get(company)
        if sheet_row is None:
            print(f"  [{i}/{len(subset)}] SKIP (no sheet for '{company}')")
            results.append({
                "source": "10q",
                "id": q.get("id"),
                "company": company,
                "depth": q.get("depth"),
                "question": question_text,
                "ground_truth": ground_truth,
                "llm_response": None,
                "predicted": None,
                "correct": False,
                "skip": True,
            })
            continue

        sheet_text = sheet_to_text_10q(sheet_row)
        prompt = build_prompt(sheet_text, question_text)

        response = run_inference(tokenizer, model, prompt)
        predicted = extract_number(response)
        correct = is_correct(predicted, ground_truth, tol)

        status = "CORRECT" if correct else "WRONG "
        print(f"  [{i}/{len(subset)}] {status} | truth={ground_truth} pred={predicted} | {question_text[:60]}...")

        results.append({
            "source": "10q",
            "id": q.get("id"),
            "company": company,
            "depth": q.get("depth"),
            "question": question_text,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "llm_response": response,
            "predicted": predicted,
            "correct": correct,
            "skip": False,
        })

    return results


def evaluate_90q(
    questions: list,
    sheet_lookup: dict,
    tokenizer,
    model,
    limit: int | None,
    tol: float,
) -> list:
    """Evaluate on 90-question dataset. Returns per-question result dicts."""
    results = []
    subset = questions[:limit] if limit else questions

    for i, q in enumerate(subset, 1):
        entity = q.get("entity") or q.get("leaf_1_entity", "")
        question_text = q.get("question", "")
        ground_truth = q.get("answer")

        sheet_rows = sheet_lookup.get(entity)
        if not sheet_rows:
            print(f"  [{i}/{len(subset)}] SKIP (no sheet for '{entity}')")
            results.append({
                "source": "90q",
                "id": q.get("id") or q.get("question_id"),
                "entity": entity,
                "depth": q.get("depth"),
                "question": question_text,
                "ground_truth": ground_truth,
                "llm_response": None,
                "predicted": None,
                "correct": False,
                "skip": True,
            })
            continue

        sheet_text = sheet_to_text_90q(sheet_rows)
        prompt = build_prompt(sheet_text, question_text)

        response = run_inference(tokenizer, model, prompt)
        predicted = extract_number(response)
        correct = is_correct(predicted, ground_truth, tol)

        status = "CORRECT" if correct else "WRONG "
        print(f"  [{i}/{len(subset)}] {status} | truth={float(ground_truth):.4f} pred={predicted} | {question_text[:60]}...")

        results.append({
            "source": "90q",
            "id": q.get("id") or q.get("question_id"),
            "entity": entity,
            "depth": q.get("depth"),
            "question": question_text,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "llm_response": response,
            "predicted": predicted,
            "correct": correct,
            "skip": False,
        })

    return results


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------

def compute_accuracy(results: list) -> dict:
    evaluated = [r for r in results if not r.get("skip")]
    if not evaluated:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(1 for r in evaluated if r["correct"])
    return {
        "total": len(evaluated),
        "skipped": len(results) - len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated),
    }


def print_summary(model_name: str, results_10q: list, results_90q: list) -> None:
    acc10 = compute_accuracy(results_10q)
    acc90 = compute_accuracy(results_90q)
    total_eval = acc10["total"] + acc90["total"]
    total_correct = acc10["correct"] + acc90["correct"]
    overall = total_correct / total_eval if total_eval else 0.0

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  10-Q  : {acc10['correct']}/{acc10['total']}  accuracy = {acc10['accuracy']:.1%}  (skipped {acc10.get('skipped',0)})")
    print(f"  90-Q  : {acc90['correct']}/{acc90['total']}  accuracy = {acc90['accuracy']:.1%}  (skipped {acc90.get('skipped',0)})")
    print(f"  Overall: {total_correct}/{total_eval}  accuracy = {overall:.1%}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(all_results: dict, path: Path, tol: float) -> None:
    """Write a flat CSV with one row per question across all models and datasets."""
    fieldnames = [
        "model", "dataset", "id", "entity", "depth",
        "question", "prompt", "ground_truth", "predicted_answer",
        "reasoning", "correct", "relative_error", "within_tol", "skipped",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model_data in all_results.items():
            for dataset_key in ("10q", "90q"):
                for r in model_data.get(dataset_key, []):
                    gt = r.get("ground_truth")
                    pred = r.get("predicted")
                    try:
                        gt = float(str(gt).replace(",", "")) if gt is not None else None
                    except (ValueError, TypeError):
                        gt = None
                    if gt is not None and pred is not None and gt != 0:
                        rel_err = abs(pred - gt) / abs(gt)
                    else:
                        rel_err = None
                    writer.writerow({
                        "model": model_name,
                        "dataset": dataset_key,
                        "id": r.get("id", ""),
                        "entity": r.get("company") or r.get("entity", ""),
                        "depth": r.get("depth", ""),
                        "question": r.get("question", ""),
                        "prompt": r.get("prompt", ""),
                        "ground_truth": gt,
                        "predicted_answer": pred,
                        "reasoning": r.get("llm_response", ""),
                        "correct": r.get("correct", False),
                        "relative_error": f"{rel_err:.6f}" if rel_err is not None else "",
                        "within_tol": (rel_err is not None and rel_err <= tol),
                        "skipped": r.get("skip", False),
                    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LLM evaluation on 10-Q and 90-Q financial QA datasets")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Model folder names under MODELS_DIR (default: Qwen3.5-4B Qwen3.5-9B)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions per dataset per model (default: all)",
    )
    parser.add_argument(
        "--tol", type=float, default=0.1,
        help="Relative tolerance for numeric correctness (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max new tokens to generate per answer (default: 512)",
    )
    parser.add_argument(
        "--output", type=str, default="output_llm/eval_results.json",
        help="Path to save per-question results JSON (default: eval_results.json)",
    )
    parser.add_argument(
        "--csv", type=str, default="output_llm/eval_results.csv",
        help="Path to save per-question results CSV (default: eval_results.csv)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["10q", "90q"], default=["10q", "90q"],
        help="Which dataset(s) to evaluate: 10q, 90q, or both (default: both)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    print("Loading datasets...")
    if "10q" in args.datasets:
        questions_10q = load_json(DATASET_10Q)
        sheet_lookup_10q = build_10q_sheet_lookup(load_json(SHEET_10Q))
        print(f"  10-Q: {len(questions_10q)} questions, {len(sheet_lookup_10q)} companies")
    else:
        questions_10q, sheet_lookup_10q = [], {}

    if "90q" in args.datasets:
        questions_90q = load_json(DATASET_90Q)
        sheet_lookup_90q = build_90q_sheet_lookup(load_json(SHEET_90Q))
        print(f"  90-Q: {len(questions_90q)} questions, {len(sheet_lookup_90q)} companies")
    else:
        questions_90q, sheet_lookup_90q = [], {}

    all_results = {}

    for model_name in args.models:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"\nWARNING: model not found at {model_path}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating model: {model_name}")
        print(f"{'='*60}")

        tokenizer, model = load_model(model_name)

        if "10q" in args.datasets:
            print(f"\n-- 10-Q evaluation ({args.limit or len(questions_10q)} questions) --")
            results_10q = evaluate_10q(
                questions_10q, sheet_lookup_10q, tokenizer, model,
                limit=args.limit, tol=args.tol,
            )
        else:
            results_10q = []

        if "90q" in args.datasets:
            print(f"\n-- 90-Q evaluation ({args.limit or len(questions_90q)} questions) --")
            results_90q = evaluate_90q(
                questions_90q, sheet_lookup_90q, tokenizer, model,
                limit=args.limit, tol=args.tol,
            )
        else:
            results_90q = []

        print_summary(model_name, results_10q, results_90q)

        all_results[model_name] = {
            "10q": results_10q,
            "90q": results_90q,
            "accuracy_10q": compute_accuracy(results_10q),
            "accuracy_90q": compute_accuracy(results_90q),
        }

        # Free memory before loading next model
        del model, tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output)
    output_path = output_path.with_stem(f"{output_path.stem}_{ts}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    csv_path = Path(args.csv)
    csv_path = csv_path.with_stem(f"{csv_path.stem}_{ts}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(all_results, csv_path, tol=args.tol)
    print(f"CSV saved to {csv_path}")

    # Final comparison across models
    if len(all_results) > 1:
        print("\n=== Model Comparison ===")
        print(f"{'Model':<20} {'10-Q Acc':>10} {'90-Q Acc':>10} {'Overall':>10}")
        print("-" * 52)
        for m, r in all_results.items():
            a10 = r["accuracy_10q"]["accuracy"]
            a90 = r["accuracy_90q"]["accuracy"]
            t10 = r["accuracy_10q"]["total"]
            t90 = r["accuracy_90q"]["total"]
            total = t10 + t90
            correct = r["accuracy_10q"]["correct"] + r["accuracy_90q"]["correct"]
            overall = correct / total if total else 0.0
            print(f"{m:<20} {a10:>10.1%} {a90:>10.1%} {overall:>10.1%}")


if __name__ == "__main__":
    main()
