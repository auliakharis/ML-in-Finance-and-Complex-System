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

for the multi turn : 
the system message (financial data) is added once at the start of history and stays there for all turns — it's never re-added. But because history is passed in full each time, the model does see it on every turn:

history = [system_msg]          # ← financial data added once here

# Turn 1: history = [system_msg, user1]                            ✓ data present
# Turn 2: history = [system_msg, user1, asst1, user2]             ✓ data present  
# Turn 3: history = [system_msg, user1, asst1, user2, asst2, user3] ✓ data present
Since system_msg is always the first element in history, it's included in every apply_chat_template call. So the model sees the financial sheets on every single turn — it's just sent once in the list rather than duplicated.

"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

_client = openai.Client(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.svc.cscs.ch/v1",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODELS_DIR = Path(f"/cluster/scratch/{os.environ.get('USER', 'user')}/models")

DATASET_10Q = BASE_DIR / "10q" / "final_qa_dataset.json"
SHEET_10Q   = BASE_DIR / "10q" / "financial_spreadsheet.json"

DATASET_90Q = BASE_DIR / "dataset_output" / "original_questions.json"
SHEET_90Q   = BASE_DIR / "90q" / "financial_spreadsheet.json"

DATASET_MT  = BASE_DIR / "dataset_output" / "multi_turn_and_augmented_questions.json"
SHEET_MT    = BASE_DIR / "90q" / "financial_spreadsheet.json"  # same synthetic companies

DEFAULT_MODELS = ["swiss-ai/Apertus-70B-Instruct-2509",
                  "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                  "meta-llama/Llama-3.3-70B-Instruct",
                  "openai/gpt-oss-120b-evMj",
                  "deepseek-ai/deepseek-coder-33b-instruct",]


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


def find_company(question_text: str, sheet_lookup: dict) -> str | None:
    """Return the first company name from sheet_lookup that appears in question_text."""
    for name in sheet_lookup:
        if name in question_text:
            return name
    return None


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

def strip_thinking_tags(text: str) -> str:
    """Extract only the final message content, discarding thinking blocks and special tokens."""
    # The format is: <|channel|>analysis<|message|>THINKING<|end|><|start|>assistant<|channel|>final<|message|>ANSWER
    # Split on <|message|> and take the last segment as the final answer content
    parts = re.split(r'<\|message\|>', text)
    if len(parts) > 1:
        last = parts[-1]
        # Strip any trailing <|end|> or other special tokens
        last = re.sub(r'<\|[^|]*\|>', '', last)
        return last.strip()
    # Fallback: strip all special token blocks and standalone tokens
    text = re.sub(r'<\|(?!end\|)[^|]*\|>.*?<\|end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|[^|]*\|>', '', text)
    return text.strip()


def _sanitize_messages(messages: list) -> list:
    """Strip thinking tags from all assistant messages before sending to API."""
    sanitized = []
    for msg in messages:
        if msg.get("role") == "assistant":
            sanitized.append({**msg, "content": strip_thinking_tags(msg["content"])})
        else:
            sanitized.append(msg)
    return sanitized


def run_inference(model_name: str, prompt: str, max_new_tokens: int = 2048) -> str:
    """Call the API with a single user prompt and return the response text."""
    response = _client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Do not show your thinking process. Output only the answer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_new_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content.strip()


def run_multiturn_inference(model_name: str, messages: list, max_new_tokens: int = 2048) -> str:
    """Call the API with a full conversation history and return the response text."""
    response = _client.chat.completions.create(
        model=model_name,
        messages=_sanitize_messages(messages),
        max_tokens=max_new_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def evaluate_10q(
    questions: list,
    sheet_lookup: dict,
    model_name: str,
    limit: int | None,
    tol: float,
    max_new_tokens: int = 2048,
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

        response = run_inference(model_name, prompt, max_new_tokens)
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
    model_name: str,
    limit: int | None,
    tol: float,
    max_new_tokens: int = 2048,
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

        response = run_inference(model_name, prompt, max_new_tokens)
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


def evaluate_multiturn(
    questions: list,
    sheet_lookup: dict,
    model_name: str,
    limit: int | None,
    tol: float,
    max_new_tokens: int = 512,
) -> list:
    """
    Evaluate multi-turn questions.

    For each question the model plays through all conversation turns
    sequentially. The financial context is injected as a system message.
    Accuracy is measured on the final turn against q['answer'].
    """
    results = []
    subset = [q for q in questions if not q.get("skipped")]
    if limit:
        subset = subset[:limit]

    for i, q in enumerate(subset, 1):
        qid = q.get("question_id")
        original_q = q.get("original_question", "")
        ground_truth = q.get("answer")
        num_turns = q.get("num_turns", 1)
        depth = q.get("depth")

        # Build system message with financial context
        company = find_company(original_q, sheet_lookup)
        system_msg = None
        if company:
            sheet_rows = sheet_lookup.get(company)
            if sheet_rows:
                sheet_text = sheet_to_text_90q(sheet_rows)
                system_content = (
                    "You are a financial analyst. Answer using ONLY the data below. "
                    "No commentary.\n\n"
                    "=== DATA ===\n"
                    f"{sheet_text}\n"
                    "=== END ===\n\n"
                    "RULES:\n"
                    "- Extract only the numbers you need.\n"
                    "- Show calculations in 1-3 lines max if needed.\n"
                    "- Last line MUST be: Answer: <value>\n"
                    "- <value> is either a number, True, or False. Nothing else.\n"
                    "- Do NOT explain or add anything after the Answer line.\n"
                    "- Do NOT show your thinking process.\n"
                )
                system_msg = {"role": "system", "content": system_content}

        if company is None:
            print(f"  [{i}/{len(subset)}] SKIP (company not found in spreadsheet) | {original_q[:60]}...")
            results.append({
                "source": "mt",
                "id": qid,
                "entity": None,
                "depth": depth,
                "num_turns": num_turns,
                "question": original_q,
                "ground_truth": ground_truth,
                "llm_response": None,
                "predicted": None,
                "correct": False,
                "skip": True,
            })
            continue

        # Replay conversation turn by turn
        history = [system_msg] if system_msg else []
        final_response = None
        turn_responses = []        # raw model output per turn
        turn_questions = []        # user question text per turn
        turn_prompt_snapshots = [] # full messages list sent to the model at each turn

        for msg in q.get("messages", []):
            if msg["role"] == "user":
                question_text = msg["content"].strip()
                history.append({"role": "user", "content": question_text})
                # Snapshot the full context sent to the model (copy before adding response)
                turn_prompt_snapshots.append(list(history))
                response = run_multiturn_inference(model_name, history, max_new_tokens)
                history.append({"role": "assistant", "content": strip_thinking_tags(response)})
                turn_responses.append(response)
                turn_questions.append(question_text)
                final_response = response
            # assistant placeholder messages are skipped — we fill them ourselves

        predicted = extract_number(final_response) if final_response else None
        correct = is_correct(predicted, ground_truth, tol)

        # Per-turn correctness against intermediate ground truths
        turns_meta = q.get("turns", [])
        turn_correctness = []
        first_failure_depth = None
        for idx, tr in enumerate(turns_meta):
            tr_gt = tr.get("ground_truth")
            tr_response = turn_responses[idx] if idx < len(turn_responses) else None
            tr_question = turn_questions[idx] if idx < len(turn_questions) else tr.get("question", "")
            tr_prompt = turn_prompt_snapshots[idx] if idx < len(turn_prompt_snapshots) else []
            tr_pred = extract_number(tr_response) if tr_response else None
            tr_correct = is_correct(tr_pred, tr_gt, tol)
            turn_correctness.append({
                "turn_number": tr.get("turn_number"),
                "op": tr.get("op"),
                "question": tr_question,
                "prompt_messages": tr_prompt,  # full messages list sent to model at this turn
                "ground_truth": tr_gt,
                "predicted": tr_pred,
                "llm_response": tr_response,
                "correct": tr_correct,
                "is_final": tr.get("is_final", False),
            })
            if not tr_correct and first_failure_depth is None:
                first_failure_depth = tr.get("turn_number")

        status = "CORRECT" if correct else "WRONG "
        try:
            gt_display = f"{float(ground_truth):.4f}"
        except (TypeError, ValueError):
            gt_display = str(ground_truth)
        failure_info = f" first_fail@turn={first_failure_depth}" if not correct and num_turns > 1 else ""
        print(f"  [{i}/{len(subset)}] {status} | turns={num_turns} depth={depth}{failure_info} | "
              f"truth={gt_display} pred={predicted} | {original_q[:50]}...")

        results.append({
            "source": "mt",
            "id": qid,
            "entity": company,
            "depth": depth,
            "num_turns": num_turns,
            "question": original_q,
            "ground_truth": ground_truth,
            "turn_responses": turn_responses,
            "turn_correctness": turn_correctness,
            "first_failure_depth": first_failure_depth,
            "llm_response": final_response,
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
    # "recovered": final answer correct but had at least one wrong intermediate turn
    recovered = sum(
        1 for r in evaluated
        if r["correct"] and r.get("first_failure_depth") is not None
    )
    return {
        "total": len(evaluated),
        "skipped": len(results) - len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated),
        "recovered": recovered,
    }


def print_summary(model_name: str, results_10q: list, results_90q: list, results_mt: list) -> None:
    acc10 = compute_accuracy(results_10q)
    acc90 = compute_accuracy(results_90q)
    accmt = compute_accuracy(results_mt)
    total_eval = acc10["total"] + acc90["total"] + accmt["total"]
    total_correct = acc10["correct"] + acc90["correct"] + accmt["correct"]
    overall = total_correct / total_eval if total_eval else 0.0

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  10-Q      : {acc10['correct']}/{acc10['total']}  accuracy = {acc10['accuracy']:.1%}  (skipped {acc10.get('skipped',0)})")
    print(f"  90-Q      : {acc90['correct']}/{acc90['total']}  accuracy = {acc90['accuracy']:.1%}  (skipped {acc90.get('skipped',0)})")
    print(f"  Multi-turn: {accmt['correct']}/{accmt['total']}  accuracy = {accmt['accuracy']:.1%}  (skipped {accmt.get('skipped',0)})")

    # Break down multi-turn accuracy by number of turns + failure origin
    if results_mt:
        from collections import defaultdict
        by_turns: dict = defaultdict(list)
        for r in results_mt:
            if not r.get("skip"):
                by_turns[r.get("num_turns", "?")].append(r)

        total_recovered = sum(
            1 for r in results_mt
            if not r.get("skip") and r["correct"] and r.get("first_failure_depth") is not None
        )
        if total_recovered:
            print(f"    recovered (correct final, wrong intermediate): {total_recovered}")

        for nt in sorted(by_turns):
            rows = by_turns[nt]
            n_correct = sum(1 for r in rows if r["correct"])
            n_recovered = sum(
                1 for r in rows
                if r["correct"] and r.get("first_failure_depth") is not None
            )
            recovered_str = f"  [{n_recovered} recovered]" if n_recovered else ""
            print(f"    turns={nt}: {n_correct}/{len(rows)}  accuracy = {n_correct/len(rows):.1%}{recovered_str}")
            # For multi-turn questions, show where failures first occur
            if nt > 1:
                failed = [r for r in rows if not r["correct"]]
                if failed:
                    fail_counts: dict = defaultdict(int)
                    for r in failed:
                        fd = r.get("first_failure_depth")
                        fail_counts[fd if fd is not None else "?"] += 1
                    parts = [f"turn {fd}: {cnt}" for fd, cnt in sorted(fail_counts.items(), key=lambda x: (x[0] is None, x[0]))]
                    print(f"      first failure at — {', '.join(parts)}  (of {len(failed)} failed)")

    print(f"  Overall   : {total_correct}/{total_eval}  accuracy = {overall:.1%}")
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
            for dataset_key in ("10q", "90q", "mt"):
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
        "--max-new-tokens", type=int, default=2048,
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
        "--datasets", nargs="+", choices=["10q", "90q", "mt"], default=["10q", "90q", "mt"],
        help="Which dataset(s) to evaluate: 10q, 90q, mt, or any combination (default: all)",
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

    if "mt" in args.datasets:
        questions_mt = load_json(DATASET_MT)
        sheet_lookup_mt = build_90q_sheet_lookup(load_json(SHEET_MT))
        non_skipped = sum(1 for q in questions_mt if not q.get("skipped"))
        print(f"  Multi-turn: {len(questions_mt)} total, {non_skipped} non-skipped, {len(sheet_lookup_mt)} companies")
    else:
        questions_mt, sheet_lookup_mt = [], {}

    all_results = {}

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  Evaluating model: {model_name}")
        print(f"{'='*60}")

        if "10q" in args.datasets:
            print(f"\n-- 10-Q evaluation ({args.limit or len(questions_10q)} questions) --")
            results_10q = evaluate_10q(
                questions_10q, sheet_lookup_10q, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
            )
        else:
            results_10q = []

        if "90q" in args.datasets:
            print(f"\n-- 90-Q evaluation ({args.limit or len(questions_90q)} questions) --")
            results_90q = evaluate_90q(
                questions_90q, sheet_lookup_90q, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
            )
        else:
            results_90q = []

        if "mt" in args.datasets:
            non_skipped = sum(1 for q in questions_mt if not q.get("skipped"))
            print(f"\n-- Multi-turn evaluation ({args.limit or non_skipped} questions) --")
            results_mt = evaluate_multiturn(
                questions_mt, sheet_lookup_mt, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
            )
        else:
            results_mt = []

        print_summary(model_name, results_10q, results_90q, results_mt)

        all_results[model_name] = {
            "10q": results_10q,
            "90q": results_90q,
            "mt": results_mt,
            "accuracy_10q": compute_accuracy(results_10q),
            "accuracy_90q": compute_accuracy(results_90q),
            "accuracy_mt": compute_accuracy(results_mt),
        }

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
        print(f"{'Model':<20} {'10-Q Acc':>10} {'90-Q Acc':>10} {'MT Acc':>10} {'Overall':>10}")
        print("-" * 62)
        for m, r in all_results.items():
            a10 = r["accuracy_10q"]["accuracy"]
            a90 = r["accuracy_90q"]["accuracy"]
            amt = r["accuracy_mt"]["accuracy"]
            t10 = r["accuracy_10q"]["total"]
            t90 = r["accuracy_90q"]["total"]
            tmt = r["accuracy_mt"]["total"]
            total = t10 + t90 + tmt
            correct = r["accuracy_10q"]["correct"] + r["accuracy_90q"]["correct"] + r["accuracy_mt"]["correct"]
            overall = correct / total if total else 0.0
            print(f"{m:<20} {a10:>10.1%} {a90:>10.1%} {amt:>10.1%} {overall:>10.1%}")


if __name__ == "__main__":
    main()
