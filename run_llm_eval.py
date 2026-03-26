"""
Run LLM evaluation on the final QA dataset.

Usage:
    python run_llm_eval.py --model /cluster/scratch/$USER/models/Qwen3.5-4B
    python run_llm_eval.py --model /cluster/scratch/$USER/models/Qwen3.5-9B
    python run_llm_eval.py --model /cluster/scratch/$USER/models/Qwen3.5-4B --limit 50
"""

import argparse
import json
import os
import re
import sys
import time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    return parser.parse_args()


def load_dataset(limit=None):
    path = os.path.join(OUTPUT_DIR, "final_qa_dataset.json")
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def build_prompt(question, result_type):
    """Build a concise prompt for extracting a factual answer."""
    if result_type == "boolean":
        instruction = "Answer with only 'Yes' or 'No'."
    elif result_type in ("numeric", "percentage", "ratio", "count"):
        instruction = "Answer with only the numeric value (no units, no explanation)."
    else:
        instruction = "Answer as concisely as possible."

    return (
        f"You are a financial data assistant. Answer the following question using only the given instruction.\n"
        f"Instruction: {instruction}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def extract_answer(raw_text, result_type):
    """Parse the model's raw output into a comparable value."""
    text = raw_text.strip()

    if result_type == "boolean":
        lower = text.lower()
        if lower.startswith("yes"):
            return "Yes"
        if lower.startswith("no"):
            return "No"
        return text

    # Try to extract a number
    # Remove commas and percent signs, then find the first number
    text_clean = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text_clean)
    if match:
        num_str = match.group()
        try:
            val = float(num_str)
            if val == int(val) and result_type == "count":
                return int(val)
            return val
        except ValueError:
            pass

    return text  # fallback: return raw text


def is_correct(predicted, ground_truth_formatted, result_type, answer_raw):
    """Check if the predicted answer matches ground truth."""
    if result_type == "boolean":
        return str(predicted).strip().lower() == str(ground_truth_formatted).strip().lower()

    # Numeric types: allow small relative tolerance
    try:
        pred_num = float(str(predicted).replace(",", ""))
        true_num = float(str(answer_raw))

        if true_num == 0:
            return abs(pred_num) < 1e-6

        rel_error = abs(pred_num - true_num) / abs(true_num)
        return rel_error <= 0.01  # 1% tolerance
    except (ValueError, TypeError):
        pass

    # String / fallback: exact match (case-insensitive)
    return str(predicted).strip().lower() == str(ground_truth_formatted).strip().lower()


def run_evaluation(model_path, dataset, batch_size, max_new_tokens):
    """Load model and run inference over the dataset."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = os.path.basename(model_path.rstrip("/"))
    print(f"\nLoading model: {model_name}")
    print(f"Path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded. Running on {next(model.parameters()).device}")
    print(f"Evaluating {len(dataset)} questions (batch_size={batch_size})...\n")

    results = []
    correct = 0
    start_time = time.time()

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [build_prompt(item["question"], item["result_type"]) for item in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        input_lengths = inputs["input_ids"].shape[1]
        for j, item in enumerate(batch):
            generated_ids = outputs[j][input_lengths:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted = extract_answer(raw_text, item["result_type"])
            ok = is_correct(predicted, item["answer_formatted"], item["result_type"], item["answer"])

            if ok:
                correct += 1

            results.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["answer_formatted"],
                "predicted_raw": raw_text.strip(),
                "predicted_parsed": str(predicted),
                "correct": ok,
                "result_type": item["result_type"],
                "operation": item["operation"],
                "depth": item["depth"],
            })

        done = min(i + batch_size, len(dataset))
        acc = correct / done * 100
        elapsed = time.time() - start_time
        print(f"  [{done:>4}/{len(dataset)}]  acc={acc:.1f}%  elapsed={elapsed:.0f}s", end="\r")

    print()  # newline after progress line

    accuracy = correct / len(dataset) * 100
    elapsed = time.time() - start_time

    print(f"\n{'='*55}")
    print(f"  Model: {model_name}")
    print(f"  Total questions : {len(dataset)}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy:.2f}%")
    print(f"  Time            : {elapsed:.1f}s")
    print(f"{'='*55}\n")

    # Breakdown by result_type
    from collections import defaultdict
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_depth = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_type[r["result_type"]]["total"] += 1
        by_depth[r["depth"]]["total"] += 1
        if r["correct"]:
            by_type[r["result_type"]]["correct"] += 1
            by_depth[r["depth"]]["correct"] += 1

    print("  Accuracy by result type:")
    for t, v in sorted(by_type.items()):
        print(f"    {t:<15}: {v['correct']}/{v['total']} = {v['correct']/v['total']*100:.1f}%")

    print("\n  Accuracy by depth:")
    for d, v in sorted(by_depth.items()):
        print(f"    Depth {d}        : {v['correct']}/{v['total']} = {v['correct']/v['total']*100:.1f}%")

    return results, accuracy, model_name


def save_results(results, accuracy, model_name):
    out = {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "predictions": results,
    }
    out_path = os.path.join(OUTPUT_DIR, f"eval_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
    return out_path


def main():
    args = parse_args()
    dataset = load_dataset(args.limit)
    print(f"Loaded {len(dataset)} QA pairs from final_qa_dataset.json")

    results, accuracy, model_name = run_evaluation(
        args.model, dataset, args.batch_size, args.max_new_tokens
    )
    save_results(results, accuracy, model_name)


if __name__ == "__main__":
    main()
