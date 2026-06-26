"""
Compare evaluation results from two models and report accuracy.

Usage:
    python compare_results.py \
        --model_a output/eval_Qwen3.5-4B.json \
        --model_b output/eval_Qwen3.5-9B.json

Run this after both run_llm_eval.py jobs finish.
"""

import argparse
import json
import os
from collections import defaultdict


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_a",
        default=os.path.join(OUTPUT_DIR, "eval_Qwen3.5-4B.json"),
        help="Path to first model's eval JSON",
    )
    parser.add_argument(
        "--model_b",
        default=os.path.join(OUTPUT_DIR, "eval_Qwen3.5-9B.json"),
        help="Path to second model's eval JSON",
    )
    return parser.parse_args()


def load_eval(path):
    with open(path) as f:
        return json.load(f)


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def compare(eval_a, eval_b):
    name_a = eval_a["model"]
    name_b = eval_b["model"]

    preds_a = {r["id"]: r for r in eval_a["predictions"]}
    preds_b = {r["id"]: r for r in eval_b["predictions"]}
    common_ids = sorted(set(preds_a) & set(preds_b))

    total = len(common_ids)
    both_correct = 0
    only_a = 0
    only_b = 0
    both_wrong = 0
    same_answer = 0  # models gave same prediction (regardless of correctness)

    by_type_a = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type_b = defaultdict(lambda: {"correct": 0, "total": 0})
    by_depth_a = defaultdict(lambda: {"correct": 0, "total": 0})
    by_depth_b = defaultdict(lambda: {"correct": 0, "total": 0})

    disagreements = []

    for qid in common_ids:
        ra = preds_a[qid]
        rb = preds_b[qid]
        rt = ra["result_type"]
        depth = ra["depth"]

        by_type_a[rt]["total"] += 1
        by_type_b[rt]["total"] += 1
        by_depth_a[depth]["total"] += 1
        by_depth_b[depth]["total"] += 1

        if ra["correct"]:
            by_type_a[rt]["correct"] += 1
            by_depth_a[depth]["correct"] += 1
        if rb["correct"]:
            by_type_b[rt]["correct"] += 1
            by_depth_b[depth]["correct"] += 1

        if ra["correct"] and rb["correct"]:
            both_correct += 1
        elif ra["correct"] and not rb["correct"]:
            only_a += 1
        elif not ra["correct"] and rb["correct"]:
            only_b += 1
        else:
            both_wrong += 1

        if ra["predicted_parsed"].strip().lower() == rb["predicted_parsed"].strip().lower():
            same_answer += 1
        else:
            disagreements.append({
                "id": qid,
                "question": ra["question"],
                "ground_truth": ra["ground_truth"],
                f"{name_a}_pred": ra["predicted_parsed"],
                f"{name_b}_pred": rb["predicted_parsed"],
                f"{name_a}_correct": ra["correct"],
                f"{name_b}_correct": rb["correct"],
                "result_type": rt,
                "depth": depth,
            })

    # ── Overall accuracy ────────────────────────────────────────
    print_header("OVERALL ACCURACY")
    print(f"  Questions evaluated : {total}")
    print(f"  {name_a:<25}: {eval_a['correct']}/{eval_a['total']}  =  {eval_a['accuracy']:.2f}%")
    print(f"  {name_b:<25}: {eval_b['correct']}/{eval_b['total']}  =  {eval_b['accuracy']:.2f}%")

    diff = eval_b["accuracy"] - eval_a["accuracy"]
    winner = name_b if diff > 0 else name_a
    print(f"\n  Difference          : {abs(diff):.2f}pp  ({winner} is better)")

    # ── Agreement matrix ────────────────────────────────────────
    print_header("AGREEMENT MATRIX (on common questions)")
    print(f"  Both correct        : {both_correct:>4}  ({both_correct/total*100:.1f}%)")
    print(f"  Only {name_a:<15} : {only_a:>4}  ({only_a/total*100:.1f}%)")
    print(f"  Only {name_b:<15} : {only_b:>4}  ({only_b/total*100:.1f}%)")
    print(f"  Both wrong          : {both_wrong:>4}  ({both_wrong/total*100:.1f}%)")
    print(f"\n  Same prediction     : {same_answer:>4}  ({same_answer/total*100:.1f}%)")
    print(f"  Different prediction: {len(disagreements):>4}  ({len(disagreements)/total*100:.1f}%)")

    # ── Accuracy by result type ─────────────────────────────────
    print_header("ACCURACY BY RESULT TYPE")
    all_types = sorted(set(by_type_a) | set(by_type_b))
    header = f"  {'Type':<15}  {'Model A':>12}  {'Model B':>12}"
    print(header)
    print(f"  {'-'*15}  {'-'*12}  {'-'*12}")
    for t in all_types:
        va = by_type_a[t]
        vb = by_type_b[t]
        acc_a = va["correct"] / va["total"] * 100 if va["total"] else 0
        acc_b = vb["correct"] / vb["total"] * 100 if vb["total"] else 0
        print(f"  {t:<15}  {acc_a:>10.1f}%  {acc_b:>10.1f}%")

    # ── Accuracy by depth ───────────────────────────────────────
    print_header("ACCURACY BY DEPTH")
    depth_labels = {0: "Lookup", 1: "Single Op", 2: "Composed"}
    print(f"  {'Depth':<12}  {'Model A':>12}  {'Model B':>12}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}")
    for d in sorted(set(by_depth_a) | set(by_depth_b)):
        va = by_depth_a[d]
        vb = by_depth_b[d]
        acc_a = va["correct"] / va["total"] * 100 if va["total"] else 0
        acc_b = vb["correct"] / vb["total"] * 100 if vb["total"] else 0
        label = f"Depth {d} ({depth_labels.get(d, '?')})"
        print(f"  {label:<22}  {acc_a:>10.1f}%  {acc_b:>10.1f}%")

    # ── Sample disagreements ─────────────────────────────────────
    print_header(f"SAMPLE DISAGREEMENTS (first 10 of {len(disagreements)})")
    for dis in disagreements[:10]:
        print(f"\n  [{dis['id']}] depth={dis['depth']} type={dis['result_type']}")
        print(f"  Q: {dis['question'][:90]}...")
        print(f"  Truth  : {dis['ground_truth']}")
        print(f"  {name_a}: {dis[f'{name_a}_pred']}  ({'✓' if dis[f'{name_a}_correct'] else '✗'})")
        print(f"  {name_b}: {dis[f'{name_b}_pred']}  ({'✓' if dis[f'{name_b}_correct'] else '✗'})")

    # ── Save comparison report ───────────────────────────────────
    report = {
        "model_a": name_a,
        "model_b": name_b,
        "total_common": total,
        "accuracy": {
            name_a: eval_a["accuracy"],
            name_b: eval_b["accuracy"],
        },
        "agreement": {
            "both_correct": both_correct,
            "only_model_a": only_a,
            "only_model_b": only_b,
            "both_wrong": both_wrong,
            "same_prediction_pct": round(same_answer / total * 100, 2),
        },
        "by_result_type": {
            t: {
                name_a: round(by_type_a[t]["correct"] / by_type_a[t]["total"] * 100, 2) if by_type_a[t]["total"] else None,
                name_b: round(by_type_b[t]["correct"] / by_type_b[t]["total"] * 100, 2) if by_type_b[t]["total"] else None,
            }
            for t in all_types
        },
        "disagreements": disagreements,
    }

    out_path = os.path.join(OUTPUT_DIR, "comparison_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n\nFull comparison report saved to: {out_path}")


def main():
    args = parse_args()

    for path in [args.model_a, args.model_b]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            print("Run run_llm_eval.py for both models first.")
            return

    eval_a = load_eval(args.model_a)
    eval_b = load_eval(args.model_b)

    print(f"Loaded: {eval_a['model']} ({eval_a['total']} predictions)")
    print(f"Loaded: {eval_b['model']} ({eval_b['total']} predictions)")

    compare(eval_a, eval_b)


if __name__ == "__main__":
    main()
