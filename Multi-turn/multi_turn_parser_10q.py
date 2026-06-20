"""
Multi-turn question generator for 10Q financial expression trees.

Adapted from multi_turn_parser.py for the 10Q compiler pipeline:
- Reads a CSV (same leaf_N_* format as random_questions_10q.csv)
- Handles quarterly period strings ("Quarter Ended March 31, 2023") instead of int years
- Does NOT embed financial sheet text in messages — the eval script injects
  the atom context as a system message at runtime.

Usage:
    python multi_turn_parser_10q.py \
        --input  ../compiler_pipeline_refactored_10Q/output/random_questions_10q.csv \
        --output ../compiler_pipeline_refactored_10Q/output/multi_turn_10q.json
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Leaf:
    key: str
    concept: str
    entity: str
    period: str
    value: float


@dataclass
class Node:
    op: str
    args: list
    derived_name: str = ""


@dataclass
class Turn:
    turn_number: int
    op: str
    question: str
    ground_truth: float
    is_final: bool = False


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def parse_period_date(period: str) -> date:
    """Parse a period string to a date for chronological sorting."""
    date_str = period.split("Ended ")[-1].strip() if "Ended " in period else period.strip()
    for fmt in ("%B %d, %Y", "%B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    return date.min


# ---------------------------------------------------------------------------
# Templates — use period strings instead of int years
# ---------------------------------------------------------------------------

TEMPLATES = {
    "max":     lambda c, e, p1, p2: f"What is the maximum of {c} for {e} from {p1} to {p2}?",
    "min":     lambda c, e, p1, p2: f"What is the minimum of {c} for {e} from {p1} to {p2}?",
    "average": lambda c, e, p1, p2: f"What is the average of {c} for {e} from {p1} to {p2}?",
    "avg":     lambda c, e, p1, p2: f"What is the average of {c} for {e} from {p1} to {p2}?",
    "growth":  lambda c, e, p1, p2: f"What is the growth of {c} for {e} from {p1} to {p2}?",
    "sum":     lambda c, e, p1, p2: f"What is the sum of {c} for {e} from {p1} to {p2}?",
    "diff":    lambda c, e, p1, p2: f"What is the difference of {c} for {e} from {p1} to {p2}?",
    "ratio":   lambda c, e, p1, p2: f"What is the ratio of {c} for {e} from {p1} to {p2}?",
    "mul":     lambda c, e, p1, p2: f"What is the product of {c} for {e} from {p1} to {p2}?",
}

CROSS_OPS = {"mul", "add", "subtract", "divide", "sum", "diff", "ratio"}


# ---------------------------------------------------------------------------
# Tree parsing
# ---------------------------------------------------------------------------

def build_tree_from_json(obj: dict):
    if "leaf" in obj:
        return Leaf(key=obj["leaf"], concept="", entity="", period="", value=0.0)
    if "literal" in obj:
        v = float(obj["literal"])
        return Leaf(key=f"__literal_{v}__", concept="", entity="", period="__literal__", value=v)
    if "derived" in obj:
        node = build_tree_from_json(obj["expanded"])
        if isinstance(node, Node):
            node.derived_name = obj["derived"]
        return node
    if "op" in obj:
        if "left" in obj and "right" in obj:
            return Node(op=obj["op"], args=[
                build_tree_from_json(obj["left"]),
                build_tree_from_json(obj["right"]),
            ])
        if "args" in obj:
            args = [
                build_tree_from_json(a) if isinstance(a, dict)
                else Leaf(key=a, concept="", entity="", period="", value=0.0)
                for a in obj["args"]
            ]
            return Node(op=obj["op"], args=args)
    raise ValueError(f"Unrecognised expression_json node: {obj}")


# ---------------------------------------------------------------------------
# Leaf map
# ---------------------------------------------------------------------------

def build_leaf_map(record: dict) -> dict[str, Leaf]:
    leaves = {}
    i = 1
    while f"leaf_{i}_key" in record:
        key = record[f"leaf_{i}_key"]
        if not key:
            i += 1
            continue
        leaves[key] = Leaf(
            key=key,
            concept=record.get(f"leaf_{i}_label", ""),
            entity=record.get(f"leaf_{i}_entity", ""),
            period=record.get(f"leaf_{i}_period", ""),
            value=float(record.get(f"leaf_{i}_value", 0)),
        )
        i += 1
    return leaves


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def compute(node, leaf_map: dict[str, Leaf]) -> float:
    if isinstance(node, Leaf):
        if node.period == "__literal__":
            return node.value
        return leaf_map[node.key].value

    vals = [compute(a, leaf_map) for a in node.args]
    if node.op == "max":                  return max(vals)
    if node.op == "min":                  return min(vals)
    if node.op in ("average", "avg"):     return sum(vals) / len(vals)
    if node.op == "growth":               return (vals[1] - vals[0]) / vals[0]
    if node.op in ("mul",):               return vals[0] * vals[1]
    if node.op in ("sum", "add"):         return vals[0] + vals[1]
    if node.op in ("diff", "subtract"):   return vals[0] - vals[1]
    if node.op in ("ratio", "divide"):    return vals[0] / vals[1]
    raise ValueError(f"Unknown op: {node.op}")


def get_periods(node, leaf_map: dict[str, Leaf]) -> list[str]:
    if isinstance(node, Leaf):
        if node.period == "__literal__":
            return []
        return [leaf_map[node.key].period]
    return [p for a in node.args for p in get_periods(a, leaf_map)]


def get_concept(node, leaf_map: dict[str, Leaf]) -> str:
    if isinstance(node, Leaf):
        return leaf_map[node.key].concept
    for a in node.args:
        c = get_concept(a, leaf_map)
        if c:
            return c
    return ""


def get_entity(node, leaf_map: dict[str, Leaf]) -> str:
    if isinstance(node, Leaf):
        return leaf_map[node.key].entity
    for a in node.args:
        e = get_entity(a, leaf_map)
        if e:
            return e
    return ""


# ---------------------------------------------------------------------------
# Turn generation
# ---------------------------------------------------------------------------

def collect_turns(node, leaf_map, original_question, turns, is_root=False, skip_turn=False):
    if isinstance(node, Leaf):
        return

    for arg in node.args:
        child_skip = (
            node.op in ("min", "max")
            and isinstance(arg, Node)
            and arg.op == node.op
        )
        collect_turns(arg, leaf_map, original_question, turns, is_root=False, skip_turn=child_skip)

    if skip_turn:
        return

    if is_root and node.op in CROSS_OPS:
        turns.append(Turn(
            turn_number=len(turns) + 1,
            op=node.op,
            question=original_question,
            ground_truth=compute(node, leaf_map),
            is_final=True,
        ))
        return

    all_periods = get_periods(node, leaf_map)
    periods_sorted = sorted(set(all_periods), key=parse_period_date)
    concept = get_concept(node, leaf_map)
    entity  = get_entity(node, leaf_map)
    gt      = compute(node, leaf_map)
    p1 = periods_sorted[0]  if periods_sorted else ""
    p2 = periods_sorted[-1] if periods_sorted else ""

    if node.derived_name:
        label = node.derived_name.replace("_", " ")
        question = f"What is {label} for {entity} in {p1}?"
    elif node.op in TEMPLATES:
        question = TEMPLATES[node.op](concept, entity, p1, p2)
    elif node.op in CROSS_OPS:
        question = original_question
    else:
        raise ValueError(f"No template for op: {node.op}")

    turns.append(Turn(
        turn_number=len(turns) + 1,
        op=node.op,
        question=question,
        ground_truth=gt,
        is_final=is_root,
    ))


def generate_turns(record: dict) -> list[Turn]:
    depth = int(record.get("bound_expression_depth", record.get("depth", 1)))
    if depth < 1:
        return []
    leaf_map = build_leaf_map(record)
    tree = build_tree_from_json(json.loads(record["expression_json"]))
    turns: list[Turn] = []
    collect_turns(tree, leaf_map, record["question"], turns, is_root=True)
    return turns


# ---------------------------------------------------------------------------
# Message builder — no sheet embedded; eval injects atom context at runtime
# ---------------------------------------------------------------------------

def build_messages(turns: list[Turn], original_question: str) -> list[dict]:
    if not turns:
        return [
            {"role": "user",      "content": original_question},
            {"role": "assistant", "content": "{{FILL_MODEL_RESPONSE}}"},
        ]
    messages = []
    for turn in turns:
        messages.append({"role": "user",      "content": turn.question})
        messages.append({"role": "assistant", "content": "{{FILL_MODEL_RESPONSE}}"})
    return messages


# ---------------------------------------------------------------------------
# Dataset processor
# ---------------------------------------------------------------------------

def process_dataset(records: list[dict]) -> list[dict]:
    results = []
    for record in records:
        question_id = record.get("question_id", "unknown")
        depth  = int(record.get("bound_expression_depth", record.get("depth", 1)))
        entity = record.get("leaf_1_entity", "")

        if depth < 1:
            results.append({
                "question_id":       question_id,
                "original_question": record["question"],
                "answer":            record.get("answer"),
                "depth":             depth,
                "entity":            entity,
                "num_turns":         1,
                "turns":             [],
                "messages":          build_messages([], record["question"]),
                "skipped":           True,
            })
            continue

        try:
            turns = generate_turns(record)
            results.append({
                "question_id":       question_id,
                "original_question": record["question"],
                "answer":            record.get("answer"),
                "depth":             depth,
                "entity":            entity,
                "num_turns":         len(turns),
                "turns":             [asdict(t) for t in turns],
                "messages":          build_messages(turns, record["question"]),
                "skipped":           False,
            })
        except Exception as e:
            results.append({
                "question_id": question_id,
                "error":       str(e),
                "entity":      entity,
                "skipped":     True,
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn chains from 10Q questions CSV."
    )
    parser.add_argument("--input",  required=True, help="Path to 10Q questions CSV")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    results = process_dataset(records)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total    = len(results)
    skipped  = sum(1 for r in results if r.get("skipped") and "error" not in r)
    errors   = sum(1 for r in results if "error" in r)
    multi    = total - skipped - errors
    avg_turns = (
        sum(r["num_turns"] for r in results if not r.get("skipped") and "error" not in r) / multi
        if multi > 0 else 0
    )

    print(f"\nProcessed : {args.input}")
    print(f"  Total    : {total}")
    print(f"  Multi-turn: {multi}  (avg {avg_turns:.1f} turns)")
    print(f"  One-shot  : {skipped}  (depth < 1, skipped)")
    print(f"  Errors    : {errors}")
    print(f"\nOutput    : {args.output}")


if __name__ == "__main__":
    main()
