"""
Multi-turn question generator for financial expression trees.

Processes a full dataset JSON file and generates ordered turn chains
for every record with bound_expression_depth >= 1.

Usage:
    python multiturn_generator.py --input dataset.json --output turns.json
    python multiturn_generator.py --input dataset.json --output turns.json --sheet sheet.txt
"""

import json
import re
import argparse
from dataclasses import dataclass, asdict


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
    derived_name: str = ""   # set when this node expands a named derived concept


@dataclass
class Turn:
    turn_number: int
    op: str
    question: str
    ground_truth: float
    is_final: bool = False


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TEMPLATES = {
    "max":     lambda c, e, y1, y2: f"What is the maximum of {c} for {e} from {y1} to {y2}?",
    "min":     lambda c, e, y1, y2: f"What is the minimum of {c} for {e} from {y1} to {y2}?",
    "average": lambda c, e, y1, y2: f"What is the average of {c} for {e} from {y1} to {y2}?",
    "avg":     lambda c, e, y1, y2: f"What is the average of {c} for {e} from {y1} to {y2}?",
    "growth":  lambda c, e, y1, y2: f"What is the growth of {c} for {e} from {y1} to {y2}?",
    "sum":      lambda c, e, y1, y2: f"What is the sum of {c} for {e} from {y1} to {y2}?",
    "diff":     lambda c, e, y1, y2: f"What is the difference of {c} for {e} from {y1} to {y2}?",
    "ratio":    lambda c, e, y1, y2: f"What is the ratio of {c} for {e} from {y1} to {y2}?",
    "add":      lambda c, e, y1, y2: f"What is the sum of {c} for {e} from {y1} to {y2}?",
    "subtract": lambda c, e, y1, y2: f"What is the difference of {c} for {e} from {y1} to {y2}?",
    "divide":   lambda c, e, y1, y2: f"What is the ratio of {c} for {e} from {y1} to {y2}?",
    "mul":      lambda c, e, y1, y2: f"What is the product of {c} for {e} from {y1} to {y2}?",
}

CROSS_OPS = {"mul", "add", "subtract", "divide", "sum", "diff", "ratio"}


# ---------------------------------------------------------------------------
# Parser — build tree from expression_json (handles literals, derived concepts)
# ---------------------------------------------------------------------------

def build_tree_from_json(obj: dict):
    """Recursively convert expression_json dict to Node/Leaf."""
    if "leaf" in obj:
        return Leaf(key=obj["leaf"], concept="", entity="", period="", value=0.0)
    if "literal" in obj:
        # Numeric literal — store value directly on the Leaf
        v = float(obj["literal"])
        return Leaf(key=f"__literal_{v}__", concept="", entity="", period="__literal__", value=v)
    if "derived" in obj:
        # Expand derived concept inline but preserve its name on the node
        node = build_tree_from_json(obj["expanded"])
        if isinstance(node, Node):
            node.derived_name = obj["derived"]
        return node
    if "op" in obj:
        # Binary form (left/right) — used by all current expression_json nodes
        if "left" in obj and "right" in obj:
            left  = build_tree_from_json(obj["left"])
            right = build_tree_from_json(obj["right"])
            return Node(op=obj["op"], args=[left, right])
        # N-ary form (args list) — used by derived concept formulas (e.g. current_assets)
        if "args" in obj:
            args = [build_tree_from_json(a) if isinstance(a, dict) else
                    Leaf(key=a, concept="", entity="", period="", value=0.0)
                    for a in obj["args"]]
            return Node(op=obj["op"], args=args)
    raise ValueError(f"Unrecognised expression_json node: {obj}")


# ---------------------------------------------------------------------------
# Leaf helpers
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
        # Literal nodes carry their value directly; named leaves use the map
        if node.period == "__literal__":
            return node.value
        return leaf_map[node.key].value

    vals = [compute(a, leaf_map) for a in node.args]

    if node.op == "max":                return max(vals)
    if node.op == "min":                return min(vals)
    if node.op in ("average", "avg"):   return sum(vals) / len(vals)
    if node.op == "growth":             return (vals[1] - vals[0]) / vals[0]
    if node.op == "mul":      return vals[0] * vals[1]
    if node.op == "add":      return vals[0] + vals[1]
    if node.op == "subtract": return vals[0] - vals[1]
    if node.op == "divide":   return vals[0] / vals[1]
    if node.op == "sum":      return vals[0] + vals[1]
    if node.op == "diff":     return vals[0] - vals[1]
    if node.op == "ratio":    return vals[0] / vals[1]

    raise ValueError(f"Unknown op: {node.op}")


def get_years(node, leaf_map: dict[str, Leaf]) -> list[int]:
    if isinstance(node, Leaf):
        if node.period == "__literal__":
            return []  # numeric literals have no associated year
        return [int(leaf_map[node.key].period)]
    return [y for a in node.args for y in get_years(a, leaf_map)]


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
        # Suppress intermediate turns for nested same-op min/max chains
        # e.g. min(min(2021,2022), 2023) → only one turn for the outermost min
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

    years   = sorted(set(get_years(node, leaf_map)))
    concept = get_concept(node, leaf_map)
    entity  = get_entity(node, leaf_map)
    gt      = compute(node, leaf_map)

    if node.derived_name:
        label = node.derived_name.replace("_", " ")
        question = f"What is {label} for {entity} in {years[0]}?"
    elif node.op in TEMPLATES:
        question = TEMPLATES[node.op](concept, entity, years[0], years[-1])
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
    depth = int(record.get("bound_expression_depth", 1))
    if depth < 1:
        return []

    leaf_map          = build_leaf_map(record)
    expr_json         = json.loads(record["expression_json"])
    tree              = build_tree_from_json(expr_json)
    original_question = record["question"]

    turns = []
    collect_turns(tree, leaf_map, original_question, turns, is_root=True)
    return turns


# ---------------------------------------------------------------------------
# Message array builder (strict mode)
# ---------------------------------------------------------------------------

def build_messages(record: dict, financial_sheet: str) -> list[dict]:
    turns = generate_turns(record)

    if not turns:
        return [{"role": "user", "content": f"{financial_sheet}\n\n{record['question']}"}]

    messages = []
    for i, turn in enumerate(turns):
        content = f"{financial_sheet}\n\n{turn.question}" if i == 0 else turn.question
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": "{{FILL_MODEL_RESPONSE}}"})

    return messages


# ---------------------------------------------------------------------------
# Dataset processor
# ---------------------------------------------------------------------------

def process_dataset(records: list[dict], financial_sheet: str = "") -> list[dict]:
    """
    Process every record in the dataset.

    Each output item contains:
      - question_id        : from the record
      - original_question  : the full compound question
      - answer             : final ground truth
      - depth              : bound_expression_depth
      - num_turns          : number of turns generated
      - turns              : list of {turn_number, op, question, ground_truth, is_final}
      - messages           : full message array ready for LLM eval (strict mode)
      - skipped            : True if depth < 1 (one-shot fallback)
    """
    results = []

    for record in records:
        question_id = record.get("question_id", "unknown")
        depth       = int(record.get("bound_expression_depth", 1))

        # depth < 1 → one-shot, no multi-turn needed
        if depth < 1:
            results.append({
                "question_id":       question_id,
                "original_question": record["question"],
                "answer":            record.get("answer"),
                "depth":             depth,
                "num_turns":         1,
                "turns":             [],
                "messages":          build_messages(record, financial_sheet),
                "skipped":           True,
            })
            continue

        try:
            turns    = generate_turns(record)
            messages = build_messages(record, financial_sheet)

            results.append({
                "question_id":       question_id,
                "original_question": record["question"],
                "answer":            record.get("answer"),
                "depth":             depth,
                "num_turns":         len(turns),
                "turns":             [asdict(t) for t in turns],
                "messages":          messages,
                "skipped":           False,
            })

        except Exception as e:
            results.append({
                "question_id": question_id,
                "error":       str(e),
                "skipped":     True,
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn chains from a financial dataset JSON file."
    )
    parser.add_argument("--input",  required=True, help="Path to input dataset JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--sheet",  default="",    help="Path to financial sheet text file (optional)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = json.load(f)

    # support both a list of records or a single record dict
    if isinstance(records, dict):
        records = [records]

    financial_sheet = ""
    if args.sheet:
        with open(args.sheet, "r") as f:
            financial_sheet = f.read()

    results = process_dataset(records, financial_sheet)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # print summary
    total     = len(results)
    skipped   = sum(1 for r in results if r.get("skipped") and "error" not in r)
    errors    = sum(1 for r in results if "error" in r)
    multi     = total - skipped - errors
    avg_turns = (
        sum(r["num_turns"] for r in results if not r.get("skipped") and "error" not in r) / multi
        if multi > 0 else 0
    )

    print(f"\nProcessed : {args.input}")
    print(f"  Total    : {total}")
    print(f"  Multi-turn: {multi}  (avg {avg_turns:.1f} turns each)")
    print(f"  One-shot : {skipped}  (depth < 1, skipped)")
    print(f"  Errors   : {errors}")
    print(f"\nOutput    : {args.output}")


if __name__ == "__main__":
    main()