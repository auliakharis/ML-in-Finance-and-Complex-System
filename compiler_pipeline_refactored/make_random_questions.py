from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from evaluator import Evaluator
from question_renderer import QuestionRenderer
from semantic_analyzer import SemanticAnalyzer
from tree import Atom, BindEnv, DERIVED_CONCEPTS, Expr, Store

T = TypeVar("T")


def validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be > 0.")
    if args.depth_min < 0 or args.depth_max < 0:
        raise ValueError("--depth-min/--depth-max must be >= 0.")
    if args.depth_min > args.depth_max:
        raise ValueError("--depth-min must be <= --depth-max.")
    if not (0.0 <= args.derived_prob_min <= 1.0 and 0.0 <= args.derived_prob_max <= 1.0):
        raise ValueError("--derived-prob-min and --derived-prob-max must be in [0, 1].")
    if args.derived_prob_min > args.derived_prob_max:
        raise ValueError("--derived-prob-min must be <= --derived-prob-max.")


def resolve_path(path_like: str, base_dir: Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_concept_metadata(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    concepts = payload.get("define")
    if not isinstance(concepts, dict) or not concepts:
        raise ValueError(f"Invalid concept metadata payload in {path}")
    return concepts


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        return list(reader.fieldnames), list(reader)


def resolve_entity_column(fieldnames: Sequence[str]) -> str:
    for candidate in ("company_name", "ticker", "entity"):
        if candidate in fieldnames:
            return candidate
    raise ValueError("CSV must contain one of: company_name, ticker, entity.")


def normalize_value(value: Any) -> float:
    if value is None:
        raise ValueError("Missing value in spreadsheet.")
    if isinstance(value, str) and not value.strip():
        raise ValueError("Missing value in spreadsheet.")
    return float(value)


def build_atoms_from_dataframe(
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
    concept_metadata: dict[str, dict[str, Any]],
) -> dict[str, Atom]:
    required = {"year", *concept_metadata.keys()}
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    entity_column = resolve_entity_column(fieldnames)
    atoms: dict[str, Atom] = {}

    for idx, row in enumerate(rows):
        for concept, meta in concept_metadata.items():
            atom = Atom(
                key=f"{idx}_{concept}",
                concept=concept,
                semantic_type=meta["semantic_type"],
                label=concept.replace("_", " "),
                entity=str(row[entity_column]),
                period=str(row["year"]),
                unit=meta["unit"],
                value=normalize_value(row[concept]),
                depth=0,
                parent_concept=meta.get("parent_concept"),
                role=meta.get("role"),
            )
            atoms[atom.key] = atom

    return atoms


def flatten_leaf_keys(expr: Expr) -> list[str]:
    return [leaf.key for leaf in expr.flatten_leaves() if leaf.key is not None]


def with_random_seed(seed: int, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    previous_state = random.getstate()
    random.seed(seed)
    try:
        return fn(*args, **kwargs)
    finally:
        random.setstate(previous_state)


def build_row(
    i: int,
    expr: Expr,
    question: str,
    answer: float,
    expr_json: dict[str, Any],
    expr_str: str,
    tree_payload: Expr,
    tree_seed: int,
    bind_seed: int,
    master_seed: int,
    derived_prob: float,
    depth: int,
    template_stats: dict[str, int],
    leaf_keys: list[str],
    atoms: dict[str, Atom],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": i + 1,
        "depth": expr.expr_depth(),
        "question": question,
        "expression": expr_str,
        "expression_json": json.dumps(expr_json, ensure_ascii=False),
        "template_expression": json.dumps(tree_payload.expr_to_json(), ensure_ascii=False),
        "answer": answer,
        "template_depth_requested": depth,
        "template_actual_depth": Expr.actual_tree_depth(tree_payload),
        "bound_expression_depth": expr.expr_depth(),
        "tree_seed": tree_seed,
        "binding_seed": bind_seed,
        "master_seed": master_seed,
        "derived_prob": derived_prob,
        "template_internal_nodes": template_stats["internal_nodes"],
        "template_leaf_slots": template_stats["leaves"],
        "bound_leaf_count": len(leaf_keys),
    }

    for j, leaf_key in enumerate(leaf_keys, start=1):
        atom = atoms[leaf_key]
        row[f"leaf_{j}_key"] = atom.key
        row[f"leaf_{j}_concept"] = atom.concept
        row[f"leaf_{j}_label"] = atom.label
        row[f"leaf_{j}_entity"] = atom.entity
        row[f"leaf_{j}_period"] = atom.period
        row[f"leaf_{j}_unit"] = atom.unit
        row[f"leaf_{j}_value"] = atom.value

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random financial questions through the v2 pipeline."
    )
    parser.add_argument("--csv", default="output/synthetic_company_data.csv", help="Input spreadsheet CSV")
    parser.add_argument(
        "--concept-metadata",
        default="config/concept_metadata.json",
        help="Concept metadata JSON used to build atoms from the CSV",
    )
    parser.add_argument("--n", type=int, default=90, help="Number of questions to generate")
    parser.add_argument("--depth-min", type=int, default=1, help="Minimum sampled tree depth")
    parser.add_argument("--depth-max", type=int, default=4, help="Maximum sampled tree depth")
    parser.add_argument(
        "--derived-prob-min",
        type=float,
        default=0.10,
        help="Minimum derived concept probability",
    )
    parser.add_argument(
        "--derived-prob-max",
        type=float,
        default=0.60,
        help="Maximum derived concept probability",
    )
    parser.add_argument("--seed", type=int, default=None, help="Master seed; default is random")
    parser.add_argument(
        "--output",
        default="output/random_questions_90.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    validate_args(args)

    base_dir = Path(__file__).resolve().parent
    csv_path = resolve_path(args.csv, base_dir)
    concept_metadata_path = resolve_path(args.concept_metadata, base_dir)
    output_path = resolve_path(args.output, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames, csv_rows = read_csv_rows(csv_path)
    concept_metadata = load_concept_metadata(concept_metadata_path)
    atoms = build_atoms_from_dataframe(fieldnames, csv_rows, concept_metadata)
    store = Store.store_from_atoms(atoms)
    analyzer = SemanticAnalyzer(atoms)
    evaluator = Evaluator(atoms)
    renderer = QuestionRenderer()

    master_seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 10**9)
    master_rng = random.Random(master_seed)

    rows: list[dict[str, Any]] = []
    max_leaf_count = 0

    for i in range(args.n):
        last_error: Exception | None = None
        for attempt in range(1, 1001):
            tree_seed = master_rng.randint(0, 10**9)
            bind_seed = master_rng.randint(0, 10**9)
            depth = master_rng.randint(args.depth_min, args.depth_max)
            derived_prob = round(
                master_rng.uniform(args.derived_prob_min, args.derived_prob_max),
                3,
            )

            try:
                tree_payload, _ = Expr.sample_tree_with_rejection(
                    max_depth=depth,
                    rng=random.Random(tree_seed),
                    derived_prob=derived_prob,
                    derived_registry=DERIVED_CONCEPTS,
                )
                expr = with_random_seed(
                    bind_seed,
                    Expr.instantiate_typed_tree,
                    tree_payload,
                    store,
                    BindEnv(),
                    DERIVED_CONCEPTS,
                )
                analysis = analyzer.analyze(expr)
                answer = evaluator.eval(expr)
                question = with_random_seed(bind_seed, renderer.render, analysis)
                expr_json = expr.expr_to_json()
                expr_str = expr.show_expr()
                template_stats = Expr.count_nodes(tree_payload)
                leaf_keys = flatten_leaf_keys(expr)
                max_leaf_count = max(max_leaf_count, len(leaf_keys))
                break
            except Exception as err:
                last_error = err
                if attempt == 1000:
                    raise RuntimeError(
                        f"Failed to generate question {i + 1} after 1000 attempts. "
                        f"Last error: {type(last_error).__name__}: {last_error}"
                    ) from last_error

        row = build_row(
            i=i,
            expr=expr,
            question=question,
            answer=answer,
            expr_json=expr_json,
            expr_str=expr_str,
            tree_payload=tree_payload,
            tree_seed=tree_seed,
            bind_seed=bind_seed,
            master_seed=master_seed,
            derived_prob=derived_prob,
            depth=depth,
            template_stats=template_stats,
            leaf_keys=leaf_keys,
            atoms=atoms,
        )
        rows.append(row)

    fieldnames = [
        "question_id",
        "depth",
        "question",
        "expression",
        "expression_json",
        "template_expression",
        "answer",
        "template_depth_requested",
        "template_actual_depth",
        "bound_expression_depth",
        "tree_seed",
        "binding_seed",
        "master_seed",
        "derived_prob",
        "template_internal_nodes",
        "template_leaf_slots",
        "bound_leaf_count",
    ]
    for j in range(1, max_leaf_count + 1):
        fieldnames.extend(
            [
                f"leaf_{j}_key",
                f"leaf_{j}_concept",
                f"leaf_{j}_label",
                f"leaf_{j}_entity",
                f"leaf_{j}_period",
                f"leaf_{j}_unit",
                f"leaf_{j}_value",
            ]
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} questions to {output_path}")
    print(f"Master seed: {master_seed}")
    print(f"Max bound leaf count: {max_leaf_count}")


if __name__ == "__main__":
    main()
