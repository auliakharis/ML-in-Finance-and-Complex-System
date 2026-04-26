from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ATOM_ALLOWED_FIELDS = {
    "key",
    "concept",
    "semantic_type",
    "label",
    "entity",
    "period",
    "unit",
    "value",
    "depth",
    "parent_concept",
    "role",
}


def load_module(module_path: Path, module_name: str):
    # Dynamically import numbered pipeline scripts by absolute path.
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def flatten_leaves(expr: Any) -> List[str]:
    # Normalize any bound expression tree into a flat list of leaf keys.
    if hasattr(expr, "key"):
        return [expr.key]
    if hasattr(expr, "expr"):
        return flatten_leaves(expr.expr)
    # tree_to_language.Literal (mean-over-years divisor): not a spreadsheet leaf.
    if hasattr(expr, "value") and not hasattr(expr, "left"):
        return []
    return flatten_leaves(expr.left) + flatten_leaves(expr.right)


def validate_args(args: argparse.Namespace) -> None:
    # Enforce sane generation ranges before any sampling starts.
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
    # Resolve relative CLI paths against this script directory.
    path = Path(path_like)
    if not path.is_absolute():
        path = base_dir / path
    return path


def build_row(
    i: int,
    expr: Any,
    question: str,
    answer: float,
    expr_json: Dict[str, Any],
    expr_str: str,
    tree_payload: Dict[str, Any],
    tree_seed: int,
    bind_seed: int,
    master_seed: int,
    derived_prob: float,
    depth: int,
    template_stats: Dict[str, int],
    leaf_keys: List[str],
    atoms: Dict[str, Any],
    mod_sampler: Any,
    mod_lang: Any,
) -> Dict[str, Any]:
    # Core row-level metadata for one generated question.
    row: Dict[str, Any] = {
        "question_id": i + 1,
        "depth": mod_lang.expr_depth(expr),
        "question": question,
        "expression": expr_str,
        "expression_json": json.dumps(expr_json, ensure_ascii=False),
        "template_expression": json.dumps(tree_payload, ensure_ascii=False),
        "answer": answer,
        "template_depth_requested": depth,
        "template_actual_depth": mod_sampler.actual_tree_depth(tree_payload),
        "bound_expression_depth": mod_lang.expr_depth(expr),
        "tree_seed": tree_seed,
        "binding_seed": bind_seed,
        "master_seed": master_seed,
        "derived_prob": derived_prob,
        "template_internal_nodes": template_stats["internal_nodes"],
        "template_leaf_slots": template_stats["leaves"],
        "bound_leaf_count": len(leaf_keys),
    }

    # Append per-leaf provenance columns so each question is auditable.
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


def clean_atom_payload(raw_atom: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only fields accepted by tree_to_language.Atom.
    return {k: v for k, v in raw_atom.items() if k in ATOM_ALLOWED_FIELDS}

def main() -> None:
    # Parse high-level generation controls.
    parser = argparse.ArgumentParser(description="Generate random financial questions through the full pipeline.")
    parser.add_argument("--csv", default="financial_spreadsheet.csv", help="Input spreadsheet CSV")
    parser.add_argument("--n", type=int, default=90, help="Number of questions to generate")
    parser.add_argument("--depth-min", type=int, default=1, help="Minimum sampled tree depth")
    parser.add_argument("--depth-max", type=int, default=4, help="Maximum sampled tree depth")
    parser.add_argument("--derived-prob-min", type=float, default=0.10, help="Minimum derived concept probability")
    parser.add_argument("--derived-prob-max", type=float, default=0.60, help="Maximum derived concept probability")
    parser.add_argument("--seed", type=int, default=None, help="Master seed; default is random")
    parser.add_argument("--output", default="output/random_questions_90.csv", help="Output CSV path")
    args = parser.parse_args()
    validate_args(args)

    # Resolve runtime paths and prepare output location.
    base_dir = Path(__file__).resolve().parent
    csv_path = resolve_path(args.csv, base_dir)
    output_path = resolve_path(args.output, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load pipeline modules so we can call their functions directly.
    mod_atoms = load_module(base_dir / "2.fixed_building_atoms.py", "fixed_building_atoms")
    mod_sampler = load_module(base_dir / "3.fixed_tree_sampler.py", "fixed_tree_sampler")
    mod_lang = load_module(base_dir / "4.tree_to_language.py", "tree_to_language")
    print("USING SAMPLER:", (base_dir / "3.fixed_tree_sampler.py").resolve())
    print("USING LANG:", (base_dir / "4.tree_to_language.py").resolve())
    print("USING FILE 5 FROM:", Path(__file__).resolve())
    # Build reusable objects once, then sample repeatedly.
    df = mod_atoms.pd.read_csv(csv_path)
    atoms_raw = mod_atoms.build_atoms(df)
    atoms = {a["key"]: mod_lang.Atom(**clean_atom_payload(a)) for a in atoms_raw}
    index = mod_lang.AtomIndex(atoms)
    analyzer = mod_lang.SemanticAnalyzer(atoms)
    evaluator = mod_lang.Evaluator(atoms)
    renderer = mod_lang.QuestionRenderer()

    master_seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 10**9)
    master_rng = random.Random(master_seed)

    rows: List[Dict[str, Any]] = []
    max_leaf_count = 0

    # Generate N questions with bounded retry loops per question.
    for i in range(args.n):
        last_error: Exception | None = None
        for attempt in range(1, 1001):
            # Draw independent seeds so tree shape and atom binding vary.
            tree_seed = master_rng.randint(0, 10**9)
            bind_seed = master_rng.randint(0, 10**9)
            depth = master_rng.randint(args.depth_min, args.depth_max)
            derived_prob = round(master_rng.uniform(args.derived_prob_min, args.derived_prob_max), 3)

            try:
                # 1) sample typed template 2) bind atoms 3) analyze/render/evaluate.
                tree_rng = random.Random(tree_seed)
                tree_payload, _ = mod_sampler.sample_tree_with_rejection(
                    max_depth=depth,
                    rng=tree_rng,
                    derived_prob=derived_prob,
                )
                expr = mod_lang.instantiate_typed_tree(
                    tree_payload,
                    index=index,
                    rng=random.Random(bind_seed),
                    env=mod_lang.BindEnv(),
                    derived_registry={
                        name: spec["formula"]
                        for name, spec in mod_sampler.DERIVED_CONCEPTS.items()
                    },
                )
                analysis = analyzer.analyze(expr)
                answer = evaluator.eval(expr)
                question = renderer.render(analysis)
                expr_json = mod_lang.expr_to_json(expr)
                expr_str = mod_lang.show_expr(expr)
                template_stats = mod_sampler.count_nodes(tree_payload)
                leaf_keys = flatten_leaves(expr)
                max_leaf_count = max(max_leaf_count, len(leaf_keys))
                break
            except Exception as err:
                # Retry transient semantic mismatches until attempt budget is exhausted.
                last_error = err
                if attempt == 1000:
                    raise RuntimeError(
                        f"Failed to generate question {i + 1} after 1000 attempts. "
                        f"Last error: {type(last_error).__name__}: {last_error}"
                    ) from last_error
                continue

        # Convert generated artifacts into one CSV record.
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
            mod_sampler=mod_sampler,
            mod_lang=mod_lang,
        )

        rows.append(row)

    # Base output columns shared by all rows.
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
    # Add dynamic leaf_* columns up to the maximum leaf count observed.
    for j in range(1, max_leaf_count + 1):
        fieldnames.extend([
            f"leaf_{j}_key",
            f"leaf_{j}_concept",
            f"leaf_{j}_label",
            f"leaf_{j}_entity",
            f"leaf_{j}_period",
            f"leaf_{j}_unit",
            f"leaf_{j}_value",
        ])

    # Write final dataset.
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} questions to {output_path}")
    print(f"Master seed: {master_seed}")
    print(f"Max bound leaf count: {max_leaf_count}")


if __name__ == "__main__":
    main()
