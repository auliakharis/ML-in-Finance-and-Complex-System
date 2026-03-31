from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_module(module_path: str):
    import sys
    spec = importlib.util.spec_from_file_location("victorie_compiler", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load compiler from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_atom_objects(raw_atoms: List[Dict[str, Any]], compiler) -> Dict[str, Any]:
    atoms = {}
    for item in raw_atoms:
        atoms[str(item["key"])] = compiler.Atom(
            key=str(item["key"]),
            concept=item["concept"],
            semantic_type=item.get("semantic_type", "amount"),
            label=item.get("label", item["concept"]),
            entity=item["entity"],
            period=str(item["period"]),
            unit=item.get("unit", "USD"),
            value=float(item["value"]),
            parent_concept=item.get("parent_concept"),
            role=item.get("role"),
        )
    return atoms


def list_internal_nodes(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    if tree["kind"] == "leaf":
        return []
    out = [tree]
    out.extend(list_internal_nodes(tree["left"]))
    out.extend(list_internal_nodes(tree["right"]))
    return out


def extract_expr_and_leaves(subtree: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    leaves: List[Dict[str, Any]] = []

    def rec(node: Dict[str, Any]) -> Dict[str, Any]:
        if node["kind"] == "leaf":
            leaves.append(
                {
                    "name": node["name"],
                    "family": node["family"],
                    "semantic_type_in": node["semantic_type_in"],
                    "context_group": node.get("context_group"),
                    "entity_group": node.get("entity_group"),
                }
            )
            return {"leaf": node["name"]}
        return {"op": node["op"], "left": rec(node["left"]), "right": rec(node["right"])}

    return rec(subtree), leaves


def matches_leaf_spec(atom: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    return atom["semantic_type"] in spec["semantic_type_in"]


def check_groups(binding: Dict[str, Dict[str, Any]], leaf_specs: List[Dict[str, Any]]) -> bool:
    by_context: Dict[str, List[Dict[str, Any]]] = {}
    by_entity: Dict[str, List[Dict[str, Any]]] = {}

    for spec in leaf_specs:
        if spec["name"] not in binding:
            continue
        atom = binding[spec["name"]]
        ctx = spec.get("context_group")
        ent = spec.get("entity_group")
        if ctx is not None:
            by_context.setdefault(ctx, []).append(atom)
        if ent is not None:
            by_entity.setdefault(ent, []).append(atom)

    for atoms in by_context.values():
        first = atoms[0]
        if any(a["entity"] != first["entity"] or a["period"] != first["period"] or a["unit"] != first["unit"] for a in atoms[1:]):
            return False

    for atoms in by_entity.values():
        first = atoms[0]
        if any(a["entity"] != first["entity"] for a in atoms[1:]):
            return False

    return True


def fill_expr(expr: Dict[str, Any], binding: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "leaf" in expr:
        return {"leaf": binding[expr["leaf"]]["key"]}
    return {
        "op": expr["op"],
        "left": fill_expr(expr["left"], binding),
        "right": fill_expr(expr["right"], binding),
    }


def safe_compile(expr_json: Dict[str, Any], compiler, atom_objects: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        expr = compiler.parse_expr(expr_json)
        analyzer = compiler.SemanticAnalyzer(atom_objects)
        evaluator = compiler.Evaluator(atom_objects)
        renderer = compiler.QuestionRenderer()
        analysis = analyzer.analyze(expr)
        answer = evaluator.eval(expr)
        question = renderer.render(analysis.meaning)
        return {
            "expression": compiler.expr_to_json(expr),
            "depth": analysis.depth,
            "question": question,
            "answer": answer,
            "entity": analysis.meaning.entity,
            "period": analysis.meaning.period,
            "concept": analysis.meaning.concept,
            "kind": analysis.meaning.kind,
            "semantic_type": analysis.meaning.semantic_type,
            "derivation": analysis.meaning.derivation,
        }
    except Exception:
        return None


def sample_binding(
    leaf_specs: List[Dict[str, Any]], raw_atoms: List[Dict[str, Any]], rng: random.Random, max_attempts: int
) -> Optional[Dict[str, Dict[str, Any]]]:
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    for spec in leaf_specs:
        pool = [a for a in raw_atoms if matches_leaf_spec(a, spec)]
        if not pool:
            return None
        candidates[spec["name"]] = pool

    order = sorted(leaf_specs, key=lambda s: len(candidates[s["name"]]))

    for _ in range(max_attempts):
        binding: Dict[str, Dict[str, Any]] = {}
        ok = True
        for spec in order:
            pool = candidates[spec["name"]][:]
            rng.shuffle(pool)
            chosen = None
            for atom in pool:
                binding[spec["name"]] = atom
                if check_groups(binding, leaf_specs):
                    chosen = atom
                    break
            if chosen is None:
                ok = False
                break
        if ok and check_groups(binding, leaf_specs):
            return binding
    return None


def normalize_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text


def contains_banned_phrase(question: str, banned_phrases: List[str]) -> bool:
    q = question.lower()
    return any(phrase.lower() in q for phrase in banned_phrases if phrase.strip())


def classify_candidate(item: Dict[str, Any]) -> Dict[str, bool]:
    entities = set(item.get("binding_entities", []))
    periods = set(item.get("binding_periods", []))
    return {
        "cross_company": len(entities) >= 2,
        "cross_year": len(periods) >= 2,
        "depth3": item.get("depth") == 3,
        "depth4plus": item.get("depth", 0) >= 4,
    }


def dedupe_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (
        normalize_question(item["question"]),
        json.dumps(item["expression"], sort_keys=True),
    )


def select_with_mix(
    candidates: List[Dict[str, Any]],
    banned_phrases: List[str],
    final_n: int,
    min_cross_company: int,
    min_cross_year: int,
    min_depth3: int,
    min_depth4plus: int,
) -> List[Dict[str, Any]]:
    filtered = [c for c in candidates if not contains_banned_phrase(c["question"], banned_phrases)]

    for c in filtered:
        c["flags"] = classify_candidate(c)

    selected: List[Dict[str, Any]] = []
    seen_q = set()
    seen_e = set()

    def can_add(item: Dict[str, Any]) -> bool:
        qk, ek = dedupe_key(item)
        return qk not in seen_q and ek not in seen_e

    def add_item(item: Dict[str, Any]) -> bool:
        if not can_add(item):
            return False
        qk, ek = dedupe_key(item)
        seen_q.add(qk)
        seen_e.add(ek)
        selected.append(item)
        return True

    def take_bucket(predicate, target: int) -> None:
        if target <= 0:
            return
        for item in filtered:
            if len(selected) >= final_n:
                return
            if predicate(item):
                add_item(item)
                if sum(1 for s in selected if predicate(s)) >= target:
                    return

    # Priority buckets
    take_bucket(lambda x: x["flags"]["depth4plus"], min_depth4plus)
    take_bucket(lambda x: x["flags"]["cross_company"], min_cross_company)
    take_bucket(lambda x: x["flags"]["cross_year"], min_cross_year)
    take_bucket(lambda x: x["flags"]["depth3"], min_depth3)

    # Prefer more depth 3 first for the rest, then deeper ones, then anything else.
    preferred = sorted(
        filtered,
        key=lambda x: (
            0 if x["flags"]["depth3"] else 1,
            0 if x["flags"]["depth4plus"] else 1,
            0 if x["flags"]["cross_company"] else 1,
            0 if x["flags"]["cross_year"] else 1,
        )
    )
    for item in preferred:
        if len(selected) >= final_n:
            break
        add_item(item)

    return selected[:final_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mixed final set of financial questions with composition constraints.")
    parser.add_argument("--compiler", default="Victorie_tree_with_depth.py")
    parser.add_argument("--atoms", default="output/atoms_data.json")
    parser.add_argument("--tree", default="output/generated_operator_tree.json")
    parser.add_argument("--output", default="output/final_questions_mixed.json")
    parser.add_argument("--n", type=int, default=90)
    parser.add_argument("--oversample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--binding_attempts", type=int, default=30)
    parser.add_argument("--subtree_attempts", type=int, default=80000)
    parser.add_argument("--min_cross_company", type=int, default=10)
    parser.add_argument("--min_cross_year", type=int, default=10)
    parser.add_argument("--min_depth3", type=int, default=40)
    parser.add_argument("--min_depth4plus", type=int, default=1)
    parser.add_argument(
        "--ban_phrase",
        action="append",
        default=[],
        help="Drop any question containing this phrase. Repeat the flag for multiple phrases.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    compiler = load_module(args.compiler)
    raw_atoms = load_json(args.atoms)
    tree_payload = load_json(args.tree)
    atom_objects = make_atom_objects(raw_atoms, compiler)

    all_nodes = list_internal_nodes(tree_payload["tree"])
    if not all_nodes:
        raise RuntimeError("The generated tree has no internal nodes to sample from.")

    raw_selected: List[Dict[str, Any]] = []
    raw_seen = set()

    for _ in range(args.subtree_attempts):
        if len(raw_selected) >= args.oversample:
            break

        sampled_node = rng.choice(all_nodes)
        expr_template, leaf_specs = extract_expr_and_leaves(sampled_node)
        binding = sample_binding(leaf_specs, raw_atoms, rng, args.binding_attempts)
        if binding is None:
            continue

        expr_json = fill_expr(expr_template, binding)
        compiled = safe_compile(expr_json, compiler, atom_objects)
        if not compiled:
            continue

        key = dedupe_key(compiled)
        if key in raw_seen:
            continue
        raw_seen.add(key)

        bound_atoms = list(binding.values())
        compiled["sampled_node_id"] = sampled_node["node_id"]
        compiled["sampled_node_family"] = sampled_node["family"]
        compiled["sampled_node_depth"] = sampled_node["depth"]
        compiled["binding"] = {k: v["key"] for k, v in binding.items()}
        compiled["binding_entities"] = sorted({a["entity"] for a in bound_atoms})
        compiled["binding_periods"] = sorted({str(a["period"]) for a in bound_atoms})
        raw_selected.append(compiled)

    selected = select_with_mix(
        raw_selected,
        args.ban_phrase,
        args.n,
        args.min_cross_company,
        args.min_cross_year,
        args.min_depth3,
        args.min_depth4plus,
    )

    for i, item in enumerate(selected, start=1):
        item["id"] = i

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    cc = sum(1 for x in selected if len(set(x.get("binding_entities", []))) >= 2)
    cy = sum(1 for x in selected if len(set(x.get("binding_periods", []))) >= 2)
    d3 = sum(1 for x in selected if x.get("depth") == 3)
    d4 = sum(1 for x in selected if x.get("depth", 0) >= 4)

    print(f"Sampled from {len(all_nodes)} internal nodes.")
    print(f"Collected {len(raw_selected)} raw candidates before final selection.")
    print(f"Kept {len(selected)} final questions.")
    print(f"Cross-company: {cc}")
    print(f"Cross-year: {cy}")
    print(f"Depth 3: {d3}")
    print(f"Depth 4+: {d4}")
    if args.ban_phrase:
        print(f"Applied banned phrases: {args.ban_phrase}")
    if len(raw_selected) < args.oversample:
        print(f"Warning: only found {len(raw_selected)} raw candidates after {args.subtree_attempts} subtree samples.")
    if len(selected) < args.n:
        print(f"Warning: only kept {len(selected)} final questions after selection.")


if __name__ == "__main__":
    main()
