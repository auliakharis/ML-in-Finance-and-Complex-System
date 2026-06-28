from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from data_prep import run_data_prep
from evaluator import Evaluator
from obstacles import (
    DATA_PREP_OBSTACLES,
    GENERATION_OBSTACLES,
    OBSTACLE_NAMES,
    USELESS_INFO_TEMPLATES_FILE,
    ObstacleContext,
    load_financial_spreadsheet,
    m_usd_concepts_from_metadata,
    obstacle_requires_spreadsheet,
    pick_unit_scale,
    scale_answer_from_display_units,
    scale_csv_rows,
    scale_financial_spreadsheet_rows,
    BIG_NUMBERS_SCALE_FACTOR,
    validate_big_numbers_factor,
    validate_obstacle_name,
    validate_useless_info_family,
    validate_useless_info_template_coverage,
)
from question_renderer import QuestionRenderer
from semantic_analyzer import SemanticAnalyzer
from tree import Atom, BindEnv, DERIVED_CONCEPTS, Expr, Store

T = TypeVar("T")

@dataclass
class RowData:
    i: int
    expr: Expr
    question: str
    answer: float
    expr_json: dict[str, Any]
    expr_str: str
    tree_payload: Expr
    tree_seed: int
    bind_seed: int
    master_seed: int
    derived_prob: float
    depth: int
    template_stats: dict[str, int]
    leaf_keys: list[str]
    atoms: dict[str, Atom]
    does_expr_contain_derived: bool
    original_question: str
    useless_info_family_used: str
    obstacle: str | None

def validate_args(args: argparse.Namespace, *, base_dir: Path | None = None) -> None:
    if args.n <= 0:
        raise ValueError("--n must be > 0.")
    if args.depth_min < 0 or args.depth_max < 0:
        raise ValueError("--depth-min/--depth-max must be >= 0.")
    if args.depth_min > args.depth_max:
        raise ValueError("--depth-min must be <= --depth-max.")
    if not (
        0.0 <= args.derived_prob_min <= 1.0 and 0.0 <= args.derived_prob_max <= 1.0
    ):
        raise ValueError("--derived-prob-min and --derived-prob-max must be in [0, 1].")
    if args.derived_prob_min > args.derived_prob_max:
        raise ValueError("--derived-prob-min must be <= --derived-prob-max.")
    if args.depth_max == 0 and args.derived_prob_min > 0.0:
        raise ValueError("--derived-prob-min must be 0 when --depth-max=0: at depth 0 the sampler always emits a plain leaf.")
    obstacle = getattr(args, "obstacle", None)
    validate_obstacle_name(obstacle)
    useless_info_family = getattr(args, "useless_info_family", None)
    if useless_info_family and obstacle != "useless_info":
        raise ValueError("--useless-info-family requires --obstacle useless_info")
    if base_dir is not None:
        templates_path = base_dir / USELESS_INFO_TEMPLATES_FILE
        if obstacle == "useless_info":
            validate_useless_info_template_coverage(templates_path)
        if useless_info_family:
            validate_useless_info_family(useless_info_family, templates_path)
    big_numbers_factor = getattr(args, "big_numbers_factor", BIG_NUMBERS_SCALE_FACTOR)
    validate_big_numbers_factor(big_numbers_factor)
    if (
        obstacle != "big_numbers"
        and big_numbers_factor != BIG_NUMBERS_SCALE_FACTOR
    ):
        raise ValueError("--big-numbers-factor requires --obstacle big_numbers")


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


def build_row(data=RowData) -> dict[str, Any]:
    (
        i,
        expr,
        question,
        answer,
        expr_json,
        expr_str,
        tree_payload,
        tree_seed,
        bind_seed,
        master_seed,
        derived_prob,
        depth,
        template_stats,
        leaf_keys,
        atoms,
        original_question,
        useless_info_family_used,
        obstacle
    ) = (
        data.i,
        data.expr,
        data.question,
        data.answer,
        data.expr_json,
        data.expr_str,
        data.tree_payload,
        data.tree_seed,
        data.bind_seed,
        data.master_seed,
        data.derived_prob,
        data.depth,
        data.template_stats,
        data.leaf_keys,
        data.atoms,
        data.original_question,
        data.useless_info_family_used,
        data.obstacle
    )
    row: dict[str, Any] = {
        "question_id": i + 1,
        "depth": expr.expr_depth(),
        "question": question,
        "question_original": original_question,
        "obstacle": obstacle or "",
        "useless_info_family_used": useless_info_family_used,
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


ORIGINAL_QUESTIONS_MAX_LEAF_SLOTS = 9

ORIGINAL_QUESTIONS_BASE_FIELDS: list[str] = [
    "question_id",
    "depth",
    "question",
    "question_original",
    "obstacle",
    "useless_info_family_used",
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

ORIGINAL_QUESTIONS_LEAF_SUFFIXES: list[str] = [
    "key",
    "concept",
    "label",
    "entity",
    "period",
    "unit",
    "value",
]


def original_questions_fieldnames(
    max_leaf_slots: int = ORIGINAL_QUESTIONS_MAX_LEAF_SLOTS,
) -> list[str]:
    """Column order matching original_questions.json."""
    fieldnames = list(ORIGINAL_QUESTIONS_BASE_FIELDS)
    for j in range(1, max_leaf_slots + 1):
        for suffix in ORIGINAL_QUESTIONS_LEAF_SUFFIXES:
            fieldnames.append(f"leaf_{j}_{suffix}")
    return fieldnames


def _stringify_original_questions_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def question_row_to_original_questions_record(
    row: dict[str, Any],
    max_leaf_slots: int = ORIGINAL_QUESTIONS_MAX_LEAF_SLOTS,
) -> dict[str, str]:
    """Convert one pipeline question row to an original_questions.json record."""
    record: dict[str, str] = {
        field: _stringify_original_questions_value(row.get(field))
        for field in ORIGINAL_QUESTIONS_BASE_FIELDS
    }
    for j in range(1, max_leaf_slots + 1):
        for suffix in ORIGINAL_QUESTIONS_LEAF_SUFFIXES:
            key = f"leaf_{j}_{suffix}"
            record[key] = _stringify_original_questions_value(row.get(key, ""))
    return record


def question_rows_to_original_questions_records(
    rows: list[dict[str, Any]],
    max_leaf_slots: int = ORIGINAL_QUESTIONS_MAX_LEAF_SLOTS,
) -> list[dict[str, str]]:
    return [
        question_row_to_original_questions_record(row, max_leaf_slots=max_leaf_slots)
        for row in rows
    ]


def write_original_questions_json(
    records: list[dict[str, str]],
    output_path: Path | str,
    max_leaf_slots: int = ORIGINAL_QUESTIONS_MAX_LEAF_SLOTS,
) -> None:
    """Write records in the same schema as original_questions.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = original_questions_fieldnames(max_leaf_slots)
    ordered_records = [{name: record[name] for name in fieldnames} for record in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered_records, f, indent=4)


def persist_unit_scaled_financial_data(
    csv_path: Path,
    financial_spreadsheet_path: Path,
    m_usd_columns: frozenset[str],
    factor: float,
) -> None:
    """Write CSV and spreadsheet with M_USD amounts divided by factor."""
    fieldnames, csv_rows = read_csv_rows(csv_path)
    scaled_csv = scale_csv_rows(csv_rows, m_usd_columns, factor)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scaled_csv)

    sheet_rows = load_financial_spreadsheet(financial_spreadsheet_path)
    scaled_sheet = scale_financial_spreadsheet_rows(sheet_rows, m_usd_columns, factor)
    financial_spreadsheet_path.parent.mkdir(parents=True, exist_ok=True)
    with open(financial_spreadsheet_path, "w", encoding="utf-8") as f:
        json.dump(scaled_sheet, f, indent=4)


def generate_question_rows(
    csv_path: Path,
    concept_metadata_path: Path,
    n: int,
    depth_min: int,
    depth_max: int,
    derived_prob_min: float,
    derived_prob_max: float,
    seed: int | None = None,
    obstacle: str | None = None,
    financial_spreadsheet_path: Path | None = None,
    base_dir: Path | None = None,
    unit_scale_label: str | None = None,
    unit_scale_factor: float | None = None,
    unit_scale_data_persisted: bool = False,
    useless_info_family: str | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Run the question-generation pipeline and return rows plus seeds/stats."""
    validate_obstacle_name(obstacle)
    pipeline_base = base_dir or Path(__file__).resolve().parent

    fieldnames, csv_rows = read_csv_rows(csv_path)
    concept_metadata = load_concept_metadata(concept_metadata_path)
    m_usd_columns = m_usd_concepts_from_metadata(concept_metadata)

    if obstacle == "unit_scale_change":
        if unit_scale_label is None or unit_scale_factor is None:
            unit_scale_label, unit_scale_factor = pick_unit_scale()
        if not unit_scale_data_persisted:
            csv_rows = scale_csv_rows(csv_rows, m_usd_columns, unit_scale_factor)

    atoms = build_atoms_from_dataframe(fieldnames, csv_rows, concept_metadata)
    store = Store.store_from_atoms(atoms)
    analyzer = SemanticAnalyzer(atoms)
    evaluator = Evaluator(atoms)
    renderer = QuestionRenderer()

    spreadsheet_rows: list[dict[str, str]] = []
    if obstacle_requires_spreadsheet(obstacle) or obstacle == "unit_scale_change":
        sheet_path = financial_spreadsheet_path or (
            pipeline_base / "output/financial_spreadsheet.json"
        )
        if not sheet_path.is_file():
            raise FileNotFoundError(
                f"Financial spreadsheet not found at {sheet_path}. "
                "Run: python data_prep.py"
            )
        spreadsheet_rows = load_financial_spreadsheet(sheet_path)
        if (
            obstacle == "unit_scale_change"
            and unit_scale_factor is not None
            and not unit_scale_data_persisted
        ):
            spreadsheet_rows = scale_financial_spreadsheet_rows(
                spreadsheet_rows, m_usd_columns, unit_scale_factor
            )

    master_seed = seed if seed is not None else random.SystemRandom().randint(0, 10**9)
    master_rng = random.Random(master_seed)

    rows: RowData = []
    max_leaf_count = 0
    balanced_tree = obstacle == "balanced_tree"

    i = 0
    derived_true_count = 0


    while i < n:
        last_error: Exception | None = None

        current_ratio = derived_true_count / (i + 1)
        target_ratio = (derived_prob_min + derived_prob_max) / 2
        need_derived = current_ratio < target_ratio

        for attempt in range(1, 1001):
            tree_seed = master_rng.randint(0, 10**9)
            bind_seed = master_rng.randint(0, 10**9)
            depth = master_rng.randint(depth_min, depth_max)
            derived_prob = round(
                master_rng.uniform(derived_prob_min, derived_prob_max),
                3,
            )

            try:
                tree_payload, _ = Expr.sample_tree_with_rejection(
                    max_depth=depth,
                    derived_prob=derived_prob,
                    derived_registry=DERIVED_CONCEPTS,
                    balanced=balanced_tree,
                )
                expr = with_random_seed(
                    bind_seed,
                    Expr.instantiate_typed_tree,
                    tree_payload,
                    store,
                    BindEnv(),
                    DERIVED_CONCEPTS,
                    balanced=balanced_tree,
                )
                analysis = analyzer.analyze(expr)
                answer = evaluator.eval(expr)
                if obstacle == "unit_scale_change" and unit_scale_factor is not None:
                    answer = scale_answer_from_display_units(
                        answer,
                        semantic_type=analysis.meaning.semantic_type,
                        factor=unit_scale_factor,
                    )
                question = with_random_seed(bind_seed, renderer.render, analysis)
                question_original = ""
                useless_info_family_used = ""
                if obstacle and obstacle not in DATA_PREP_OBSTACLES and obstacle not in GENERATION_OBSTACLES:
                    question_original = question
                    ctx = ObstacleContext(
                        question=question,
                        analysis=analysis,
                        expr=expr,
                        leaf_atoms=[atoms[k] for k in flatten_leaf_keys(expr)],
                        spreadsheet_rows=spreadsheet_rows,
                        base_dir=pipeline_base,
                        unit_scale_label=unit_scale_label,
                        useless_info_family=useless_info_family,
                    )
                    question = ctx.apply(obstacle)
                    if obstacle == "useless_info" and ctx.useless_info_family_used:
                        useless_info_family_used = ctx.useless_info_family_used
                expr_json = expr.expr_to_json()
                expr_str = expr.show_expr()
                template_stats = Expr.count_nodes(tree_payload)
                leaf_keys = flatten_leaf_keys(expr)
                max_leaf_count = max(max_leaf_count, len(leaf_keys))

                does_expr_contain_derived = expr.contains_a_derived_concept()
                actual_depth = expr.expr_depth()
                
                if need_derived and not does_expr_contain_derived:
                    continue
                if not (depth_min <= actual_depth <= depth_max):
                    continue


                break
            except Exception as err:
                last_error = err
                if attempt == 1000:
                    raise RuntimeError(
                        f"Failed to generate question {i + 1} after 1000 attempts. "
                        f"Last error: {type(last_error).__name__}: {last_error}"
                    ) from last_error

        i += 1
        if does_expr_contain_derived:
            derived_true_count += 1

        row_data = RowData(
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
            does_expr_contain_derived=does_expr_contain_derived,
            original_question=question_original,
            useless_info_family_used=useless_info_family_used,
            obstacle=obstacle,
        )

        row = build_row(
            row_data
        )
        rows.append(row)
    
    return rows, master_seed, max_leaf_count


def write_questions_csv(rows: list[dict[str, Any]], output_path: Path) -> int:
    """Write question rows to CSV; return max leaf count."""
    max_leaf_count = 0
    for row in rows:
        bound_leaf_count = row.get("bound_leaf_count", 0)
        if isinstance(bound_leaf_count, str) and bound_leaf_count.isdigit():
            max_leaf_count = max(max_leaf_count, int(bound_leaf_count))
        elif isinstance(bound_leaf_count, int):
            max_leaf_count = max(max_leaf_count, bound_leaf_count)

    fieldnames = list(ORIGINAL_QUESTIONS_BASE_FIELDS)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return max_leaf_count


def generate_original_questions_json(
    output_path: Path | str,
    csv_path: Path | str = "output/financial_spreadsheet.csv",
    concept_metadata_path: Path | str = "config/concept_metadata.json",
    n: int = 90,
    depth_min: int = 1,
    depth_max: int = 4,
    derived_prob_min: float = 0.10,
    derived_prob_max: float = 0.60,
    seed: int | None = None,
    max_leaf_slots: int = ORIGINAL_QUESTIONS_MAX_LEAF_SLOTS,
    obstacle: str | None = None,
    financial_spreadsheet_path: Path | str | None = None,
    useless_info_family: str | None = None,
) -> list[dict[str, str]]:
    """Generate and write original_questions.json from synthetic spreadsheet data."""
    args = argparse.Namespace(
        n=n,
        depth_min=depth_min,
        depth_max=depth_max,
        derived_prob_min=derived_prob_min,
        derived_prob_max=derived_prob_max,
        obstacle=obstacle,
        useless_info_family=useless_info_family,
    )
    validate_args(args, base_dir=Path(__file__).resolve().parent)

    base_dir = Path(__file__).resolve().parent
    resolved_csv_path = resolve_path(str(csv_path), base_dir)
    resolved_concept_metadata_path = resolve_path(str(concept_metadata_path), base_dir)
    resolved_output_path = resolve_path(str(output_path), base_dir)
    resolved_sheet_path = (
        resolve_path(str(financial_spreadsheet_path), base_dir)
        if financial_spreadsheet_path
        else None
    )

    rows, _, _ = generate_question_rows(
        csv_path=resolved_csv_path,
        concept_metadata_path=resolved_concept_metadata_path,
        n=n,
        depth_min=depth_min,
        depth_max=depth_max,
        derived_prob_min=derived_prob_min,
        derived_prob_max=derived_prob_max,
        seed=seed,
        obstacle=obstacle,
        financial_spreadsheet_path=resolved_sheet_path,
        base_dir=base_dir,
        useless_info_family=useless_info_family,
    )
    records = question_rows_to_original_questions_records(
        rows, max_leaf_slots=max_leaf_slots
    )
    write_original_questions_json(
        records, resolved_output_path, max_leaf_slots=max_leaf_slots
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random financial questions through the v2 pipeline."
    )
    parser.add_argument("--csv", default="output/financial_spreadsheet.csv", help="Input spreadsheet CSV")
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
    parser.add_argument(
        "--obstacle",
        choices=OBSTACLE_NAMES,
        default=None,
        help=(
            "Apply one obstacle (big_numbers / unit_scale_change scale persisted data; "
            "balanced_tree changes tree shape; others mainly modify question text)"
        ),
    )
    parser.add_argument(
        "--financial-spreadsheet",
        default="output/financial_spreadsheet.json",
        help="Spreadsheet JSON for conditional obstacle (and unit_scale_change data scaling)",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help=(
            "With --obstacle big_numbers, skip regenerating CSV/spreadsheet; "
            "with unit_scale_change, skip scaling persisted CSV/spreadsheet"
        ),
    )
    parser.add_argument(
        "--useless-info-family",
        default=None,
        metavar="FAMILY",
        help=(
            "With --obstacle useless_info, force this concept family from "
            "config/useless_info_templates.json (default: auto-match from expression)"
        ),
    )
    parser.add_argument(
        "--big-numbers-factor",
        type=float,
        default=BIG_NUMBERS_SCALE_FACTOR,
        help=(
            "With --obstacle big_numbers, multiply numeric sampling ranges by this "
            f"factor during data prep (default: {BIG_NUMBERS_SCALE_FACTOR:g})"
        ),
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    validate_args(args, base_dir=base_dir)
    csv_path = resolve_path(args.csv, base_dir)
    concept_metadata_path = resolve_path(args.concept_metadata, base_dir)
    output_path = resolve_path(args.output, base_dir)
    financial_spreadsheet_path = resolve_path(args.financial_spreadsheet, base_dir)

    unit_scale_label: str | None = None
    unit_scale_factor: float | None = None

    if args.obstacle == "big_numbers" and not args.skip_data_prep:
        import os

        def _path_for_data_prep(path: Path) -> str:
            try:
                return str(path.relative_to(base_dir))
            except ValueError:
                return str(path)

        prev_cwd = os.getcwd()
        os.chdir(base_dir)
        try:
            run_data_prep(
                obstacle="big_numbers",
                big_numbers_factor=args.big_numbers_factor,
                csv_path=_path_for_data_prep(csv_path),
                financial_spreadsheet_path=_path_for_data_prep(
                    financial_spreadsheet_path
                ),
            )
        finally:
            os.chdir(prev_cwd)
        print(
            "Regenerated synthetic data with big_numbers obstacle "
            f"(factor={args.big_numbers_factor:g})."
        )

    if args.obstacle == "unit_scale_change":
        unit_scale_label, unit_scale_factor = pick_unit_scale()
        if not args.skip_data_prep:
            concept_metadata = load_concept_metadata(concept_metadata_path)
            m_usd_columns = m_usd_concepts_from_metadata(concept_metadata)
            if not csv_path.is_file():
                import os

                prev_cwd = os.getcwd()
                os.chdir(base_dir)
                try:
                    run_data_prep()
                finally:
                    os.chdir(prev_cwd)
            persist_unit_scaled_financial_data(
                csv_path,
                financial_spreadsheet_path,
                m_usd_columns,
                unit_scale_factor,
            )
            print(
                f"Scaled financial data for unit_scale_change "
                f"({unit_scale_label}, factor={unit_scale_factor:g})."
            )

    rows, master_seed, max_leaf_count = generate_question_rows(
        csv_path=csv_path,
        concept_metadata_path=concept_metadata_path,
        n=args.n,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        derived_prob_min=args.derived_prob_min,
        derived_prob_max=args.derived_prob_max,
        seed=args.seed,
        obstacle=args.obstacle,
        financial_spreadsheet_path=financial_spreadsheet_path,
        base_dir=base_dir,
        unit_scale_label=unit_scale_label,
        unit_scale_factor=unit_scale_factor,
        unit_scale_data_persisted=(
            args.obstacle == "unit_scale_change" and not args.skip_data_prep
        ),
        useless_info_family=args.useless_info_family,
    )

    max_leaf_count = write_questions_csv(rows, output_path)

    original_questions_json_path = resolve_path(
        "output/original_questions.json", base_dir
    )
    original_questions_records = question_rows_to_original_questions_records(rows)
    write_original_questions_json(original_questions_records, original_questions_json_path)

    print(f"Saved {len(rows)} questions to {output_path}")
    print(f"Saved {len(original_questions_records)} questions to {original_questions_json_path}")
    if args.obstacle:
        print(f"Obstacle: {args.obstacle}")
    if args.useless_info_family:
        print(f"Useless-info family: {args.useless_info_family}")
    print(f"Master seed: {master_seed}")
    print(f"Max bound leaf count: {max_leaf_count}")


if __name__ == "__main__":
    main()
