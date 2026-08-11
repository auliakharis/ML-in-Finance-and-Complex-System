"""Generate all query obstacle variants and adversarial corruptions.

Reads data produced by data_obstacles.py (default: output/data/baseline/).
Generates one questions file per query obstacle variant, then generates
adversarial corruptions of the spreadsheet based on the baseline questions.

Query obstacle variants:
  baseline          — no modification to question text
  unit_scale_change — data presented in thousands/millions/billions/trillions
  useless_info      — irrelevant financial sentence injected into each question
  conditional       — question prefixed with a revenue threshold condition
  negation          — question wrapped with misleading double-negation phrasing

Adversarial corruptions (5 variants, based on baseline questions):
  missing_values.csv, garbage.csv, lookalike.csv,
  cross_contaminated.csv, combined_adversarial.csv

Run:
  python query_obstacles.py                              # all variants
  python query_obstacles.py --obstacles baseline useless_info
  python query_obstacles.py --n 20 --seed 42
  python query_obstacles.py --skip-adversarial
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import csv as _csv

from adversarial import (
    PROMPT_INJECTION_RATE,
    _OUTPUT_FILES,
    collect_used_atom_keys,
    inject_prompt_injections,
    load_concept_names,
    read_csv_rows,
    write_corrupted_json,
)
from make_random_questions_adversarial import (
    generate_question_rows,
    question_rows_to_original_questions_records,
    write_original_questions_json,
    write_questions_csv,
)

BASE_DIR = Path(__file__).resolve().parent

QUERY_VARIANTS: dict[str, dict] = {
    "baseline":          {"obstacle": None},
    "unit_scale_change": {"obstacle": "unit_scale_change"},
    "useless_info":      {"obstacle": "useless_info"},
    "conditional":       {"obstacle": "conditional"},
    "negation":          {"obstacle": "negation"},
}


def generate_variant(
    name: str,
    obstacle: str | None,
    data_dir: Path,
    out_dir: Path,
    n: int,
    seed: int | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "synthetic_company_data.csv"
    sheet_path = data_dir / "financial_spreadsheet.json"
    concept_path = BASE_DIR / "config" / "concept_metadata.json"

    prev = os.getcwd()
    os.chdir(BASE_DIR)
    try:
        rows, _, _ = generate_question_rows(
            csv_path=csv_path,
            concept_metadata_path=concept_path,
            n=n,
            depth_min=1,
            depth_max=4,
            derived_prob_min=0.10,
            derived_prob_max=0.60,
            seed=seed,
            obstacle=obstacle,
            financial_spreadsheet_path=sheet_path,
            base_dir=BASE_DIR,
        )
    finally:
        os.chdir(prev)

    csv_out = out_dir / "questions.csv"
    json_out = out_dir / "questions.json"
    write_questions_csv(rows, csv_out)
    write_original_questions_json(
        question_rows_to_original_questions_records(rows), json_out
    )
    print(f"  [{name}] {len(rows)} questions -> {csv_out.relative_to(BASE_DIR)}")
    return csv_out


def _apply_injection_to_csv(path: Path, injection_rate: float | None) -> None:
    """Read a written CSV, apply prompt injection to its rows, and overwrite it."""
    rate = injection_rate if injection_rate is not None else PROMPT_INJECTION_RATE
    fieldnames, rows = read_csv_rows(path)
    injected = inject_prompt_injections(rows, rate=rate)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(injected)


def generate_adversarial(
    questions_csv: Path,
    data_dir: Path,
    adv_dir: Path,
    corruption_rate: float,
    seed: int | None,
    apply_prompt_injection: bool = False,
    injection_rate: float | None = None,
) -> None:
    adv_dir.mkdir(parents=True, exist_ok=True)
    concept_names = load_concept_names(BASE_DIR / "config" / "concept_metadata.json")
    used_keys = collect_used_atom_keys(questions_csv)
    csv_path = data_dir / "synthetic_company_data.csv"

    for mode, filename in _OUTPUT_FILES.items():
        out_path = adv_dir / filename
        write_corrupted_json(
            csv_path=csv_path,
            used_atom_keys=used_keys,
            concept_names=concept_names,
            output_path=out_path,
            mode=mode,
            corruption_rate=corruption_rate,
            seed=seed,
        )
        if apply_prompt_injection:
            _apply_injection_to_csv(out_path, injection_rate=injection_rate)
    print(f"  [adversarial] 5 variants -> {adv_dir.relative_to(BASE_DIR)}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all query obstacle variants and adversarial corruptions."
    )
    parser.add_argument(
        "--obstacles", nargs="+",
        choices=list(QUERY_VARIANTS.keys()),
        default=list(QUERY_VARIANTS.keys()),
        help="Which query variants to generate (default: all)",
    )
    parser.add_argument(
        "--n", type=int, default=90,
        help="Number of questions per variant (default: 90)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master random seed (default: random)",
    )
    parser.add_argument(
        "--data-dir", default=str(BASE_DIR / "output" / "data" / "baseline"),
        help="Data directory to read from, produced by data_obstacles.py (default: output/data/baseline/)",
    )
    parser.add_argument(
        "--output-dir", default=str(BASE_DIR / "output" / "questions"),
        help="Root output directory for question variants (default: output/questions/)",
    )
    parser.add_argument(
        "--adversarial-dir", default=str(BASE_DIR / "output" / "adversarial"),
        help="Output directory for adversarial corruptions (default: output/adversarial/)",
    )
    parser.add_argument(
        "--corruption-rate", type=float, default=0.4,
        help="Fraction of unused cells to corrupt in adversarial variants (default: 0.4)",
    )
    parser.add_argument(
        "--skip-adversarial", action="store_true",
        help="Skip generating adversarial corruptions",
    )
    parser.add_argument(
        "--prompt-injection", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Apply prompt injection to adversarial CSVs. "
            "Auto-detected when --data-dir contains 'prompt_injection'. "
            "Use --no-prompt-injection to force off."
        ),
    )
    parser.add_argument(
        "--injection-rate", type=float, default=None,
        help="Fraction of rows to inject (default: uses PROMPT_INJECTION_RATE from adversarial.py)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "synthetic_company_data.csv").exists():
        print(f"ERROR: data not found at {data_dir}")
        print("       Run data_obstacles.py first.")
        raise SystemExit(1)

    questions_dir = Path(args.output_dir)
    adv_dir = Path(args.adversarial_dir)

    print("Generating question variants...")
    baseline_csv: Path | None = None
    for name in args.obstacles:
        csv_out = generate_variant(
            name=name,
            obstacle=QUERY_VARIANTS[name]["obstacle"],
            data_dir=data_dir,
            out_dir=questions_dir / name,
            n=args.n,
            seed=args.seed,
        )
        if name == "baseline":
            baseline_csv = csv_out

    if not args.skip_adversarial:
        if baseline_csv is None:
            baseline_csv = questions_dir / "baseline" / "questions.csv"
        if baseline_csv.exists():
            # Auto-detect prompt injection from the data-dir name if flag not set explicitly.
            apply_pi = (
                args.prompt_injection
                if args.prompt_injection is not None
                else "prompt_injection" in str(data_dir)
            )
            print("\nGenerating adversarial corruptions...")
            if apply_pi:
                print("  (prompt injection will be layered on top of each corruption variant)")
            generate_adversarial(
                questions_csv=baseline_csv,
                data_dir=data_dir,
                adv_dir=adv_dir,
                corruption_rate=args.corruption_rate,
                seed=args.seed,
                apply_prompt_injection=apply_pi,
                injection_rate=args.injection_rate,
            )
        else:
            print(f"\nWARNING: baseline questions not found at {baseline_csv}")
            print("         Re-run with --obstacles baseline or use --skip-adversarial.")

    print("\nDone.")


if __name__ == "__main__":
    main()
