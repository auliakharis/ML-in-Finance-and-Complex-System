"""Adversarial CSV generator.

Reads a questions CSV (produced by make_random_questions.py) and the original
synthetic company data CSV, then writes five corrupted variants where every cell
NOT referenced by any question is randomly replaced:

  missing_values.csv      — empty string
  garbage.csv             — extreme/nonsense numeric values or "ERROR"
  lookalike.csv           — digits swapped with visually similar characters (e.g. 1→I, 0→O)
  cross_contaminated.csv  — real value stolen from a different company/year
  combined_adversarial.csv — all four types mixed equally

Usage:
    python adversarial.py \\
        --questions output/random_questions_90.csv \\
        --csv       output/synthetic_company_data.csv \\
        --output-dir output/adversarial/

Optional:
    --corruption-rate  fraction of unused cells to corrupt  (default 0.4)
    --seed             random seed for reproducibility
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Literal

Mode = Literal["missing", "garbage", "lookalike", "cross", "combined"]

_OUTPUT_FILES: dict[Mode, str] = {
    "missing":  "missing_values.csv",
    "garbage":  "garbage.csv",
    "lookalike": "lookalike.csv",
    "cross":    "cross_contaminated.csv",
    "combined": "combined_adversarial.csv",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        return list(reader.fieldnames), list(reader)


_LEAF_PATTERN = re.compile(r'"leaf":\s*"(\d+_\w+)"')


def collect_used_atom_keys(questions_path: Path) -> set[str]:
    """Extract every leaf atom key from the expression_json column of each question."""
    _, rows = read_csv_rows(questions_path)
    used: set[str] = set()
    for row in rows:
        used.update(_LEAF_PATTERN.findall(row.get("expression_json", "")))
    return used


def load_concept_names(concept_metadata_path: Path) -> list[str]:
    with open(concept_metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["define"].keys())


# ---------------------------------------------------------------------------
# Corruption helpers
# ---------------------------------------------------------------------------

_LOOKALIKE_MAP: dict[str, list[str]] = {
    "0": ["O", "o"],
    "1": ["I", "l"],
    "2": ["Z"],
    "5": ["S"],
    "6": ["b"],
    "8": ["B"],
    "9": ["g"],
}

_HARD_GARBAGE: list[Any] = [-9999999, 999999999999, -1, "ERROR"]


def make_lookalike(value: str, rng: random.Random) -> str:
    """Replace 1-2 digits with visually similar characters. Returns value unchanged if none apply."""
    chars = list(value.split(".")[0])
    candidates = [i for i, c in enumerate(chars) if c in _LOOKALIKE_MAP]
    if not candidates:
        return value
    for pos in rng.sample(candidates, k=min(2, len(candidates))):
        chars[pos] = rng.choice(_LOOKALIKE_MAP[chars[pos]])
    return "".join(chars)


def build_concept_pool(
    csv_rows: list[dict[str, str]], concept_names: list[str]
) -> dict[str, list[tuple[int, str]]]:
    """For each concept, list of (row_idx, value) for all non-empty rows."""
    pool: dict[str, list[tuple[int, str]]] = {c: [] for c in concept_names}
    for idx, row in enumerate(csv_rows):
        for concept in concept_names:
            v = row.get(concept, "").strip()
            if v:
                pool[concept].append((idx, v))
    return pool


def corrupt_cell(
    mode: Mode,
    row_idx: int,
    concept: str,
    original_value: str,
    rng: random.Random,
    concept_pool: dict[str, list[tuple[int, str]]],
) -> tuple[str, str]:
    """Return (corrupted_value, corruption_kind) for a single cell."""
    if mode == "missing":
        return "", "missing"

    if mode == "garbage":
        return str(rng.choice(_HARD_GARBAGE)), "garbage"

    if mode == "lookalike":
        return make_lookalike(original_value, rng), "lookalike"

    if mode == "cross":
        candidates = [(i, v) for i, v in concept_pool[concept] if i != row_idx]
        if candidates:
            return rng.choice(candidates)[1], "cross"
        return "", "missing"  # fallback if only one row exists for this concept

    # combined: pick one of the four types with equal probability
    roll = rng.random()
    if roll < 0.25:
        return "", "missing"
    if roll < 0.50:
        return str(rng.choice(_HARD_GARBAGE)), "garbage"
    if roll < 0.75:
        return make_lookalike(original_value, rng), "lookalike"
    candidates = [(i, v) for i, v in concept_pool[concept] if i != row_idx]
    if candidates:
        return rng.choice(candidates)[1], "cross"
    return "", "missing"


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

def write_corrupted_csv(
    csv_path: Path,
    used_atom_keys: set[str],
    concept_names: list[str],
    output_path: Path,
    mode: Mode,
    corruption_rate: float,
    seed: int | None,
) -> None:
    rng = random.Random(seed)
    fieldnames, csv_rows = read_csv_rows(csv_path)
    concept_pool = build_concept_pool(csv_rows, concept_names)

    corrupted_rows = []
    stats: dict[str, int] = {"protected": 0, "missing": 0, "garbage": 0, "lookalike": 0, "cross": 0, "untouched": 0}

    for idx, row in enumerate(csv_rows):
        new_row = dict(row)
        for concept in concept_names:
            if f"{idx}_{concept}" in used_atom_keys:
                stats["protected"] += 1
            elif rng.random() < corruption_rate:
                new_val, kind = corrupt_cell(mode, idx, concept, row[concept], rng, concept_pool)
                new_row[concept] = new_val
                stats[kind] += 1
            else:
                stats["untouched"] += 1
        corrupted_rows.append(new_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corrupted_rows)

    print(f"\n[{mode}] -> {output_path.name}")
    print(f"  Protected:          {stats['protected']}")
    print(f"  Missing:            {stats['missing']}")
    print(f"  Garbage:            {stats['garbage']}")
    print(f"  Lookalike:          {stats['lookalike']}")
    print(f"  Cross-contaminated: {stats['cross']}")
    print(f"  Untouched:          {stats['untouched']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adversarially corrupted data CSVs.")
    parser.add_argument(
        "--questions",
        default="output/random_questions_90.csv",
        help="Questions CSV produced by make_random_questions.py",
    )
    parser.add_argument(
        "--csv",
        default="output/synthetic_company_data.csv",
        help="Original synthetic company data CSV to corrupt",
    )
    parser.add_argument(
        "--concept-metadata",
        default="config/concept_metadata.json",
        help="Concept metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="output/adversarial",
        help="Directory to write all five output CSVs (default: output/adversarial/)",
    )
    parser.add_argument(
        "--corruption-rate",
        type=float,
        default=0.4,
        help="Fraction of unused cells to corrupt (default 0.4)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else base_dir / path

    used_keys = collect_used_atom_keys(resolve(args.questions))
    concept_names = load_concept_names(resolve(args.concept_metadata))
    output_dir = resolve(args.output_dir)

    for mode, filename in _OUTPUT_FILES.items():
        write_corrupted_csv(
            csv_path=resolve(args.csv),
            used_atom_keys=used_keys,
            concept_names=concept_names,
            output_path=output_dir / filename,
            mode=mode,
            corruption_rate=args.corruption_rate,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
