"""Adversarial CSV generator.

Reads a questions CSV (produced by make_random_questions.py) and the original
synthetic company data CSV, then writes a corrupted copy of the data CSV where
every cell NOT referenced by any question is randomly replaced with either a
missing value (empty string) or a garbage value.

Usage:
    python adversarial.py \
        --questions output/random_questions_90.csv \
        --csv       output/synthetic_company_data.csv \
        --output    output/adversarial_company_data.csv

Optional:
    --corruption-rate  fraction of unused cells to corrupt  (default 0.4)
    --garbage-fraction of those, fraction that become garbage vs empty (default 0.5)
    --seed             random seed for reproducibility
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        return list(reader.fieldnames), list(reader)


_LEAF_PATTERN = re.compile(r'"leaf":\s*"(\d+_\w+)"')


def collect_used_atom_keys(questions_path: Path) -> set[str]:
    """Extract every leaf atom key from the expression_json column of each question.

    expression_json always contains the fully expanded tree (including derived
    concepts), so every leaf atom key appears as {"leaf": "<key>"} in that JSON.
    """
    _, rows = read_csv_rows(questions_path)
    used: set[str] = set()
    for row in rows:
        used.update(_LEAF_PATTERN.findall(row.get("expression_json", "")))
    return used


def load_concept_names(concept_metadata_path: Path) -> list[str]:
    with open(concept_metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["define"].keys())


# Digits that have a visually similar character (OCR-confusion style)
_LOOKALIKE_MAP: dict[str, list[str]] = {
    "0": ["O", "o"],
    "1": ["I", "l"],
    "2": ["Z"],
    "5": ["S"],
    "6": ["b"],
    "8": ["B"],
    "9": ["g"],
}


def make_lookalike(value: str, rng: random.Random) -> str:
    """Replace 1-2 digits in value with a visually similar character.

    E.g. "12920" -> "l2920" or "12O20" — looks numeric but isn't parseable as float.
    Returns the original value unchanged if no digit has a lookalike.
    """
    chars = list(value.split(".")[0])  # work on the integer part only
    candidates = [i for i, c in enumerate(chars) if c in _LOOKALIKE_MAP]
    if not candidates:
        return value
    for pos in rng.sample(candidates, k=min(2, len(candidates))):
        chars[pos] = rng.choice(_LOOKALIKE_MAP[chars[pos]])
    return "".join(chars)


def build_concept_pool(
    csv_rows: list[dict[str, str]], concept_names: list[str]
) -> dict[str, list[tuple[int, str]]]:
    """For each concept, build a list of (row_idx, value) from all rows."""
    pool: dict[str, list[tuple[int, str]]] = {c: [] for c in concept_names}
    for idx, row in enumerate(csv_rows):
        for concept in concept_names:
            v = row.get(concept, "").strip()
            if v:
                pool[concept].append((idx, v))
    return pool


def write_corrupted_csv(
    csv_path: Path,
    used_atom_keys: set[str],
    concept_names: list[str],
    output_path: Path,
    corruption_rate: float,
    garbage_fraction: float,
    seed: int | None,
) -> None:
    rng = random.Random(seed)
    fieldnames, csv_rows = read_csv_rows(csv_path)

    hard_garbage: list[Any] = [-9999999, 999999999999, -1, "ERROR"]
    concept_pool = build_concept_pool(csv_rows, concept_names)

    corrupted_rows = []
    stats = {
        "protected": 0,
        "corrupted_missing": 0,
        "corrupted_garbage": 0,
        "corrupted_lookalike": 0,
        "corrupted_cross": 0,
        "untouched": 0,
    }

    for idx, row in enumerate(csv_rows):
        new_row = dict(row)
        for concept in concept_names:
            atom_key = f"{idx}_{concept}"
            if atom_key in used_atom_keys:
                stats["protected"] += 1
            elif rng.random() < corruption_rate:
                roll = rng.random()
                if roll < garbage_fraction / 3:
                    new_row[concept] = rng.choice(hard_garbage)
                    stats["corrupted_garbage"] += 1
                elif roll < 2 * garbage_fraction / 3:
                    new_row[concept] = make_lookalike(row[concept], rng)
                    stats["corrupted_lookalike"] += 1
                elif roll < garbage_fraction:
                    # cross-contamination: real value from a different row
                    candidates = [(i, v) for i, v in concept_pool[concept] if i != idx]
                    if candidates:
                        _, cross_val = rng.choice(candidates)
                        new_row[concept] = cross_val
                        stats["corrupted_cross"] += 1
                    else:
                        new_row[concept] = ""
                        stats["corrupted_missing"] += 1
                else:
                    new_row[concept] = ""
                    stats["corrupted_missing"] += 1
            else:
                stats["untouched"] += 1
        corrupted_rows.append(new_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corrupted_rows)

    print(f"Adversarial CSV written to {output_path}")
    print(f"  Protected (used by questions):   {stats['protected']}")
    print(f"  Corrupted — missing:             {stats['corrupted_missing']}")
    print(f"  Corrupted — garbage:             {stats['corrupted_garbage']}")
    print(f"  Corrupted — lookalike:           {stats['corrupted_lookalike']}")
    print(f"  Corrupted — cross-contaminated:  {stats['corrupted_cross']}")
    print(f"  Left untouched:                  {stats['untouched']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an adversarially corrupted data CSV.")
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
        help="Concept metadata JSON (used to enumerate concept column names)",
    )
    parser.add_argument(
        "--output",
        default="output/adversarial_company_data.csv",
        help="Output path for the corrupted CSV",
    )
    parser.add_argument(
        "--corruption-rate",
        type=float,
        default=0.4,
        help="Fraction of unused cells to corrupt (default 0.4)",
    )
    parser.add_argument(
        "--garbage-fraction",
        type=float,
        default=0.5,
        help="Of corrupted cells, fraction that become a garbage value vs empty string (default 0.5)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else base_dir / path

    used_keys = collect_used_atom_keys(resolve(args.questions))
    concept_names = load_concept_names(resolve(args.concept_metadata))

    write_corrupted_csv(
        csv_path=resolve(args.csv),
        used_atom_keys=used_keys,
        concept_names=concept_names,
        output_path=resolve(args.output),
        corruption_rate=args.corruption_rate,
        garbage_fraction=args.garbage_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
