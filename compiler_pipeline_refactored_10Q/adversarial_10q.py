"""Obstacle and adversarial corruption pipeline for the 10Q compiler pipeline.

Two independent axes of difficulty:

  Data corruptions  — corrupt unused atom values in atoms_10q.json:
      missing     replace unused atom values with None (shown as blank)
      garbage     replace unused atom values with extreme nonsense floats
      lookalike   replace unused atom values with a string that looks numeric
                  but contains visually similar non-digit characters
      cross       replace unused atom values with a real value from a different entity
      combined    all four types mixed equally

  Query obstacles  — modify the question text (data unchanged):
      useless_info      inject an irrelevant financial sentence
      unit_scale_change rescale all M_USD atom values; prepend unit hint to question
      negation          wrap question in misleading double-negation phrasing
      conditional       prefix question with a revenue threshold condition

Usage:
    from adversarial_10q import write_corrupted_atoms, ObstacleContext10Q
"""
from __future__ import annotations

import json
import random
import re
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_HERE = os.path.dirname(os.path.abspath(__file__))
_REFACTORED = os.path.join(_HERE, "..", "compiler_pipeline_refactored")
sys.path.insert(0, _HERE)
sys.path.insert(0, _REFACTORED)

from adversarial import (
    USELESS_INFO_TEMPLATES_FILE,
    UNIT_SCALE_OPTIONS,
    NEGATION_PREFIXES,
    NEGATION_SUFFIXES,
    format_useless_info_clause,
    load_useless_info_clauses,
    pick_useless_info_clause,
    select_useless_info_families,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORRUPTION_MODES = ("missing", "garbage", "lookalike", "cross", "combined")
QUERY_OBSTACLES  = ("useless_info", "unit_scale_change", "negation", "conditional")
ALL_OBSTACLES    = CORRUPTION_MODES + QUERY_OBSTACLES

Mode = Literal["missing", "garbage", "lookalike", "cross", "combined"]

_HARD_GARBAGE: list[float] = [-9999999.0, 999999999999.0, -1.0]

_LOOKALIKE_MAP: dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "6": "b", "8": "B", "9": "g",
}

M_USD_UNIT = "M_USD"

# ---------------------------------------------------------------------------
# Atom I/O
# ---------------------------------------------------------------------------

def load_atoms(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_atoms(atoms: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(atoms, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Collect used keys from questions CSV
# ---------------------------------------------------------------------------

_LEAF_PATTERN = re.compile(r'"leaf":\s*"([\w]+)"')


def collect_used_atom_keys(questions_path: Path) -> set[str]:
    """Return every leaf atom key referenced in the questions CSV."""
    import csv
    used: set[str] = set()
    with open(questions_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            used.update(_LEAF_PATTERN.findall(row.get("expression_json", "")))
    return used


# ---------------------------------------------------------------------------
# Corruption helpers
# ---------------------------------------------------------------------------

def _make_lookalike_str(value: float) -> str:
    """Convert float to string and replace 1-2 digits with visually similar chars."""
    s = str(int(abs(value)))
    candidates = [i for i, c in enumerate(s) if c in _LOOKALIKE_MAP]
    if not candidates:
        return s
    rng = random.Random()
    for pos in rng.sample(candidates, k=min(2, len(candidates))):
        s = s[:pos] + _LOOKALIKE_MAP[s[pos]] + s[pos + 1:]
    prefix = "-" if value < 0 else ""
    return prefix + s


def _build_concept_pool(atoms: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """concept -> list of (entity, value) for cross-corruption."""
    pool: dict[str, list[tuple[str, float]]] = {}
    for a in atoms:
        if a.get("value") is not None and a.get("semantic_type") == "amount":
            pool.setdefault(a["concept"], []).append((a["entity"], float(a["value"])))
    return pool


def _corrupt_atom_value(
    mode: Mode,
    atom: dict,
    rng: random.Random,
    concept_pool: dict[str, list[tuple[str, float]]],
) -> dict:
    """Return a copy of atom with its value corrupted according to mode."""
    a = dict(atom)
    original = float(a.get("value", 0))
    concept = a["concept"]
    entity = a["entity"]

    if mode == "missing":
        a["value"] = None
        a["_corruption"] = "missing"

    elif mode == "garbage":
        a["value"] = rng.choice(_HARD_GARBAGE)
        a["_corruption"] = "garbage"

    elif mode == "lookalike":
        a["value"] = None
        a["value_display"] = _make_lookalike_str(original)
        a["_corruption"] = "lookalike"

    elif mode == "cross":
        candidates = [(e, v) for e, v in concept_pool.get(concept, []) if e != entity]
        if candidates:
            a["value"] = rng.choice(candidates)[1]
            a["_corruption"] = "cross"
        else:
            a["value"] = None
            a["_corruption"] = "missing"

    elif mode == "combined":
        roll = rng.random()
        if roll < 0.25:
            return _corrupt_atom_value("missing", atom, rng, concept_pool)
        elif roll < 0.50:
            return _corrupt_atom_value("garbage", atom, rng, concept_pool)
        elif roll < 0.75:
            return _corrupt_atom_value("lookalike", atom, rng, concept_pool)
        else:
            return _corrupt_atom_value("cross", atom, rng, concept_pool)

    return a


# ---------------------------------------------------------------------------
# Main corruption writer
# ---------------------------------------------------------------------------

def write_corrupted_atoms(
    atoms_path: Path,
    used_keys: set[str],
    output_path: Path,
    mode: Mode,
    corruption_rate: float = 0.4,
    seed: int | None = None,
) -> dict[str, int]:
    """Write a corrupted copy of atoms_10q.json to output_path.

    Only atoms whose key is NOT in used_keys are eligible for corruption.
    Returns a stats dict.
    """
    rng = random.Random(seed)
    atoms = load_atoms(atoms_path)
    concept_pool = _build_concept_pool(atoms)

    stats: dict[str, int] = {"protected": 0, "corrupted": 0, "untouched": 0}
    corrupted: list[dict] = []

    for atom in atoms:
        key = atom.get("key", "")
        if key in used_keys:
            stats["protected"] += 1
            corrupted.append(dict(atom))
        elif atom.get("semantic_type") == "amount" and rng.random() < corruption_rate:
            corrupted.append(_corrupt_atom_value(mode, atom, rng, concept_pool))
            stats["corrupted"] += 1
        else:
            stats["untouched"] += 1
            corrupted.append(dict(atom))

    save_atoms(corrupted, output_path)
    print(f"[{mode}] -> {output_path.name}")
    print(f"  Protected: {stats['protected']}  Corrupted: {stats['corrupted']}  Untouched: {stats['untouched']}")
    return stats


# ---------------------------------------------------------------------------
# Unit scale helpers
# ---------------------------------------------------------------------------

def pick_unit_scale() -> tuple[str, float]:
    return random.choice(UNIT_SCALE_OPTIONS)


def scale_atoms(atoms: list[dict], factor: float) -> list[dict]:
    """Return atoms with all M_USD values divided by factor."""
    scaled = []
    for a in atoms:
        if a.get("unit") == M_USD_UNIT and a.get("value") is not None:
            s = dict(a)
            s["value"] = float(a["value"]) / factor
            scaled.append(s)
        else:
            scaled.append(dict(a))
    return scaled


def unit_scale_question_prefix(unit_label: str) -> str:
    return (
        f"The numbers in the financial data are all given in {unit_label}. "
        f"Please answer in numbers, not {unit_label}. "
    )


# ---------------------------------------------------------------------------
# ObstacleContext10Q — query obstacle application
# ---------------------------------------------------------------------------

@dataclass
class ObstacleContext10Q:
    question: str
    leaf_atoms: list[dict]
    all_atoms: list[dict]
    base_dir: Path
    unit_scale_label: str | None = None
    useless_info_family: str | None = None

    def _entity(self) -> str | None:
        for a in self.leaf_atoms:
            if a.get("entity"):
                return a["entity"]
        return None

    def _prepend_or_append(self, snippet: str) -> str:
        snippet = snippet.strip()
        if not snippet.endswith((".", "!", "?")):
            snippet += "."
        if random.random() < 0.5:
            return f"{snippet} {self.question}"
        return f"{self.question} {snippet}"

    def apply_useless_info(self) -> str:
        templates_path = self.base_dir / ".." / "compiler_pipeline_refactored" / USELESS_INFO_TEMPLATES_FILE
        families = select_useless_info_families(
            load_useless_info_clauses(templates_path),
            self.useless_info_family,
        )
        snippet = format_useless_info_clause(pick_useless_info_clause(families))
        return self._prepend_or_append(snippet)

    def apply_negation(self) -> str:
        if random.random() < 0.5:
            return random.choice(NEGATION_PREFIXES) + self.question
        return self.question + random.choice(NEGATION_SUFFIXES)

    def apply_unit_scale_change(self) -> str:
        if not self.unit_scale_label:
            raise ValueError("unit_scale_label is required for unit_scale_change")
        return unit_scale_question_prefix(self.unit_scale_label) + self.question

    def apply_conditional(self) -> str:
        entity = self._entity()
        revenue_atoms = [
            a for a in self.all_atoms
            if a.get("concept") == "total_revenues"
            and (entity is None or a.get("entity") == entity)
            and a.get("value") is not None
        ]
        if not revenue_atoms:
            revenue_atoms = [
                a for a in self.all_atoms
                if a.get("concept") == "total_revenues" and a.get("value") is not None
            ]
        if not revenue_atoms:
            return self.question

        atom = random.choice(revenue_atoms)
        entity_name = atom.get("entity", "the company")
        period = atom.get("period", "the period")
        revenue = float(atom["value"])
        threshold = int(revenue * random.uniform(0.3, 0.7))

        prefix = f"If total revenues for {entity_name} in {period} exceed {threshold:,}, "
        return prefix + self.question[0].lower() + self.question[1:]

    def apply(self, name: str) -> str:
        if name not in QUERY_OBSTACLES:
            raise ValueError(f"Unknown query obstacle {name!r}. Choose from: {QUERY_OBSTACLES}")
        return getattr(self, f"apply_{name}")()


# ---------------------------------------------------------------------------
# Generate all corruption variants
# ---------------------------------------------------------------------------

def generate_all_corruptions(
    atoms_path: Path,
    questions_path: Path,
    output_dir: Path,
    corruption_rate: float = 0.4,
    seed: int | None = None,
) -> None:
    """Write one corrupted atoms JSON per mode into output_dir."""
    used_keys = collect_used_atom_keys(questions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_seed = seed or 0
    for i, mode in enumerate(("missing", "garbage", "lookalike", "cross", "combined")):
        write_corrupted_atoms(
            atoms_path=atoms_path,
            used_keys=used_keys,
            output_path=output_dir / f"atoms_{mode}.json",
            mode=mode,
            corruption_rate=corruption_rate,
            seed=base_seed + i,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate adversarial 10Q atom corruptions.")
    parser.add_argument("--atoms", default="output/atoms_10q.json")
    parser.add_argument("--questions", default="output/random_questions_10q.csv")
    parser.add_argument("--output-dir", default="output/adversarial")
    parser.add_argument("--corruption-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    base = Path(__file__).parent
    generate_all_corruptions(
        atoms_path=base / args.atoms,
        questions_path=base / args.questions,
        output_dir=base / args.output_dir,
        corruption_rate=args.corruption_rate,
        seed=args.seed,
    )
