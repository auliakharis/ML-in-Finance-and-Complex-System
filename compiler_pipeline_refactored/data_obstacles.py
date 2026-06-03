"""Generate all data obstacle variants.

Produces three spreadsheet variants, each in its own subdirectory under output/data/:

  baseline/     — clean synthetic data (no scaling)
  big_numbers/  — numeric sampling ranges scaled ×100 (default)
  multi_factor/ — revenue, shares, price, and employees scaled ×1,000,000 (default)

Each subdirectory contains:
  synthetic_company_data.csv, financial_spreadsheet.json, schema.json, atoms.json

Run:
  python data_obstacles.py                          # all three variants
  python data_obstacles.py --obstacles baseline big_numbers
  python data_obstacles.py --big-numbers-factor 500 --multi-factor 500000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from data_prep_adversarial import run_data_prep

BASE_DIR = Path(__file__).resolve().parent

VARIANTS: dict[str, dict] = {
    "baseline":     {"obstacle": None,          "multi_factor": 1.0},
    "big_numbers":  {"obstacle": "big_numbers", "multi_factor": 1.0},
    "multi_factor": {"obstacle": None,          "multi_factor": 1_000_000.0},
}


def run_variant(
    name: str,
    out_dir: Path,
    obstacle: str | None,
    multi_factor: float,
    big_numbers_factor: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    os.chdir(BASE_DIR)
    try:
        run_data_prep(
            obstacle=obstacle,
            big_numbers_factor=big_numbers_factor,
            multi_factor=multi_factor,
            csv_path=str(out_dir / "synthetic_company_data.csv"),
            schema_path=str(out_dir / "schema.json"),
            atoms_path=str(out_dir / "atoms.json"),
            financial_spreadsheet_path=str(out_dir / "financial_spreadsheet.json"),
        )
    finally:
        os.chdir(prev)
    print(f"  [{name}] -> {out_dir.relative_to(BASE_DIR)}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all data obstacle variants.")
    parser.add_argument(
        "--obstacles", nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help="Which variants to generate (default: all)",
    )
    parser.add_argument(
        "--big-numbers-factor", type=float, default=100.0,
        help="Scale factor for the big_numbers variant (default: 100)",
    )
    parser.add_argument(
        "--multi-factor", type=float, default=1_000_000.0,
        help="Scale factor for the multi_factor variant (default: 1000000)",
    )
    parser.add_argument(
        "--output-dir", default=str(BASE_DIR / "output" / "data"),
        help="Root output directory (default: output/data/)",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    print("Generating data variants...")

    for name in args.obstacles:
        v = dict(VARIANTS[name])
        v["big_numbers_factor"] = args.big_numbers_factor if name == "big_numbers" else 100.0
        v["multi_factor"] = args.multi_factor if name == "multi_factor" else v["multi_factor"]
        run_variant(name, out_root / name, **v)

    print(f"\nDone. All data written to {out_root}/")


if __name__ == "__main__":
    main()
