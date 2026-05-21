# Adversarial & Obstacle Pipeline

All commands are run from the `compiler_pipeline_refactored/` directory.

---

## Overview

There are two independent axes of difficulty:

| Category | What changes | Goal |
|---|---|---|
| **Data obstacles** | The numbers in the spreadsheet | Test if the LLM can extract the right values from noisy/scaled data |
| **Query obstacles** | The wording of the question | Test if the LLM can parse tricky question phrasing |

The two can be combined freely: pick any data obstacle + any query obstacle.

---

## The two-step pipeline

```bash
# Step 1 — generate all data variants (baseline, big_numbers, multi_factor)
python data_obstacles.py

# Step 2 — generate all question variants + adversarial corruptions
python query_obstacles.py
```

Everything below explains what each step produces and how to run subsets.

---

## Category 1 — Data Obstacles

These modify the spreadsheet that the LLM receives as context. The questions and ground-truth answers are unchanged.

Run with `data_obstacles.py`:

### 1a. `big_numbers` — scale all financial figures by a large factor

Multiplies numeric sampling ranges by 100× (default) so all values are very large.

```bash
python data_obstacles.py --obstacles big_numbers

# Custom scale factor
python data_obstacles.py --obstacles big_numbers --big-numbers-factor 1000
```

### 1b. `multi_factor` — multiply revenue/shares/price/employees at generation time

A direct multiplier applied during row generation (independent of `big_numbers`).

```bash
python data_obstacles.py --obstacles multi_factor

# Custom multiplier
python data_obstacles.py --obstacles multi_factor --multi-factor 500000
```

### 1c. Adversarial cell corruption

Corrupt cells that are **not** needed to answer any question. Generated automatically at the end of `query_obstacles.py` — it needs the questions file to know which cells are protected, so it runs after question generation.

```bash
# Runs automatically after question generation (default):
python query_obstacles.py

# Custom corruption rate and seed:
python query_obstacles.py --corruption-rate 0.6 --seed 42

# Skip adversarial if you only want the question variants:
python query_obstacles.py --skip-adversarial
```

| Output file | Corruption type |
|---|---|
| `missing_values.csv` | Unused cells replaced with empty strings |
| `garbage.csv` | Unused cells replaced with extreme values (`-9999999`, `999999999999`, `-1`, `"ERROR"`) |
| `lookalike.csv` | Digits replaced with visually similar characters (`1→I`, `0→O`, `2→Z`, `5→S`, `6→b`, `8→B`, `9→g`) |
| `cross_contaminated.csv` | Unused cells replaced with real values from a different company/year |
| `combined_adversarial.csv` | All four types mixed equally |

---

## Category 2 — Query Obstacles

These modify the **question text** after generation. The data and ground-truth answers are unchanged.

Run with `query_obstacles.py`:

### 2a. `unit_scale_change` — data presented in thousands/millions/billions/trillions

Values in the spreadsheet are divided by a random unit factor. Each question is prefixed with e.g. *"The numbers in the financial data are all given in billions. Please answer in numbers, not billions."*

```bash
python query_obstacles.py --obstacles unit_scale_change
```

### 2b. `useless_info` — inject irrelevant financial sentences

An irrelevant financial clause is randomly prepended or appended to each question, e.g. *"The company's dividend policy has remained unchanged since the 1980s. What is the ratio of..."*

```bash
python query_obstacles.py --obstacles useless_info
```

Available useless-info families (pass via `make_random_questions_adversarial.py --useless-info-family` for fine-grained control):
`governance_management`, `performance_sentiment`, `bankruptcy_distress`,
`future_numeric_adjustments`, `tax_discount_policy`, `inflation_macro`,
`peer_comparison`, `sector_country_context`, `accounting_adjustment`, `temporal_irrelevant`

### 2c. `conditional` — prepend a revenue threshold condition

Adds a conditional clause using real spreadsheet values, e.g. *"If revenue for Apex Corp in 2021 exceeds 54,320, what is the..."* The condition is always true so the answer is unchanged, but the LLM must parse it correctly.

```bash
python query_obstacles.py --obstacles conditional
```

### 2d. `negation` — wrap question with misleading double-negation phrasing

Adds a prefix or suffix that sounds confusing but does not change the required computation, e.g. *"Do not apply any adjustment; simply compute: What is the revenue..."*

```bash
python query_obstacles.py --obstacles negation
```

---

## Combining data and query obstacles

Generate a data variant, then point `query_obstacles.py` at it.

Example — `big_numbers` data + all query obstacles:

```bash
# Step 1: generate big_numbers data
python data_obstacles.py --obstacles big_numbers

# Step 2: generate all question variants from that data
python query_obstacles.py --data-dir output/data/big_numbers --seed 42
```

This produces questions and adversarial corruptions all based on the big_numbers spreadsheet.

---

## Selective runs

```bash
# Only specific data variants
python data_obstacles.py --obstacles baseline big_numbers

# Only specific query variants, no adversarial
python query_obstacles.py --obstacles baseline useless_info --skip-adversarial

# Use big_numbers data for question generation
python query_obstacles.py --data-dir output/data/big_numbers

# Quick test with 10 questions and fixed seed
python data_obstacles.py --obstacles baseline
python query_obstacles.py --n 10 --seed 42
```

---

## Output structure

```
output/
├── data/
│   ├── baseline/
│   │   ├── synthetic_company_data.csv
│   │   ├── financial_spreadsheet.json
│   │   ├── schema.json
│   │   └── atoms.json
│   ├── big_numbers/
│   │   └── (same files, scaled ×100)
│   └── multi_factor/
│       └── (same files, scaled ×1,000,000)
│
├── questions/
│   ├── baseline/
│   │   ├── questions.csv
│   │   └── questions.json
│   ├── unit_scale_change/
│   ├── useless_info/
│   ├── conditional/
│   └── negation/
│
└── adversarial/          (cell-level corruptions, based on baseline questions)
    ├── missing_values.csv
    ├── garbage.csv
    ├── lookalike.csv
    ├── cross_contaminated.csv
    └── combined_adversarial.csv
```
