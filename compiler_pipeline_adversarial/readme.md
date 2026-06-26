# Compiler pipeline — overview

## Layout

**`__init__.py`** defines the public modules and symbols used across the codebase. Do not move, delete, or rename it.

**`5.make_random_questions.py`** exists only so older scripts that call that filename keep working. It delegates to **`make_random_questions.py`** (run that file for the real entry point).

**`utils.py`** holds small shared helpers (e.g. Oxford-style list joining, picking a contiguous period window for time aggregates).

**`config/`** holds JSON templates for building the initial spreadsheet: companies, categorical columns, yearly numeric columns, and concept metadata.

**`tree.py`** defines the expression-tree model (nodes, leaves, derived concepts, binding, sampling). It is not a pipeline “step” on its own; it provides the abstract structure and the functions that create, validate, bind, and serialize trees.

Other core modules:

- **`semantic_analyzer.py`** — walks a bound tree and builds human-readable meanings at each node.
- **`evaluator.py`** — computes the numeric answer for a bound tree.
- **`question_renderer.py`** — turns semantic analysis into a natural-language question string.
- **`data_prep.py`** —  data generation (spreadsheet + atoms + schema); run separately before question generation if you need fresh synthetic data.

Outputs go under **`compiler_pipeline/output/`** (CSV, atoms, questions).

---

## How to run

From the **`compiler_pipeline/`** directory (with `config/` available via paths relative to the repo — run from the project root or ensure `config/` resolves correctly):

```bash
python make_random_questions.py
# or, for backward compatibility:
python 5.make_random_questions.py
```

Optional: regenerate synthetic data first:

```bash
python data_prep.py
```

That writes `output/synthetic_company_data.csv`, `output/schema.json`, and `output/atoms.json`.

Question generation defaults: read `output/synthetic_company_data.csv`, write `output/random_questions_90.csv`, 90 questions. See `python make_random_questions.py --help` for `--n`, depth, seeds, and paths.

---

## The pipeline

### 0. Data prep (optional, `data_prep.py`)

Run when you need a new synthetic dataset. This step:

1. Loads **`config/`** (companies, column definitions, concept metadata).
2. Generates **`output/synthetic_company_data.csv`** — one row per company per year with financial fields.
3. Writes **`output/schema.json`** describing columns.
4. Builds **`output/atoms.json`** — one atom per spreadsheet cell (entity, period, concept, value, units, etc.).

`make_random_questions.py` can build atoms in memory from the CSV + concept metadata, so you do not have to run `data_prep.py` before every question run if the CSV already exists.

### 1. Load data and build atoms (`make_random_questions.py`)

Reads the spreadsheet CSV and concept metadata, then builds an in-memory `Atom` index used by binding, analysis, and evaluation.

### 2. Sample a symbolic tree (`tree.py`)

For each question, randomly samples a **template** expression tree (depth, derived-concept probability, operators) with rejection until the tree is valid (no forbidden duplicate derived forms, etc.).

### 3. Bind the tree to real data (`tree.py`)

**`Expr.instantiate_typed_tree`** maps template leaves to concrete atoms (company, year, concept), expands derived concepts, and instantiates time aggregates (min / max / avg over a window of periods). The result is a fully **bound** expression tree tied to the spreadsheet.

### 4. Semantic analysis (`semantic_analyzer.py`)

**`SemanticAnalyzer.analyze`** walks the bound tree and produces an **`AnalysisResult`** at each node: natural-language **meanings** (e.g. “revenue for Apple in 2021”, “sum of …”) that describe what the expression represents.

### 5. Evaluation (`evaluator.py`)

**`Evaluator.eval`** recursively computes the numeric **answer** from atom values and operators (sum, diff, ratio, mul, growth, min, max).

### 6. Question rendering (`question_renderer.py`)

**`QuestionRenderer.render`** picks a phrasing template and formats the root meaning into an English **question** string.

### 7. Write dataset row

Each successful question becomes one row in the output CSV: question text, answer, expression JSON/string, template metadata, seeds, and per-leaf atom details (`leaf_1_*`, `leaf_2_*`, …).

Failed samples are retried (up to 1000 attempts per question) if binding, analysis, or evaluation raises.

---

## Tests

From `compiler_pipeline/`:

```bash
pytest unit_tests/
```

Steps are grouped as `test_step1_generate` (data prep), `test_step2_build_atoms`, `test_step3_tree_sampler`, `test_step4_tree_to_language` (bind, analyze, evaluate, render), and `test_step5_make_questions` (end-to-end row building).
