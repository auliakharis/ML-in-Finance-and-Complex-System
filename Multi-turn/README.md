# Multi-turn

Converts single compositional financial questions into ordered multi-turn conversation chains by decomposing the expression tree into sequential sub-questions.

## Contents

| File | Description |
|---|---|
| `multi_turn_parser.py` | Main parser — builds turn chains from expression trees |
| `output.json` | Multi-turn output for the original 90q dataset |
| `output_augmented.json` | Multi-turn output for the augmented 90q dataset |

## How It Works

Given a compound question like:
> "What is the growth of revenue for Acme from 2021 to 2023 divided by the average net income from 2020 to 2022?"

The parser traverses the expression tree post-order and emits one turn per operator node:

| Turn | Question | Ground Truth |
|---|---|---|
| 1 | What is the growth of revenue for Acme from 2021 to 2023? | 0.15 |
| 2 | What is the average net income for Acme from 2020 to 2022? | 120M |
| 3 (final) | *original compound question* | 0.00125 |

**Collapsing nested min/max:** Nested binary `min`/`max` nodes of the same op (e.g., `min(min(2021,2022),2023)`) are collapsed into a single turn covering the full year range.

## Usage

```bash
python multi_turn_parser.py --input ../90q/random_questions_90.json --output output.json

# With a financial context sheet prepended to the first turn:
python multi_turn_parser.py \
  --input ../90q/random_questions_90_augmented.json \
  --output output_augmented.json \
  --sheet ../90q/financial_spreadsheet.csv
```

## Output Format

Each record in the output JSON contains:
- `question_id`, `original_question`, `answer`, `depth`
- `num_turns` — number of turns generated
- `turns` — list of `{turn_number, op, question, ground_truth, is_final}`
- `messages` — ready-to-use message array for LLM evaluation (`role: user/assistant` pairs with `{{FILL_MODEL_RESPONSE}}` placeholders)
