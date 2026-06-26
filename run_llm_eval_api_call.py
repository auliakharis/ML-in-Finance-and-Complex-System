"""
run_llm_eval.py
====================
Runs LLM evaluation on two Q&A datasets:

  10q  — 10q/final_qa_dataset.json
         context: one spreadsheet row per company from 10q/financial_spreadsheet.json

  90q  — 90q/random_questions_90.json
         context: all yearly rows for the entity from 90q/financial_spreadsheet.json

For each question, a prompt is built:
  "You are a financial data assistant. Given the following financial sheets statements below
   [company financial sheet]
   Answer the following question using only the given instruction.
   [question]
   Answer only the final numeric value and give your reasoning."

Models evaluated (loaded from /cluster/scratch/$USER/models/):
  - Qwen3.5-4B
  - Qwen3.5-9B

Accuracy is computed as the fraction of answers within a relative tolerance
of the ground-truth numeric answer (default ±1%).

Usage:
  python run_llm_eval.py
  python run_llm_eval.py --limit 20 --tol 0.1
  python run_llm_eval.py --models Qwen3.5-4B --limit 10
  python run_llm_eval.py --output results.json

for the multi turn : 
the system message (financial data) is added once at the start of history and stays there for all turns — it's never re-added. But because history is passed in full each time, the model does see it on every turn:

history = [system_msg]          # ← financial data added once here

# Turn 1: history = [system_msg, user1]                            ✓ data present
# Turn 2: history = [system_msg, user1, asst1, user2]             ✓ data present  
# Turn 3: history = [system_msg, user1, asst1, user2, asst2, user3] ✓ data present
Since system_msg is always the first element in history, it's included in every apply_chat_template call. So the model sees the financial sheets on every single turn — it's just sent once in the list rather than duplicated.

"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

_client = openai.Client(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.svc.cscs.ch/v1",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODELS_DIR = Path(f"/cluster/scratch/{os.environ.get('USER', 'user')}/models")

DATASET_10Q    = BASE_DIR / "compiler_pipeline_refactored_10Q" / "output" / "random_questions_10q.csv"
ATOMS_10Q      = BASE_DIR / "compiler_pipeline_refactored_10Q" / "output" / "atoms_10q.json"
DATASET_MT_10Q = BASE_DIR / "compiler_pipeline_refactored_10Q" / "output" / "multi_turn_10q.json"

_ADV_DIR_10Q = BASE_DIR / "compiler_pipeline_refactored_10Q" / "output" / "adversarial"
ADV_ATOMS_10Q: dict[str, Path] = {
    "10q_missing":   _ADV_DIR_10Q / "atoms_missing.json",
    "10q_garbage":   _ADV_DIR_10Q / "atoms_garbage.json",
    "10q_lookalike": _ADV_DIR_10Q / "atoms_lookalike.json",
    "10q_cross":     _ADV_DIR_10Q / "atoms_cross.json",
    "10q_combined":  _ADV_DIR_10Q / "atoms_combined.json",
}

DATASET_90Q = BASE_DIR / "compiler_pipeline_refactored" / "output" / "random_questions_90_1000.csv"
SHEET_90Q   = BASE_DIR / "compiler_pipeline_refactored" / "output" / "synthetic_company_data.csv"

DATASET_MT  = BASE_DIR / "dataset_output" / "multi_turn_and_augmented_questions.json"
SHEET_MT    = BASE_DIR / "90q" / "financial_spreadsheet.json"  # same synthetic companies

DEFAULT_MODELS = ["Qwen/Qwen3.5-27B"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def _parse_period_date(period: str):
    from datetime import datetime, date
    date_str = period.split("Ended ")[-1].strip() if "Ended " in period else period.strip()
    for fmt in ("%B %d, %Y", "%B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    return date.min


def build_10q_sheet_lookup(atoms: list) -> dict:
    """entity -> list of atom dicts, grouped for 10Q prompt context."""
    lookup: dict = {}
    for atom in atoms:
        lookup.setdefault(atom["entity"], []).append(atom)
    return lookup


def load_10q_questions(path: Path) -> list:
    """Load 10Q compiler pipeline questions from CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_90q_sheet_lookup(sheet: list) -> dict:
    """company_name -> list of rows (one per year), sorted by year."""
    lookup: dict = {}
    for row in sheet:
        name = row["company_name"]
        lookup.setdefault(name, []).append(row)
    for name in lookup:
        lookup[name].sort(key=lambda r: r.get("year", "0"))
    return lookup


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

# Balance sheet concept → section mapping (derived from BS_ACCESSORS in data_prep_10q.py)
_BS_SECTION: dict[str, str] = {
    "cash_and_cash_equivalents":         "Current Assets",
    "short_term_investments":            "Current Assets",
    "accounts_receivable_net":           "Current Assets",
    "inventories":                       "Current Assets",
    "prepaid_expenses_and_other":        "Current Assets",
    "long_term_marketable_securities":   "Non-Current Assets",
    "property_plant_and_equipment_net":  "Non-Current Assets",
    "goodwill":                          "Non-Current Assets",
    "other_non_current_assets":          "Non-Current Assets",
    "accounts_payable":                  "Current Liabilities",
    "deferred_revenue_current":          "Current Liabilities",
    "accrued_expenses_and_other":        "Current Liabilities",
    "current_portion_of_long_term_debt": "Current Liabilities",
    "long_term_debt":                    "Non-Current Liabilities",
    "other_non_current_liabilities":     "Non-Current Liabilities",
    "common_stock_and_additional_paid_in_capital": "Shareholders Equity",
    "retained_earnings":                 "Shareholders Equity",
    "accumulated_other_comprehensive_income_loss": "Shareholders Equity",
    # derived totals (shown when leaf_only=False)
    "total_current_assets":              "Current Assets",
    "total_non_current_assets":          "Non-Current Assets",
    "total_assets":                      "Total Assets",
    "total_current_liabilities":         "Current Liabilities",
    "total_non_current_liabilities":     "Non-Current Liabilities",
    "total_liabilities":                 "Total Liabilities",
    "total_shareholders_equity":         "Shareholders Equity",
}

_BS_SECTION_ORDER = [
    "Current Assets", "Non-Current Assets", "Total Assets",
    "Current Liabilities", "Non-Current Liabilities", "Total Liabilities",
    "Shareholders Equity",
]

# Explicit item order within each balance sheet section (matches real 10-Q layout)
_BS_ITEM_ORDER: dict[str, list] = {
    "Current Assets":       ["cash_and_cash_equivalents", "short_term_investments", "accounts_receivable_net", "inventories", "prepaid_expenses_and_other", "total_current_assets"],
    "Non-Current Assets":   ["long_term_marketable_securities", "property_plant_and_equipment_net", "goodwill", "other_non_current_assets", "total_non_current_assets"],
    "Total Assets":         ["total_assets"],
    "Current Liabilities":  ["accounts_payable", "deferred_revenue_current", "accrued_expenses_and_other", "current_portion_of_long_term_debt", "total_current_liabilities"],
    "Non-Current Liabilities": ["long_term_debt", "other_non_current_liabilities", "total_non_current_liabilities"],
    "Total Liabilities":    ["total_liabilities"],
    "Shareholders Equity":  ["common_stock_and_additional_paid_in_capital", "retained_earnings", "accumulated_other_comprehensive_income_loss", "total_shareholders_equity"],
}

# Income statement concept → section (for "Ended" periods)
_OPS_SECTION: dict[str, str] = {
    "total_revenues":                    "Revenue",
    "cost_of_sales":                     "Cost of Sales",
    "gross_profit":                      "Gross Profit",
    "research_and_development":          "Operating Expenses",
    "selling_general_and_administrative":"Operating Expenses",
    "total_costs_and_expenses":          "Total Costs and Expenses",
    "operating_income":                  "Operating Income",
    "other_income_expense_net":          "Other Income / Expense",
    "provision_for_income_taxes":        "Income Tax",
    "net_income":                        "Net Income",
    "earnings_per_share_basic":          "Earnings Per Share",
    "earnings_per_share_diluted":        "Earnings Per Share",
    "shares_used_basic":                 "Shares Used in Computing EPS",
    "shares_used_diluted":               "Shares Used in Computing EPS",
}

_OPS_SECTION_ORDER = [
    "Revenue", "Cost of Sales", "Gross Profit", "Operating Expenses",
    "Total Costs and Expenses", "Operating Income",
    "Other Income / Expense", "Income Tax", "Net Income",
    "Earnings Per Share", "Shares Used in Computing EPS",
]

# Explicit item order within each income statement section
_OPS_ITEM_ORDER: dict[str, list] = {
    # Revenue and Cost of Sales: aggregates go last (breakdowns first)
    "Gross Profit":              ["gross_profit"],
    "Operating Expenses":        ["research_and_development", "selling_general_and_administrative"],
    "Total Costs and Expenses":  ["total_costs_and_expenses"],
    "Operating Income":          ["operating_income"],
    "Other Income / Expense":    ["other_income_expense_net"],
    "Income Tax":                ["provision_for_income_taxes"],
    "Net Income":                ["net_income"],
    "Earnings Per Share":        ["earnings_per_share_basic", "earnings_per_share_diluted"],
    "Shares Used in Computing EPS": ["shares_used_basic", "shares_used_diluted"],
}

# Cash flow concept → section
_CF_SECTION: dict[str, str] = {
    "depreciation_and_amortization":          "Operating Activities",
    "stock_based_compensation":               "Operating Activities",
    "change_in_accounts_receivable":          "Operating Activities",
    "change_in_inventories":                  "Operating Activities",
    "change_in_accounts_payable":             "Operating Activities",
    "change_in_other_working_capital":        "Operating Activities",
    "net_cash_from_operating":                "Operating Activities",
    "capital_expenditures":                   "Investing Activities",
    "purchases_of_marketable_securities":     "Investing Activities",
    "proceeds_from_maturities_of_securities": "Investing Activities",
    "net_cash_from_investing":                "Investing Activities",
    "repayments_of_debt":                     "Financing Activities",
    "share_repurchases":                      "Financing Activities",
    "proceeds_from_stock_option_exercises":   "Financing Activities",
    "dividends_paid":                         "Financing Activities",
    "net_cash_from_financing":                "Financing Activities",
    "effect_of_exchange_rate_on_cash":        "Net Change in Cash",
    "opening_cash_and_equivalents":           "Net Change in Cash",
    "net_change_in_cash":                     "Net Change in Cash",
}

_CF_SECTION_ORDER = [
    "Operating Activities", "Investing Activities",
    "Financing Activities", "Net Change in Cash",
]

# Explicit item order within each cash flow section (totals last)
_CF_ITEM_ORDER: dict[str, list] = {
    "Operating Activities": ["depreciation_and_amortization", "stock_based_compensation", "change_in_accounts_receivable", "change_in_inventories", "change_in_accounts_payable", "change_in_other_working_capital", "net_cash_from_operating"],
    "Investing Activities": ["capital_expenditures", "purchases_of_marketable_securities", "proceeds_from_maturities_of_securities", "net_cash_from_investing"],
    "Financing Activities": ["repayments_of_debt", "share_repurchases", "proceeds_from_stock_option_exercises", "dividends_paid", "net_cash_from_financing"],
    "Net Change in Cash":   ["effect_of_exchange_rate_on_cash", "opening_cash_and_equivalents", "net_change_in_cash"],
}

_CF_CONCEPTS = set(_CF_SECTION.keys())

# Comprehensive income statement concept → section
_CI_SECTION: dict[str, str] = {
    "foreign_currency_translation":          "Other Comprehensive Income",
    "unrealized_gains_losses_on_securities": "Other Comprehensive Income",
    "total_other_comprehensive_income":      "Total OCI",
    "comprehensive_income":                  "Total Comprehensive Income",
}

_CI_SECTION_ORDER = [
    "Other Comprehensive Income",
    "Total OCI",
    "Total Comprehensive Income",
]

_CI_ITEM_ORDER: dict[str, list] = {
    "Other Comprehensive Income": ["foreign_currency_translation", "unrealized_gains_losses_on_securities"],
    "Total OCI":                  ["total_other_comprehensive_income"],
    "Total Comprehensive Income": ["comprehensive_income"],
}

_CI_CONCEPTS = set(_CI_SECTION.keys())


def _atom_section(a: dict, is_balance_sheet: bool) -> str:
    """Return the display section for an atom."""
    concept = a["concept"]
    parent = a.get("parent_concept") or ""
    if is_balance_sheet:
        return _BS_SECTION.get(concept, "Other")
    if concept in _CF_CONCEPTS:
        return _CF_SECTION[concept]
    if concept in _CI_CONCEPTS:
        return _CI_SECTION[concept]
    if concept in _OPS_SECTION:
        return _OPS_SECTION[concept]
    if parent == "total_revenues":
        return "Revenue"
    if parent == "cost_of_sales":
        return "Cost of Sales"
    return "Other"


def _sort_section_items(items: list, section: str, is_bs: bool) -> list:
    """Sort atoms within a section in logical financial order (matching real 10-Q layout)."""
    if is_bs:
        order = _BS_ITEM_ORDER.get(section, [])
    elif section in _CF_ITEM_ORDER:
        order = _CF_ITEM_ORDER[section]
    elif section in _CI_ITEM_ORDER:
        order = _CI_ITEM_ORDER[section]
    elif section in _OPS_ITEM_ORDER:
        order = _OPS_ITEM_ORDER[section]
    elif section == "Revenue":
        # Breakdown items alphabetically, total_revenues last
        return sorted(items, key=lambda a: (1 if a["concept"] == "total_revenues" else 0, a.get("label") or a["concept"]))
    elif section == "Cost of Sales":
        # Breakdown items alphabetically, cost_of_sales last
        return sorted(items, key=lambda a: (1 if a["concept"] == "cost_of_sales" else 0, a.get("label") or a["concept"]))
    else:
        order = []

    def key(a):
        try:
            return (order.index(a["concept"]), a.get("label") or a["concept"])
        except ValueError:
            return (len(order), a.get("label") or a["concept"])

    return sorted(items, key=key)


def _render_sections(lines: list, period_atoms: list, section_order: list, item_order_map: dict, is_bs: bool, indent: str) -> None:
    """Group atoms by section and render with headers and ordered items."""
    from collections import defaultdict
    by_section: dict = defaultdict(list)
    for a in period_atoms:
        by_section[_atom_section(a, is_bs)].append(a)

    seen: set = set()
    ordered = [s for s in section_order if s in by_section]
    remaining = [s for s in by_section if s not in section_order]
    for section in ordered + remaining:
        if section in seen:
            continue
        seen.add(section)
        lines.append(f"{indent}[{section}]")
        for a in _sort_section_items(by_section[section], section, is_bs):
            unit = f" ({a['unit']})" if a.get("unit") else ""
            display = a.get("value_display") or (a["value"] if a.get("value") is not None else "")
            lines.append(f"{indent}  {a.get('label') or a['concept']}{unit}: {display}")


def sheet_to_text_10q(atoms: list, leaf_only: bool = False) -> str:
    """Format 10Q atoms as four labelled statement blocks matching real 10-Q structure.

    Statements: Income Statement → Balance Sheet → Cash Flow → Comprehensive Income.
    Within each section items follow real filing order (not alphabetical).
    If leaf_only=True, derived atoms (role='derived') are excluded.
    """
    from collections import defaultdict
    entity = None
    ops_by_period: dict = defaultdict(list)
    bs_by_period:  dict = defaultdict(list)
    cf_by_period:  dict = defaultdict(list)
    ci_by_period:  dict = defaultdict(list)

    for a in atoms:
        if entity is None:
            entity = a.get("entity")
        if leaf_only and a.get("role") == "derived":
            continue
        period = a["period"]
        if "Ended" not in period:
            bs_by_period[period].append(a)
        elif a["concept"] in _CF_CONCEPTS:
            cf_by_period[period].append(a)
        elif a["concept"] in _CI_CONCEPTS:
            ci_by_period[period].append(a)
        else:
            ops_by_period[period].append(a)

    lines = []
    if entity:
        lines.append(f"  Entity: {entity}")
        lines.append("")

    if ops_by_period:
        lines.append("  === Income Statement ===")
        for period in sorted(ops_by_period, key=_parse_period_date):
            lines.append(f"    {period}:")
            _render_sections(lines, ops_by_period[period], _OPS_SECTION_ORDER, _OPS_ITEM_ORDER, is_bs=False, indent="      ")

    if bs_by_period:
        lines.append("")
        lines.append("  === Balance Sheet ===")
        for period in sorted(bs_by_period, key=_parse_period_date):
            lines.append(f"    {period}:")
            _render_sections(lines, bs_by_period[period], _BS_SECTION_ORDER, _BS_ITEM_ORDER, is_bs=True, indent="      ")

    if cf_by_period:
        lines.append("")
        lines.append("  === Cash Flow Statement ===")
        for period in sorted(cf_by_period, key=_parse_period_date):
            lines.append(f"    {period}:")
            _render_sections(lines, cf_by_period[period], _CF_SECTION_ORDER, _CF_ITEM_ORDER, is_bs=False, indent="      ")

    if ci_by_period:
        lines.append("")
        lines.append("  === Comprehensive Income Statement ===")
        for period in sorted(ci_by_period, key=_parse_period_date):
            lines.append(f"    {period}:")
            _render_sections(lines, ci_by_period[period], _CI_SECTION_ORDER, _CI_ITEM_ORDER, is_bs=False, indent="      ")

    return "\n".join(lines)


def sheet_to_text_90q(rows: list) -> str:
    """Format multiple annual rows for a company as a table-like text."""
    if not rows:
        return "(no data)"
    lines = []
    for row in rows:
        year = row.get("year", "?")
        lines.append(f"  Year {year}:")
        for k, v in row.items():
            if k not in ("company_name", "year"):
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def find_company(question_text: str, sheet_lookup: dict) -> str | None:
    """Return the first company name from sheet_lookup that appears in question_text."""
    for name in sheet_lookup:
        if name in question_text:
            return name
    return None


def build_prompt(sheet_text: str, question: str) -> str:
    return (
    "ROLE: Financial analyst. Answer using ONLY the data below. No commentary.\n\n"
    "=== DATA ===\n"
    f"{sheet_text}\n"
    "=== END ===\n\n"
    f"QUESTION: {question}\n\n"
    "RULES:\n"
    "- Extract only the numbers you need.\n"
    "- Show calculations in 1–3 lines max if needed.\n"
    "- Last line MUST be: Answer: <value>\n"
    "- <value> is either a number, True, or False. Nothing else.\n"
    "- Do NOT explain, summarize, or add anything after the Answer line.\n\n"
    "- Do NOT show your thinking process.\n\n" 
    "SOLVE NOW:"
    )

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_number(text: str) -> float | bool | None:
    """
    Extract the value from the 'Answer: <value>' line.
    Handles numbers, True/False. Falls back to last number if no Answer line found.
    """
    # 1. Look for explicit "Answer: <value>" line (case-insensitive)
    match = re.search(r'Answer\s*:\s*(.+)', text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip().rstrip(".,;")

        # Check for True/False first
        if raw.lower() == "true":
            return True
        if raw.lower() == "false":
            return False

        # Try to parse as number
        cleaned = re.sub(r'(\d),(\d)', r'\1\2', raw)
        num_match = re.search(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned)
        if num_match:
            return float(num_match.group())

    # 2. Fallback: last number in text (less reliable)
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
    matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned)
    return float(matches[-1]) if matches else None


def is_correct(predicted, ground_truth, tol: float) -> bool:
    if predicted is None or ground_truth is None:
        return False

    # Boolean comparison
    if isinstance(predicted, bool) or isinstance(ground_truth, bool):
        return predicted == ground_truth

    # Coerce string ground truth to float; skip if not numeric
    if isinstance(ground_truth, str):
        try:
            ground_truth = float(ground_truth.replace(",", ""))
        except ValueError:
            return False

    # Numeric comparison
    if ground_truth == 0:
        return abs(predicted) < 1e-6
    return abs(predicted - ground_truth) / abs(ground_truth) <= tol


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------

def strip_thinking_tags(text: str) -> str:
    """Extract only the final message content, discarding thinking blocks and special tokens."""
    # The format is: <|channel|>analysis<|message|>THINKING<|end|><|start|>assistant<|channel|>final<|message|>ANSWER
    # Split on <|message|> and take the last segment as the final answer content
    parts = re.split(r'<\|message\|>', text)
    if len(parts) > 1:
        last = parts[-1]
        # Strip any trailing <|end|> or other special tokens
        last = re.sub(r'<\|[^|]*\|>', '', last)
        return last.strip()
    # Fallback: strip all special token blocks and standalone tokens
    text = re.sub(r'<\|(?!end\|)[^|]*\|>.*?<\|end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|[^|]*\|>', '', text)
    return text.strip()


def _sanitize_messages(messages: list) -> list:
    """Strip thinking tags from all assistant messages before sending to API."""
    sanitized = []
    for msg in messages:
        if msg.get("role") == "assistant":
            sanitized.append({**msg, "content": strip_thinking_tags(msg["content"])})
        else:
            sanitized.append(msg)
    return sanitized


def run_inference(model_name: str, prompt: str, max_new_tokens: int = 2048, thinking: bool = False) -> str:
    """Call the API with a single user prompt and return the response text."""
    system_msg = (
        "You are a financial analyst. Think carefully before answering."
        if thinking else
        "Do not show your thinking process. Output only the answer."
    )
    response = _client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_new_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    content = response.choices[0].message.content.strip()
    if thinking:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


def run_multiturn_inference(model_name: str, messages: list, max_new_tokens: int = 2048) -> str:
    """Call the API with a full conversation history and return the response text."""
    response = _client.chat.completions.create(
        model=model_name,
        messages=_sanitize_messages(messages),
        max_tokens=max_new_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def evaluate_10q(
    questions: list,
    sheet_lookup: dict,
    model_name: str,
    limit: int | None,
    tol: float,
    max_new_tokens: int = 2048,
    thinking: bool = False,
    leaf_only: bool = False,
) -> list:
    """Evaluate on 10Q compiler pipeline questions. Returns per-question result dicts."""
    results = []
    subset = questions[:limit] if limit else questions

    for i, q in enumerate(subset, 1):
        entity = q.get("leaf_1_entity", "")
        question_text = q.get("question", "")
        ground_truth = q.get("answer")

        atoms = sheet_lookup.get(entity)
        if not atoms:
            print(f"  [{i}/{len(subset)}] SKIP (no atoms for '{entity}')")
            results.append({
                "source": "10q",
                "id": q.get("question_id"),
                "entity": entity,
                "depth": q.get("depth"),
                "question": question_text,
                "ground_truth": ground_truth,
                "llm_response": None,
                "predicted": None,
                "correct": False,
                "skip": True,
            })
            continue

        sheet_text = sheet_to_text_10q(atoms, leaf_only=leaf_only)
        prompt = build_prompt(sheet_text, question_text)

        response = run_inference(model_name, prompt, max_new_tokens, thinking=thinking)
        predicted = extract_number(response)
        correct = is_correct(predicted, ground_truth, tol)

        status = "CORRECT" if correct else "WRONG "
        try:
            gt_display = f"{float(ground_truth):.4f}"
        except (TypeError, ValueError):
            gt_display = str(ground_truth)
        print(f"  [{i}/{len(subset)}] {status} | truth={gt_display} pred={predicted} | {question_text[:60]}...")

        results.append({
            "source": "10q",
            "id": q.get("question_id"),
            "entity": entity,
            "depth": q.get("depth"),
            "question": question_text,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "llm_response": response,
            "predicted": predicted,
            "correct": correct,
            "skip": False,
        })

    return results


def evaluate_90q(
    questions: list,
    sheet_lookup: dict,
    model_name: str,
    limit: int | None,
    tol: float,
    max_new_tokens: int = 2048,
) -> list:
    """Evaluate on 90-question dataset. Returns per-question result dicts."""
    results = []
    subset = questions[:limit] if limit else questions

    for i, q in enumerate(subset, 1):
        entity = q.get("entity") or q.get("leaf_1_entity", "")
        question_text = q.get("question", "")
        ground_truth = q.get("answer")

        sheet_rows = sheet_lookup.get(entity)
        if not sheet_rows:
            print(f"  [{i}/{len(subset)}] SKIP (no sheet for '{entity}')")
            results.append({
                "source": "90q",
                "id": q.get("id") or q.get("question_id"),
                "entity": entity,
                "depth": q.get("depth"),
                "question": question_text,
                "ground_truth": ground_truth,
                "llm_response": None,
                "predicted": None,
                "correct": False,
                "skip": True,
            })
            continue

        sheet_text = sheet_to_text_90q(sheet_rows)
        prompt = build_prompt(sheet_text, question_text)

        response = run_inference(model_name, prompt, max_new_tokens)
        predicted = extract_number(response)
        correct = is_correct(predicted, ground_truth, tol)

        status = "CORRECT" if correct else "WRONG "
        print(f"  [{i}/{len(subset)}] {status} | truth={float(ground_truth):.4f} pred={predicted} | {question_text[:60]}...")

        results.append({
            "source": "90q",
            "id": q.get("id") or q.get("question_id"),
            "entity": entity,
            "depth": q.get("depth"),
            "question": question_text,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "llm_response": response,
            "predicted": predicted,
            "correct": correct,
            "skip": False,
        })

    return results


def evaluate_multiturn(
    questions: list,
    sheet_lookup: dict,
    model_name: str,
    limit: int | None,
    tol: float,
    max_new_tokens: int = 512,
) -> list:
    """
    Evaluate multi-turn questions.

    For each question the model plays through all conversation turns
    sequentially. The financial context is injected as a system message.
    Accuracy is measured on the final turn against q['answer'].
    """
    results = []
    subset = [q for q in questions if not q.get("skipped")]
    if limit:
        subset = subset[:limit]

    for i, q in enumerate(subset, 1):
        qid = q.get("question_id")
        original_q = q.get("original_question", "")
        ground_truth = q.get("answer")
        num_turns = q.get("num_turns", 1)
        depth = q.get("depth")

        # Build system message with financial context
        company = find_company(original_q, sheet_lookup)
        system_msg = None
        if company:
            sheet_rows = sheet_lookup.get(company)
            if sheet_rows:
                sheet_text = sheet_to_text_90q(sheet_rows)
                system_content = (
                    "You are a financial analyst. Answer using ONLY the data below. "
                    "No commentary.\n\n"
                    "=== DATA ===\n"
                    f"{sheet_text}\n"
                    "=== END ===\n\n"
                    "RULES:\n"
                    "- Extract only the numbers you need.\n"
                    "- Show calculations in 1-3 lines max if needed.\n"
                    "- Last line MUST be: Answer: <value>\n"
                    "- <value> is either a number, True, or False. Nothing else.\n"
                    "- Do NOT explain or add anything after the Answer line.\n"
                    "- Do NOT show your thinking process.\n"
                )
                system_msg = {"role": "system", "content": system_content}

        if company is None:
            print(f"  [{i}/{len(subset)}] SKIP (company not found in spreadsheet) | {original_q[:60]}...")
            results.append({
                "source": "mt",
                "id": qid,
                "entity": None,
                "depth": depth,
                "num_turns": num_turns,
                "question": original_q,
                "ground_truth": ground_truth,
                "llm_response": None,
                "predicted": None,
                "correct": False,
                "skip": True,
            })
            continue

        # Replay conversation turn by turn
        history = [system_msg] if system_msg else []
        final_response = None
        turn_responses = []        # raw model output per turn
        turn_questions = []        # user question text per turn
        turn_prompt_snapshots = [] # full messages list sent to the model at each turn

        for msg in q.get("messages", []):
            if msg["role"] == "user":
                question_text = msg["content"].strip()
                history.append({"role": "user", "content": question_text})
                # Snapshot the full context sent to the model (copy before adding response)
                turn_prompt_snapshots.append(list(history))
                response = run_multiturn_inference(model_name, history, max_new_tokens)
                history.append({"role": "assistant", "content": strip_thinking_tags(response)})
                turn_responses.append(response)
                turn_questions.append(question_text)
                final_response = response
            # assistant placeholder messages are skipped — we fill them ourselves

        predicted = extract_number(final_response) if final_response else None
        correct = is_correct(predicted, ground_truth, tol)

        # Per-turn correctness against intermediate ground truths
        turns_meta = q.get("turns", [])
        turn_correctness = []
        first_failure_depth = None
        for idx, tr in enumerate(turns_meta):
            tr_gt = tr.get("ground_truth")
            tr_response = turn_responses[idx] if idx < len(turn_responses) else None
            tr_question = turn_questions[idx] if idx < len(turn_questions) else tr.get("question", "")
            tr_prompt = turn_prompt_snapshots[idx] if idx < len(turn_prompt_snapshots) else []
            tr_pred = extract_number(tr_response) if tr_response else None
            tr_correct = is_correct(tr_pred, tr_gt, tol)
            turn_correctness.append({
                "turn_number": tr.get("turn_number"),
                "op": tr.get("op"),
                "question": tr_question,
                "prompt_messages": tr_prompt,  # full messages list sent to model at this turn
                "ground_truth": tr_gt,
                "predicted": tr_pred,
                "llm_response": tr_response,
                "correct": tr_correct,
                "is_final": tr.get("is_final", False),
            })
            if not tr_correct and first_failure_depth is None:
                first_failure_depth = tr.get("turn_number")

        status = "CORRECT" if correct else "WRONG "
        try:
            gt_display = f"{float(ground_truth):.4f}"
        except (TypeError, ValueError):
            gt_display = str(ground_truth)
        failure_info = f" first_fail@turn={first_failure_depth}" if not correct and num_turns > 1 else ""
        print(f"  [{i}/{len(subset)}] {status} | turns={num_turns} depth={depth}{failure_info} | "
              f"truth={gt_display} pred={predicted} | {original_q[:50]}...")

        results.append({
            "source": "mt",
            "id": qid,
            "entity": company,
            "depth": depth,
            "num_turns": num_turns,
            "question": original_q,
            "ground_truth": ground_truth,
            "turn_responses": turn_responses,
            "turn_correctness": turn_correctness,
            "first_failure_depth": first_failure_depth,
            "llm_response": final_response,
            "predicted": predicted,
            "correct": correct,
            "skip": False,
        })

    return results


def evaluate_multiturn_10q(
    questions: list,
    sheet_lookup: dict,
    model_name: str,
    limit: int | None,
    tol: float,
    max_new_tokens: int = 2048,
) -> list:
    """Evaluate multi-turn questions on the 10Q dataset using atom context."""
    results = []
    subset = [q for q in questions if not q.get("skipped")]
    if limit:
        subset = subset[:limit]

    for i, q in enumerate(subset, 1):
        qid         = q.get("question_id")
        original_q  = q.get("original_question", "")
        ground_truth = q.get("answer")
        num_turns   = q.get("num_turns", 1)
        depth       = q.get("depth")
        entity      = q.get("entity", "")

        atoms = sheet_lookup.get(entity)
        if not atoms:
            print(f"  [{i}/{len(subset)}] SKIP (no atoms for '{entity}')")
            results.append({
                "source": "mt_10q", "id": qid, "entity": entity, "depth": depth,
                "num_turns": num_turns, "question": original_q, "ground_truth": ground_truth,
                "llm_response": None, "predicted": None, "correct": False, "skip": True,
            })
            continue

        sheet_text = sheet_to_text_10q(atoms)
        system_content = (
            "You are a financial analyst. Answer using ONLY the data below. No commentary.\n\n"
            "=== DATA ===\n"
            f"{sheet_text}\n"
            "=== END ===\n\n"
            "RULES:\n"
            "- Extract only the numbers you need.\n"
            "- Show calculations in 1-3 lines max if needed.\n"
            "- Last line MUST be: Answer: <value>\n"
            "- <value> is either a number, True, or False. Nothing else.\n"
            "- Do NOT explain or add anything after the Answer line.\n"
            "- Do NOT show your thinking process.\n"
        )
        system_msg = {"role": "system", "content": system_content}

        history = [system_msg]
        final_response = None
        turn_responses: list = []
        turn_questions: list = []

        for msg in q.get("messages", []):
            if msg["role"] == "user":
                question_text = msg["content"].strip()
                history.append({"role": "user", "content": question_text})
                response = run_multiturn_inference(model_name, history, max_new_tokens)
                history.append({"role": "assistant", "content": strip_thinking_tags(response)})
                turn_responses.append(response)
                turn_questions.append(question_text)
                final_response = response

        predicted = extract_number(final_response) if final_response else None
        correct   = is_correct(predicted, ground_truth, tol)

        turns_meta = q.get("turns", [])
        turn_correctness = []
        first_failure_depth = None
        for idx, tr in enumerate(turns_meta):
            tr_gt      = tr.get("ground_truth")
            tr_response = turn_responses[idx] if idx < len(turn_responses) else None
            tr_pred    = extract_number(tr_response) if tr_response else None
            tr_correct = is_correct(tr_pred, tr_gt, tol)
            turn_correctness.append({
                "turn_number": tr.get("turn_number"), "op": tr.get("op"),
                "question": turn_questions[idx] if idx < len(turn_questions) else "",
                "ground_truth": tr_gt, "predicted": tr_pred,
                "llm_response": tr_response, "correct": tr_correct,
                "is_final": tr.get("is_final", False),
            })
            if not tr_correct and first_failure_depth is None:
                first_failure_depth = tr.get("turn_number")

        status = "CORRECT" if correct else "WRONG "
        try:
            gt_display = f"{float(ground_truth):.4f}"
        except (TypeError, ValueError):
            gt_display = str(ground_truth)
        failure_info = f" first_fail@turn={first_failure_depth}" if not correct and num_turns > 1 else ""
        print(f"  [{i}/{len(subset)}] {status} | turns={num_turns} depth={depth}{failure_info} | "
              f"truth={gt_display} pred={predicted} | {original_q[:50]}...")

        results.append({
            "source": "mt_10q", "id": qid, "entity": entity, "depth": depth,
            "num_turns": num_turns, "question": original_q, "ground_truth": ground_truth,
            "turn_responses": turn_responses, "turn_correctness": turn_correctness,
            "first_failure_depth": first_failure_depth, "llm_response": final_response,
            "predicted": predicted, "correct": correct, "skip": False,
        })

    return results


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------

def compute_accuracy(results: list) -> dict:
    evaluated = [r for r in results if not r.get("skip")]
    if not evaluated:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(1 for r in evaluated if r["correct"])
    # "recovered": final answer correct but had at least one wrong intermediate turn
    recovered = sum(
        1 for r in evaluated
        if r["correct"] and r.get("first_failure_depth") is not None
    )
    return {
        "total": len(evaluated),
        "skipped": len(results) - len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated),
        "recovered": recovered,
    }


def print_summary(model_name: str, results_10q: list, results_90q: list, results_mt: list, results_mt_10q: list | None = None) -> None:
    acc10    = compute_accuracy(results_10q)
    acc90    = compute_accuracy(results_90q)
    accmt    = compute_accuracy(results_mt)
    accmt10q = compute_accuracy(results_mt_10q) if results_mt_10q else None
    total_eval    = acc10["total"] + acc90["total"] + accmt["total"] + (accmt10q["total"] if accmt10q else 0)
    total_correct = acc10["correct"] + acc90["correct"] + accmt["correct"] + (accmt10q["correct"] if accmt10q else 0)
    overall = total_correct / total_eval if total_eval else 0.0

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  10-Q          : {acc10['correct']}/{acc10['total']}  accuracy = {acc10['accuracy']:.1%}  (skipped {acc10.get('skipped',0)})")
    print(f"  Simple        : {acc90['correct']}/{acc90['total']}  accuracy = {acc90['accuracy']:.1%}  (skipped {acc90.get('skipped',0)})")
    print(f"  Multi-turn    : {accmt['correct']}/{accmt['total']}  accuracy = {accmt['accuracy']:.1%}  (skipped {accmt.get('skipped',0)})")
    if accmt10q is not None:
        print(f"  Multi-turn 10Q: {accmt10q['correct']}/{accmt10q['total']}  accuracy = {accmt10q['accuracy']:.1%}  (skipped {accmt10q.get('skipped',0)})")

    # Break down multi-turn accuracy by number of turns + failure origin
    if results_mt:
        from collections import defaultdict
        by_turns: dict = defaultdict(list)
        for r in results_mt:
            if not r.get("skip"):
                by_turns[r.get("num_turns", "?")].append(r)

        total_recovered = sum(
            1 for r in results_mt
            if not r.get("skip") and r["correct"] and r.get("first_failure_depth") is not None
        )
        if total_recovered:
            print(f"    recovered (correct final, wrong intermediate): {total_recovered}")

        for nt in sorted(by_turns):
            rows = by_turns[nt]
            n_correct = sum(1 for r in rows if r["correct"])
            n_recovered = sum(
                1 for r in rows
                if r["correct"] and r.get("first_failure_depth") is not None
            )
            recovered_str = f"  [{n_recovered} recovered]" if n_recovered else ""
            print(f"    turns={nt}: {n_correct}/{len(rows)}  accuracy = {n_correct/len(rows):.1%}{recovered_str}")
            # For multi-turn questions, show where failures first occur
            if nt > 1:
                failed = [r for r in rows if not r["correct"]]
                if failed:
                    fail_counts: dict = defaultdict(int)
                    for r in failed:
                        fd = r.get("first_failure_depth")
                        fail_counts[fd if fd is not None else "?"] += 1
                    parts = [f"turn {fd}: {cnt}" for fd, cnt in sorted(fail_counts.items(), key=lambda x: (x[0] is None, int(x[0]) if x[0] is not None else 0))]
                    print(f"      first failure at — {', '.join(parts)}  (of {len(failed)} failed)")

    print(f"  Overall   : {total_correct}/{total_eval}  accuracy = {overall:.1%}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(all_results: dict, path: Path, tol: float) -> None:
    """Write a flat CSV with one row per question across all models and datasets."""
    fieldnames = [
        "model", "dataset", "id", "entity", "depth",
        "question", "prompt", "ground_truth", "predicted_answer",
        "reasoning", "correct", "relative_error", "within_tol", "skipped",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model_data in all_results.items():
            adv_keys = [k for k in model_data if k.startswith("10q_")]
            for dataset_key in ["10q", "10q_leaf", "90q", "mt", "mt_10q"] + adv_keys:
                for r in model_data.get(dataset_key, []):
                    gt = r.get("ground_truth")
                    pred = r.get("predicted")
                    try:
                        gt = float(str(gt).replace(",", "")) if gt is not None else None
                    except (ValueError, TypeError):
                        gt = None
                    if gt is not None and pred is not None and gt != 0:
                        rel_err = abs(pred - gt) / abs(gt)
                    else:
                        rel_err = None
                    writer.writerow({
                        "model": model_name,
                        "dataset": dataset_key,
                        "id": r.get("id", ""),
                        "entity": r.get("company") or r.get("entity", ""),
                        "depth": r.get("depth", ""),
                        "question": r.get("question", ""),
                        "prompt": r.get("prompt", ""),
                        "ground_truth": gt,
                        "predicted_answer": pred,
                        "reasoning": r.get("llm_response", ""),
                        "correct": r.get("correct", False),
                        "relative_error": f"{rel_err:.6f}" if rel_err is not None else "",
                        "within_tol": (rel_err is not None and rel_err <= tol),
                        "skipped": r.get("skip", False),
                    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LLM evaluation on 10-Q and 90-Q financial QA datasets")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Model folder names under MODELS_DIR (default: Qwen3.5-4B Qwen3.5-9B)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions per dataset per model (default: all)",
    )
    parser.add_argument(
        "--tol", type=float, default=0.1,
        help="Relative tolerance for numeric correctness (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Max new tokens to generate per answer (default: 512)",
    )
    parser.add_argument(
        "--output", type=str, default="output_llm/eval_results.json",
        help="Path to save per-question results JSON (default: eval_results.json)",
    )
    parser.add_argument(
        "--csv", type=str, default="output_llm/eval_results.csv",
        help="Path to save per-question results CSV (default: eval_results.csv)",
    )
    parser.add_argument(
        "--thinking", action="store_true", default=False,
        help="Enable thinking mode for models that support it (e.g. Qwen3)",
    )
    _adv_choices = ["10q_missing", "10q_garbage", "10q_lookalike", "10q_cross", "10q_combined"]
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["10q", "10q_leaf", "90q", "mt", "mt_10q"] + _adv_choices,
        default=["10q", "90q", "mt"],
        help="Which dataset(s) to evaluate: 10q, 10q_leaf (no derived atoms), 90q, mt, mt_10q, or adversarial 10Q variants (default: all)",
    )
    parser.add_argument(
        "--questions-10q", type=str, default=None,
        help="Override path to 10Q questions CSV (default: random_questions_10q_1000.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    print("Loading datasets...")
    _adv_keys = {"10q_missing", "10q_garbage", "10q_lookalike", "10q_cross", "10q_combined"}
    if "10q" in args.datasets or "10q_leaf" in args.datasets or any(k in args.datasets for k in _adv_keys):
        dataset_10q_path = Path(args.questions_10q) if args.questions_10q else DATASET_10Q
        questions_10q = load_10q_questions(dataset_10q_path)
        sheet_lookup_10q = build_10q_sheet_lookup(load_json(ATOMS_10Q))
        print(f"  10-Q: {len(questions_10q)} questions, {len(sheet_lookup_10q)} entities")
    else:
        questions_10q, sheet_lookup_10q = [], {}

    if "90q" in args.datasets:
        questions_90q = load_10q_questions(DATASET_90Q)
        sheet_lookup_90q = build_90q_sheet_lookup(load_10q_questions(SHEET_90Q))
        print(f"  Simple: {len(questions_90q)} questions, {len(sheet_lookup_90q)} companies")
    else:
        questions_90q, sheet_lookup_90q = [], {}

    if "mt" in args.datasets:
        questions_mt = load_json(DATASET_MT)
        sheet_lookup_mt = build_90q_sheet_lookup(load_json(SHEET_MT))
        non_skipped = sum(1 for q in questions_mt if not q.get("skipped"))
        print(f"  Multi-turn: {len(questions_mt)} total, {non_skipped} non-skipped, {len(sheet_lookup_mt)} companies")
    else:
        questions_mt, sheet_lookup_mt = [], {}

    if "mt_10q" in args.datasets:
        questions_mt_10q = load_json(DATASET_MT_10Q)
        sheet_lookup_mt_10q = build_10q_sheet_lookup(load_json(ATOMS_10Q))
        non_skipped_10q = sum(1 for q in questions_mt_10q if not q.get("skipped"))
        print(f"  Multi-turn 10Q: {len(questions_mt_10q)} total, {non_skipped_10q} non-skipped, {len(sheet_lookup_mt_10q)} entities")
    else:
        questions_mt_10q, sheet_lookup_mt_10q = [], {}

    # Pre-load adversarial atom lookups
    adv_lookups: dict[str, dict] = {}
    for adv_key, adv_path in ADV_ATOMS_10Q.items():
        if adv_key in args.datasets:
            if not adv_path.exists():
                print(f"  WARNING: {adv_path} not found — run adversarial_10q.py first")
            else:
                adv_lookups[adv_key] = build_10q_sheet_lookup(load_json(adv_path))
                print(f"  {adv_key}: {len(adv_lookups[adv_key])} entities")

    all_results = {}

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  Evaluating model: {model_name}")
        print(f"{'='*60}")

        if "10q" in args.datasets:
            print(f"\n-- 10-Q evaluation ({args.limit or len(questions_10q)} questions) --")
            results_10q = evaluate_10q(
                questions_10q, sheet_lookup_10q, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
                thinking=args.thinking, leaf_only=False,
            )
        else:
            results_10q = []

        if "10q_leaf" in args.datasets:
            print(f"\n-- 10-Q (leaf-only, no derived atoms) evaluation ({args.limit or len(questions_10q)} questions) --")
            results_10q_leaf = evaluate_10q(
                questions_10q, sheet_lookup_10q, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
                thinking=args.thinking, leaf_only=True,
            )
        else:
            results_10q_leaf = []

        # Adversarial 10Q variants
        results_adv: dict[str, list] = {}
        for adv_key, adv_lookup in adv_lookups.items():
            print(f"\n-- {adv_key} evaluation ({args.limit or len(questions_10q)} questions) --")
            results_adv[adv_key] = evaluate_10q(
                questions_10q, adv_lookup, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
                thinking=args.thinking, leaf_only=True,
            )

        if "90q" in args.datasets:
            print(f"\n-- Simple evaluation ({args.limit or len(questions_90q)} questions) --")
            results_90q = evaluate_90q(
                questions_90q, sheet_lookup_90q, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
            )
        else:
            results_90q = []

        if "mt" in args.datasets:
            non_skipped = sum(1 for q in questions_mt if not q.get("skipped"))
            print(f"\n-- Multi-turn evaluation ({args.limit or non_skipped} questions) --")
            results_mt = evaluate_multiturn(
                questions_mt, sheet_lookup_mt, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
            )
        else:
            results_mt = []

        if "mt_10q" in args.datasets:
            non_skipped_10q = sum(1 for q in questions_mt_10q if not q.get("skipped"))
            print(f"\n-- Multi-turn 10Q evaluation ({args.limit or non_skipped_10q} questions) --")
            results_mt_10q = evaluate_multiturn_10q(
                questions_mt_10q, sheet_lookup_mt_10q, model_name,
                limit=args.limit, tol=args.tol, max_new_tokens=args.max_new_tokens,
            )
        else:
            results_mt_10q = []

        print_summary(model_name, results_10q, results_90q, results_mt, results_mt_10q)

        # Print adversarial accuracy summary
        for adv_key, adv_results in results_adv.items():
            acc = compute_accuracy(adv_results)
            print(f"  {adv_key}: {acc['correct']}/{acc['total']}  accuracy = {acc['accuracy']:.1%}  (skipped {acc.get('skipped', 0)})")

        if results_10q_leaf:
            acc_leaf = compute_accuracy(results_10q_leaf)
            print(f"  10q_leaf: {acc_leaf['correct']}/{acc_leaf['total']}  accuracy = {acc_leaf['accuracy']:.1%}  (skipped {acc_leaf.get('skipped', 0)})")

        all_results[model_name] = {
            "10q": results_10q,
            "10q_leaf": results_10q_leaf,
            "90q": results_90q,
            "mt": results_mt,
            "mt_10q": results_mt_10q,
            "accuracy_10q": compute_accuracy(results_10q),
            "accuracy_10q_leaf": compute_accuracy(results_10q_leaf),
            "accuracy_90q": compute_accuracy(results_90q),
            "accuracy_mt": compute_accuracy(results_mt),
            "accuracy_mt_10q": compute_accuracy(results_mt_10q),
            **{adv_key: adv_results for adv_key, adv_results in results_adv.items()},
            **{f"accuracy_{adv_key}": compute_accuracy(adv_results) for adv_key, adv_results in results_adv.items()},
        }

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output)
    output_path = output_path.with_stem(f"{output_path.stem}_{ts}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    csv_path = Path(args.csv)
    csv_path = csv_path.with_stem(f"{csv_path.stem}_{ts}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(all_results, csv_path, tol=args.tol)
    print(f"CSV saved to {csv_path}")

    # Final comparison across models
    if len(all_results) > 1:
        print("\n=== Model Comparison ===")
        print(f"{'Model':<20} {'10-Q Acc':>10} {'Simple Acc':>10} {'MT Acc':>10} {'Overall':>10}")
        print("-" * 62)
        for m, r in all_results.items():
            a10 = r["accuracy_10q"]["accuracy"]
            a90 = r["accuracy_90q"]["accuracy"]
            amt = r["accuracy_mt"]["accuracy"]
            t10 = r["accuracy_10q"]["total"]
            t90 = r["accuracy_90q"]["total"]
            tmt = r["accuracy_mt"]["total"]
            total = t10 + t90 + tmt
            correct = r["accuracy_10q"]["correct"] + r["accuracy_90q"]["correct"] + r["accuracy_mt"]["correct"]
            overall = correct / total if total else 0.0
            print(f"{m:<20} {a10:>10.1%} {a90:>10.1%} {amt:>10.1%} {overall:>10.1%}")


if __name__ == "__main__":
    main()
