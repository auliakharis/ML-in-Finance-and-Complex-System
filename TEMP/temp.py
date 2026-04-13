# """
# LoongBench-Style Financial Dataset Generator
# Generates finance QA entries from Level 1 (simple ratios) to Level 7 (DCF valuation)
# Each entry includes: id, question, rationale (executable code), final_answer, metadata
# """

# import json
# import random
# import math
# from dataclasses import dataclass, field, asdict
# from typing import Any
# from datetime import date


# # ─────────────────────────────────────────────
# # Data Structures
# # ─────────────────────────────────────────────

# @dataclass
# class CompanyFinancials:
#     name: str
#     revenue: float
#     cogs: float
#     opex: float
#     interest_expense: float
#     tax_rate: float
#     total_assets: float
#     total_equity: float
#     total_debt: float
#     current_assets: float
#     current_liabilities: float
#     inventory: float
#     operating_cash_flow: float
#     capex: float
#     dividends_paid: float
#     shares_outstanding: float
#     stock_price: float

#     # Derived fields (auto-computed)
#     gross_profit: float = field(init=False)
#     ebit: float = field(init=False)
#     ebt: float = field(init=False)
#     tax: float = field(init=False)
#     net_income: float = field(init=False)
#     fcf: float = field(init=False)
#     eps: float = field(init=False)

#     def __post_init__(self):
#         self.gross_profit = self.revenue - self.cogs
#         self.ebit = self.gross_profit - self.opex
#         self.ebt = self.ebit - self.interest_expense
#         self.tax = self.ebt * self.tax_rate
#         self.net_income = self.ebt - self.tax
#         self.fcf = self.operating_cash_flow - self.capex
#         self.eps = self.net_income / self.shares_outstanding


# @dataclass
# class DatasetEntry:
#     id: str
#     level: int
#     question: str
#     rationale: str
#     final_answer: str
#     metadata: dict


# # ─────────────────────────────────────────────
# # Company Generator
# # ─────────────────────────────────────────────

# def generate_company(seed: int) -> CompanyFinancials:
#     """Generate a realistic company with randomized but internally consistent financials."""
#     rng = random.Random(seed)

#     company_names = [
#         "TechCorp Inc.", "FinancePro Ltd.", "AlphaVentures SA",
#         "BetaGroup PLC", "GammaIndustries Co.", "DeltaCapital LLC",
#         "OmegaSystems Inc.", "ZenithHoldings Ltd.", "ApexFinance Corp.",
#         "NovaTech Enterprises"
#     ]

#     revenue = rng.randint(3_000_000, 15_000_000)
#     cogs_ratio = rng.uniform(0.35, 0.55)
#     opex_ratio = rng.uniform(0.15, 0.28)
#     interest_rate = rng.uniform(0.04, 0.08)
#     tax_rate = rng.choice([0.20, 0.21, 0.25, 0.28, 0.30])
#     equity_ratio = rng.uniform(0.50, 0.70)
#     total_assets = revenue * rng.uniform(1.3, 2.2)
#     total_equity = total_assets * equity_ratio
#     total_debt = total_assets - total_equity
#     current_ratio = rng.uniform(1.5, 3.0)
#     current_liabilities = total_assets * rng.uniform(0.10, 0.18)
#     current_assets = current_liabilities * current_ratio
#     inventory = current_assets * rng.uniform(0.20, 0.35)
#     ocf_ratio = rng.uniform(0.10, 0.18)
#     capex_ratio = rng.uniform(0.05, 0.10)
#     shares = rng.randint(500_000, 5_000_000)
#     pe_ratio = rng.uniform(12, 28)

#     cogs = revenue * cogs_ratio
#     opex = revenue * opex_ratio
#     ebit = revenue - cogs - opex
#     interest_expense = total_debt * interest_rate
#     ebt = ebit - interest_expense
#     net_income = ebt * (1 - tax_rate)
#     eps = net_income / shares
#     stock_price = round(eps * pe_ratio, 2)
#     ocf = revenue * ocf_ratio
#     capex = revenue * capex_ratio
#     dividends = net_income * rng.uniform(0.15, 0.35)

#     return CompanyFinancials(
#         name=rng.choice(company_names) + f" (ID-{seed})",
#         revenue=round(revenue, 2),
#         cogs=round(cogs, 2),
#         opex=round(opex, 2),
#         interest_expense=round(interest_expense, 2),
#         tax_rate=tax_rate,
#         total_assets=round(total_assets, 2),
#         total_equity=round(total_equity, 2),
#         total_debt=round(total_debt, 2),
#         current_assets=round(current_assets, 2),
#         current_liabilities=round(current_liabilities, 2),
#         inventory=round(inventory, 2),
#         operating_cash_flow=round(ocf, 2),
#         capex=round(capex, 2),
#         dividends_paid=round(dividends, 2),
#         shares_outstanding=shares,
#         stock_price=stock_price,
#     )


# def fmt(n: float) -> str:
#     """Format number as dollar string."""
#     return f"${n:,.2f}"


# def company_context(c: CompanyFinancials) -> str:
#     """Generate the financial statement context block embedded in each question."""
#     return f"""
# Company: {c.name}
# Fiscal Year 2023

# INCOME STATEMENT
# Revenue:                {fmt(c.revenue)}
# Cost of Goods Sold:     {fmt(c.cogs)}
# Gross Profit:           {fmt(c.gross_profit)}
# Operating Expenses:     {fmt(c.opex)}
# EBIT:                   {fmt(c.ebit)}
# Interest Expense:       {fmt(c.interest_expense)}
# EBT:                    {fmt(c.ebt)}
# Tax ({int(c.tax_rate*100)}%):              {fmt(c.tax)}
# Net Income:             {fmt(c.net_income)}

# BALANCE SHEET
# Total Assets:           {fmt(c.total_assets)}
# Total Equity:           {fmt(c.total_equity)}
# Total Debt:             {fmt(c.total_debt)}
# Current Assets:         {fmt(c.current_assets)}
# Current Liabilities:    {fmt(c.current_liabilities)}
# Inventory:              {fmt(c.inventory)}

# CASH FLOW
# Operating Cash Flow:    {fmt(c.operating_cash_flow)}
# Capital Expenditure:    {fmt(c.capex)}
# Dividends Paid:         {fmt(c.dividends_paid)}

# MARKET DATA
# Shares Outstanding:     {c.shares_outstanding:,}
# Stock Price:            {fmt(c.stock_price)}
# """.strip()


# # ─────────────────────────────────────────────
# # Level Generators
# # ─────────────────────────────────────────────

# def level_1(c: CompanyFinancials, idx: int) -> DatasetEntry:
#     """Net Profit Margin — single formula, direct lookup."""
#     result = round((c.net_income / c.revenue) * 100, 2)
#     rationale = f"""\
# # Given from financial statements
# net_income = {c.net_income}
# revenue = {c.revenue}

# # Net Profit Margin = Net Income / Revenue
# result = round((net_income / revenue) * 100, 2)
# print(result)  # → {result}%"""

#     return DatasetEntry(
#         id=f"finance_fs_{idx:03d}_L1",
#         level=1,
#         question=f"Using the financial statements below, calculate the Net Profit Margin.\n\n{company_context(c)}",
#         rationale=rationale,
#         final_answer=str(result),
#         metadata={"concept": "Net Profit Margin", "formula": "Net Income / Revenue"}
#     )


# def level_2(c: CompanyFinancials, idx: int) -> DatasetEntry:
#     """Gross and Operating Margin — two ratios + comparison."""
#     gpm = round((c.gross_profit / c.revenue) * 100, 2)
#     opm = round((c.ebit / c.revenue) * 100, 2)
#     diff = round(gpm - opm, 2)
#     result = f"{gpm},{opm},{diff}"

#     rationale = f"""\
# revenue = {c.revenue}
# gross_profit = {c.gross_profit}
# ebit = {c.ebit}

# gpm = round((gross_profit / revenue) * 100, 2)   # Gross Profit Margin
# opm = round((ebit / revenue) * 100, 2)            # Operating Profit Margin
# difference = round(gpm - opm, 2)

# result = f"{{gpm}},{{opm}},{{difference}}"
# print(result)  # → {result}"""

#     return DatasetEntry(
#         id=f"finance_fs_{idx:03d}_L2",
#         level=2,
#         question=(
#             f"Using the financial statements below, calculate the Gross Profit Margin and "
#             f"Operating Profit Margin. Report both values and their difference in the format "
#             f"'GPM,OPM,DIFF'.\n\n{company_context(c)}"
#         ),
#         rationale=rationale,
#         final_answer=result,
#         metadata={"concept": "Margin Analysis", "formula": "Gross Profit/Revenue, EBIT/Revenue"}
#     )


# def level_3(c: CompanyFinancials, idx: int) -> DatasetEntry:
#     """Free Cash Flow + FCF Margin — derived input not labeled on sheet."""
#     fcf = round(c.operating_cash_flow - c.capex, 2)
#     fcf_margin = round((fcf / c.revenue) * 100, 2)
#     result = f"{fcf},{fcf_margin}"

#     rationale = f"""\
# # FCF is not directly listed — must be derived
# operating_cash_flow = {c.operating_cash_flow}
# capex = {c.capex}
# revenue = {c.revenue}

# fcf = operating_cash_flow - capex
# fcf_margin = round((fcf / revenue) * 100, 2)

# result = f"{{round(fcf, 2)}},{{fcf_margin}}"
# print(result)  # → {result}"""

#     return DatasetEntry(
#         id=f"finance_fs_{idx:03d}_L3",
#         level=3,
#         question=(
#             f"Using the financial statements below, calculate Free Cash Flow (FCF) and FCF Margin. "
#             f"Report as 'FCF_VALUE,FCF_MARGIN_PCT'.\n\n{company_context(c)}"
#         ),
#         rationale=rationale,
#         final_answer=result,
#         metadata={"concept": "Free Cash Flow", "formula": "OCF - CapEx"}
#     )


# def level_4(c: CompanyFinancials, idx: int) -> DatasetEntry:
#     """ROE vs ROA — cross-statement, leverage effect."""
#     roe = round((c.net_income / c.total_equity) * 100, 2)
#     roa = round((c.net_income / c.total_assets) * 100, 2)
#     leverage_effect = round(roe - roa, 2)
#     result = f"{roe},{roa},{leverage_effect}"

#     rationale = f"""\
# # Requires both Income Statement and Balance Sheet
# net_income   = {c.net_income}
# total_equity = {c.total_equity}
# total_assets = {c.total_assets}

# roe = round((net_income / total_equity) * 100, 2)
# roa = round((net_income / total_assets) * 100, 2)
# leverage_effect = round(roe - roa, 2)   # debt amplification

# result = f"{{roe}},{{roa}},{{leverage_effect}}"
# print(result)  # → {result}"""

#     return DatasetEntry(
#         id=f"finance_fs_{idx:03d}_L4",
#         level=4,
#         question=(
#             f"Using the financial statements below, calculate ROE and ROA. "
#             f"Then compute the leverage effect (ROE - ROA). "
#             f"Report as 'ROE,ROA,LEVERAGE_EFFECT'.\n\n{company_context(c)}"
#         ),
#         rationale=rationale,
#         final_answer=result,
#         metadata={"concept": "ROE / ROA / Leverage", "formula": "Net Income / Equity, Net Income / Assets"}
#     )


# def level_5(c: CompanyFinancials, idx: int) -> DatasetEntry:
#     """DuPont decomposition — chained ratios, must verify equality."""
#     net_margin = c.net_income / c.revenue
#     asset_turnover = c.revenue / c.total_assets
#     equity_multiplier = c.total_assets / c.total_equity
#     dupont_roe = round(net_margin * asset_turnover * equity_multiplier * 100, 2)
#     result = f"{round(net_margin*100,4)},{round(asset_turnover,4)},{round(equity_multiplier,4)},{dupont_roe}"

#     rationale = f"""\
# net_income   = {c.net_income}
# revenue      = {c.revenue}
# total_assets = {c.total_assets}
# total_equity = {c.total_equity}

# # DuPont: ROE = Net Margin × Asset Turnover × Equity Multiplier
# net_margin        = net_income / revenue
# asset_turnover    = revenue / total_assets
# equity_multiplier = total_assets / total_equity

# dupont_roe = round(net_margin * asset_turnover * equity_multiplier * 100, 2)

# result = f"{{round(net_margin*100,4)}},{{round(asset_turnover,4)}},{{round(equity_multiplier,4)}},{{dupont_roe}}"
# print(result)  # → {result}"""

#     return DatasetEntry(
#         id=f"finance_fs_{idx:03d}_L5",
#         level=5,
#         question=(
#             f"Using the financial statements below, perform a DuPont decomposition of ROE. "
#             f"Report Net Margin (%), Asset Turnover, Equity Multiplier, and final ROE (%) "
#             f"as 'NET_MARGIN,ASSET_TURNOVER,EQUITY_MULTIPLIER,ROE'.\n\n{company_context(c)}"
#         ),
#         rationale=rationale,
#         final_answer=result,
#         metadata={"concept": "DuPont Analysis", "formula": "NM × AT × EM"}
#     )


# def level_6(c: CompanyFinancials, idx: int) -> DatasetEntry:
#     """Scenario analysis — new debt + revenue expansion, rebuild P&L."""
#     rng = random.Random(idx + 999)
#     new_debt = round(c.revenue * rng.uniform(0.15, 0.25), 2)
#     new_interest_rate = rng.uniform(0.05, 0.08)
#     revenue_growth = rng.uniform(0.10, 0.20)

#     new_revenue = round(c.revenue * (1 + revenue_growth), 2)
#     new_interest = round(c.interest_expense + new_debt * new_interest_rate, 2)
#     new_gross = round(new_revenue - c.cogs, 2)   # COGS stays same
#     new_ebit = round(new_gross - c.opex, 2)
#     new_ebt = round(new_ebit - new_interest, 2)
#     new_net = round(new_ebt * (1 - c.tax_rate), 2)

#     old_roe = round((c.net_income / c.total_equity) * 100, 2)
#     new_roe = round((new_net / c.total_equity) * 100, 2)
#     roe_delta = round(new_roe - old_roe, 2)
#     result = f"{new_roe},{old_roe},{roe_delta}"

#     rationale = f"""\
# # Current state
# net_income   = {c.net_income}
# total_equity = {c.total_equity}
# cogs         = {c.cogs}
# opex         = {c.opex}
# tax_rate     = {c.tax_rate}

# # Scenario: raise ${fmt(new_debt)} at {round(new_interest_rate*100,1)}% interest, revenue +{round(revenue_growth*100,1)}%
# new_debt            = {new_debt}
# new_interest_rate   = {round(new_interest_rate, 4)}
# revenue_growth      = {round(revenue_growth, 4)}

# new_revenue  = round({c.revenue} * (1 + revenue_growth), 2)
# new_interest = round({c.interest_expense} + new_debt * new_interest_rate, 2)
# new_ebt      = round(new_revenue - cogs - opex - new_interest, 2)
# new_net      = round(new_ebt * (1 - tax_rate), 2)

# old_roe = round((net_income / total_equity) * 100, 2)
# new_roe = round((new_net / total_equity) * 100, 2)
# roe_delta = round(new_roe - old_roe, 2)

# result = f"{{new_roe}},{{old_roe}},{{roe_delta}}"
# print(result)  # → {result}"""

#     return DatasetEntry(
#         id=f"finance_fs_{idx:03d}_L6",
#         level=6,
#         question=(
#             f"Using the financial statements below, perform a scenario analysis. "
#             f"The company raises {fmt(new_debt)} in new debt at {round(new_interest_rate*100,1)}% interest "
#             f"and uses it to fund an expansion that grows revenue by {round(revenue_growth*100,1)}%. "
#             f"COGS and OpEx remain unchanged. Calculate the new ROE, old ROE, and the difference. "
#             f"Report as 'NEW_ROE,OLD_ROE,DELTA'.\n\n{company_context(c)}"
#         ),
#         rationale=rationale,
#         final_answer=result,
#         metadata={"concept": "Scenario Analysis / Leverage", "formula": "Rebuilt P&L under new assumptions"}
#     )


# def level_7(c: CompanyFinancials, idx: int) -> DatasetEntry:
#     """DCF Valuation — multi-year projection, terminal value, WACC discounting."""
#     rng = random.Random(idx + 7777)
#     fcf_growth = round(rng.uniform(0.06, 0.12), 3)
#     terminal_growth = round(rng.uniform(0.02, 0.04), 3)
#     wacc = round(rng.uniform(0.08, 0.13), 3)

#     fcf = c.fcf
#     fcf_projections = [round(fcf * (1 + fcf_growth)**t, 2) for t in range(1, 6)]
#     terminal_value = round(fcf_projections[-1] * (1 + terminal_growth) / (wacc - terminal_growth), 2)

#     pv_fcfs = sum([cf / (1 + wacc)**t for t, cf in enumerate(fcf_projections, 1)])
#     pv_terminal = terminal_value / (1 + wacc)**5
#     intrinsic_value = round(pv_fcfs + pv_terminal, 2)

#     # Per share
#     intrinsic_per_share = round(intrinsic_value / c.shares_outstanding, 2)
#     result = f"{intrinsic_value},{intrinsic_per_share}"

#     projections_str = ", ".join([str(v) for v in fcf_projections])

#     rationale = f"""\
# import math

# # Base FCF from financial statements
# fcf = {round(fcf, 2)}

# # DCF assumptions
# fcf_growth      = {fcf_growth}    # {round(fcf_growth*100,1)}% growth for 5 years
# terminal_growth = {terminal_growth}   # {round(terminal_growth*100,1)}% perpetuity growth
# wacc            = {wacc}   # Discount rate

# # Project 5 years of FCF
# fcf_projections = [round(fcf * (1 + fcf_growth)**t, 2) for t in range(1, 6)]
# # → [{projections_str}]

# # Terminal value (Gordon Growth at year 5)
# terminal_value = round(fcf_projections[-1] * (1 + terminal_growth) / (wacc - terminal_growth), 2)

# # Discount everything to present value
# pv_fcfs = sum([cf / (1 + wacc)**t for t, cf in enumerate(fcf_projections, 1)])
# pv_terminal = terminal_value / (1 + wacc)**5

# intrinsic_value = round(pv_fcfs + pv_terminal, 2)
# shares = {c.shares_outstanding}
# intrinsic_per_share = round(intrinsic_value / shares, 2)

# result = f"{{intrinsic_value}},{{intrinsic_per_share}}"
# print(result)  # → {result}"""

#     return DatasetEntry(
#         id=f"finance_fs_{idx:03d}_L7",
#         level=7,
#         question=(
#             f"Using the financial statements below, perform a DCF valuation. "
#             f"Assume FCF grows at {round(fcf_growth*100,1)}% annually for 5 years, "
#             f"then {round(terminal_growth*100,1)}% in perpetuity. WACC = {round(wacc*100,1)}%. "
#             f"Calculate the total intrinsic value and intrinsic value per share. "
#             f"Report as 'INTRINSIC_VALUE,VALUE_PER_SHARE'.\n\n{company_context(c)}"
#         ),
#         rationale=rationale,
#         final_answer=result,
#         metadata={"concept": "DCF Valuation", "formula": "PV of FCFs + PV of Terminal Value"}
#     )


# # ─────────────────────────────────────────────
# # Main Generator
# # ─────────────────────────────────────────────

# LEVEL_GENERATORS = [level_1, level_2, level_3, level_4, level_5, level_6, level_7]


# def generate_dataset(n_companies: int = 10) -> list[dict]:
#     """
#     Generate a full dataset with n_companies × 7 levels = n_companies*7 entries.
#     Each company gets one question per difficulty level.
#     """
#     dataset = []
#     for i in range(n_companies):
#         company = generate_company(seed=i * 42 + 7)
#         for level_fn in LEVEL_GENERATORS:
#             entry = level_fn(company, idx=i)
#             dataset.append(asdict(entry))
#     return dataset


# def verify_entry(entry: dict) -> tuple[bool, Any]:
#     """Execute an entry's rationale code and verify the output matches final_answer."""
#     namespace = {}
#     try:
#         exec(entry["rationale"], namespace)
#         # Capture printed output
#         import io, sys
#         buf = io.StringIO()
#         sys.stdout = buf
#         exec(entry["rationale"], {})
#         sys.stdout = sys.__stdout__
#         output = buf.getvalue().strip().split("\n")[-1]  # last printed line
#         # Strip comment
#         output = output.split("#")[0].strip()
#         match = str(entry["final_answer"]) in output or output in str(entry["final_answer"])
#         return match, output
#     except Exception as e:
#         return False, str(e)


# def print_summary(dataset: list[dict]):
#     print("\n" + "="*60)
#     print("DATASET GENERATION SUMMARY")
#     print("="*60)

#     level_counts = {}
#     for entry in dataset:
#         lvl = entry["level"]
#         level_counts[lvl] = level_counts.get(lvl, 0) + 1

#     print(f"Total entries: {len(dataset)}")
#     print(f"\nBreakdown by level:")
#     labels = {
#         1: "Direct lookup (net margin)",
#         2: "Two ratios + compare",
#         3: "Derived input (FCF)",
#         4: "Cross-statement (ROE/ROA)",
#         5: "DuPont decomposition",
#         6: "Scenario analysis",
#         7: "DCF valuation",
#     }
#     for lvl in sorted(level_counts):
#         bar = "█" * level_counts[lvl]
#         print(f"  L{lvl} [{labels[lvl]:<30}]: {level_counts[lvl]:>3} entries  {bar}")

#     print("\nSample entry (L1):")
#     sample = next(e for e in dataset if e["level"] == 1)
#     print(f"  ID:     {sample['id']}")
#     print(f"  Answer: {sample['final_answer']}")
#     print(f"  Q:      {sample['question'][:80]}...")


# if __name__ == "__main__":
#     import sys

#     n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
#     print(f"Generating dataset with {n} companies × 7 levels = {n*7} entries...")

#     dataset = generate_dataset(n_companies=n)

#     # Save to JSON
#     output_path = "output/finance_dataset.json"
#     with open(output_path, "w") as f:
#         json.dump(dataset, f, indent=2)

#     print(f"Saved to {output_path}")
#     print_summary(dataset)

#     # Quick verification spot-check
#     print("\nVerification spot-check (5 random entries):")
#     import random
#     samples = random.sample(dataset, min(5, len(dataset)))
#     for s in samples:
#         ok, out = verify_entry(s)
#         status = "✅" if ok else "❌"
#         print(f"  {status} {s['id']} → expected={s['final_answer']}, got={out}")



import csv
import json

data = []  # ← missing this

with open("90q/random_questions_90_new.csv", 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        data.append(dict(row))

with open("90q/random_questions_90_new.json", 'w', encoding='utf-8') as json_file:  # ← fix extension
    json.dump(data, json_file, indent=4)

print(f"Done! {len(data)} rows converted.")