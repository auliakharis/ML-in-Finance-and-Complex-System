import csv
import json
from pathlib import Path

YEARS = ["2022", "2023", "2024"]

METRIC_LABEL_TO_KEY = {
    "Revenue": "revenue",
    "Cost Of Goods Sold": "cost_of_goods_sold",
    "Gross Profit": "gross_profit",
    "Operating Expenses": "operating_expenses",
    "Operating Income": "operating_income",
    "Net Income": "net_income",
    "Total Assets": "total_assets",
    "Total Liabilities": "total_liabilities",
    "Total Equity": "total_equity",
    "Current Assets": "current_assets",
    "Current Liabilities": "current_liabilities",
    "Cash": "cash",
    "Long Term Debt": "long_term_debt",
    "Accounts Receivable": "accounts_receivable",
    "Inventories": "inventories",
    "Capex": "capex",
    "Dividends Paid": "dividends_paid",
    "Shares Outstanding": "shares_outstanding",
    "Stock Price": "stock_price",
    "Employees": "employees",
}

INFO_LABEL_TO_KEY = {
    "Company Name": "company_name",
    "Ticker": "ticker",
    "Sector": "sector",
    "Country": "country",
    "Exchange": "exchange",
    "Credit Rating": "credit_rating",
}

SECTION_HEADERS = {"Income Statement", "Balance Sheet", "Per Share & Other"}


def _parse_number(raw: str):
    s = raw.strip().replace(",", "")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return raw.strip()


def parse_csv(csv_path: str) -> dict:
    result: dict = {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    year_indices: dict = {}

    for row in rows:
        if not row or all(cell.strip() == "" for cell in row):
            year_indices = {}  # reset on blank row
            continue

        first = row[0].strip()

        # Info block header — skip
        if first == "Field" and len(row) >= 2 and row[1].strip() == "Value":
            continue

        # Company info row
        if first in INFO_LABEL_TO_KEY:
            result[INFO_LABEL_TO_KEY[first]] = row[1].strip() if len(row) > 1 else ""
            continue

        # Section header row e.g. ["Income Statement", "2022", "2023", "2024"]
        if first in SECTION_HEADERS:
            header = [c.strip() for c in row]
            year_indices = {yr: i for i, c in enumerate(header) if (yr := c) in YEARS}
            continue

        # Metric data row
        metric_key = METRIC_LABEL_TO_KEY.get(first)
        if metric_key and year_indices:
            for yr, idx in year_indices.items():
                if idx < len(row) and row[idx].strip():
                    result[f"{metric_key}_{yr}"] = _parse_number(row[idx])

    return result


def main():
    input_dir = Path("output/csvs")
    output_dir = Path("output/parsed_json_csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSVs found in {input_dir}/")
        return

    all_companies = []
    for csv_path in csv_files:
        print(f"Parsing: {csv_path.name} ...", end=" ")
        data = parse_csv(str(csv_path))
        ticker = csv_path.stem
        out_path = output_dir / f"{ticker}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        all_companies.append(data)
        print(f"saved -> {out_path.name}")

    combined_path = output_dir / "financial_spreadsheet_parsed.json"
    with open(combined_path, "w") as f:
        json.dump(all_companies, f, indent=2)

    print(f"\nDone. {len(all_companies)} companies parsed.")
    print(f"Combined JSON: {combined_path}")


if __name__ == "__main__":
    main()
