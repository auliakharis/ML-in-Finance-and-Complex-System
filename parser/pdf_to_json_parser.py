import json
from pathlib import Path

import pdfplumber

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
    "Ticker": "ticker",
    "Sector": "sector",
    "Country": "country",
    "Exchange": "exchange",
    "Credit Rating": "credit_rating",
}


def _parse_number(raw: str):
    """Convert a formatted number string back to int or float."""
    s = raw.strip().lstrip("$").replace(",", "")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return raw.strip()


def _cell(row, idx) -> str:
    if idx >= len(row) or row[idx] is None:
        return ""
    return str(row[idx]).strip()


def parse_pdf(pdf_path: str) -> dict:
    result: dict = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract company name from the first non-empty text line
            if "company_name" not in result:
                text = page.extract_text() or ""
                first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), None)
                if first_line:
                    result["company_name"] = first_line

            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                header = [_cell(table[0], i) for i in range(len(table[0]))]

                # Company info table: headers are ["Field", "Value"]
                if header[:2] == ["Field", "Value"]:
                    for row in table[1:]:
                        field_label = _cell(row, 0)
                        value = _cell(row, 1)
                        key = INFO_LABEL_TO_KEY.get(field_label)
                        if key and value:
                            result[key] = value

                # Financial table: headers are ["Metric", "2022", "2023", "2024"]
                elif header[0] == "Metric" and all(yr in header for yr in YEARS):
                    year_idx = {yr: header.index(yr) for yr in YEARS}
                    for row in table[1:]:
                        metric_label = _cell(row, 0)
                        key = METRIC_LABEL_TO_KEY.get(metric_label)
                        if not key:
                            continue
                        for yr, idx in year_idx.items():
                            raw = _cell(row, idx)
                            if raw and raw != "N/A":
                                result[f"{key}_{yr}"] = _parse_number(raw)

    return result


def main():
    input_dir = Path("output/pdfs")
    output_dir = Path("output/parsed_json")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}/")
        return

    all_companies = []
    for pdf_path in pdf_files:
        print(f"Parsing: {pdf_path.name} ...", end=" ")
        data = parse_pdf(str(pdf_path))
        ticker = pdf_path.stem
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
