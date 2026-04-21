import csv
import json
from pathlib import Path

YEARS = [2022, 2023, 2024]

COMPANY_INFO_FIELDS = ["company_name", "ticker", "sector", "country", "exchange", "credit_rating"]

METRICS = {
    "Income Statement": [
        "revenue",
        "cost_of_goods_sold",
        "gross_profit",
        "operating_expenses",
        "operating_income",
        "net_income",
    ],
    "Balance Sheet": [
        "total_assets",
        "total_liabilities",
        "total_equity",
        "current_assets",
        "current_liabilities",
        "cash",
        "long_term_debt",
        "accounts_receivable",
        "inventories",
    ],
    "Per Share & Other": [
        "capex",
        "dividends_paid",
        "shares_outstanding",
        "stock_price",
        "employees",
    ],
}


def label(field: str) -> str:
    return field.replace("_", " ").title()


def generate_csv(company: dict, output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Company info block
        writer.writerow(["Field", "Value"])
        for field in COMPANY_INFO_FIELDS:
            writer.writerow([label(field), company.get(field, "")])

        writer.writerow([])  # blank separator

        # Financial sections
        for section, metrics in METRICS.items():
            writer.writerow([section, *[str(y) for y in YEARS]])
            for metric in metrics:
                row = [label(metric)]
                for year in YEARS:
                    row.append(company.get(f"{metric}_{year}", ""))
                writer.writerow(row)
            writer.writerow([])  # blank separator between sections


def main():
    input_path = Path("output/financial_spreadsheet.json")
    output_dir = Path("output/csvs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        companies = json.load(f)

    for company in companies:
        ticker = company["ticker"]
        out_path = output_dir / f"{ticker}.csv"
        generate_csv(company, str(out_path))
        print(f"Generated: {out_path}")

    print(f"\nDone. {len(companies)} CSVs saved to {output_dir}/")


if __name__ == "__main__":
    main()
