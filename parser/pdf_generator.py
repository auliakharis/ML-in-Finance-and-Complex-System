import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

YEARS = [2022, 2023, 2024]

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

COMPANY_INFO_FIELDS = ["ticker", "sector", "country", "exchange", "credit_rating"]

HEADER_COLOR = colors.HexColor("#2C3E50")
ROW_ALT_COLOR = colors.HexColor("#ECF0F1")


def label(field: str) -> str:
    return field.replace("_", " ").title()


def format_value(metric: str, value) -> str:
    if value == "N/A":
        return "N/A"
    if metric == "stock_price":
        return f"${float(value):,.2f}"
    if metric == "employees":
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{int(value):,}"


def _base_table_style() -> list:
    return [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT_COLOR]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]


def generate_pdf(company: dict, output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "title", parent=styles["Heading1"], alignment=TA_CENTER, fontSize=16, spaceAfter=6
    )
    section_style = ParagraphStyle(
        "section", parent=styles["Heading2"], fontSize=11, spaceAfter=4, spaceBefore=8
    )

    story.append(Paragraph(company["company_name"], title_style))
    story.append(Spacer(1, 0.1 * inch))

    # Company info table
    info_data = [["Field", "Value"]]
    for field in COMPANY_INFO_FIELDS:
        info_data.append([label(field), str(company.get(field, ""))])

    info_table = Table(info_data, colWidths=[2.5 * inch, 4.5 * inch])
    style = _base_table_style()
    style.append(("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"))
    info_table.setStyle(TableStyle(style))
    story.append(info_table)
    story.append(Spacer(1, 0.15 * inch))

    # Financial sections
    col_widths = [2.5 * inch, 1.6 * inch, 1.6 * inch, 1.6 * inch]
    for section_name, metrics in METRICS.items():
        story.append(Paragraph(section_name, section_style))

        table_data = [["Metric"] + [str(y) for y in YEARS]]
        for metric in metrics:
            row = [label(metric)]
            for year in YEARS:
                val = company.get(f"{metric}_{year}", "N/A")
                row.append(format_value(metric, val))
            table_data.append(row)

        style = _base_table_style()
        style.append(("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"))
        style.append(("ALIGN", (1, 0), (-1, -1), "RIGHT"))

        tbl = Table(table_data, colWidths=col_widths)
        tbl.setStyle(TableStyle(style))
        story.append(tbl)

    doc.build(story)


def main():
    input_path = Path("output/financial_spreadsheet.json")
    output_dir = Path("output/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        companies = json.load(f)

    for company in companies:
        ticker = company["ticker"]
        out_path = output_dir / f"{ticker}.pdf"
        generate_pdf(company, str(out_path))
        print(f"Generated: {out_path}")

    print(f"\nDone. {len(companies)} PDFs saved to {output_dir}/")


if __name__ == "__main__":
    main()
