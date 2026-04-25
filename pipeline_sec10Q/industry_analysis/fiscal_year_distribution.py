"""
Fiscal Year End Distribution by Industry
Pulls data from SEC EDGAR bulk datasets — free, official, no API key needed.

Run from anywhere:
    pip install requests pandas
    python fiscal_year_distribution.py

To recompute from saved CSV without hitting the network:
    python fiscal_year_distribution.py --from-csv fiscal_year_raw.csv

Output: fiscal_year_distribution.csv and a printed summary table.
"""

import sys
import time
import requests
import pandas as pd

HEADERS = {"User-Agent": "research@example.com"}


# ---------------------------------------------------------------------------
# SIC code -> broad industry group
# Non-overlapping, checked in order — first match wins.
# Full SIC list: https://www.sec.gov/info/edgar/siccodes.htm
# ---------------------------------------------------------------------------

SIC_RANGES = [
    (100,    999,  "Agriculture"),
    (1000,  1499,  "Mining & Energy"),
    (1500,  1799,  "Construction"),
    (2000,  2099,  "Food & Beverage Manufacturing"),
    (2100,  2199,  "Tobacco"),
    (2800,  2899,  "Chemicals & Pharma"),
    (2900,  2999,  "Petroleum Refining"),
    (3600,  3699,  "Electronics & Tech Hardware"),
    (3700,  3799,  "Transportation Equipment"),
    (3800,  3899,  "Instruments & Medical Devices"),
    (2200,  3999,  "Manufacturing (Other)"),
    (4500,  4599,  "Airlines"),
    (4900,  4999,  "Utilities"),
    (4000,  4999,  "Transportation"),
    (5000,  5199,  "Wholesale"),
    (5200,  5299,  "Retail (Building/Home)"),
    (5400,  5499,  "Retail (Food/Grocery)"),
    (5500,  5599,  "Retail (Auto)"),
    (5600,  5699,  "Retail (Apparel)"),
    (5900,  5999,  "Retail (Misc)"),
    (5200,  5999,  "Retail (Other)"),
    (6000,  6099,  "Banking"),
    (6100,  6199,  "Credit & Finance"),
    (6200,  6299,  "Securities & Brokers"),
    (6300,  6411,  "Insurance"),
    (6500,  6599,  "Real Estate"),
    (6600,  6999,  "Holding Companies"),
    (7000,  7099,  "Hotels & Hospitality"),
    (7370,  7379,  "Software & IT Services"),
    (8000,  8099,  "Healthcare Services"),
    (8200,  8299,  "Education"),
    (7000,  7999,  "Personal Services"),
    (8000,  8999,  "Professional Services"),
    (9000,  9999,  "Government & Other"),
]


def sic_to_industry(sic: int) -> str:
    for lo, hi, name in SIC_RANGES:
        if lo <= sic <= hi:
            return name
    return "Unknown"


# ---------------------------------------------------------------------------
# Fetch from SEC EDGAR
# ---------------------------------------------------------------------------

def fetch_company_list() -> pd.DataFrame:
    print("Downloading company list from SEC EDGAR...")
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["data"], columns=data["fields"])
    print(f"  Found {len(df):,} companies with exchange listings")
    return df


def fetch_company_facts_batch(ciks: list, sample_size: int = 3000) -> pd.DataFrame:
    import random
    sampled = random.sample(ciks, min(sample_size, len(ciks)))
    records = []
    print(f"Fetching metadata for {len(sampled):,} companies...")

    for i, cik in enumerate(sampled):
        try:
            url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            d = resp.json()
            records.append({
                "cik":             cik,
                "name":            d.get("name", ""),
                "sic":             int(d.get("sic", 0) or 0),
                "sic_description": d.get("sicDescription", ""),
                "fiscal_year_end": d.get("fiscalYearEnd", ""),
                "state":           d.get("stateOfIncorporation", ""),
                "entity_type":     d.get("entityType", ""),
            })
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sampled)} fetched...")
            time.sleep(0.5)

    print(f"  Successfully fetched {len(records):,} records")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analyze — works on both freshly fetched and CSV-loaded data
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame) -> tuple:
    # Parse fiscal year end month
    df = df[df["fiscal_year_end"].astype(str).str.len() == 4].copy()
    df["fy_month"] = df["fiscal_year_end"].astype(str).str[:2].astype(int)
    df["fy_month_name"] = df["fy_month"].map({
        1: "Jan", 2: "Feb",  3: "Mar", 4: "Apr",
        5: "May", 6: "Jun",  7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    })
    df["industry"] = df["sic"].apply(sic_to_industry)

    # Drop duplicates — each company counted once
    df = df.drop_duplicates(subset="cik")

    # Overall distribution
    print("\n-- Overall Fiscal Year End Distribution ---------------------")
    overall = df["fy_month_name"].value_counts()
    total = len(df)
    for month, count in overall.items():
        bar = "#" * int(count / total * 40)
        print(f"  {month:>3}  {bar:<40}  {count:>5,} ({count/total*100:.1f}%)")

    # By industry pivot
    pivot = (
        df.groupby(["industry", "fy_month_name"])
        .size()
        .unstack(fill_value=0)
    )
    pivot["Total"] = pivot.sum(axis=1)
    month_cols = [m for m in ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                  if m in pivot.columns]
    pivot["Most common"]   = pivot[month_cols].idxmax(axis=1)
    pivot["Most common %"] = (
        pivot[month_cols].max(axis=1) / pivot["Total"] * 100
    ).round(1)
    pivot = pivot.sort_values("Total", ascending=False)

    return pivot, df


def print_summary(pivot: pd.DataFrame) -> None:
    month_cols = [m for m in ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                  if m in pivot.columns]

    print("\n-- Fiscal Year End by Industry ------------------------------")
    print(f"  {'Industry':<33} {'N':>6}  {'Most common':>11}  {'%':>5}  Top 3 months")
    print("  " + "-" * 85)

    seen = set()
    for industry, row in pivot.iterrows():
        if industry in seen:
            continue
        seen.add(industry)

        total = int(row["Total"])
        if total < 5:
            continue

        top3 = sorted(month_cols, key=lambda m: row.get(m, 0), reverse=True)[:3]
        top3_str = "  ".join(
            f"{m}({row[m]/total*100:.0f}%)" for m in top3 if row.get(m, 0) > 0
        )
        print(f"  {industry:<33} {total:>6}  {row['Most common']:>11}  "
              f"{row['Most common %']:>4.1f}%  {top3_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_or_fetch(csv_path: str = "fiscal_year_raw.csv", sample_size: int = 3000) -> pd.DataFrame:
    """
    Load from CSV if it exists, otherwise fetch from SEC EDGAR and save.
    This means the first run fetches, every subsequent run loads from disk.
    Pass --fetch to force a fresh fetch even if the CSV exists.
    """
    import os
    force_fetch = "--fetch" in sys.argv

    if not force_fetch and os.path.exists(csv_path):
        print(f"Found {csv_path} — loading from disk (pass --fetch to re-fetch)...")
        df = pd.read_csv(csv_path, dtype={
            "cik":             str,
            "sic":             "Int64",
            "fiscal_year_end": str,
        })
        df["sic"] = df["sic"].fillna(0).astype(int)
        df["fiscal_year_end"] = df["fiscal_year_end"].astype(str).str.zfill(4)
        print(f"  Loaded {len(df):,} rows")
        return df
    else:
        if force_fetch:
            print("--fetch flag detected — fetching fresh data from SEC EDGAR...")
        else:
            print(f"{csv_path} not found — fetching from SEC EDGAR...")
        companies_df = fetch_company_list()
        ciks = companies_df["cik"].tolist()
        return fetch_company_facts_batch(ciks, sample_size=sample_size)


if __name__ == "__main__":
    df = load_or_fetch()

    pivot, full_df = analyze(df)
    print_summary(pivot)

    pivot.to_csv("fiscal_year_distribution.csv")
    full_df.to_csv("fiscal_year_raw.csv", index=False)
    print("\nSaved: fiscal_year_distribution.csv, fiscal_year_raw.csv")