# -*- coding: utf-8 -*-
"""
ARENA Score Analysis - Data Cleaning and Exploration

Reads the raw anonymized Excel workbook (rawdata1.xlsx), standardises
column names, drops invalid rows (empty student ID, marked absent,
all-zero scores), and writes a tidy CSV (data_clean.csv) used by the
downstream analysis and figure-generation scripts.
"""
import pandas as pd
import numpy as np
import openpyxl
from pathlib import Path

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC = "rawdata1.xlsx"


def load_sheet(sheet_name):
    """Load a single sheet and rename columns to a canonical short form.

    The raw workbook uses descriptive English column headers
    (Student ID, Name, Plan Comparison, Plan Generation, Total,
    Logins, Notes).  We map them to short codes (sid, name, C, D, E,
    F, note) so the rest of the pipeline can refer to variables by
    letter, matching the paper's notation.
    """
    df = pd.read_excel(SRC, sheet_name=sheet_name, header=None)
    # Standard layout: row 0 is the header, row 1+ is the data.
    header = df.iloc[0].tolist()
    data = df.iloc[1:].copy()
    data.columns = [str(c).strip() if c is not None else f"col{i}" for i, c in enumerate(header)]
    rename = {}
    for c in data.columns:
        cs = str(c)
        if "Student ID" in cs:
            rename[c] = "sid"
        elif "Name" in cs:
            rename[c] = "name"
        elif "Plan Comparison" in cs:
            rename[c] = "C"
        elif "Plan Generation" in cs:
            rename[c] = "D"
        elif "Total" in cs:
            rename[c] = "E"
        elif "#log" in cs.lower() or "Logins" in cs:
            rename[c] = "F"
        elif "Notes" in cs:
            rename[c] = "note"
    data = data.rename(columns=rename)
    keep = [c for c in ["sid", "name", "C", "D", "E", "F", "note"] if c in data.columns]
    data = data[keep].copy()
    data["year"] = sheet_name
    return data


def clean(df):
    """Drop invalid rows: empty student ID, marked absent, all-zero scores."""
    n0 = len(df)
    # Empty student ID -> aggregate / blank row.
    df = df[df["sid"].astype(str).str.strip() != ""].copy()
    n1 = len(df)
    # "absent" in notes -> drop.
    if "note" in df.columns:
        df["note"] = df["note"].astype(str).fillna("")
        df = df[~df["note"].str.contains("absent", na=False)].copy()
    n2 = len(df)
    # Cast C/D/E/F to numeric.
    for c in ["C", "D", "E", "F"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Keep rows with at least one non-null score.
    df = df[df[["C", "D", "E", "F"]].notna().any(axis=1)].copy()
    n3 = len(df)
    # Drop rows where C = D = E = F = 0 (treated as non-participation).
    mask_all_zero = (
        (df["C"].fillna(0) == 0)
        & (df["D"].fillna(0) == 0)
        & (df["E"].fillna(0) == 0)
        & (df["F"].fillna(0) == 0)
    )
    df = df[~mask_all_zero].copy()
    n4 = len(df)
    # Fill remaining NaN scores with 0.
    for c in ["C", "D", "E", "F"]:
        df[c] = df[c].fillna(0)
    print(
        f"  {df['year'].iloc[0]}: raw={n0}  has_sid={n1}  "
        f"no_absence={n2}  has_score={n3}  no_allzero={n4}"
    )
    return df


def main():
    pieces = []
    for y in ["2023", "2024", "2025"]:
        d = load_sheet(y)
        print(f"=== {y} raw ===")
        print(d.head(3).to_string())
        d = clean(d)
        pieces.append(d)
    full = pd.concat(pieces, ignore_index=True)
    # Recompute E and verify it matches the recorded value.
    full["E_check"] = full["C"] + full["D"]
    full["E_diff"] = (full["E"] - full["E_check"]).abs()
    print(f"\n=== sanity: E - (C+D) max abs diff = {full['E_diff'].max()} ===")
    full = full.drop(columns=["E_check", "E_diff"])
    full.to_csv(OUT_DIR / "data_clean.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_DIR / 'data_clean.csv'}  rows={len(full)}")
    print("\n=== final head ===")
    print(full.head().to_string())
    print("\n=== describe by year ===")
    for y in ["2023", "2024", "2025"]:
        sub = full[full.year == y]
        print(f"\n--- {y} (n={len(sub)}) ---")
        print(sub[["C", "D", "E", "F"]].describe().to_string())


if __name__ == "__main__":
    main()