#!/usr/bin/env python3
"""
Summarise LDP analysis values from every *.log file under OUTPUT_SAMPL/.

Extracted fields
----------------
- HASC_1
- ISO_1 (extracted from GADM)
- English province name (from a GADM Admin-1 vector file)
- Points CSF min/mean/max [ppm]
- Points MSL min/mean/max [m]
- LDP population coverage statistics

Example
-------
python3 analyse_ldp_logs.py OUTPUT_SAMPL \
    --output OUTPUT_SAMPL/ldp_log_summary.csv

The CSV is written as standard UTF-8. Use --excel-bom for UTF-8 with BOM.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


NUMBER = r"[+-]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?"

CSF_RE = re.compile(
    rf"Points\s+CSF\s+min/mean/max\s*\[ppm\]\s*:\s*"
    rf"({NUMBER})\s*/\s*({NUMBER})\s*/\s*({NUMBER})",
    re.IGNORECASE,
)

MSL_RE = re.compile(
    rf"Points\s+MSL\s+min/mean/max\s*\[(?:m\.?|metres?|meters?)\]\s*:\s*"
    rf"({NUMBER})\s*/\s*({NUMBER})\s*/\s*({NUMBER})",
    re.IGNORECASE,
)

COVERAGE_RE = re.compile(
    rf"\|\s*(TH\.[A-Z0-9]+)\s*"
    rf"\|\s*({NUMBER})\s*"
    rf"\|\s*({NUMBER})\s*"
    rf"\|\s*({NUMBER})\s*"
    rf"\|\s*({NUMBER})\s*"
    rf"\|\s*({NUMBER})\s*"
    rf"\|\s*({NUMBER})\s*\|",
    re.IGNORECASE,
)

HASC_RE = re.compile(r"\b(TH\.[A-Z0-9]+)\b", re.IGNORECASE)

OUTPUT_COLUMNS = [
    "HASC_1",
    "ISO_1",
    "Province_EN",
    "CSF_Min_ppm",
    "CSF_Mean_ppm",
    "CSF_Max_ppm",
    "MSL_Min_m",
    "MSL_Mean_m",
    "MSL_Max_m",
    "Total_Pts",
    "Valid_Pts",
    "Pt_Coverage_pct",
    "Total_POP",
    "Valid_POP",
    "POP_Coverage_pct",
    "Log_File",
]


def parse_number(value: str) -> float:
    """Convert a signed/comma-formatted numeric string to float."""
    return float(value.replace(",", ""))


def integer_or_float(value: str) -> int | float:
    """Return int when the value is integral; otherwise return float."""
    number = parse_number(value)
    return int(number) if number.is_integer() else number


def infer_hasc_from_path(log_path: Path) -> str | None:
    """Try to obtain HASC_1 from the filename or one of its parent folders."""
    for text in [log_path.stem, log_path.name, *[p.name for p in log_path.parents]]:
        match = HASC_RE.search(text)
        if match:
            return match.group(1).upper()
    return None


def parse_log(log_path: Path) -> dict[str, Any]:
    """Read one log and extract its LDP statistics."""
    text = log_path.read_text(encoding="utf-8", errors="replace")

    result: dict[str, Any] = {
        "HASC_1": infer_hasc_from_path(log_path),
        "Log_File": str(log_path),
    }

    csf_matches = list(CSF_RE.finditer(text))
    if csf_matches:
        values = [parse_number(v) for v in csf_matches[-1].groups()]
        result["CSF_Min_ppm"], result["CSF_Mean_ppm"], result["CSF_Max_ppm"] = values

    msl_matches = list(MSL_RE.finditer(text))
    if msl_matches:
        values = [parse_number(v) for v in msl_matches[-1].groups()]
        result["MSL_Min_m"], result["MSL_Mean_m"], result["MSL_Max_m"] = values

    coverage_matches = list(COVERAGE_RE.finditer(text))
    if coverage_matches:
        match = coverage_matches[-1]
        (
            hasc,
            total_pts,
            valid_pts,
            pt_coverage,
            total_pop,
            valid_pop,
            pop_coverage,
        ) = match.groups()

        result.update(
            {
                "HASC_1": hasc.upper(),
                "Total_Pts": integer_or_float(total_pts),
                "Valid_Pts": integer_or_float(valid_pts),
                "Pt_Coverage_pct": parse_number(pt_coverage),
                "Total_POP": integer_or_float(total_pop),
                "Valid_POP": integer_or_float(valid_pop),
                "POP_Coverage_pct": parse_number(pop_coverage),
            }
        )

    return result


def choose_eng_name(row: pd.Series) -> str | None:
    """Choose the English province name from GADM NAME_1 column."""
    if "NAME_1" in row.index and pd.notna(row["NAME_1"]):
        raw = str(row["NAME_1"]).strip()
        if raw:
            return raw
    return None


def load_gadm_data(gadm_path: Path, layer: str | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """Load HASC_1 -> English province name & ISO_1 from a GADM Admin-1 vector dataset."""
    eng_mapping: dict[str, str] = {}
    iso_mapping: dict[str, str] = {}
    
    if not gadm_path.exists():
        print(f"WARNING: GADM file not found at {gadm_path}. Skipping province names/ISO.")
        return eng_mapping, iso_mapping

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "geopandas is required for reading GADM. Install it with:\n"
            "  conda install -c conda-forge geopandas"
        ) from exc

    kwargs: dict[str, Any] = {}
    if layer:
        kwargs["layer"] = layer
        
    try:
        kwargs["engine"] = "pyogrio"
        gdf = gpd.read_file(gadm_path, **kwargs)
    except:
        kwargs.pop("engine", None)
        gdf = gpd.read_file(gadm_path, **kwargs)

    if "HASC_1" not in gdf.columns:
        raise ValueError(
            f"{gadm_path} (Layer: {layer}) does not contain an HASC_1 column. "
            f"Available columns: {', '.join(map(str, gdf.columns))}"
        )

    for _, row in gdf.iterrows():
        if pd.isna(row["HASC_1"]):
            continue

        hasc = str(row["HASC_1"]).strip().upper()
        
        name = choose_eng_name(row)
        if name:
            eng_mapping[hasc] = name
            
        if "ISO_1" in row.index and pd.notna(row["ISO_1"]):
            iso_val = str(row["ISO_1"]).strip()
            if iso_val:
                iso_mapping[hasc] = iso_val

    return eng_mapping, iso_mapping


def load_name_csv(
    csv_path: Path,
    hasc_column: str,
    eng_column: str,
    iso_column: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Load HASC_1 mappings from a UTF-8 CSV file."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    eng_mapping: dict[str, str] = {}
    iso_mapping: dict[str, str] = {}

    if hasc_column not in df.columns:
        raise ValueError(f"Missing HASC mapping column '{hasc_column}' in {csv_path}.")

    for _, row in df.iterrows():
        hasc = row.get(hasc_column)
        if pd.isna(hasc):
            continue
        hasc_str = str(hasc).strip().upper()
        
        if eng_column in df.columns and pd.notna(row[eng_column]):
            eng_mapping[hasc_str] = str(row[eng_column]).strip()
            
        if iso_column in df.columns and pd.notna(row[iso_column]):
            iso_mapping[hasc_str] = str(row[iso_column]).strip()

    return eng_mapping, iso_mapping


def build_summary(
    root: Path,
    eng_mapping: dict[str, str],
    iso_mapping: dict[str, str],
) -> pd.DataFrame:
    """Parse all recursive *.log files and build the base result DataFrame."""
    log_files = sorted(root.rglob("*.log"))

    if not log_files:
        raise FileNotFoundError(f"No *.log files found under: {root}")

    records = [parse_log(path) for path in log_files]
    df = pd.DataFrame.from_records(records)

    # Apply mappings
    df["Province_EN"] = df["HASC_1"].map(eng_mapping) if eng_mapping else None
    df["ISO_1"] = df["HASC_1"].map(iso_mapping) if iso_mapping else None

    analysis_columns = [
        "CSF_Min_ppm",
        "CSF_Mean_ppm",
        "CSF_Max_ppm",
        "MSL_Min_m",
        "MSL_Mean_m",
        "MSL_Max_m",
        "Total_Pts",
        "Valid_Pts",
        "Pt_Coverage_pct",
        "Total_POP",
        "Valid_POP",
        "POP_Coverage_pct",
    ]
    df = df.loc[df[analysis_columns].notna().any(axis=1)].copy()

    if df.empty:
        raise ValueError(
            "Log files were found, but none contained the expected CSF, MSL, "
            "or LDP Population Coverage lines."
        )

    # Ensure all output columns exist (fills missing with NaNs)
    df = df.reindex(columns=OUTPUT_COLUMNS)

    return df.reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    """Print a readable summary table to the terminal."""
    # Restrict columns shown in the terminal
    display_cols = [
        "HASC_1", "ISO_1", "Province_EN",
        "CSF_Min_ppm", "CSF_Mean_ppm", "CSF_Max_ppm",
        "MSL_Min_m", "MSL_Mean_m", "MSL_Max_m",
        "Pt_Coverage_pct", "POP_Coverage_pct"
    ]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()

    # Limit Province_EN to 10 characters
    if "Province_EN" in display_df.columns:
        display_df["Province_EN"] = display_df["Province_EN"].apply(
            lambda x: str(x)[:10] if pd.notna(x) else ""
        )

    # Format CSF and MSL columns (force 1 decimal place with a sign)
    for column in ("CSF_Min_ppm", "CSF_Mean_ppm", "CSF_Max_ppm", "MSL_Min_m", "MSL_Mean_m", "MSL_Max_m"):
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda x: f"{float(x):+.1f}" if pd.notna(x) else ""
            )

    # Format percentages (2 decimal places)
    for column in ("Pt_Coverage_pct", "POP_Coverage_pct"):
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else ""
            )

    try:
        # stralign="right" keeps the numbers nicely justified 
        print(display_df.to_markdown(index=False, stralign="right"))
    except ImportError:
        print(display_df.to_string(index=False, justify="right"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise CSF, MSL and population coverage from LDP logs."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("OUTPUT_SAMPL"),
        help="Root directory containing recursive *.log files "
        "(default: OUTPUT_SAMPL)",
    )
    parser.add_argument(
        "--gadm",
        type=Path,
        default=Path("DATA/gadm41_THA.gpkg"),
        help="GADM Admin-1 vector file containing HASC_1, ISO_1, and province names (default: DATA/gadm41_THA.gpkg)",
    )
    parser.add_argument(
        "--layer",
        default="ADM_ADM_1",
        help="Optional layer name when --gadm points to a GeoPackage (default: ADM_ADM_1)",
    )
    parser.add_argument(
        "--name-csv",
        type=Path,
        help="Optional UTF-8 CSV containing HASC_1, ISO_1, and English province names",
    )
    parser.add_argument(
        "--hasc-column",
        default="HASC_1",
        help="HASC column in --name-csv (default: HASC_1)",
    )
    parser.add_argument(
        "--eng-column",
        default="Province_EN",
        help="English-name column in --name-csv (default: Province_EN)",
    )
    parser.add_argument(
        "--iso-column",
        default="ISO_1",
        help="ISO column in --name-csv (default: ISO_1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("OUTPUT_SAMPL/ldp_log_summary.csv"),
        help="Output CSV path (default: OUTPUT_SAMPL/ldp_log_summary.csv)",
    )
    parser.add_argument(
        "--excel-bom",
        action="store_true",
        help="Write UTF-8 with BOM for older Excel versions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not args.root.is_dir():
        print(f"ERROR: Directory does not exist: {args.root}", file=sys.stderr)
        return 2

    try:
        eng_mapping = {}
        iso_mapping = {}

        if args.name_csv:
            eng_mapping, iso_mapping = load_name_csv(
                args.name_csv,
                args.hasc_column,
                args.eng_column,
                args.iso_column
            )
        else:
            eng_mapping, iso_mapping = load_gadm_data(args.gadm, args.layer)

        # 1. Build the base DataFrame
        df_base = build_summary(args.root, eng_mapping, iso_mapping)

        # 2. Sort DataFrame 1: POP_Coverage_pct (Descending)
        df_pop = df_base.assign(_sort_pop=df_base["POP_Coverage_pct"].fillna(-1)).sort_values(
            by=["_sort_pop", "HASC_1"], 
            ascending=[False, True]
        ).drop(columns="_sort_pop").reset_index(drop=True)
        
        # 3. Sort DataFrame 2: ISO_1 (Ascending)
        df_iso = df_base.assign(_sort_iso=df_base["ISO_1"].fillna("ZZZ")).sort_values(
            by=["_sort_iso", "HASC_1"], 
            ascending=[True, True]
        ).drop(columns="_sort_iso").reset_index(drop=True)

        # Save the primary sorted table (POP_Coverage_pct) to CSV (with full raw columns)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        encoding = "utf-8-sig" if args.excel_bom else "utf-8"
        df_pop.to_csv(args.output, index=False, encoding=encoding)

        # Print Output Table 1
        print("\n" + "="*50 + " Table 1: Sorted by POP_Coverage_pct (Descending) " + "="*50)
        print_summary(df_pop)
        
        # Print Output Table 2
        print("\n" + "="*56 + " Table 2: Sorted by ISO_1 (Ascending) " + "="*56)
        print_summary(df_iso)

        # Final System Logs
        print(f"\nLogs parsed : {len(df_base):,}")
        print(f"CSV output  : {args.output}")
        print(f"Encoding    : {encoding}")

        # Check for missing province names
        if not eng_mapping:
            print(
                "\nNOTE: Province_EN is blank because GADM file could not be read "
                "or mapping was not supplied.",
                file=sys.stderr,
            )
        else:
            missing_names = df_base.loc[df_base["Province_EN"].isna(), "HASC_1"].dropna().unique()
            if len(missing_names):
                print(
                    "\nWARNING: No English province name found for: "
                    + ", ".join(sorted(missing_names)),
                    file=sys.stderr,
                )
                
        # Check for missing ISO_1 codes
        if iso_mapping:
            missing_iso = df_base.loc[df_base["ISO_1"].isna(), "HASC_1"].dropna().unique()
            if len(missing_iso):
                print(
                    "WARNING: No ISO_1 code found for: "
                    + ", ".join(sorted(missing_iso)),
                    file=sys.stderr,
                )

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
