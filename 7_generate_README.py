import json
import sqlite3
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

TARGET_COLUMNS = [
    "ISO_1", "HASC_1", "NAME_1", "area_sqkm", 
    "EW_km", "NS_km", "LDP", "CM_CP", "POP_Coverage(%)"
]

PATHS = {
    "gadm_csv": Path("OUTPUT_PROV/gadm41_THA.csv"),
    "logs_dir": Path("OUTPUT_SAMPL"),
    "sqlite_db": Path("OUTPUT_LDP/LDP_Province.sqlite"),
    "out_csv": Path("OUTPUT_LDP/ProvinceLDP.csv"),
    "readme": Path("README.md")
}

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================

def read_gadm_data(csv_path: Path) -> pd.DataFrame:
    """Reads the GADM province CSV file into a pandas DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    return pd.read_csv(csv_path)

def parse_single_log(log_path: Path, hasc: str) -> Dict[str, Any]:
    """Parses a single pipeline log file for MSL stats and JSONL metadata."""
    row_data = {"HASC_1": hasc, "MSL_min_mean_max": None}
    
    try:
        with log_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Extract MSL stats
                if line.startswith("Points MSL min/mean/max [m.]"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        row_data["MSL_min_mean_max"] = parts[1].strip()
                
                # Extract LDP and Coverage metadata
                elif line.startswith('{"meta": "LDP_Definition"') or line.startswith('{"meta": "Coverage_Analysis"'):
                    try:
                        row_data.update(json.loads(line))
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse JSON in {log_path.name}")
                        
    except Exception as e:
        logging.error(f"Failed to read log file {log_path}: {e}")

    return row_data

def parse_all_pipeline_logs(hasc_series: pd.Series, base_dir: Path) -> pd.DataFrame:
    """Iterates through all province identifiers and extracts log data."""
    parsed_data = []

    for hasc in hasc_series:
        # Normalize folder name format (e.g., "TH.AC" -> "TH_AC")
        folder_name = hasc.replace('.', '_')
        log_path = base_dir / folder_name / f"{folder_name}_pipeline.log"
        
        if not log_path.exists():
            logging.warning(f"Log file missing for {hasc}: {log_path}")
            continue

        row_data = parse_single_log(log_path, hasc)
        parsed_data.append(row_data)

    return pd.DataFrame(parsed_data)

def export_sqlite(df: pd.DataFrame, output_path: Path) -> None:
    """Writes the merged DataFrame to a standard SQLite database."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with sqlite3.connect(output_path) as conn:
            df.to_sql("LDP_Province", conn, if_exists="replace", index=False)
        logging.info(f"Saved SQLite database: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save SQLite database: {e}")

def highlight_low_coverage(val: Any) -> str:
    """Formatter to highlight coverage below 75% with a red exclamation emoji."""
    if pd.isna(val):
        return str(val)
    if isinstance(val, (int, float)) and val < 75.0:
        return f'❗ {val}'
    return str(val)

def export_summary_files(df: pd.DataFrame, csv_path: Path, readme_path: Path) -> None:
    """Exports sorted raw CSV and a GitHub-flavored Markdown README."""
    available_cols = [col for col in TARGET_COLUMNS if col in df.columns]
    df_out = df[available_cols].copy()

    # Sort ascending FIRST so that both the CSV and README are ordered correctly
    if "POP_Coverage(%)" in df_out.columns:
        df_out["POP_Coverage(%)"] = pd.to_numeric(df_out["POP_Coverage(%)"], errors='coerce')
        df_out = df_out.sort_values(by="POP_Coverage(%)", ascending=False)

    # 1. Export standard CSV (Now natively sorted, no emojis)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")

    # 2. Format and Export README.md
    if "POP_Coverage(%)" in df_out.columns:
        # Apply the emoji warning for the Markdown table only
        df_out["POP_Coverage(%)"] = df_out["POP_Coverage(%)"].apply(highlight_low_coverage)

    try:
        with readme_path.open('w', encoding='utf-8') as f:
            f.write("# Province LDP & Population Coverage Summary\n\n")
            f.write("*(Table is sorted by population coverage ascending. Values below 75% are highlighted with ❗.)*\n\n")
            f.write(df_out.to_markdown(index=False, tablefmt="github"))
        logging.info(f"Generated README: {readme_path}")
    except Exception as e:
        logging.error(f"Failed to write README.md: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    logging.info("Starting log parsing pipeline...")

    # Step 1: Read reference
    logging.info(f"Reading GADM reference data from {PATHS['gadm_csv']}...")
    df_gadm = read_gadm_data(PATHS['gadm_csv'])

    # Steps 2-4: Parse logs
    logging.info("Parsing pipeline logs for metadata and stats...")
    df_logs = parse_all_pipeline_logs(df_gadm['HASC_1'], PATHS['logs_dir'])

    # Merge data
    if not df_logs.empty:
        df_merged = pd.merge(df_gadm, df_logs, on="HASC_1", how="left")
    else:
        logging.warning("No log data extracted. Proceeding with base GADM data only.")
        df_merged = df_gadm.copy()

    # Steps 5-6: Exports
    export_sqlite(df_merged, PATHS['sqlite_db'])
    export_summary_files(df_merged, PATHS['out_csv'], PATHS['readme'])
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
