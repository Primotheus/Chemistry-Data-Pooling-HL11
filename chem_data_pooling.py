#!/usr/bin/env python3
"""
chem_data_pooling.py
Finalized version — IB Chemistry HL 11 Data Pooling (Density, Reaction Time)
Author: Luke (John Sneppor)
Updated by Gemini to handle wide-format 'Density Cubes' sheet.
"""

import os
import re
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
EXCEL_FILES = ["density_stations.xlsx", "rubberstopper.xlsx"]
OUT_DIR = Path("plots")
SUMMARY_XLSX = "summary_report.xlsx"

SMALL_N_MIN = 3
SMALL_N_MAX = 9

plt.style.use('dark_background')  # Apply dark theme
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True
})
# ----------------------------


def ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_sheets(path: str):
    """Load all sheets from an Excel file into a dict of DataFrames."""
    # Load all sheets, using header=0 by default
    return pd.read_excel(path, sheet_name=None)


def looks_like_date_series(s: pd.Series) -> bool:
    """Heuristic check for date-like columns."""
    sstr = s.astype(str).str.strip()
    # Regex to match common date-like formats
    date_like = sstr.str.match(r'^\s*\d{1,4}\s*[-/]\s*\d{1,4}\s*[-/]\s*\d{2,4}\s*$')
    return date_like.sum() >= max(1, int(0.4 * len(s)))


def clean_df(df: pd.DataFrame):
    """Clean DataFrame: drop empty rows/cols, strip headers, coerce numerics."""
    df = df.copy().dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].astype(str)
            # Try to extract numbers, even with text
            extracted = s.str.extract(r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
            num_ok = extracted[0].notna().sum()
            # If a good portion of the column is numeric-like, convert it
            if num_ok >= (len(df) * 0.4):
                df[col] = pd.to_numeric(extracted[0], errors="coerce")
                continue
            
            # Try to convert date-like columns
            try:
                if looks_like_date_series(df[col]):
                    df[col] = pd.to_datetime(s, errors="coerce", dayfirst=False)
            except Exception:
                pass  # Ignore conversion errors
    return df


def sanitize_sheet_name(writer, desired_name: str) -> str:
    """Ensure Excel sheet name <= 31 chars and unique."""
    # Remove invalid Excel characters
    safe = re.sub(r'[:\\/?*\[\]]', '_', desired_name)
    max_len = 31
    base = safe[:max_len]
    
    # Check if sheet name already exists
    existing = set(writer.sheets.keys())
    if base not in existing:
        return base
        
    # If it exists, append a number
    for i in range(1, 1000):
        suffix = f"_{i}"
        allowed_len = max_len - len(suffix)
        candidate = (safe[:allowed_len] + suffix)
        if candidate not in existing:
            return candidate
            
    # Fallback if we somehow fail
    return base[:max_len]


def find_uncert_col(df: pd.DataFrame):
    """Find the most likely uncertainty column."""
    cols = list(df.columns)
    lc = [str(c).lower() for c in cols]
    
    # Prioritize columns with explicit uncertainty keywords
    for kw in ["uncert", "unc", "error", "err", "±", "plusminus", "sd_inst", "instrument"]:
        for i, name in enumerate(lc):
            if kw in name:
                return cols[i], True
                
    # Check for percentage columns as a secondary option
    for i, name in enumerate(lc):
        if "%" in name or "percent" in name:
            return cols[i], True
            
    return None, False


def extract_values_and_uncert(df: pd.DataFrame):
    """Find the main data column and uncertainty column."""
    cols = list(df.columns)
    lc = [str(c).lower() for c in cols]
    value_candidates = []
    
    # Find likely main data columns
    for kw in ["density", "time", "mass", "length", "volume", "reading"]:
        for i, name in enumerate(lc):
            if kw in name:
                value_candidates.append(cols[i])
                
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if value_candidates:
        # Use the first candidate found
        value_col = value_candidates[0]
    elif numeric_cols:
        # Fallback: use the numeric column with the most data
        counts = {c: df[c].notna().sum() for c in numeric_cols}
        value_col = max(counts, key=counts.get)
    else:
        # Final fallback: just use the first column
        value_col = cols[0] if cols else None

    unc_col, found = find_uncert_col(df)
    uncert_series = None
    if found and unc_col in df.columns:
        # Found an explicit uncertainty column
        uncert_series = pd.to_numeric(df[unc_col], errors="coerce")
    else:
        # Fallback: check other columns for '±' symbols
        for c in cols:
            s = df[c].astype(str)
            pm = s.str.extract(r"±\s*([0-9]*\.?[0-9]+)")
            if pm.notna().sum().sum() > 0:
                uncert_series = pd.to_numeric(pm[0], errors="coerce")
                break

    values = pd.to_numeric(df[value_col], errors="coerce") if value_col in df.columns else pd.Series(dtype=float)
    
    return values.dropna().reset_index(drop=True), (uncert_series.dropna().reset_index(drop=True)
            if isinstance(uncert_series, pd.Series) else (pd.Series(uncert_series)
            if uncert_series is not None else None))


def compute_uncertainty_in_mean(values: pd.Series, per_point_uncert: pd.Series = None):
    """Calculate mean and combined uncertainty using IB methods."""
    n = int(values.size)
    if n == 0:
        return None
        
    mean = float(values.mean())
    std_sample = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = std_sample / math.sqrt(n) if n > 0 else float("nan") # (σ / sqrt(N))
    R = float(values.max() - values.min())
    delta_range = R / (2.0 * n) if n > 0 else float("nan") # (Range / 2N)

    # Choose statistical uncertainty based on N
    if SMALL_N_MIN <= n <= SMALL_N_MAX:
        chosen_stat_unc = delta_range
        method = f"range_method (R/(2N)), N={n}"
    elif n >= SMALL_N_MAX + 1:
        chosen_stat_unc = sem
        method = f"statistical (σ/√N), N={n}"
    else: # n < SMALL_N_MIN
        chosen_stat_unc = max(sem, delta_range) if n > 1 else 0.0
        method = f"low_N (N={n}); using max(sem,range_est)"

    u_instr_mean = None
    if per_point_uncert is not None:
        per_point_uncert = pd.to_numeric(per_point_uncert, errors="coerce").dropna().reset_index(drop=True)
        if per_point_uncert.size == n:
            # Add instrument uncertainties in quadrature, divide by N for mean
            u_instr_mean = math.sqrt(np.sum(np.array(per_point_uncert, dtype=float) ** 2)) / n
        elif per_point_uncert.size > 0:
            # Handle cases where uncertainty data is incomplete
            u_instr_mean = math.sqrt(np.sum(np.array(per_point_uncert, dtype=float) ** 2)) / max(1, per_point_uncert.size)

    # Combine statistical and instrument uncertainty in quadrature
    if u_instr_mean is not None:
        combined_unc = math.sqrt(chosen_stat_unc ** 2 + u_instr_mean ** 2)
    else:
        combined_unc = chosen_stat_unc
        
    rel_combined_pct = 100.0 * combined_unc / mean if mean != 0 else float("nan")

    return {
        "n": n, "mean": mean, "std_sample": std_sample, "sem": sem,
        "R": R, "delta_range": delta_range, "chosen_stat_unc": chosen_stat_unc,
        "method": method, "u_instr_mean": u_instr_mean, "combined_unc": combined_unc,
        "rel_combined_pct": rel_combined_pct, "min": float(values.min()), "max": float(values.max()),
        "median": float(values.median()), "q1": float(values.quantile(0.25)), "q3": float(values.quantile(0.75))
    }


def plot_values(values: pd.Series, metrics: dict, name: str):
    """Generate and save histogram, boxplot, and index plot."""
    n = metrics["n"]
    mean = metrics["mean"]
    std = metrics["std_sample"]
    combined_unc = metrics["combined_unc"]

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(values, bins=12, edgecolor="black", alpha=0.75)
    ax.axvline(mean, linestyle="--", linewidth=1.2, label=f"mean = {mean:.4g}")
    if n > 1:
        ax.axvline(mean + std, linestyle=":", linewidth=1, label=f"mean ± 1σ")
        ax.axvline(mean - std, linestyle=":", linewidth=1)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} — Histogram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_hist.png")
    plt.close(fig)

    # Boxplot
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.boxplot(values, vert=True, tick_labels=[name])  # Changed 'labels' to 'tick_labels'
    ax.set_title(f"{name} — Boxplot")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_box.png")
    plt.close(fig)

    # Value plot (Index Plot)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(values))
    ax.plot(x, values.values, "o-", markersize=4, label="Measurements")
    ax.axhline(mean, color="green", linestyle="--", label=f"mean = {mean:.4g}")
    if not math.isnan(combined_unc):
        ax.fill_between([-0.5, len(values) - 0.5], mean - combined_unc, mean + combined_unc,
                        color="orange", alpha=0.12, label=f"mean ± combined_unc ({combined_unc:.3g})")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.set_title(f"{name} — Values and Mean ± Combined Uncertainty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_index.png")
    plt.close(fig)


def plot_metal_density_scatter(df: pd.DataFrame, name_prefix: str):
    """Create a scatter plot for a single metal's density by group."""
    print(f"[scatter] Creating density plot for {name_prefix}")
    
    dfc = df.copy()

    # Find group and density columns
    group_col = next((c for c in dfc.columns if "group" in str(c).lower() or "name" in str(c).lower()), None)
    density_col = next((c for c in dfc.columns if "density" in str(c).lower()), None)

    if not group_col or not density_col:
        print(f"[scatter] Skipping {name_prefix}: Could not find Group and Density columns.")
        return
        
    dfc[density_col] = pd.to_numeric(dfc[density_col], errors='coerce')
    # Drop rows where group or density is missing (filters out summary rows)
    dfc = dfc.dropna(subset=[group_col, density_col]) 

    if dfc.empty:
        print(f"[scatter] No valid data found for density plot {name_prefix}.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(dfc[group_col], dfc[density_col], s=50, alpha=0.8, edgecolors="w", linewidth=0.5)
    
    # Add mean line
    mean_density = dfc[density_col].mean()
    if not pd.isna(mean_density):
        plt.axhline(mean_density, color='red', linestyle='--', label=f"Mean = {mean_density:.4f} g/cm³")
    
    plt.title(f"{name_prefix} — Density by Group")
    plt.xlabel("Group (Names)")
    plt.ylabel("Density (g/cm³)")
    plt.xticks(rotation=75, fontsize=8)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    outpath = OUT_DIR / f"{name_prefix}_density_scatter.png"
    plt.savefig(outpath)
    plt.close()
    print(f"[scatter] Saved density plot -> {outpath}")


def plot_reaction_time_scatter(df: pd.DataFrame, name_prefix: str):
    """Create a scatter plot for reaction time by group."""
    print(f"[scatter] Creating reaction time plot for {name_prefix}")
    
    dfc = df.copy()
    
    # Find group and time columns
    group_col = next((c for c in dfc.columns if "group" in str(c).lower() or "name" in str(c).lower()), None)
    time_col = next((c for c in dfc.columns if "time" in str(c).lower()), None)
    
    if not group_col or not time_col:
        print(f"[scatter] Skipping {name_prefix}: Could not find Group and Time columns.")
        return
        
    dfc[time_col] = pd.to_numeric(dfc[time_col], errors='coerce')
    dfc = dfc.dropna(subset=[group_col, time_col]) # Drop rows with missing data

    if dfc.empty:
        print(f"[scatter] No valid data found for reaction time plot {name_prefix}.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(dfc[group_col], dfc[time_col], s=50, alpha=0.8, edgecolors="w", linewidth=0.5)
    
    # Add mean line
    mean_time = dfc[time_col].mean()
    if not pd.isna(mean_time):
        plt.axhline(mean_time, color='red', linestyle='--', label=f"Mean = {mean_time:.3f} s")
    
    plt.title(f"{name_prefix} — Reaction Time by Group")
    plt.xlabel("Group (Names)")
    plt.ylabel("Reaction Time (s)")
    plt.xticks(rotation=75, fontsize=8)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    outpath = OUT_DIR / f"{name_prefix}_reaction_time_scatter.png"
    plt.savefig(outpath)
    plt.close()
    print(f"[scatter] Saved reaction time plot -> {outpath}")


def process_sheet(df: pd.DataFrame, sheet_name: str, writer: pd.ExcelWriter, create_standard_plots: bool = True):
    """Clean, analyze, plot, and report on a single DataFrame."""
    dfc = clean_df(df)
    
    # Try to find a 'group' column to filter out summary rows (which have blank group)
    group_col = next((c for c in dfc.columns if "group" in str(c).lower() or "name" in str(c).lower()), None)
    
    if group_col:
        # Drop rows where the group name is missing, as these are likely summary rows
        dfc = dfc.dropna(subset=[group_col])

    values, per_unc = extract_values_and_uncert(dfc)
    
    if values.empty:
        print(f"Sheet '{sheet_name}': no numeric values found; skipping.")
        return None

    metrics = compute_uncertainty_in_mean(values, per_point_uncert=per_unc)
    if metrics is None:
        print(f"Sheet '{sheet_name}': could not compute metrics.")
        return None

    summary_table = pd.DataFrame({
        "metric": list(metrics.keys()),
        "value": list(metrics.values())
    })

    # Sanitize sheet names for Excel
    values_sheet = sanitize_sheet_name(writer, f"{sheet_name}_values")
    summary_sheet = sanitize_sheet_name(writer, f"{sheet_name}_summary")

    # Write processed data and summary to Excel
    parsed_values_df = pd.DataFrame({
        "value": values,
        "per_point_uncert": (per_unc if per_unc is not None else pd.Series([None] * len(values)))
    })
    parsed_values_df.to_excel(writer, sheet_name=values_sheet, index=False)
    summary_table.to_excel(writer, sheet_name=summary_sheet, index=False)

    # Generate standard plots (hist, box, index) if requested
    if create_standard_plots:
        plot_values(values, metrics, sheet_name.replace(" ", "_")[:40])
    
    print(f"Processed '{sheet_name}': n={metrics['n']}, mean={metrics['mean']:.5g}, combined_unc={metrics['combined_unc']:.5g} ({metrics['method']})")
    return metrics


def main():
    ensure_out()
    results = {}
    writer = pd.ExcelWriter(SUMMARY_XLSX, engine="xlsxwriter")

    for f in EXCEL_FILES:
        if not Path(f).exists():
            print(f"File not found (skipping): {f}")
            continue
        
        # Load all sheets with the default header=0 to inspect them
        try:
            sheets = load_all_sheets(f)
        except Exception as e:
            print(f"Could not read Excel file {f}. Error: {e}")
            continue
            
        for sname, df in sheets.items():
            # Skip the "Stations" sheet entirely
            if "station" in sname.lower():
                print(f"Skipping sheet '{sname}' (raw station data).")
                continue

            # --- [GEMINI] MODIFICATION START ---
            # Special handling for the "Density Cubes" sheet
            if "density cubes" in sname.lower():
                print(f"Found '{sname}'. Applying special multi-metal processing.")
                
                # Based on the CSV snippet, the layout is 3 blocks of 4 columns
                # The actual headers are on row 1 (iloc[0])
                metal_blocks = {
                    "Aluminum": df.iloc[:, 0:4],
                    "Copper":   df.iloc[:, 4:8],
                    "Brass":    df.iloc[:, 8:12]
                }

                for metal_name, metal_df_raw in metal_blocks.items():
                    # The headers are on row 0 of the *sliced* df
                    new_header = metal_df_raw.iloc[0].astype(str)
                    # The data starts from row 1 of the *sliced* df
                    metal_df = metal_df_raw.iloc[1:].copy() 
                    metal_df.columns = new_header # Set the correct headers
                    metal_df = metal_df.reset_index(drop=True)
                    
                    # Create a unique key for this sub-dataset
                    key = f"{Path(f).stem}__{sname}__{metal_name}"
                    print(f"  Processing sub-sheet: {key}")
                    
                    # --- [GEMINI] NEW ---
                    # Create the individual scatter plot for this metal
                    plot_metal_density_scatter(metal_df, key.replace(" ", "_")[:40])
                    # --- [GEMINI] END ---
                    
                    # Process this individual metal's DataFrame
                    # --- [GEMINI] MODIFIED ---
                    # Set create_standard_plots=False to skip hist, box, index plots
                    metrics = process_sheet(metal_df, key, writer, create_standard_plots=False)
                    
                    # Report stats to console
                    if metrics:
                        print(f"  > {metal_name} Stats: Mean = {metrics['mean']:.4g}, Std Dev = {metrics['std_sample']:.4g}")
                    # --- [GEMINI] END ---
                    results[key] = metrics
                
                # We've handled this sheet, so skip the default processing
                continue 
            # --- [GEMINI] MODIFICATION END ---

            # Default processing for all other sheets
            key = f"{Path(f).stem}__{sname}"
            
            # --- [GEMINI] MODIFIED ---
            # Default to True, but set to False for 'reaction time'
            create_plots = True
            if "reaction time" in sname.lower():
                create_plots = False
                # Call the dedicated scatter plot function for reaction time
                plot_reaction_time_scatter(df, key.replace(" ", "_")[:40])
                
            metrics = process_sheet(df, key, writer, create_standard_plots=create_plots)
            # --- [GEMINI] END ---
            results[key] = metrics

    # Create the final summary tab
    overall = []
    for k, m in results.items():
        if m: # Only include if metrics were successfully computed
            overall.append({
                "dataset": k,
                "n": m["n"],
                "mean": m["mean"],
                "combined_unc": m["combined_unc"],
                "rel_combined_pct": m["rel_combined_pct"],
                "method": m["method"]
            })
            
    overall_df = pd.DataFrame(overall)
    if not overall_df.empty:
        overall_df.to_excel(writer, sheet_name=sanitize_sheet_name(writer, "overall_summary"), index=False)
        
    writer.close()
    print(f"Saved summary Excel -> {SUMMARY_XLSX}")
    print(f"Plots saved to folder -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually")
    main()



