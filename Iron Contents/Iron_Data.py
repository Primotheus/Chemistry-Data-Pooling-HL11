#!/usr/bin/env python3

'''
Runs:
Liad
A
B
C
D
E
Khoa
Eric
Wilson
Mohammad
Angus
'''

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
# Try locating the input CSV relative to this script first, then fallback to CWD and common subfolders
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "All_Iron_Data.csv"
if not INPUT_CSV.exists():
    alt = Path.cwd() / "All_Iron_Data.csv"
    if alt.exists():
        INPUT_CSV = alt
    else:
        alt2 = Path.cwd() / "Iron Contents" / "All_Iron_Data.csv"
        if alt2.exists():
            INPUT_CSV = alt2

OUT_DIR = Path("iron_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using input CSV: {INPUT_CSV}")

# Helper: find column by keywords (case-insensitive)
def find_col_by_keywords(columns, keywords):
    cols_l = [str(c).lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(cols_l):
            if kw in c:
                return columns[i]
    return None

# Load data (support wide paired columns like "Run 1: Wavelength", "Run 1: Absorbance")
df_raw = pd.read_csv(INPUT_CSV)

# look for pairs of columns where a prefix (e.g. 'Run 1' or 'Latest') has a 'Wavelength' and an 'Absorbance' column
pair_wl_regex = re.compile(r'^(?P<prefix>.+?):\s*Wavelength', re.I)
pairs = []
for col in df_raw.columns:
    m = pair_wl_regex.match(col)
    if not m:
        continue
    prefix = m.group('prefix').strip()
    wl_col = col
    # find absorb column with same prefix
    absorb_col = None
    for c in df_raw.columns:
        if c.lower().startswith(prefix.lower()) and 'absorb' in c.lower():
            absorb_col = c
            break
    if absorb_col:
        pairs.append((prefix, wl_col, absorb_col))

if pairs:
    long_rows = []
    for prefix, wl_col, absorb_col in pairs:
        tmp = pd.DataFrame({
            'Run': prefix,
            'Wavelength': pd.to_numeric(df_raw[wl_col], errors='coerce'),
            'Absorbance': pd.to_numeric(df_raw[absorb_col], errors='coerce')
        })
        long_rows.append(tmp)
    df = pd.concat(long_rows, ignore_index=True)

    # normalize run names to the requested labels
    rename_map = {
        'run 1': 'Solution A',
        'run 2': 'Solution B',
        'run 3': 'Solution C',
        'run 4': 'Solution D',
        'run 5': 'Solution E',
        'run 6': 'Unknown Khoa',
        'run 7': 'Unknown Eric',
        'run 8': 'Unknown Wilson',
        'run 9': 'Unknown Mohammad',
        'run 10': 'Unknown Angus',
        'latest': 'Unknown Liad',
    }

    def map_run_name(r):
        if pd.isna(r):
            return r
        rl = str(r).strip().lower()
        return rename_map.get(rl, r)

    df['Run'] = df['Run'].apply(map_run_name)

    wavelength_col = 'Wavelength'
    absorb_col = 'Absorbance'
    run_col = 'Run'
    print(f"Detected paired run columns: {[p[0] for p in pairs]}")
else:
    # fallback to original long-format CSV (one Wavelength/Absorbance column per row)
    df = df_raw

# Convert types
df[wavelength_col] = pd.to_numeric(df[wavelength_col], errors="coerce")
df[absorb_col] = pd.to_numeric(df[absorb_col], errors="coerce")
df = df.dropna(subset=[wavelength_col, absorb_col])

# Basic grouping
groups = list(df[run_col].unique())
print(f"Detected runs: {groups}")

# Per-run plots
def sanitize(s):
    return re.sub(r"[^\w\-]+", "_", str(s)).strip("_")

for run, g in df.groupby(run_col):
    g = g.sort_values(wavelength_col)
    plt.figure(figsize=(6, 3))
    plt.plot(g[wavelength_col], g[absorb_col], '-k', lw=1)
    # use the run label directly as the title (no 'Run:' prefix)
    plt.title(f"{run}")
    plt.xlabel(f"{wavelength_col}")
    plt.ylabel(f"{absorb_col}")
    plt.grid(alpha=0.3)
    # save using the sanitized run label (no 'run_' prefix)
    fn = OUT_DIR / f"{sanitize(run)}.png"
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()

# Overlay all runs
plt.figure(figsize=(8, 4))
for run, g in df.groupby(run_col):
    g = g.sort_values(wavelength_col)
    plt.plot(g[wavelength_col], g[absorb_col], alpha=0.6, label=str(run))
plt.xlabel(f"{wavelength_col}")
plt.ylabel(f"{absorb_col}")
plt.title("Overlay of all runs")
plt.legend(ncol=2, fontsize="small")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "all_unk_runs.png", dpi=150)
plt.close()

# ----- New: overlay and mean±std for Solution A-E only (Run 1-5) -----
subset_requested = ['Solution A', 'Solution B', 'Solution C', 'Solution D', 'Solution E']
available = [r for r in subset_requested if r in df[run_col].unique()]
if available:
    # overlay
    plt.figure(figsize=(8, 4))
    for run in available:
        g = df[df[run_col] == run].sort_values(wavelength_col)
        plt.plot(g[wavelength_col], g[absorb_col], alpha=0.8, label=str(run))
    plt.xlabel(f"{wavelength_col}")
    plt.ylabel(f"{absorb_col}")
    plt.title("Overlay: Solution A-E")
    plt.legend(ncol=2, fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "all_ABCDE_runs.png", dpi=150)
    plt.close()

    # mean ± std across the selected solutions
    pivot_sub = df[df[run_col].isin(available)].pivot_table(index=wavelength_col, columns=run_col, values=absorb_col, aggfunc='mean')
    pivot_sub = pivot_sub.sort_index()
    mean_spec_sub = pivot_sub.mean(axis=1)
    std_spec_sub = pivot_sub.std(axis=1, ddof=1)
    plt.figure(figsize=(8, 4))
    plt.plot(mean_spec_sub.index, mean_spec_sub.values, color='darkgreen', lw=1.5, label='mean (A-E)')
    plt.fill_between(mean_spec_sub.index, mean_spec_sub - std_spec_sub, mean_spec_sub + std_spec_sub, color='darkgreen', alpha=0.25, label='±1σ')
    plt.xlabel(f"{wavelength_col}")
    plt.ylabel(f"{absorb_col}")
    plt.title("Mean spectrum (Solutions A-E) ± 1σ")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_std_ABCDE.png", dpi=150)
    plt.close()
else:
    print("No Solution A-E runs found to plot subset.")

# Pivot to common wavelength index (mean if duplicates)
pivot = df.pivot_table(index=wavelength_col, columns=run_col, values=absorb_col, aggfunc="mean")
pivot = pivot.sort_index()

# Mean spectrum ± std
mean_spec = pivot.mean(axis=1)
std_spec = pivot.std(axis=1, ddof=1)
plt.figure(figsize=(8, 4))
plt.plot(mean_spec.index, mean_spec.values, color="navy", lw=1.5, label="mean")
plt.fill_between(mean_spec.index, mean_spec - std_spec, mean_spec + std_spec, color="navy", alpha=0.2, label="±1σ")
plt.xlabel(f"{wavelength_col}")
plt.ylabel(f"{absorb_col}")
plt.title("Mean spectrum ± 1σ")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "mean_std_all.png", dpi=150)
plt.close()

# Heatmap (runs × wavelength) — reorder runs for display
heatmap_data = pivot.T.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
plt.figure(figsize=(10, max(3, 0.25 * len(heatmap_data))))
plt.imshow(heatmap_data.values, aspect="auto", origin="lower", cmap="viridis",
           extent=[heatmap_data.columns.min(), heatmap_data.columns.max(), 0, heatmap_data.shape[0]])
plt.yticks(np.arange(0.5, heatmap_data.shape[0] + 0.5), heatmap_data.index)
plt.xlabel(f"{wavelength_col}")
plt.title("Heatmap: runs (y) vs wavelength (x) — color = absorbance")
cbar = plt.colorbar()
cbar.set_label(f"{absorb_col}")
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_runs_wavelength.png", dpi=150)
plt.close()

# Simple peak detection (local maxima) per run
def find_local_peaks(x, y):
    # returns indices of local maxima (simple neighbor comparison)
    if len(y) < 3:
        return np.array([], dtype=int)
    mid = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    peaks_idx = np.where(mid)[0] + 1
    return peaks_idx

summary_rows = []
for run, g in df.groupby(run_col):
    g = g.sort_values(wavelength_col)
    x = g[wavelength_col].to_numpy()
    y = g[absorb_col].to_numpy()
    n = len(x)
    mean_run = np.nanmean(y)
    std_run = np.nanstd(y, ddof=1) if n > 1 else float("nan")
    max_idx = np.nanargmax(y) if n > 0 else None
    max_wl = x[max_idx] if max_idx is not None else np.nan
    max_val = y[max_idx] if max_idx is not None else np.nan

    peaks_idx = find_local_peaks(x, y)
    peaks_wl = x[peaks_idx].tolist()
    peaks_val = y[peaks_idx].tolist()

    summary_rows.append({
        "run": run,
        "n_points": n,
        "mean_absorbance": mean_run,
        "std_absorbance": std_run,
        "max_wavelength": max_wl,
        "max_absorbance": max_val,
        "n_peaks_found": len(peaks_idx),
        "peaks_wavelengths": ";".join(map(str, peaks_wl)),
        "peaks_values": ";".join(map(str, peaks_val)),
    })

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "iron_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary -> {summary_path}")

# Annotated overlay with detected main peak markers (first peak per run)
plt.figure(figsize=(8, 4))
for row in summary_rows:
    run = row["run"]
    g = df[df[run_col] == run].sort_values(wavelength_col)
    plt.plot(g[wavelength_col], g[absorb_col], alpha=0.5, label=str(run))
    # mark max
    if not np.isnan(row["max_wavelength"]):
        plt.scatter([row["max_wavelength"]], [row["max_absorbance"]], s=30, marker="x", color="red")
plt.xlabel(f"{wavelength_col}")
plt.ylabel(f"{absorb_col}")
plt.title("All Runs with Peak Markers")
plt.legend(ncol=2, fontsize="small")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "all_unk_peak_runs.png", dpi=150)
plt.close()

print("Plots saved in:", OUT_DIR)
print("Per-run summary saved to:", summary_path)

