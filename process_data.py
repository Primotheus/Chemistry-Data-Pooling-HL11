import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#Chemistry Data Pooling (Density Lab & Rubber Stopper)

# prefer seaborn darkgrid if available, otherwise fall back
try:
    plt.style.use("seaborn-darkgrid")
except Exception:
    try:
        plt.style.use("seaborn")
    except Exception:
        plt.style.use("default")
 
EXCEL_FILES = ["density_stations.xlsx", "rubberstopper.xlsx"]
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

def load_all_sheets(path):
    # returns dict: sheet_name -> DataFrame
    sheets = pd.read_excel(path, sheet_name=None)
    return {k: v for k, v in sheets.items()}


def clean_df(df):
    # drop fully empty rows/cols
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # try convert numeric-like columns
    for col in df.columns:
        if df[col].dtype == object:
            # try numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_na = coerced.notna().sum()
            if non_na >= (len(df) * 0.5):  # if many values parse as numeric
                df[col] = coerced
            else:
                # try datetime
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    dt = pd.to_datetime(df[col], errors="coerce")
                if dt.notna().sum() >= (len(df) * 0.5):
                    df[col] = dt
    return df


def summarize(df, name):
    print(f"=== Summary: {name} ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Dtypes:")
    print(df.dtypes)
    print("\nNumeric describe:")
    print(df.select_dtypes(include=[np.number]).describe().T)
    print("\nSample rows:")
    print(df.head(), "\n")


def plot_numeric(df, name, max_cols=3):
    """Create a concise summary plot for a DataFrame's numeric columns.
    - Select up to `max_cols` numeric columns ranked by non-null count.
    - Save a single combined histogram figure and a boxplot (one file each).
    - Run scatter matrix only when there are 2-4 selected columns with enough rows.
    """
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return

    # choose top columns by number of non-null values
    counts = num.notna().sum().sort_values(ascending=False)
    selected = list(counts.index[:max_cols])

    # prepare histograms in a single figure
    n_plots = len(selected)
    if n_plots > 0:
        fig, axes = plt.subplots(n_plots, 1, figsize=(6, 3 * n_plots), squeeze=False)
        for i, col in enumerate(selected):
            series = num[col].dropna()
            ax = axes[i, 0]
            if series.size == 0:
                ax.text(0.5, 0.5, 'no valid data', ha='center', va='center')
                ax.set_axis_off()
                continue
            ax.hist(series, bins=20, color="#2b8cbe", alpha=0.85, edgecolor='black')
            ax.set_title(f"{col} — histogram")
            ax.set_xlabel(col)
            ax.set_ylabel("count")
        fig.suptitle(f"{name} — top {n_plots} numeric columns")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(OUT_DIR, f"{name}_hist_summary.png"))
        plt.close(fig)

    # boxplot for the same selected columns
    box_data = [num[c].dropna() for c in selected if num[c].dropna().size > 0]
    box_labels = [c for c in selected if num[c].dropna().size > 0]
    if box_data:
        fig, ax = plt.subplots(figsize=(min(12, 3 * len(box_data)), 5))
        try:
            ax.boxplot(box_data, tick_labels=box_labels, vert=True)
        except TypeError:
            ax.boxplot(box_data, labels=box_labels, vert=True)
        ax.set_title(f"{name} — boxplot")
        ax.set_ylabel("value")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{name}_boxplot.png"))
        plt.close(fig)

    # scatter matrix only for small selections with enough rows
    if 2 <= len(selected) <= 4:
        df_sel = num[selected].dropna()
        if df_sel.shape[0] >= 3:
            try:
                axs = scatter_matrix(df_sel, figsize=(6 + 2 * len(selected), 6), diagonal="kde")
                fig = axs[0, 0].figure
                fig.suptitle(f"{name} — scatter matrix")
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(os.path.join(OUT_DIR, f"{name}_scattermatrix.png"))
                plt.close(fig)
            except Exception as e:
                print(f"Warning: could not create scatter matrix for {name}: {e}")


def try_merge(dfs_map):
    # find common columns across the two files (any sheet)
    # flatten to single DataFrame per file by concatenating sheets (if multiple)
    flat = {}
    for file_key, sheets in dfs_map.items():
        dfs = []
        for name, df in sheets.items():
            tmp = df.copy()
            tmp["_sheet"] = name
            dfs.append(tmp)
        flat[file_key] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    keys = []
    common = set(flat[EXCEL_FILES[0]].columns).intersection(flat[EXCEL_FILES[1]].columns)
    # ignore the sentinel _sheet column
    common.discard("_sheet")
    if common:
        keys = list(common)
        # use the first shared column as merge key
        key = keys[0]
        merged = pd.merge(flat[EXCEL_FILES[0]], flat[EXCEL_FILES[1]], on=key, suffixes=("_a", "_b"))
        out_path = "merged_on_" + key + ".xlsx"
        merged.to_excel(out_path, index=False)
        print(f"Merged on '{key}' -> saved to {out_path} (shape {merged.shape})")
    else:
        print("No common columns found to merge the two files automatically.")


def process_rubberstopper_df(df, out_prefix):
    """Extract density and uncertainty from a rubberstopper DataFrame, convert units if needed,
    compute statistics, save a CSV summary and a plot with density + relative uncertainty.
    """
    import re
    from math import isnan

    # helper to find a column by keywords (case-insensitive)
    def find_col_by_keywords(cols, keywords):
        cols_lc = [str(c).lower() for c in cols]
        for kw in keywords:
            for i, c in enumerate(cols_lc):
                if kw in c:
                    return cols[i]
        return None

    cols = list(df.columns)
    # candidate keywords
    density_kw = ["density", "g/cm", "g cm", "gcm", "g cm3", "g/cm3", "g cm^-3", "kg/m3", "kg/m^3"]
    uncert_kw = ["uncert", "unc", "±", "error", "err", "%"]

    den_col = find_col_by_keywords(cols, density_kw)
    unc_col = find_col_by_keywords(cols, uncert_kw)

    # fallback: first numeric column
    if den_col is None:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        den_col = numcols[0] if numcols else None

    # try to extract numeric density values even if stored as strings with ±
    density_raw = None
    if den_col is not None:
        density_raw = df[den_col].astype(str)
        # try extracting the left side before ± or parentheses
        dens_extracted = density_raw.str.extract(r"([+-]?[0-9]*\.?[0-9]+)")
        dens_series = pd.to_numeric(dens_extracted[0], errors="coerce")
    else:
        dens_series = pd.Series(dtype=float)

    # uncertainty handling
    unc_series = None
    if unc_col is not None:
        unc_series = pd.to_numeric(df[unc_col], errors="coerce")
    else:
        # try to extract ± value from the density raw strings
        if density_raw is not None:
            plusminus = density_raw.str.extract(r"±\s*([0-9]*\.?[0-9]+)")
            if plusminus.notna().any().any():
                unc_series = pd.to_numeric(plusminus[0], errors="coerce")

    # If unc_series still None, look for percent values in any column
    if unc_series is None:
        for c in cols:
            s = df[c].astype(str)
            pct = s.str.contains('%')
            if pct.any():
                # extract percent number
                extracted = s.str.extract(r"([0-9]*\.?[0-9]+)\s*%")
                unc_series = pd.to_numeric(extracted[0], errors="coerce")
                # mark that these are percentages; will convert later
                unc_is_percent = True
                break
        else:
            unc_is_percent = False
    else:
        unc_is_percent = False

    # align series lengths and convert to numeric
    dens = dens_series if isinstance(dens_series, pd.Series) else pd.to_numeric(dens_series, errors='coerce')
    # if all densities are NaN, try pulling any numeric columns
    if dens.dropna().empty:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numcols:
            dens = pd.to_numeric(df[numcols[0]], errors='coerce')

    if unc_series is None:
        # no explicit uncertainty found; estimate using instrument precision heuristics
        # use 1% as a conservative relative uncertainty if nothing provided
        unc = pd.Series(np.nan, index=dens.index)
        unc.fillna((0.01 * dens).where(dens.notna()), inplace=True)
    else:
        unc = pd.to_numeric(unc_series, errors='coerce')
        if unc_is_percent:
            # convert percent to absolute uncertainty
            unc = (unc / 100.0) * dens
        # if unc values look too large compared to densities (e.g., > 10x), assume percent
        mask_large = (unc.notna()) & (dens.notna()) & (unc > 10 * dens.abs())
        if mask_large.any():
            unc.loc[mask_large] = (unc.loc[mask_large] / 100.0) * dens.loc[mask_large]

    # Unit conversion: detect if densities are in kg/m3 (typical values ~1000) and convert to g/cm3
    dens_mean = dens.dropna().mean() if dens.notna().any() else float('nan')
    converted_from = None
    if not isnan(dens_mean) and dens_mean > 50:  # heuristic: >50 => likely kg/m3
        dens = dens / 1000.0
        unc = unc / 1000.0
        converted_from = 'kg/m3->g/cm3'

    # final cleaned series
    valid_mask = dens.notna()
    dens_clean = dens[valid_mask]
    unc_clean = unc[valid_mask]

    n = dens_clean.size
    if n == 0:
        print(f"No numeric density data found for {out_prefix}")
        return

    mean = dens_clean.mean()
    std_sample = dens_clean.std(ddof=1)
    sem = std_sample / np.sqrt(n) if n > 0 else float('nan')

    # relative uncertainty (absolute unc / density)
    rel_unc = (unc_clean / dens_clean).replace([np.inf, -np.inf], np.nan)

    # -- additional diagnostics and outputs --
    # assemble parsed values into a DataFrame and save
    parsed_df = pd.DataFrame({
        'density_g_per_cm3': dens_clean.values,
        'abs_unc_g_per_cm3': unc_clean.values,
        'rel_unc_frac': rel_unc.values,
        'rel_unc_pct': (rel_unc * 100.0).values
    })
    parsed_path = os.path.join(OUT_DIR, f"{out_prefix}_rubberstopper_parsed_values.csv")
    parsed_df.to_csv(parsed_path, index=False)

    # compute additional statistics
    median = dens_clean.median()
    q1 = dens_clean.quantile(0.25)
    q3 = dens_clean.quantile(0.75)
    iqr = q3 - q1
    data_min = dens_clean.min()
    data_max = dens_clean.max()
    missing_count = int((~valid_mask).sum())

    # outliers by 1.5*IQR rule
    outlier_mask = (dens_clean < (q1 - 1.5 * iqr)) | (dens_clean > (q3 + 1.5 * iqr))
    outlier_count = int(outlier_mask.sum())
    outliers = dens_clean[outlier_mask]

    # save summary (expanded)
    summary = pd.DataFrame({
        'metric': [
            'n', 'mean_g_per_cm3', 'std_sample_ddof1_g_per_cm3', 'sem_g_per_cm3',
            'median_g_per_cm3', 'iqr_g_per_cm3', 'min_g_per_cm3', 'max_g_per_cm3',
            'missing_count', 'outlier_count', 'converted_from', 'density_column', 'uncertainty_column'
        ],
        'value': [
            n, mean, std_sample, sem,
            median, iqr, data_min, data_max,
            missing_count, outlier_count, converted_from or '', den_col or '', unc_col or ''
        ]
    })
    summary_path = os.path.join(OUT_DIR, f"{out_prefix}_rubberstopper_summary.csv")
    summary.to_csv(summary_path, index=False)

    # print parsed values and stats to console for quick inspection
    print(f"Parsed densities saved to: {parsed_path}")
    print(parsed_df.head(10).to_string(index=False))
    print(f"Statistics: n={n}, mean={mean:.5f} g/cm3, median={median:.5f} g/cm3, std(ddof=1)={std_sample:.5f} g/cm3, SEM={sem:.5f} g/cm3")
    print(f"IQR={iqr:.5f} g/cm3, min={data_min:.5f}, max={data_max:.5f}, missing_count={missing_count}, outlier_count={outlier_count}")
    if outlier_count > 0:
        print("Outliers:\n", outliers.to_string())

    # create a plot: density (with absolute uncertainty) and relative uncertainty (%) on twin axis
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dens_clean))
    ax1.errorbar(x, dens_clean.values, yerr=unc_clean.values if unc_clean.notna().any() else None,
                 fmt='o', color='#2b8cbe', ecolor='gray', capsize=3, label='density (g/cm³)')
    ax1.set_xlabel('sample index')
    ax1.set_ylabel('density (g/cm³)')
    ax1.set_title(f'{out_prefix} — Rubber stopper density and relative uncertainty')

    # Draw mean and ±1 standard deviation lines and shaded region
    try:
        mean_val = float(mean)
        std_val = float(std_sample)
        ax1.axhline(mean_val, color='green', linestyle='--', linewidth=1.25, label=f'Mean = {mean_val:.4f} g/cm³')
        ax1.axhline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, label=f'Mean + 1σ = {mean_val + std_val:.4f}')
        ax1.axhline(mean_val - std_val, color='orange', linestyle=':', linewidth=1, label=f'Mean - 1σ = {mean_val - std_val:.4f}')
        # shaded area between mean±std
        xmin, xmax = -0.5, max(len(dens_clean) - 0.5, 0.5)
        ax1.fill_between([xmin, xmax], [mean_val - std_val, mean_val - std_val], [mean_val + std_val, mean_val + std_val],
                         color='orange', alpha=0.08, edgecolor='none')
        ax1.set_xlim(xmin, xmax)
    except Exception:
        # if something goes wrong converting values, skip drawing
        pass

    ax2 = ax1.twinx()
    rel_pct = (rel_unc * 100.0)
    ax2.plot(x, rel_pct.values, 'r.-', label='relative uncertainty (%)')
    ax2.set_ylabel('relative uncertainty (%)')

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"{out_prefix}_rubberstopper_density_uncertainty.png")
    fig.savefig(plot_path)
    plt.close(fig)

    # print summary to console
    print(f"Rubber stopper ({out_prefix}) — n={n}, mean={mean:.5f} g/cm3, std(ddof=1)={std_sample:.5f} g/cm3, SEM={sem:.5f} g/cm3")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved plot -> {plot_path}")


def main():
    dfs_map = {}
    for f in EXCEL_FILES:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return
        sheets = load_all_sheets(f)
        cleaned = {}
        for sname, df in sheets.items():
            dfc = clean_df(df)
            key = os.path.splitext(os.path.basename(f))[0] + "__" + sname
            cleaned[key] = dfc
            summarize(dfc, key)
            plot_numeric(dfc, key)
            # If this is the rubberstopper workbook, produce a dedicated summary/plot
            if "rubberstopper" in os.path.basename(f).lower():
                try:
                    process_rubberstopper_df(dfc, key)
                except Exception as e:
                    print(f"Warning: rubberstopper processing failed for {key}: {e}")
        dfs_map[f] = sheets

    # attempt to merge the two files (by any shared column)
    try_merge(dfs_map)

    # Special processing for rubberstopper file: extract density and uncertainty, compute stats, plot
    rubber_files = [f for f in EXCEL_FILES if "rubberstopper" in f.lower()]
    if rubber_files:
        for rf in rubber_files:
            for sname, df in dfs_map[rf].items():
                dfc = clean_df(df)
                key = os.path.splitext(os.path.basename(rf))[0] + "__" + sname
                process_rubberstopper_df(dfc, key)

    print(f"All plots saved to '{OUT_DIR}'.")


if __name__ == "__main__":
    main()