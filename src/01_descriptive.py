"""
01_descriptive.py
─────────────────
Phase 1: Weighted Descriptive Statistics
BLS Consumer Expenditure Survey — FMLI 2025 Q1 (fmli251.csv)

Each chart is saved as its own image file to outputs/figures/.
No multi-panel composite figures — every output is standalone.

Output files (19 total):
  01a_income_histogram.png
  01a_median_income_by_quintile.png
  01a_median_income_by_region.png
  01b_mean_spending_by_quintile.png
  01b_expense_ratio_by_quintile.png
  01b_housing_burden_by_quintile.png
  01c_spending_heatmap.png
  01d_sex.png
  01d_age.png
  01d_household_type.png
  01d_education_level.png
  01d_marital_status.png
  01d_housing_tenure.png
  01d_region.png
  01d_earner_count.png
  01e_expense_ratio.png
  01e_lifestyle_ratio.png
  01e_housing_ratio.png
  01e_retirement_ratio.png

Design decisions:
  - No gridlines — bar labels carry all numeric information
  - Bar charts suppress y-axis tick labels for the same reason
  - All statistics use FINLWT21 survey weights
  - Geography-suppressed rows (FLAG_REGION_UNKNOWN=1) excluded from
    region charts only; retained in all other analyses
  - Q1 ratio charts capped at y=1.5 with truncation annotation

Run from project root:
    python src/01_descriptive.py
"""

import os
import sys

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from data_prep import (
    prepare_data,
    get_analysis_sample,
    get_share_col,
    get_quintile_labels,
    weighted_quantile,
    weighted_gini,
    SPEND_CATEGORIES,
    QUINTILE_LABELS,
    QUINTILE_LABELS_SHORT,
    REGION_MAP,
    EDUC_MAP,
    EDUC_ORDER,
    MARITAL_MAP,
    FAM_TYPE_MAP,
    CUTENURE_MAP,
    URBAN_MAP,
    SEX_MAP,
)


# ══════════════════════════════════════════════════════════════════════════════
# THEME AND PALETTE
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "outputs/figures"

PALETTE = {
    "blue":       "#2E86AB",
    "teal":       "#1B998B",
    "orange":     "#F46036",
    "red":        "#E84855",
    "purple":     "#7B2D8B",
    "gray":       "#6C757D",
    "light_gray": "#D0D3D4",
    "dark":       "#212529",
    "bg":         "#FAFAFA",
}

QUINTILE_COLORS = ["#C9E8F5", "#2E86AB", "#1B998B", "#F46036", "#E84855"]

REGION_COLORS = {
    "Northeast": PALETTE["blue"],
    "Midwest":   PALETTE["teal"],
    "South":     PALETTE["orange"],
    "West":      PALETTE["red"],
}


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    PALETTE["light_gray"],
        "axes.labelcolor":   PALETTE["dark"],
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.grid":         False,       # No gridlines — labels carry the data
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,       # Remove left spine for bar charts
        "xtick.color":       PALETTE["gray"],
        "ytick.color":       PALETTE["gray"],
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   9,
        "legend.framealpha": 0.9,
        "font.family":       "sans-serif",
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": PALETTE["bg"],
    })


apply_theme()


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna()
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    return weighted_quantile(values, weights, 0.5)


def weighted_share(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna()
    return float(np.average(series[mask], weights=weights[mask]))


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_dollar(v):
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


def fmt_pct(v):
    return f"{v:.1f}%"


def fmt_ratio(v):
    return f"{v:.2f}"


def save_fig(fig: plt.Figure, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def quintile_order(df: pd.DataFrame) -> list:
    present = df["INCOME_QUINTILE"].dropna().unique()
    return [q for q in QUINTILE_LABELS if q in present]


def q_short_labels(q_order: list) -> list:
    return [QUINTILE_LABELS_SHORT[QUINTILE_LABELS.index(q)] for q in q_order]


def clean_bar_axes(ax):
    """
    Remove y-axis ticks and spine from a bar chart.
    Bar labels carry all numeric information so axis ticks are redundant.
    """
    ax.set_yticks([])
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)


def source_note(ax, extra: str = ""):
    note = "BLS Consumer Expenditure Survey, Q1 2025. Survey-weighted estimates."
    if extra:
        note += f"  {extra}"
    ax.annotate(
        note,
        xy=(0, -0.12), xycoords="axes fraction",
        fontsize=7.5, color=PALETTE["gray"], ha="left",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1a — INCOME DISTRIBUTION (3 separate charts)
# ══════════════════════════════════════════════════════════════════════════════

def chart_1a_income_histogram(df: pd.DataFrame):
    """Weighted histogram of annual household income with quintile boundaries."""
    print("\n  [1a-1] Income histogram")
    pos    = df[df["FINCBTAX"] > 0]
    p99    = pos["FINCBTAX"].quantile(0.99)
    display = pos["FINCBTAX"].clip(upper=p99)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(display, bins=60, weights=pos["FINLWT21"],
            color=PALETTE["blue"], alpha=0.82, edgecolor="white", linewidth=0.3)

    # Quintile boundary lines
    breaks = df.attrs.get("quintile_breaks", [])
    for i, brk in enumerate(breaks[1:-1], start=1):
        if brk <= p99:
            ax.axvline(brk, color=PALETTE["orange"], lw=1.3, ls="--", alpha=0.8)
            ax.text(brk, ax.get_ylim()[1] * 0.97,
                    f" Q{i}|Q{i+1}\n {fmt_dollar(brk)}",
                    fontsize=7.5, color=PALETTE["orange"], va="top")

    w_med  = weighted_median(pos["FINCBTAX"], pos["FINLWT21"])
    w_mean = weighted_mean(pos["FINCBTAX"],   pos["FINLWT21"])
    gini   = weighted_gini(pos["FINCBTAX"],   pos["FINLWT21"])

    ax.axvline(w_med,  color=PALETTE["teal"], lw=2, ls="-",
               label=f"Wtd. Median: {fmt_dollar(w_med)}")
    ax.axvline(w_mean, color=PALETTE["red"],  lw=2, ls=":",
               label=f"Wtd. Mean: {fmt_dollar(w_mean)}")

    ax.legend(fontsize=9, frameon=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: fmt_dollar(x)))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )
    ax.set_xlabel("Annual Income Before Taxes")
    ax.set_ylabel("Weighted Household Count")
    ax.set_title("U.S. Household Income Distribution", loc="left")

    ax.text(0.97, 0.95,
            f"Gini: {gini:.3f}\nClipped at 99th pct ({fmt_dollar(p99)})",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=PALETTE["light_gray"], alpha=0.9))
    source_note(ax)
    fig.tight_layout()
    save_fig(fig, "01a_income_histogram.png")


def chart_1a_median_by_quintile(df: pd.DataFrame):
    """Weighted median income by income quintile."""
    print("  [1a-2] Median income by quintile")
    pos     = df[df["FINCBTAX"] > 0]
    q_order = quintile_order(pos)
    labels  = q_short_labels(q_order)

    medians = [weighted_median(pos[pos["INCOME_QUINTILE"] == q]["FINCBTAX"],
                               pos[pos["INCOME_QUINTILE"] == q]["FINLWT21"])
               for q in q_order]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(range(len(q_order)), medians,
                  color=QUINTILE_COLORS[:len(q_order)],
                  alpha=0.87, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(q_order)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Weighted Median Income by Income Quintile", loc="left")
    ax.annotate("Quintile boundaries are data-driven weighted percentiles of FINCBTAX.",
                xy=(0.5, 1.01), xycoords="axes fraction",
                fontsize=8.5, color=PALETTE["gray"], ha="center")
    clean_bar_axes(ax)

    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(medians) * 0.01,
                fmt_dollar(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    source_note(ax)
    fig.tight_layout()
    save_fig(fig, "01a_median_income_by_quintile.png")


def chart_1a_median_by_region(df: pd.DataFrame):
    """Weighted median income by Census region."""
    print("  [1a-3] Median income by region")
    region_df    = df[(df["FINCBTAX"] > 0) & (df["FLAG_REGION_UNKNOWN"] == 0)]
    n_suppressed = df["FLAG_REGION_UNKNOWN"].sum()

    region_data = {label: weighted_median(
                        region_df[region_df["REGION"] == code]["FINCBTAX"],
                        region_df[region_df["REGION"] == code]["FINLWT21"])
                   for code, label in REGION_MAP.items()
                   if (region_df["REGION"] == code).any()}

    fig, ax = plt.subplots(figsize=(9, 5))
    labels  = list(region_data.keys())
    vals    = list(region_data.values())
    colors  = [REGION_COLORS.get(r, PALETTE["gray"]) for r in labels]

    bars = ax.barh(labels, vals, color=colors, alpha=0.87,
                   edgecolor="white", height=0.5)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Weighted Median Income by Census Region", loc="left")

    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                fmt_dollar(val), va="center", fontsize=10, fontweight="bold")

    source_note(ax, f"{n_suppressed} households with suppressed geography excluded.")
    fig.tight_layout()
    save_fig(fig, "01a_median_income_by_region.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1b — SPENDING PROFILE (3 separate charts)
# ══════════════════════════════════════════════════════════════════════════════

def chart_1b_mean_spending(df: pd.DataFrame):
    """Weighted mean quarterly spending by income quintile."""
    print("\n  [1b-1] Mean spending by quintile")
    q_order = quintile_order(df)
    labels  = q_short_labels(q_order)

    means = [weighted_mean(df[df["INCOME_QUINTILE"] == q]["TOTEXP_AVG_Q"],
                           df[df["INCOME_QUINTILE"] == q]["FINLWT21"])
             for q in q_order]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(range(len(q_order)), means,
                  color=QUINTILE_COLORS[:len(q_order)],
                  alpha=0.87, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(q_order)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Weighted Mean Quarterly Spending by Income Quintile", loc="left")
    ax.annotate("Dollar spending rises with income — see Expense Ratio chart for proportional view.",
                xy=(0.5, 1.01), xycoords="axes fraction",
                fontsize=8.5, color=PALETTE["gray"], ha="center")
    clean_bar_axes(ax)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(means) * 0.01,
                fmt_dollar(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    source_note(ax)
    fig.tight_layout()
    save_fig(fig, "01b_mean_spending_by_quintile.png")


def chart_1b_expense_ratio(df: pd.DataFrame):
    """
    Expense ratio (annual spend / annual income) by income quintile.
    Box plots with weighted mean diamonds. The 1.0 line marks spending = income.
    This is the primary visual for the Income Fallacy hypothesis.
    """
    print("  [1b-2] Expense ratio by quintile")
    q_order = quintile_order(df)
    labels  = q_short_labels(q_order)

    groups    = []
    wtd_means = []
    for q in q_order:
        sub     = df[(df["INCOME_QUINTILE"] == q) & df["EXPENSE_RATIO"].notna()]
        clipped = sub["EXPENSE_RATIO"].clip(upper=3.0)
        groups.append(clipped.values)
        wtd_means.append(weighted_mean(clipped, sub["FINLWT21"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2.5},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.2})

    for patch, color in zip(bp["boxes"], QUINTILE_COLORS[:len(q_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.87)

    ax.scatter(range(1, len(q_order) + 1), wtd_means,
               marker="D", color=PALETTE["dark"], s=45, zorder=5,
               label="Wtd. mean")
    ax.axhline(1.0, color=PALETTE["red"], lw=1.8, ls="--", alpha=0.75,
               label="Spending = Income")

    for i, val in enumerate(wtd_means):
        ax.text(i + 1, val + 0.03, fmt_ratio(val),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=PALETTE["dark"])

    ax.set_xticks(range(1, len(q_order) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Annual Spend / Annual Income")
    ax.set_title("Expense Ratio by Income Quintile", loc="left")
    ax.annotate("Clipped at 3.0 for display. Box plots show unweighted within-quintile distributions.",
                xy=(0.5, 1.01), xycoords="axes fraction",
                fontsize=8.5, color=PALETTE["gray"], ha="center")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["left"].set_visible(True)
    source_note(ax)
    fig.tight_layout()
    save_fig(fig, "01b_expense_ratio_by_quintile.png")


def chart_1b_housing_burden(df: pd.DataFrame):
    """Housing cost burden rate (housing > 30% of income) by income quintile."""
    print("  [1b-3] Housing burden by quintile")
    q_order = quintile_order(df)
    labels  = q_short_labels(q_order)

    rates = [weighted_share(
                df[(df["INCOME_QUINTILE"] == q) & df["HOUSING_BURDENED"].notna()]["HOUSING_BURDENED"],
                df[(df["INCOME_QUINTILE"] == q) & df["HOUSING_BURDENED"].notna()]["FINLWT21"]) * 100
             for q in q_order]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(range(len(q_order)), rates,
                  color=QUINTILE_COLORS[:len(q_order)],
                  alpha=0.87, edgecolor="white", width=0.6)
    ax.axhline(30, color=PALETTE["gray"], lw=1.2, ls=":",
               label="30% national reference")
    ax.set_xticks(range(len(q_order)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Housing Cost Burden Rate by Income Quintile", loc="left")
    ax.annotate("Households spending more than 30% of income on housing (HUD standard threshold).",
                xy=(0.5, 1.01), xycoords="axes fraction",
                fontsize=8.5, color=PALETTE["gray"], ha="center")
    ax.legend(fontsize=9)
    clean_bar_axes(ax)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rates) * 0.01,
                fmt_pct(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    source_note(ax)
    fig.tight_layout()
    save_fig(fig, "01b_housing_burden_by_quintile.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1c — SPENDING COMPOSITION HEATMAP (1 chart)
# ══════════════════════════════════════════════════════════════════════════════

def chart_1c_spending_heatmap(df: pd.DataFrame):
    """
    Heatmap of spending category shares by income quintile.
    Rows = categories, columns = quintiles.
    Cell values = weighted mean share of total quarterly spend.
    """
    print("\n  [1c] Spending composition heatmap")
    q_order    = quintile_order(df)
    col_labels = q_short_labels(q_order)
    bases      = list(SPEND_CATEGORIES.keys())
    row_labels = list(SPEND_CATEGORIES.values())

    matrix = np.full((len(bases), len(q_order)), np.nan)
    for j, q in enumerate(q_order):
        sub = df[df["INCOME_QUINTILE"] == q]
        for i, base in enumerate(bases):
            share_col = get_share_col(base)
            if share_col not in sub.columns:
                continue
            valid = sub[share_col].notna()
            if valid.sum() > 0:
                matrix[i, j] = weighted_mean(sub.loc[valid, share_col],
                                              sub.loc[valid, "FINLWT21"])

    heatmap_df = pd.DataFrame(matrix * 100, index=row_labels, columns=col_labels)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(heatmap_df, ax=ax,
                annot=True, fmt=".1f",
                annot_kws={"size": 10, "weight": "bold"},
                cmap="RdYlBu_r",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Wtd. Mean Share of Total Spending (%)",
                          "shrink": 0.75},
                vmin=0, vmax=np.nanmax(matrix * 100) * 1.05)

    ax.set_xlabel("Income Quintile", fontsize=11, labelpad=10)
    ax.set_ylabel("Spending Category", fontsize=11, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0,  fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=10)
    ax.set_title("How American Households Allocate Spending by Income Quintile",
                 loc="left", fontsize=13, pad=16)
    ax.annotate(
        "Each cell = weighted mean share of total quarterly spending. "
        "Darker red = larger share.\n"
        "BLS Consumer Expenditure Survey, Q1 2025. Survey-weighted estimates.",
        xy=(0, 1.02), xycoords="axes fraction", fontsize=8.5,
        color=PALETTE["gray"], ha="left")

    # Engel's Law annotation
    food_idx = row_labels.index("Food at Home")
    ax.annotate("Engel's Law: food share\ndeclines as income rises",
                xy=(len(q_order) - 0.3, food_idx + 0.5), xycoords="data",
                xytext=(len(q_order) + 0.15, food_idx + 0.5), textcoords="data",
                fontsize=9, color=PALETTE["teal"], va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE["teal"], lw=1.2))

    fig.tight_layout()
    save_fig(fig, "01c_spending_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1d — DEMOGRAPHIC PROFILE (6 separate charts)
# ══════════════════════════════════════════════════════════════════════════════

def _demographic_bar(df: pd.DataFrame, raw_col: str, decode_map,
                     title: str, filename: str, color: str,
                     note: str, region_filter: bool = False,
                     n_suppressed: int = 0,
                     sort_order: list = None,
                     is_histogram: bool = False,
                     hist_bins: int = 15):
    """
    Shared helper for single demographic chart.

    Parameters
    ----------
    sort_order : list, optional
        If provided, categories are displayed in this order instead of
        descending by weighted count. Use for ordinal variables like
        education level or earner count where display order has meaning.
    is_histogram : bool
        If True, treats raw_col as a continuous variable and plots a
        weighted histogram instead of a bar chart. Used for age.
    hist_bins : int
        Number of bins for histogram mode.
    """
    panel_df = df[df["FLAG_REGION_UNKNOWN"] == 0] if region_filter else df

    if raw_col not in panel_df.columns:
        return

    # ── Histogram mode for continuous variables (e.g., age) ────────────
    if is_histogram:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = panel_df[raw_col].dropna()
        wts  = panel_df.loc[vals.index, "FINLWT21"]
        ax.hist(vals, bins=hist_bins, weights=wts,
                color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
        wm  = float(np.average(vals, weights=wts))
        wmd = weighted_quantile(vals, wts, 0.5)
        ax.axvline(wmd, color=PALETTE["teal"], lw=2, ls="-",
                   label=f"Wtd. Median: {wmd:.0f}")
        ax.axvline(wm,  color=PALETTE["red"],  lw=2, ls=":",
                   label=f"Wtd. Mean: {wm:.0f}")
        ax.legend(fontsize=9)
        ax.set_xlabel(raw_col.replace("_", " ").title())
        ax.set_ylabel("Weighted Household Count")
        ax.set_title(title, loc="left")
        source_note(ax, f"Phase 2 covariate: {note}")
        fig.tight_layout()
        save_fig(fig, filename)
        return

    # ── Bar chart mode ─────────────────────────────────────────────────
    labels_series = (panel_df[raw_col].map(decode_map).fillna("Unknown")
                     if decode_map else panel_df[raw_col].astype(str))

    wt_counts = (panel_df.assign(_label=labels_series)
                 .groupby("_label", observed=True)["FINLWT21"].sum())

    # Apply sort order if specified; otherwise sort by descending count
    if sort_order is not None:
        wt_counts = wt_counts.reindex([s for s in sort_order
                                         if s in wt_counts.index])
    else:
        wt_counts = wt_counts.sort_values(ascending=False)

    total_wt = wt_counts.sum()
    pcts     = wt_counts / total_wt * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(pcts)), pcts.values,
                  color=color, alpha=0.82, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels(pcts.index, rotation=25, ha="right", fontsize=9)
    ax.set_title(title, loc="left")
    clean_bar_axes(ax)

    for bar, val in zip(bars, pcts.values):
        if val > 1.5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(pcts.values) * 0.01,
                    fmt_pct(val),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    extra = f"{n_suppressed} geography-suppressed rows excluded." if region_filter else ""
    source_note(ax, f"Phase 2 covariate: {note}  {extra}")
    fig.tight_layout()
    save_fig(fig, filename)


def charts_1d_demographics(df: pd.DataFrame):
    """Eight individual demographic charts (6 bar + 1 histogram + 1 sex)."""
    print("\n  [1d] Demographic profile (8 charts)")
    n_sup = int(df["FLAG_REGION_UNKNOWN"].sum())

    # Education labels in ascending order of attainment
    educ_sort_order = [EDUC_MAP[code] for code in EDUC_ORDER
                       if code in EDUC_MAP]

    # Earner count in numeric order
    earner_sort_order = [str(i) for i in range(0, 10)]

    # (raw_col, decode_map, title, filename, color, note, region_filter,
    #  sort_order, is_histogram, hist_bins)
    panels = [
        ("SEX_REF",     SEX_MAP,      "Sex of Reference Person",
         "01d_sex.png",             PALETTE["blue"],   "SEX_REF",
         False, None, False, None),
        ("AGE_REF",     None,         "Age of Reference Person",
         "01d_age.png",             PALETTE["blue"],   "AGE_REF",
         False, None, True, 25),
        ("FAM_TYPE",    FAM_TYPE_MAP, "Household Type",
         "01d_household_type.png",  PALETTE["blue"],   "FAM_TYPE_BROAD",
         False, None, False, None),
        ("EDUC_REF",    EDUC_MAP,     "Education Level (Reference Person)",
         "01d_education_level.png", PALETTE["teal"],   "EDUC_REF_ORD",
         False, educ_sort_order, False, None),
        ("MARITAL1",    MARITAL_MAP,  "Marital Status (Reference Person)",
         "01d_marital_status.png",  PALETTE["orange"], "MARITAL1",
         False, None, False, None),
        ("CUTENURE",    CUTENURE_MAP, "Housing Tenure",
         "01d_housing_tenure.png",  PALETTE["red"],    "HOMEOWNER flag",
         False, None, False, None),
        ("region_label",None,         "Census Region",
         "01d_region.png",          PALETTE["purple"], "REGION / BLS_URBN",
         True, None, False, None),
        ("NO_EARNR",    None,         "Number of Earners in Household",
         "01d_earner_count.png",    PALETTE["gray"],   "NO_EARNR",
         False, earner_sort_order, False, None),
    ]

    for (raw_col, decode_map, title, fname, color, note,
         rfilt, sorder, is_hist, hbins) in panels:
        _demographic_bar(df, raw_col, decode_map, title, fname,
                         color, note, rfilt, n_sup,
                         sort_order=sorder,
                         is_histogram=is_hist,
                         hist_bins=hbins if hbins else 15)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1e — FOUR ANALYTICAL RATIOS (4 separate charts)
# ══════════════════════════════════════════════════════════════════════════════

def _ratio_bar(df: pd.DataFrame, col: str, title: str, subtitle: str,
               filename: str, color: str, ref_line,
               ref_label: str, good_direction: str, y_cap: float = 1.0):
    """
    Shared helper for a single ratio bar chart by income quintile.
    Weighted means shown as bars, weighted IQR as error bars.
    Y-axis capped at y_cap — Q1 extreme values from near-zero income
    denominators are annotated rather than displayed.
    """
    if col not in df.columns:
        return

    q_order = quintile_order(df)
    labels  = q_short_labels(q_order)

    means, p25s, p75s = [], [], []
    for q in q_order:
        sub = df[(df["INCOME_QUINTILE"] == q) & df[col].notna()]
        means.append(weighted_mean(sub[col], sub["FINLWT21"]))
        p25s.append(weighted_quantile(sub[col], sub["FINLWT21"], 0.25))
        p75s.append(weighted_quantile(sub[col], sub["FINLWT21"], 0.75))

    means = np.array(means)
    p25s  = np.array(p25s)
    p75s  = np.array(p75s)
    x     = np.arange(len(q_order))

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(x, np.minimum(means, y_cap),
                  color=color, alpha=0.82, edgecolor="white", width=0.6, zorder=2)

    ax.errorbar(x, np.minimum(means, y_cap),
                yerr=[np.maximum(0, np.minimum(means, y_cap) - p25s),
                      np.maximum(0, p75s - np.minimum(means, y_cap))],
                fmt="none", color=PALETTE["dark"],
                capsize=5, capthick=1.3, linewidth=1.3, zorder=3,
                label="Wtd. IQR (P25–P75)")

    if ref_line is not None:
        ax.axhline(ref_line, color=PALETTE["gray"], lw=1.5, ls="--",
                   alpha=0.8, label=ref_label)

    # Value labels — show actual value even if bar is truncated
    for i, val in enumerate(means):
        display_val = min(val, y_cap)
        label_txt   = fmt_ratio(val) + (" ▲" if val > y_cap else "")
        ax.text(i, display_val + y_cap * 0.02, label_txt,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color=PALETTE["red"] if val > y_cap else PALETTE["dark"])

    ax.set_ylim(0, y_cap * 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, loc="left")
    ax.annotate(subtitle, xy=(0.5, 1.01), xycoords="axes fraction",
                fontsize=8.5, color=PALETTE["gray"], ha="center")

    direction_txt = ("Lower = better financial discipline"
                     if good_direction == "lower"
                     else "Higher = stronger savings behavior")
    ax.text(0.97, 0.97, direction_txt,
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            color=PALETTE["gray"])

    # Truncation note if any bar hit the cap
    n_trunc = (means > y_cap).sum()
    if n_trunc:
        ax.text(0.03, 0.97,
                f"{n_trunc} quintile(s) truncated at {y_cap}\n"
                "(near-zero income denominator — see report)",
                transform=ax.transAxes, fontsize=8, ha="left", va="top",
                color=PALETTE["red"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
                          edgecolor=PALETTE["orange"], alpha=0.9))

    legend_items = ["Wtd. IQR (P25–P75)"]
    if ref_line is not None:
        legend_items.append(ref_label)
    ax.legend(fontsize=9, loc="upper left" if n_trunc == 0 else "lower right")
    ax.spines["left"].set_visible(True)
    ax.set_ylabel("Ratio (annual $ / annual income)")
    source_note(ax, "▲ = value exceeds chart cap; actual value shown." if n_trunc else "")
    fig.tight_layout()
    save_fig(fig, filename)


def charts_1e_four_ratios(df: pd.DataFrame):
    """Four individual ratio charts — one per analytical ratio."""
    print("\n  [1e] Four analytical ratios (4 charts)")

    ratio_configs = [
        {
            "col":      "EXPENSE_RATIO",
            "title":    "Total Expense Ratio by Income Quintile",
            "subtitle": "Annual total spending / annual income  —  broadest behavioral measure",
            "filename": "01e_expense_ratio.png",
            "color":    PALETTE["blue"],
            "ref_line": 1.0,
            "ref_label":"Spending = Income",
            "good_direction": "lower",
        },
        {
            "col":      "LIFESTYLE_RATIO",
            "title":    "Lifestyle Ratio by Income Quintile",
            "subtitle": "Discretionary spend / annual income  —  entertainment, dining out, apparel, alcohol, tobacco, cash contributions",
            "filename": "01e_lifestyle_ratio.png",
            "color":    PALETTE["orange"],
            "ref_line": None,
            "ref_label": None,
            "good_direction": "lower",
        },
        {
            "col":      "HOUSING_RATIO",
            "title":    "Housing Cost Ratio by Income Quintile",
            "subtitle": "Annual housing spend / annual income  —  structural cost burden",
            "filename": "01e_housing_ratio.png",
            "color":    PALETTE["red"],
            "ref_line": 0.30,
            "ref_label":"30% burden threshold",
            "good_direction": "lower",
        },
        {
            "col":      "RETIRE_RATIO",
            "title":    "Retirement Contribution Ratio by Income Quintile",
            "subtitle": "Annual retirement contributions / annual income  —  forward-looking savings behavior",
            "filename": "01e_retirement_ratio.png",
            "color":    PALETTE["teal"],
            "ref_line": None,
            "ref_label": None,
            "good_direction": "higher",
        },
    ]

    for cfg in ratio_configs:
        _ratio_bar(df, **cfg)


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, analysis_df: pd.DataFrame):
    pos = df[df["FINCBTAX"] > 0]
    w   = df["FINLWT21"]

    print("\n" + "=" * 65)
    print("PHASE 1 — KEY WEIGHTED STATISTICS")
    print("=" * 65)
    print(f"\n  Full sample:                       {len(df):,} consumer units")
    print(f"  Analysis sample (income > 0):      {len(analysis_df):,} consumer units")
    print(f"  Survey-weighted households (full): {w.sum():,.0f}")
    print(f"  January households (PQ only):      {df['FLAG_CQ_MISSING'].sum():,} ({df['FLAG_CQ_MISSING'].mean():.1%})")
    print(f"  Geography-suppressed rows:         {df['FLAG_REGION_UNKNOWN'].sum():,}")

    print(f"\n  Wtd. median income:        {fmt_dollar(weighted_median(pos['FINCBTAX'], pos['FINLWT21']))}")
    print(f"  Wtd. mean income:          {fmt_dollar(weighted_mean(pos['FINCBTAX'], pos['FINLWT21']))}")
    print(f"  Gini coefficient:          {weighted_gini(pos['FINCBTAX'], pos['FINLWT21']):.3f}")

    breaks = df.attrs.get("quintile_breaks", [])
    if breaks:
        print(f"\n  Weighted income quintile breakpoints:")
        for i, lbl in enumerate(QUINTILE_LABELS):
            lo = fmt_dollar(breaks[i])
            hi = fmt_dollar(breaks[i + 1])
            n  = (df["INCOME_QUINTILE"] == lbl).sum()
            print(f"    {lbl:<22} {lo:>10} – {hi:<12} n={n:,}")

    print(f"\n  Wtd. median quarterly spend:   {fmt_dollar(weighted_median(df['TOTEXP_AVG_Q'], w))}")
    print(f"  Wtd. mean quarterly spend:     {fmt_dollar(weighted_mean(df['TOTEXP_AVG_Q'], w))}")

    ratio_meta = [
        ("EXPENSE_RATIO",   "Total expense ratio    "),
        ("LIFESTYLE_RATIO", "Lifestyle ratio        "),
        ("HOUSING_RATIO",   "Housing ratio          "),
        ("RETIRE_RATIO",    "Retirement ratio       "),
    ]
    print()
    for col, label in ratio_meta:
        if col not in analysis_df.columns:
            continue
        valid = analysis_df[analysis_df[col].notna()]
        wmed  = weighted_median(valid[col], valid["FINLWT21"])
        wmean = weighted_mean(valid[col],   valid["FINLWT21"])
        print(f"  {label}  wtd. median={wmed:.3f}  wtd. mean={wmean:.3f}  n={len(valid):,}")

    burdened    = analysis_df[analysis_df["HOUSING_BURDENED"].notna()]
    burden_rate = weighted_share(burdened["HOUSING_BURDENED"], burdened["FINLWT21"])
    severe      = analysis_df[analysis_df["HOUSING_SEVERE"].notna()]
    severe_rate = weighted_share(severe["HOUSING_SEVERE"], severe["FINLWT21"])
    print(f"\n  Housing burdened (>30%):       {df['HOUSING_BURDENED'].sum():,} rows ({burden_rate:.1%} weighted)")
    print(f"  Severely burdened (>50%):      {df['HOUSING_SEVERE'].sum():,} rows ({severe_rate:.1%} weighted)")
    print("\n" + "=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1():
    print("=" * 65)
    print("PHASE 1 — WEIGHTED DESCRIPTIVE STATISTICS")
    print("=" * 65)

    print("\n[00] Preparing data...")
    df          = prepare_data()
    analysis_df = get_analysis_sample(df)

    print_summary(df, analysis_df)

    print("\n[1a] Income distribution")
    chart_1a_income_histogram(analysis_df)
    chart_1a_median_by_quintile(analysis_df)
    chart_1a_median_by_region(analysis_df)

    print("\n[1b] Spending profile")
    chart_1b_mean_spending(analysis_df)
    chart_1b_expense_ratio(analysis_df)
    chart_1b_housing_burden(analysis_df)

    print("\n[1c] Spending composition heatmap")
    chart_1c_spending_heatmap(analysis_df)

    chart_charts_1d = charts_1d_demographics
    chart_charts_1d(df)   # uses full df for demographic counts

    charts_1e_four_ratios(analysis_df)

    print(f"\n  Phase 1 complete. 19 figures saved to {OUTPUT_DIR}/")
    print("\n  Sanity checks before Phase 2:")
    print("    Gini should be 0.45 - 0.55")
    print("    Wtd. median income should be below wtd. mean")
    print("    Housing burden rate should decline Q1 -> Q5")
    print("    Food share in heatmap should decline Q1 -> Q5  (Engel's Law)")
    print("    Lifestyle ratio should rise Q2 -> Q5           (Income Fallacy signal)")
    print("    Retirement ratio should rise Q1 -> Q5          (savings capacity)\n")


if __name__ == "__main__":
    run_phase1()