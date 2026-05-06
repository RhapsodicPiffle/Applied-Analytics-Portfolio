"""
02_observational.py
───────────────────
Phase 2: Observational Statistical Analyses
BLS Consumer Expenditure Survey — FMLI 2025 Q1 (fmli251.csv)

Hypothesis:
  The Income Fallacy — financial discipline (as measured by expense-to-income
  ratios) does not automatically improve with higher income. Lifestyle creep
  and poor expense ratios are present across the full income spectrum.

Analytical approach:
  Tests escalate in rigor. Each section answers a harder version of the
  same question.

  2a. Assumption checks
      Shapiro-Wilk normality test per quintile per ratio.
      Levene's test for homogeneity of variance.
      Results justify using Kruskal-Wallis alongside ANOVA.

  2b. Weighted ANOVA + Kruskal-Wallis
      Primary hypothesis test: do ratio distributions differ by quintile?
      Implemented as Weighted Least Squares regression with quintile dummies
      (mathematically equivalent to weighted ANOVA).
      Effect size: eta-squared (proportion of variance explained by quintile).
      Sensitivity: repeat with Q1 excluded to isolate denominator artifacts.

  2c. Post-hoc pairwise comparisons
      Tukey HSD (ANOVA followup) and Bonferroni Mann-Whitney U
      (Kruskal-Wallis followup). Cohen's d for key pairs.

  2d. Multiple regression
      EXPENSE_RATIO and LIFESTYLE_RATIO as outcomes.
      LOG_INCOME coefficient sign is the key test: positive = higher income
      independently predicts more discretionary spending, supporting the
      Income Fallacy hypothesis.
      VIF check for multicollinearity.

  2e. Cluster analysis
      K-means on behavioral variables. k selected by elbow + silhouette.
      Cross-tab of cluster membership by income quintile is the headline
      exhibit: if the high-spender cluster includes Q4 and Q5, the
      Income Fallacy has direct visual support.

Survey weights:
  ANOVA/regression: weights normalized to sum to sample size, applied as
  WLS case weights. This is standard practice for CE published tables.
  Kruskal-Wallis: unweighted (scipy limitation — noted in output).
  Clustering: normalized weights passed as sample_weight to KMeans.

Q1 note:
  Q1 households show extreme ratios from near-zero income denominators,
  not extreme spending. All tests run with and without Q1. If findings
  hold in both specifications, the result is robust. FLAG_EXTREME_RATIO
  households (ratio > 5.0) excluded from regression by default.

Run from project root:
    python src/02_observational.py
"""

import os
import sys

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import scikit_posthocs as sp
    HAS_SCIKIT_POSTHOCS = True
except ImportError:
    HAS_SCIKIT_POSTHOCS = False
    print("  NOTE: scikit-posthocs not available. "
          "Dunn's test replaced with Bonferroni Mann-Whitney U.")

from data_prep import (
    prepare_data,
    get_analysis_sample,
    weighted_quantile,
    QUINTILE_LABELS,
    QUINTILE_LABELS_SHORT,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "outputs/figures"
ALPHA      = 0.05     # Significance threshold throughout

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
    "green":      "#2D9B5A",
}

# The four analytical ratios with display metadata
# Colorblind-safe palette (Wong 2011 — Nature Methods)
# These colors are distinguishable by individuals with all common forms
# of color vision deficiency (protanopia, deuteranopia, tritanopia).
CB_BLUE      = "#0072B2"
CB_ORANGE    = "#E69F00"
CB_GREEN     = "#009E73"
CB_VERMILLION = "#D55E00"
CB_SKY       = "#56B4E9"
CB_PINK      = "#CC79A7"
CB_YELLOW    = "#F0E442"

QUINTILE_COLORS = [CB_SKY, CB_BLUE, CB_GREEN, CB_ORANGE, CB_VERMILLION]

# The four analytical ratios — mutually exclusive, collectively exhaustive.
# HOUSING + NECESSITY + DISCRETIONARY + RETIREMENT ≈ TOTAL EXPENSES.
# EXPENSE_RATIO is retained as a headline total but NOT used in clustering.
RATIOS = {
    "HOUSING_RATIO":       {
        "label":     "Housing Ratio",
        "short":     "Housing",
        "color":     CB_VERMILLION,
        "direction": "lower = better",
    },
    "NECESSITY_RATIO":     {
        "label":     "Necessities Ratio",
        "short":     "Necessities",
        "color":     CB_BLUE,
        "direction": "lower = better",
    },
    "DISCRETIONARY_RATIO": {
        "label":     "Discretionary Ratio",
        "short":     "Discretionary",
        "color":     CB_ORANGE,
        "direction": "lower = better",
    },
    "RETIRE_RATIO":        {
        "label":     "Retirement Ratio",
        "short":     "Retirement",
        "color":     CB_GREEN,
        "direction": "higher = better",
    },
}

# EXPENSE_RATIO metadata — not in RATIOS (not used in decomposition ANOVA/clustering)
# but still needed by regression which tests it as a headline outcome.
EXPENSE_META = {
    "label": "Total Expense Ratio",
    "short": "Expense",
    "color": CB_SKY,
    "direction": "lower = better",
}

def get_ratio_meta(col: str) -> dict:
    """Return display metadata for a ratio column, including EXPENSE_RATIO."""
    if col in RATIOS:
        return RATIOS[col]
    if col == "EXPENSE_RATIO":
        return EXPENSE_META
    return {"label": col, "short": col, "color": "#888888", "direction": ""}


# ── Restricted re-analysis config ─────────────────────────────────────────────
# After the initial clustering identifies minority/outlier clusters, list their
# 0-based indices here to trigger a second pass that excludes those households.
# Cluster N displayed in output = index N-1 here. Set to [] to skip.
RESTRICTED_EXCLUDE_CLUSTER_INDICES: list = [2, 4]   # Clusters 3 & 5

# Names for the restricted-run clusters.  Fill in after reviewing the
# restricted cluster profiles that are printed during that second pass.
RESTRICTED_CLUSTER_NAMES: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_theme():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    PALETTE["light_gray"],
        "axes.labelcolor":   PALETTE["dark"],
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.grid":         False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
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


def save_fig(fig: plt.Figure, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def source_note(ax, extra: str = ""):
    note = "BLS Consumer Expenditure Survey, Q1 2025. Survey-weighted estimates."
    if extra:
        note = f"{note}  {extra}"
    ax.annotate(note, xy=(0, -0.13), xycoords="axes fraction",
                fontsize=7.5, color=PALETTE["gray"], ha="left")


def normalize_weights(weights: pd.Series) -> pd.Series:
    """
    Normalize survey weights to sum to sample size.
    This is the standard approach for using CE weights in regression
    and WLS-based ANOVA — preserves relative weighting while keeping
    the effective sample size comparable to the raw row count.
    """
    n = len(weights)
    return weights / weights.sum() * n


def get_quintile_groups(df: pd.DataFrame, col: str,
                        exclude_q1: bool = False,
                        cap: float = None) -> dict:
    """
    Return {quintile_label: Series} dict of ratio values per quintile.
    Filters to non-null values only.
    Optionally excludes Q1 for sensitivity analysis.
    Optionally caps values for outlier robustness.
    """
    groups = {}
    labels = QUINTILE_LABELS[1:] if exclude_q1 else QUINTILE_LABELS
    for q in labels:
        sub = df[(df["INCOME_QUINTILE"] == q) & df[col].notna()][col]
        if cap is not None:
            sub = sub.clip(upper=cap)
        if len(sub) > 0:
            groups[q] = sub
    return groups


def cohen_d(s1: pd.Series, s2: pd.Series) -> float:
    """
    Cohen's d for two independent samples.
    Uses pooled standard deviation.
    """
    n1, n2   = len(s1), len(s2)
    var1, var2 = s1.var(ddof=1), s2.var(ddof=1)
    pooled_sd  = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((s1.mean() - s2.mean()) / pooled_sd)


def eta_squared_from_wls(result) -> float:
    """
    Compute eta-squared from a statsmodels OLS/WLS result.
    η² = SS_between / SS_total
       = (SS_total - SS_residual) / SS_total
    """
    ss_total = result.centered_tss
    ss_resid  = result.ssr
    if ss_total == 0:
        return 0.0
    return float((ss_total - ss_resid) / ss_total)


def interpret_eta_sq(eta_sq: float) -> str:
    if eta_sq < 0.01:
        return "negligible"
    if eta_sq < 0.06:
        return "small"
    if eta_sq < 0.14:
        return "medium"
    return "large"


def sep(title: str = "", width: int = 65):
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
    else:
        print("-" * width)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2a — ASSUMPTION CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def run_assumption_checks(df: pd.DataFrame) -> dict:
    """
    Test normality (Shapiro-Wilk) and homogeneity of variance (Levene).
    Returns dict of results for downstream use.
    Produces: 02a_qq_plots.png
    """
    sep("SECTION 2a — ASSUMPTION CHECKS")

    results = {"normality": {}, "levene": {}}
    ratio_keys = list(RATIOS.keys())

    # ── Shapiro-Wilk normality per quintile per ratio ──────────────────────
    print(f"\n  Shapiro-Wilk Normality Test (H0: sample drawn from normal distribution)")
    print(f"  Rejection at α={ALPHA} justifies using Kruskal-Wallis alongside ANOVA")
    print(f"  {'Ratio':<20} {'Quintile':<22} {'W stat':>8} {'p-value':>10} {'Normal?':>9}")
    sep()

    for col in ratio_keys:
        cap        = 5.0
        norm_res   = {}
        for q in QUINTILE_LABELS:
            sub = df[(df["INCOME_QUINTILE"] == q) & df[col].notna()][col].clip(upper=cap)
            if len(sub) < 3:
                continue
            # Shapiro-Wilk is unreliable for n > 5000; subsample if needed
            sample = sub.sample(min(len(sub), 2000), random_state=42)
            w, p   = stats.shapiro(sample)
            norm_res[q] = {"W": w, "p": p, "normal": p >= ALPHA}
            q_short = QUINTILE_LABELS_SHORT[QUINTILE_LABELS.index(q)]
            flag    = "" if p >= ALPHA else "  ← non-normal"
            print(f"  {get_ratio_meta(col)['short']:<20} {q_short:<22} {w:>8.4f} {p:>10.4f}{flag}")

        results["normality"][col] = norm_res

    # ── Levene's test for homogeneity of variance ──────────────────────────
    print(f"\n  Levene's Test for Equal Variances (H0: group variances are equal)")
    print(f"  Rejection means ANOVA variance assumption is violated")
    print(f"  {'Ratio':<20} {'F stat':>10} {'p-value':>12} {'Equal var?':>12}")
    sep()

    for col in ratio_keys:
        groups = list(get_quintile_groups(df, col, cap=5.0).values())
        if len(groups) < 2:
            continue
        f_stat, p = stats.levene(*groups)
        equal_var  = p >= ALPHA
        results["levene"][col] = {"F": f_stat, "p": p, "equal_var": equal_var}
        flag = "" if equal_var else "  ← unequal variances"
        print(f"  {get_ratio_meta(col)['short']:<20} {f_stat:>10.3f} {p:>12.4f}"
              f"  {'Yes' if equal_var else 'No':>10}{flag}")

    # ── Q-Q plots ──────────────────────────────────────────────────────────
    print("\n  Generating Q-Q plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, ratio_keys):
        all_vals = df[df[col].notna()][col].clip(upper=5.0)
        stats.probplot(all_vals, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {get_ratio_meta(col)['label']}", loc="left", fontsize=11)
        ax.get_lines()[0].set(color=get_ratio_meta(col)["color"], alpha=0.5,
                               markersize=2, markeredgewidth=0)
        ax.get_lines()[1].set(color=PALETTE["dark"], lw=1.5)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.annotate("Departure from line = non-normality",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    fontsize=8, color=PALETTE["gray"], va="top")

    fig.suptitle("Normality Check — Q-Q Plots by Ratio\n"
                 "Clipped at 5.0 for display. Strong departures justify Kruskal-Wallis.",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "02a_qq_plots.png")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2b — WEIGHTED ANOVA + KRUSKAL-WALLIS
# ══════════════════════════════════════════════════════════════════════════════

def run_anova(df: pd.DataFrame) -> dict:
    """
    Weighted ANOVA (as WLS) and Kruskal-Wallis for all four ratios.
    Produces: 02b_anova_effect_sizes.png, 02b_group_means.png
    Returns dict of results for post-hoc use.
    """
    sep("SECTION 2b — WEIGHTED ANOVA + KRUSKAL-WALLIS")

    results = {}
    cap     = 5.0   # Clip extreme ratios before testing

    print(f"\n  Ratios clipped at {cap} before testing.")
    print(f"  ANOVA implemented as Weighted Least Squares with quintile dummies.")
    print(f"  Weights normalized to sum to N (CE standard practice).")
    print(f"  Kruskal-Wallis uses unweighted ranks (scipy limitation — noted).\n")

    header = (f"  {'Ratio':<20} {'Spec':<12} "
              f"{'F / H':>8} {'df':>6} {'p-value':>10} "
              f"{'η²':>7} {'Effect':>12} {'Sig?':>6}")
    print(header)
    sep()

    eta_sq_all   = {}
    anova_full   = {}
    kw_full      = {}

    for col in RATIOS:
        meta = get_ratio_meta(col)

        # ── Full sample (all quintiles) ────────────────────────────────────
        for spec_label, excl_q1 in [("All Q", False), ("Q2-Q5", True)]:
            sub = df[df["INCOME_QUINTILE"].notna() & df[col].notna()].copy()
            if excl_q1:
                sub = sub[sub["INCOME_QUINTILE"] != "Q1 (Bottom 20%)"]
            sub[col] = sub[col].clip(upper=cap)
            sub["w_norm"] = normalize_weights(sub["FINLWT21"])

            # Weighted ANOVA as WLS
            model  = smf.wls(f"{col} ~ C(INCOME_QUINTILE)",
                             data=sub, weights=sub["w_norm"])
            result = model.fit()
            f_stat = result.fvalue
            f_p    = result.f_pvalue
            df_num = result.df_model
            eta_sq = eta_squared_from_wls(result)
            eff    = interpret_eta_sq(eta_sq)
            sig    = "Yes" if f_p < ALPHA else "No"

            # Kruskal-Wallis (unweighted)
            groups = list(get_quintile_groups(sub, col, excl_q1).values())
            h_stat, h_p = stats.kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)

            # Store for downstream
            if spec_label == "All Q":
                anova_full[col] = {"result": result, "f": f_stat, "p": f_p,
                                   "eta_sq": eta_sq, "df": df_num}
                kw_full[col]    = {"H": h_stat, "p": h_p}
                eta_sq_all[col] = eta_sq

            # Print ANOVA row
            print(f"  {meta['short']:<20} {spec_label:<12} "
                  f"{f_stat:>8.2f} {int(df_num):>6} {f_p:>10.4f} "
                  f"{eta_sq:>7.4f} {eff:>12} {sig:>6}")

        # Print K-W row
        h_stat_all, h_p_all = kw_full[col]["H"], kw_full[col]["p"]
        print(f"  {meta['short']:<20} {'K-W (unwtd)':<12} "
              f"{h_stat_all:>8.2f} {'—':>6} {h_p_all:>10.4f} "
              f"{'—':>7} {'—':>12} "
              f"{'Yes' if h_p_all < ALPHA else 'No':>6}")
        sep()

    results["anova"] = anova_full
    results["kw"]    = kw_full

    # ── Effect size chart ──────────────────────────────────────────────────
    ratio_labels = [RATIOS[c]["short"] for c in RATIOS]
    eta_values   = [eta_sq_all[c] for c in RATIOS]
    colors       = [RATIOS[c]["color"] for c in RATIOS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ratio_labels, eta_values, color=colors,
                  alpha=0.85, edgecolor="white", width=0.55)

    # Reference lines for effect size thresholds
    for thresh, label, ls in [(0.01, "Small (0.01)", ":"),
                               (0.06, "Medium (0.06)", "--"),
                               (0.14, "Large (0.14)", "-.")]:
        ax.axhline(thresh, color=PALETTE["gray"], lw=1.2,
                   ls=ls, alpha=0.7, label=label)

    for bar, val in zip(bars, eta_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"η²={val:.4f}\n({interpret_eta_sq(val)})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_yticks([])
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("ANOVA Effect Size (η²) by Ratio\n"
                 "Proportion of ratio variance explained by income quintile",
                 loc="left")
    ax.legend(fontsize=8, title="Effect size thresholds")
    source_note(ax, "All quintiles included. Ratios clipped at 5.0.")
    fig.tight_layout()
    save_fig(fig, "02b_anova_effect_sizes.png")

    # ── Group means chart ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, RATIOS):
        meta     = get_ratio_meta(col)
        q_labels = []
        means    = []
        cis      = []

        for q in QUINTILE_LABELS:
            sub = df[(df["INCOME_QUINTILE"] == q) & df[col].notna()].copy()
            sub[col] = sub[col].clip(upper=cap)
            if len(sub) == 0:
                continue
            sub["w_norm"] = normalize_weights(sub["FINLWT21"])
            wm = np.average(sub[col], weights=sub["w_norm"])
            # Bootstrap 95% CI on weighted mean (500 iterations)
            boot_means = []
            for _ in range(500):
                idx  = np.random.choice(len(sub), len(sub), replace=True)
                bsub = sub.iloc[idx]
                boot_means.append(np.average(bsub[col], weights=bsub["w_norm"]))
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
            q_labels.append(QUINTILE_LABELS_SHORT[QUINTILE_LABELS.index(q)])
            means.append(wm)
            cis.append((wm - ci_lo, ci_hi - wm))

        x     = np.arange(len(q_labels))
        means = np.array(means)
        yerr  = np.array(cis).T

        bars = ax.bar(x, means, color=meta["color"],
                      alpha=0.82, edgecolor="white", width=0.55)
        ax.errorbar(x, means, yerr=yerr,
                    fmt="none", color=PALETTE["dark"],
                    capsize=5, capthick=1.2, linewidth=1.2)

        for i, val in enumerate(means):
            ax.text(i, val + max(means) * 0.03, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=10)
        ax.set_yticks([])
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(meta["label"], loc="left", fontsize=11)
        ax.annotate(f"η²={eta_sq_all[col]:.4f}  ({interpret_eta_sq(eta_sq_all[col])})  "
                    f"p={anova_full[col]['p']:.4f}",
                    xy=(0.97, 0.97), xycoords="axes fraction",
                    fontsize=8.5, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=PALETTE["light_gray"], alpha=0.9))

    fig.suptitle("Weighted Group Means by Income Quintile\n"
                 "Error bars = bootstrapped 95% CI (500 iterations). "
                 "Ratios clipped at 5.0.",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02b_group_means.png")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2c — POST-HOC PAIRWISE COMPARISONS
# ══════════════════════════════════════════════════════════════════════════════

def run_posthoc(df: pd.DataFrame, anova_results: dict):
    """
    Tukey HSD post-hoc following ANOVA.
    Bonferroni-corrected Mann-Whitney U as non-parametric alternative.
    Cohen's d for key pairwise comparisons.
    Produces: 02c_posthoc_heatmap.png
    """
    sep("SECTION 2c — POST-HOC PAIRWISE COMPARISONS")

    cap           = 5.0
    key_pairs     = [
        ("Q1 (Bottom 20%)", "Q5 (Top 20%)",  "Q1 vs Q5"),
        ("Q3",               "Q5 (Top 20%)", "Q3 vs Q5"),
        ("Q4",               "Q5 (Top 20%)", "Q4 vs Q5"),
        ("Q3",               "Q4",           "Q3 vs Q4"),
    ]

    print(f"\n  Tukey HSD: controls familywise error rate across all pairwise comparisons")
    print(f"  Mann-Whitney U: Bonferroni-corrected non-parametric alternative")
    print(f"  Cohen's d: effect size for key pairs (|d|<0.2=trivial, 0.2-0.5=small, "
          f"0.5-0.8=medium, >0.8=large)\n")

    all_tukey      = {}
    all_cohens_d   = {}

    for col in RATIOS:
        meta = get_ratio_meta(col)
        sub  = df[df["INCOME_QUINTILE"].notna() & df[col].notna()].copy()
        sub[col] = sub[col].clip(upper=cap)

        # ── Tukey HSD ─────────────────────────────────────────────────────
        tukey = pairwise_tukeyhsd(
            endog=sub[col].values,
            groups=sub["INCOME_QUINTILE"].values,
            alpha=ALPHA
        )
        all_tukey[col] = tukey

        # ── Cohen's d for key pairs ────────────────────────────────────────
        print(f"  {meta['label']}")
        print(f"  {'Pair':<20} {'Cohen d':>9} {'Size':>10} "
              f"{'MW p (adj)':>12} {'Sig?':>6}")
        print(f"  {'-'*60}")

        cd_row = {}
        q_groups = get_quintile_groups(sub, col, cap=cap)
        n_pairs  = len(key_pairs)

        for q1_lbl, q2_lbl, pair_name in key_pairs:
            # Match flexible quintile label format
            g1 = next((v for k, v in q_groups.items() if q1_lbl in k), None)
            g2 = next((v for k, v in q_groups.items() if q2_lbl in k), None)
            if g1 is None or g2 is None:
                continue

            d        = cohen_d(g1, g2)
            mw_stat, mw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            mw_p_adj = min(mw_p * n_pairs, 1.0)   # Bonferroni correction

            size = ("trivial" if abs(d) < 0.2 else
                    "small"   if abs(d) < 0.5 else
                    "medium"  if abs(d) < 0.8 else "large")
            sig  = "Yes" if mw_p_adj < ALPHA else "No"
            cd_row[pair_name] = d

            print(f"  {pair_name:<20} {d:>9.3f} {size:>10} "
                  f"{mw_p_adj:>12.4f} {sig:>6}")

        all_cohens_d[col] = cd_row
        print()

    # ── Pairwise significance heatmap (Tukey HSD p-values) ────────────────
    print("  Generating post-hoc significance heatmap...")

    # Build significance matrix for each ratio
    q_labels_short = QUINTILE_LABELS_SHORT
    n_q            = len(q_labels_short)
    ratio_keys     = list(RATIOS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for ax, col in zip(axes, ratio_keys):
        meta      = get_ratio_meta(col)
        tukey     = all_tukey[col]
        tukey_df  = pd.DataFrame(data=tukey._results_table.data[1:],
                                  columns=tukey._results_table.data[0])
        tukey_df["p-adj"] = pd.to_numeric(tukey_df["p-adj"])

        # Map full quintile labels to short labels
        lbl_to_short = dict(zip(QUINTILE_LABELS, QUINTILE_LABELS_SHORT))

        # Build n×n p-value matrix
        p_matrix = pd.DataFrame(np.ones((n_q, n_q)),
                                  index=q_labels_short,
                                  columns=q_labels_short)
        for _, row in tukey_df.iterrows():
            g1 = lbl_to_short.get(row["group1"], row["group1"])
            g2 = lbl_to_short.get(row["group2"], row["group2"])
            if g1 in q_labels_short and g2 in q_labels_short:
                p_matrix.loc[g1, g2] = row["p-adj"]
                p_matrix.loc[g2, g1] = row["p-adj"]

        # Color: green = sig, red = not sig, diagonal = white
        cell_colors = np.where(p_matrix.values < ALPHA, "#D4EDDA", "#F8D7DA")
        np.fill_diagonal(cell_colors, "#FFFFFF")

        ax.imshow(np.zeros_like(p_matrix.values, dtype=float),
                  cmap="RdYlGn", vmin=0, vmax=1, alpha=0)
        for i in range(n_q):
            for j in range(n_q):
                color = cell_colors[i, j]
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            color=color, zorder=0))
                if i != j:
                    p_val = p_matrix.iloc[i, j]
                    txt   = f"{p_val:.3f}" if p_val >= 0.001 else "<.001"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=9,
                            color=PALETTE["dark"] if p_val >= 0.001 else PALETTE["red"],
                            fontweight="bold" if p_val < ALPHA else "normal")

        ax.set_xticks(range(n_q))
        ax.set_yticks(range(n_q))
        ax.set_xticklabels(q_labels_short, fontsize=9)
        ax.set_yticklabels(q_labels_short, fontsize=9)
        ax.set_title(meta["label"], loc="left", fontsize=11)

        sig_patch   = mpatches.Patch(color="#D4EDDA", label=f"p < {ALPHA} (significant)")
        nosig_patch = mpatches.Patch(color="#F8D7DA", label=f"p ≥ {ALPHA} (not significant)")
        ax.legend(handles=[sig_patch, nosig_patch], fontsize=8,
                  loc="lower right", framealpha=0.9)

    fig.suptitle("Tukey HSD Post-Hoc: Pairwise Adjusted p-Values\n"
                 "Green = statistically significant difference at α=0.05",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "02c_posthoc_heatmap.png")

    return all_cohens_d


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2d — MULTIPLE REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(df: pd.DataFrame, suffix: str = ""):
    """
    Multiple regression for all five expense buckets plus total expenses.
    Excludes FLAG_EXTREME_RATIO rows (ratio > 5.0) and FLAG_INCOME_INVALID.
    Uses WLS with normalized survey weights.
    suffix: appended to output filenames; "_restricted" for the second pass.
    """
    label = "RESTRICTED — " if suffix else ""
    sep(f"SECTION 2d — {label}MULTIPLE REGRESSION")

    # ── Sample preparation ─────────────────────────────────────────────────
    # Exclude extreme ratios — driven by near-zero income denominators,
    # not behavioral patterns. Retaining them inflates residuals and masks
    # real coefficient signals.
    excl_cols = ["FLAG_INCOME_INVALID", "FLAG_EXTREME_RATIO"]
    excl_mask = (df["FLAG_INCOME_INVALID"] == 1)
    if "FLAG_EXTREME_RATIO" in df.columns:
        excl_mask |= (df["FLAG_EXTREME_RATIO"] == 1)

    reg_df = df[~excl_mask].copy()
    print(f"\n  Regression sample: {len(reg_df):,} rows "
          f"(excluded {excl_mask.sum():,} extreme/invalid)")

    # Encode region as dummies (South = reference, largest group)
    reg_df["region_label"] = reg_df["region_label"].fillna("Suppressed")
    reg_df = pd.get_dummies(reg_df, columns=["region_label"], drop_first=False)
    region_dummies = [c for c in reg_df.columns if c.startswith("region_label_")
                      and "South" not in c and "Suppressed" not in c]

    # Build predictor string — LOG_INCOME is key; all others are controls
    controls = (["LOG_INCOME", "FAM_SIZE", "HOMEOWNER", "NO_EARNR",
                 "HAS_CHILDREN", "HAS_ELDERLY", "EDUC_REF_ORD", "AGE_REF"]
                + region_dummies)

    formula_base = " + ".join(controls)

    outcomes = [
        "EXPENSE_RATIO",
        "HOUSING_RATIO",
        "NECESSITY_RATIO",
        "DISCRETIONARY_RATIO",
        "RETIRE_RATIO",
    ]

    print(f"\n  Model: outcome ~ {formula_base}")
    print(f"  Reference categories: South region")
    print(f"  LOG_INCOME sign: negative = discipline (ratio shrinks with income); "
          f"positive = creep/growth")

    model_results = {}

    for col in outcomes:
        meta = get_ratio_meta(col)
        sub  = reg_df[reg_df[col].notna()].copy()
        sub["w_norm"] = normalize_weights(sub["FINLWT21"])

        formula = f"{col} ~ {formula_base}"
        model   = smf.wls(formula, data=sub, weights=sub["w_norm"])
        result  = model.fit()

        model_results[col] = result

        sep(f"  {meta['label']} Model")
        print(f"  n = {int(result.nobs):,}  |  "
              f"R² = {result.rsquared:.4f}  |  "
              f"Adj R² = {result.rsquared_adj:.4f}  |  "
              f"F = {result.fvalue:.2f}  p = {result.f_pvalue:.4f}")
        print()
        print(f"  {'Variable':<30} {'Coef':>9} {'Std Err':>9} "
              f"{'t':>8} {'p-value':>10} {'Sig':>5}")
        print(f"  {'-'*65}")

        params = result.params
        bse    = result.bse
        tvals  = result.tvalues
        pvals  = result.pvalues

        # Compute standardized (Beta) coefficients
        # Beta = B * (SD_x / SD_y) — strips unit scale to compare predictor importance
        y_sd = sub[col].std()
        beta_coefs = {}
        for var in params.index:
            if var == "Intercept":
                beta_coefs[var] = np.nan
                continue
            # Get the underlying predictor name (handle dummies)
            if var.endswith("[T.True]"):
                # Dummy predictor — use 0/1 SD
                pred_name = var.replace("[T.True]", "")
                if pred_name in sub.columns:
                    x_sd = sub[pred_name].astype(float).std()
                else:
                    x_sd = np.nan
            elif var in sub.columns:
                x_sd = sub[var].astype(float).std()
            else:
                x_sd = np.nan
            beta_coefs[var] = params[var] * (x_sd / y_sd) if y_sd > 0 and not np.isnan(x_sd) else np.nan

        print(f"  {'Variable':<30} {'B':>9} {'Beta':>8} {'Std Err':>9} "
              f"{'t':>8} {'p-value':>10} {'Sig':>5}")
        print(f"  {'-'*72}")
        for var in params.index:
            sig    = ("***" if pvals[var] < 0.001 else
                      "**"  if pvals[var] < 0.01  else
                      "*"   if pvals[var] < 0.05  else "")
            flag   = " ← KEY" if var == "LOG_INCOME" else ""
            v_name = var.replace("region_label_", "Region: ").replace("[T.True]", "")
            beta_str = f"{beta_coefs[var]:>8.3f}" if not np.isnan(beta_coefs[var]) else f"{'—':>8}"
            print(f"  {v_name:<30} {params[var]:>9.4f} {beta_str} {bse[var]:>9.4f} "
                  f"{tvals[var]:>8.3f} {pvals[var]:>10.4f} {sig:>5}{flag}")

        # ── Q2-Q5 Sensitivity Check ────────────────────────────────────────
        # Re-run the same regression excluding Q1 households.
        # If LOG_INCOME coefficient stays similar in sign and significance,
        # the finding is robust to Q1's denominator distortion.
        sub_q25 = sub[sub["INCOME_QUINTILE"] != "Q1 (Bottom 20%)"].copy()
        sub_q25["w_norm"] = normalize_weights(sub_q25["FINLWT21"])
        result_q25 = smf.wls(formula, data=sub_q25, weights=sub_q25["w_norm"]).fit()

        log_inc_full   = result.params.get("LOG_INCOME", np.nan)
        log_inc_q25    = result_q25.params.get("LOG_INCOME", np.nan)
        log_inc_p_full = result.pvalues.get("LOG_INCOME", np.nan)
        log_inc_p_q25  = result_q25.pvalues.get("LOG_INCOME", np.nan)

        print(f"\n  Q2-Q5 SENSITIVITY (Q1 excluded):")
        print(f"    n = {int(result_q25.nobs):,}  R² = {result_q25.rsquared:.4f}")
        print(f"    LOG_INCOME coefficient:  full={log_inc_full:>8.4f}  Q2-Q5={log_inc_q25:>8.4f}")
        print(f"    LOG_INCOME p-value:      full={log_inc_p_full:.4f}  Q2-Q5={log_inc_p_q25:.4f}")
        if log_inc_q25 != 0 and not np.isnan(log_inc_q25):
            ratio_change = (log_inc_q25 - log_inc_full) / abs(log_inc_full) * 100
            print(f"    Magnitude change:        {ratio_change:+.1f}%  "
                  f"(positive = Q1 was suppressing the coefficient)")

        # VIF
        print(f"\n  Variance Inflation Factors (VIF > 5 = concern, > 10 = serious):")
        # Cast to float64 explicitly — Int8 boolean flags break statsmodels VIF
        X_vif = sub[controls].dropna().astype(np.float64)
        X_vif = X_vif.assign(const=1.0)
        try:
            vif_data = pd.DataFrame({
                "Variable": X_vif.columns,
                "VIF":       [variance_inflation_factor(X_vif.values, i)
                               for i in range(X_vif.shape[1])]
            }).query("Variable != 'const'").sort_values("VIF", ascending=False)
            for _, row in vif_data.iterrows():
                flag = "  ← HIGH" if row["VIF"] > 5 else ""
                print(f"    {row['Variable']:<30} {row['VIF']:>7.2f}{flag}")
        except Exception as e:
            print(f"    VIF calculation failed: {e}")
        print()

    # ── Coefficient plots ──────────────────────────────────────────────────
    for col in outcomes:
        meta   = get_ratio_meta(col)
        result = model_results[col]
        params = result.params.drop("Intercept", errors="ignore")
        ci_lo  = result.conf_int()[0].drop("Intercept", errors="ignore")
        ci_hi  = result.conf_int()[1].drop("Intercept", errors="ignore")
        pvals  = result.pvalues.drop("Intercept", errors="ignore")

        # Exclude region dummies from the main plot for clarity
        mask = ~params.index.str.startswith("region_label_")
        params, ci_lo, ci_hi, pvals = (params[mask], ci_lo[mask],
                                        ci_hi[mask], pvals[mask])

        y_pos   = np.arange(len(params))
        colors  = [meta["color"] if pvals.iloc[i] < ALPHA
                   else PALETTE["light_gray"]
                   for i in range(len(params))]
        labels  = [v.replace("region_label_", "Region: ")
                   for v in params.index]

        fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.5 + 2)))
        ax.barh(y_pos, params.values, color=colors,
                alpha=0.85, edgecolor="white", height=0.6)
        ax.errorbar(params.values, y_pos,
                    xerr=[params.values - ci_lo.values,
                          ci_hi.values - params.values],
                    fmt="none", color=PALETTE["dark"],
                    capsize=4, capthick=1.2, linewidth=1.2)
        ax.axvline(0, color=PALETTE["dark"], lw=1.2, ls="--", alpha=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Coefficient (change in ratio per unit change in predictor)")
        ax.set_title(f"Regression Coefficients — {meta['label']}", loc="left")
        ax.annotate(
            f"n={int(result.nobs):,}  R²={result.rsquared:.4f}  "
            f"Adj R²={result.rsquared_adj:.4f}\n"
            f"Colored bars = significant at α=0.05. Error bars = 95% CI.\n"
            f"Region dummies omitted for clarity.",
            xy=(0.97, 0.03), xycoords="axes fraction",
            fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PALETTE["light_gray"], alpha=0.9))

        sig_patch   = mpatches.Patch(color=meta["color"], alpha=0.85,
                                      label="Significant (p < 0.05)")
        nosig_patch = mpatches.Patch(color=PALETTE["light_gray"], alpha=0.85,
                                      label="Not significant")
        ax.legend(handles=[sig_patch, nosig_patch], fontsize=9)
        source_note(ax, "WLS with normalized survey weights. Extreme ratios excluded.")
        fig.tight_layout()

        fname_map = {
            "EXPENSE_RATIO":       f"02d_regression_expense{suffix}.png",
            "HOUSING_RATIO":       f"02d_regression_housing{suffix}.png",
            "NECESSITY_RATIO":     f"02d_regression_necessity{suffix}.png",
            "DISCRETIONARY_RATIO": f"02d_regression_discretionary{suffix}.png",
            "RETIRE_RATIO":        f"02d_regression_retirement{suffix}.png",
        }
        save_fig(fig, fname_map[col])

    # ── Cross-outcome comparison: the housing creep test ──────────────────
    # If LOG_INCOME has a small/negative coefficient on lifestyle ratio but a
    # positive coefficient on housing ratio, that supports the housing creep
    # hypothesis — selective behavioral discipline.
    sep("CROSS-OUTCOME COMPARISON: LOG_INCOME COEFFICIENT")
    print(f"\n  This is the housing creep test:")
    print(f"  Negative coefficient = ratio shrinks as income rises (discipline)")
    print(f"  Positive coefficient = ratio grows as income rises (creep)\n")

    print(f"  {'Outcome':<22} {'B (unstd)':>12} {'Beta (std)':>12} {'p-value':>10} {'Direction':>14}")
    print(f"  {'-'*72}")
    for col in outcomes:
        result = model_results[col]
        b      = result.params.get("LOG_INCOME", np.nan)
        p      = result.pvalues.get("LOG_INCOME", np.nan)
        # Recompute beta locally
        sub_full = reg_df[reg_df[col].notna()].copy()
        beta = b * (sub_full["LOG_INCOME"].std() / sub_full[col].std()) if sub_full[col].std() > 0 else np.nan
        if col == "RETIRE_RATIO":
            direction = "MORE SAVING" if b > 0 else "LESS SAVING"
        else:
            direction = "DISCIPLINE" if b < 0 else "CREEP"
        print(f"  {get_ratio_meta(col)['label']:<22} {b:>12.4f} {beta:>12.4f} {p:>10.4f} {direction:>14}")

    return model_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2e — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(df: pd.DataFrame, suffix: str = ""):
    """
    K-means behavioral cluster analysis.
    k selected by elbow (inertia) + silhouette score.
    Key exhibit: cluster membership by income quintile.
    suffix: appended to output filenames; "_restricted" for the second pass.
    """
    label = "RESTRICTED — " if suffix else ""
    sep(f"SECTION 2e — {label}CLUSTER ANALYSIS")

    # ── Feature set ────────────────────────────────────────────────────────
    # Behavioral and structural variables — income included as context,
    # not as the primary grouping variable
    # Clustering uses the four-bucket decomposition ratios — mutually exclusive
    # and collectively exhaustive. EXPENSE_RATIO is excluded because it is the
    # sum of its components and would double-count spending patterns.
    cluster_features = [
        "HOUSING_RATIO",
        "NECESSITY_RATIO",
        "DISCRETIONARY_RATIO",
        "RETIRE_RATIO",
        "LOG_INCOME",
        "FAM_SIZE",
        "HOMEOWNER",
        "NO_EARNR",
        "AGE_REF",
        "EDUC_REF_ORD",
    ]

    # Sample: complete cases only, extreme ratios excluded
    excl_mask = (df["FLAG_INCOME_INVALID"] == 1)
    if "FLAG_EXTREME_RATIO" in df.columns:
        excl_mask |= (df["FLAG_EXTREME_RATIO"] == 1)

    clust_df = df[~excl_mask].copy()
    clust_df = clust_df.dropna(subset=cluster_features + ["INCOME_QUINTILE"])

    # Clip ratios for clustering stability
    for col in ["HOUSING_RATIO", "NECESSITY_RATIO", "DISCRETIONARY_RATIO", "RETIRE_RATIO"]:
        clust_df[col] = clust_df[col].clip(upper=3.0)

    print(f"\n  Cluster sample: {len(clust_df):,} rows "
          f"(excluded {excl_mask.sum():,} extreme/invalid, "
          f"{df[cluster_features + ['INCOME_QUINTILE']].isnull().any(axis=1).sum():,} incomplete)")

    X_raw = clust_df[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Normalize weights
    w_norm = normalize_weights(clust_df["FINLWT21"]).values

    # ── k selection: elbow + silhouette ────────────────────────────────────
    print("\n  Running elbow and silhouette analysis for k = 2 to 8...")
    k_range    = range(2, 9)
    inertias   = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled, sample_weight=w_norm)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_,
                                sample_size=min(len(X_scaled), 3000),
                                random_state=42)
        silhouettes.append(sil)
        print(f"    k={k}  inertia={km.inertia_:,.1f}  silhouette={sil:.4f}")

    # ── Elbow + silhouette chart ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(list(k_range), inertias, "o-", color=PALETTE["blue"],
             lw=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow Method", loc="left")
    ax1.set_xticks(list(k_range))
    ax1.spines["left"].set_visible(True)

    ax2.plot(list(k_range), silhouettes, "o-", color=PALETTE["orange"],
             lw=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score (higher = better separation)")
    ax2.set_title("Silhouette Score", loc="left")
    ax2.set_xticks(list(k_range))
    ax2.spines["left"].set_visible(True)

    # Mark optimal k on both charts
    optimal_k = int(k_range[np.argmax(silhouettes)])
    ax2.axvline(optimal_k, color=PALETTE["red"], lw=1.5, ls="--",
                label=f"Optimal k={optimal_k}")
    ax1.axvline(optimal_k, color=PALETTE["red"], lw=1.5, ls="--",
                label=f"Optimal k={optimal_k}")
    ax1.legend(fontsize=9)
    ax2.legend(fontsize=9)

    fig.suptitle("K-Means Cluster Count Selection\n"
                 "Choose k at elbow in inertia curve and peak in silhouette score.",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, f"02e_elbow_silhouette{suffix}.png")

    # ── Fit final model ────────────────────────────────────────────────────
    print(f"\n  Fitting final model with k={optimal_k}...")
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    km_final.fit(X_scaled, sample_weight=w_norm)
    clust_df["CLUSTER"] = km_final.labels_

    # ── Profile clusters ───────────────────────────────────────────────────
    print(f"\n  Cluster profiles (unscaled means):")
    print(f"  {'Feature':<22}", end="")
    for k in range(optimal_k):
        print(f"  {'Cluster ' + str(k+1):>12}", end="")
    print()
    sep()

    profile = clust_df.groupby("CLUSTER")[cluster_features].mean()

    for feat in cluster_features:
        print(f"  {feat:<22}", end="")
        for k in range(optimal_k):
            val = profile.loc[k, feat]
            print(f"  {val:>12.3f}", end="")
        print()

    # ── Cluster labeling ─────────────────────────────────────────────────────
    # MANUAL OVERRIDE: Edit the CLUSTER_NAMES dict below after reviewing the
    # cluster profiles printed above. Map cluster index (0-based) to your
    # chosen label. Run the script once to see profiles, then assign names.
    #
    # If a cluster index is not in the dict, it falls back to "Cluster N".
    # The dict is applied EVERY run, so your labels persist across runs as
    # long as k and random_state don't change.
    #
    # ──────────────────────────────────────────────────────────────────────
    # EDIT THESE LABELS:
    # Use RESTRICTED_CLUSTER_NAMES for the second pass; CLUSTER_NAMES for the first.
    # Review the profiles printed above, fill in the appropriate dict, then re-run.
    CLUSTER_NAMES = {
        # 0: "...",
        # 1: "...",
    }
    # ──────────────────────────────────────────────────────────────────────

    names_dict = RESTRICTED_CLUSTER_NAMES if suffix else CLUSTER_NAMES
    cluster_labels = {}
    for k in range(optimal_k):
        cluster_labels[k] = names_dict.get(k, f"Cluster {k+1}")

    print(f"\n  Cluster labels (n = sample, N = estimated U.S. households):")
    for k, lbl in cluster_labels.items():
        n     = (clust_df["CLUSTER"] == k).sum()
        N_wtd = clust_df.loc[clust_df["CLUSTER"] == k, "FINLWT21"].sum()
        print(f"    Cluster {k+1}: {lbl}  (n={n:,}, N≈{N_wtd/1e6:.1f}M households)")

    clust_df["CLUSTER_LABEL"] = clust_df["CLUSTER"].map(cluster_labels)

    # ── Cluster profiles heatmap ────────────────────────────────────────────
    # More space-efficient than a radar chart and directly comparable across
    # clusters and features simultaneously.
    # Rows = clusters (ordered by population, largest on top)
    # Columns = all clustering features
    # Cell values = unscaled cluster means
    heatmap_features = cluster_features
    heatmap_labels   = {
        "HOUSING_RATIO":       "Housing\nRatio",
        "NECESSITY_RATIO":     "Necessities\nRatio",
        "DISCRETIONARY_RATIO": "Discretionary\nRatio",
        "RETIRE_RATIO":        "Retirement\nRatio",
        "LOG_INCOME":          "Avg\nIncome",
        "FAM_SIZE":            "Family\nSize",
        "HOMEOWNER":           "Home-\nowner %",
        "NO_EARNR":            "Avg\nEarners",
        "AGE_REF":             "Avg\nAge",
        "EDUC_REF_ORD":        "Educ\nLevel",
    }

    # Order clusters by population (largest first)
    cluster_pop = (clust_df.groupby("CLUSTER")["FINLWT21"].sum()
                   .sort_values(ascending=False))
    ordered_clusters = cluster_pop.index.tolist()

    hm_data = profile.loc[ordered_clusters, heatmap_features].copy()
    # Explicit float cast — Int8 columns (HOMEOWNER) propagate as object dtype
    # through pandas operations, which seaborn/matplotlib cannot render
    hm_data = hm_data.astype(np.float64)
    hm_data.index = [cluster_labels[k] for k in ordered_clusters]
    hm_data.columns = [heatmap_labels.get(c, c) for c in heatmap_features]

    # Standardize columns for color mapping (so no single column dominates)
    # but display raw values in annotations
    hm_display = hm_data.copy()
    hm_normed  = (hm_data - hm_data.min()) / (hm_data.max() - hm_data.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(15, max(5, optimal_k * 1.2 + 2)))

    import seaborn as sns
    # Build custom annotation array with human-readable formatting per column
    # Ratios → whole percentages (0.26 → "26%")
    # LOG_INCOME → actual dollars (4.7 → "$50K")
    # FAM_SIZE → average with 1 decimal ("1.8")
    # HOMEOWNER → percentage of homeowners ("90%")
    # NO_EARNR → average with 1 decimal ("2.2")
    annot_strings = np.empty_like(hm_display.values, dtype=object)
    col_names = list(hm_data.columns)
    feature_names = list(heatmap_features)  # original column names before rename

    for i in range(hm_display.shape[0]):
        for j in range(hm_display.shape[1]):
            raw_val  = float(hm_display.values[i, j])
            feat     = feature_names[j]

            if feat in ("HOUSING_RATIO", "NECESSITY_RATIO",
                        "DISCRETIONARY_RATIO", "RETIRE_RATIO"):
                annot_strings[i, j] = f"{raw_val * 100:.0f}%"
            elif feat == "LOG_INCOME":
                actual_income = 10 ** raw_val
                if actual_income >= 1_000_000:
                    annot_strings[i, j] = f"${actual_income/1e6:.1f}M"
                elif actual_income >= 1_000:
                    annot_strings[i, j] = f"${actual_income/1e3:.0f}K"
                else:
                    annot_strings[i, j] = f"${actual_income:.0f}"
            elif feat == "HOMEOWNER":
                annot_strings[i, j] = f"{raw_val * 100:.0f}%"
            elif feat == "FAM_SIZE":
                annot_strings[i, j] = f"{raw_val:.1f}"
            elif feat == "NO_EARNR":
                annot_strings[i, j] = f"{raw_val:.1f}"
            elif feat == "AGE_REF":
                annot_strings[i, j] = f"{raw_val:.0f} yr"
            elif feat == "EDUC_REF_ORD":
                annot_strings[i, j] = f"{raw_val:.1f}"
            else:
                annot_strings[i, j] = f"{raw_val:.2f}"

    sns.heatmap(
        hm_normed,
        ax=ax,
        annot=annot_strings,
        fmt="",
        annot_kws={"size": 10, "weight": "bold"},
        cmap="RdYlBu_r",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Relative magnitude (min-max scaled)", "shrink": 0.6},
        vmin=0, vmax=1,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_title(f"Behavioral Cluster Profiles (k={optimal_k})",
                 loc="left", fontsize=11, fontweight="bold", pad=14)
    ax.annotate(
        "Ratios shown as % of income. Color = min-max normalized within each column.",
        xy=(0, 1.01), xycoords="axes fraction",
        fontsize=8.5, color=PALETTE["gray"], ha="left")

    # Population count annotations on the right margin (sample n + weighted N)
    for i, k in enumerate(ordered_clusters):
        n     = (clust_df["CLUSTER"] == k).sum()
        N_wtd = clust_df.loc[clust_df["CLUSTER"] == k, "FINLWT21"].sum()
        ax.text(len(heatmap_features) + 0.1, i + 0.5,
                f"n={n:,}\n~{N_wtd/1e6:.1f}M HH",
                va="center", fontsize=8.5, color=PALETTE["gray"])

    source_note(ax, "Ordered by weighted population (largest cluster on top).")
    fig.tight_layout()
    save_fig(fig, f"02e_cluster_profiles{suffix}.png")

    # ── THE KEY EXHIBIT: cluster composition by income quintile ───────────
    # This is the direct test of the Income Fallacy:
    # if high-spender clusters have substantial Q4/Q5 representation,
    # the hypothesis holds.
    print(f"\n  Cluster composition by income quintile:")
    ct = pd.crosstab(clust_df["INCOME_QUINTILE"],
                      clust_df["CLUSTER_LABEL"],
                      normalize="index") * 100
    # Reorder rows to Q1→Q5
    ct = ct.reindex([q for q in QUINTILE_LABELS if q in ct.index])
    print(ct.round(1).to_string())

    # Order columns (clusters) by total weighted population — most populous
    # cluster on the bottom of the stack so it forms the visual baseline.
    cluster_pop_order = (
        clust_df.groupby("CLUSTER_LABEL")["FINLWT21"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Reorder columns to most→least populous (bottom→top in the stack)
    ct = ct[[c for c in cluster_pop_order if c in ct.columns]]

    fig, ax = plt.subplots(figsize=(11, 6))
    cluster_colors_map = {
        lbl: [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"],
              PALETTE["red"], PALETTE["purple"], PALETTE["gray"]][i % 6]
        for i, lbl in enumerate(cluster_pop_order)
    }

    ct_short = ct.copy()
    ct_short.index = [QUINTILE_LABELS_SHORT[QUINTILE_LABELS.index(i)]
                       for i in ct_short.index]

    bottom = np.zeros(len(ct_short))
    for clust_lbl in ct_short.columns:
        color = cluster_colors_map.get(clust_lbl, PALETTE["gray"])
        vals  = ct_short[clust_lbl].values
        bars  = ax.bar(range(len(ct_short)), vals,
                       bottom=bottom, color=color,
                       alpha=0.87, edgecolor="white",
                       label=clust_lbl, width=0.6)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 6:
                ax.text(j, bottom[j] + val / 2,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(range(len(ct_short)))
    ax.set_xticklabels(ct_short.index, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([])
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Behavioral Cluster Composition by Income Quintile",
                 loc="left", fontsize=13)
    ax.annotate(
        "Each bar = 100% of that quintile's households.\n"
        "If 'High Spender' cluster is present in Q4 and Q5, that supports "
        "the Income Fallacy hypothesis.",
        xy=(0.5, 1.01), xycoords="axes fraction",
        fontsize=8.5, color=PALETTE["gray"], ha="center")
    ax.legend(title="Behavioral Cluster", fontsize=9,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    source_note(ax, "K-means clustering on behavioral variables. "
                    "Extreme ratios excluded.")
    fig.tight_layout()
    save_fig(fig, f"02e_cluster_by_quintile{suffix}.png")

    return clust_df, cluster_labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2f — CLUSTER-BASED ANOVA + PAIRWISE EFFECT SIZES
# ══════════════════════════════════════════════════════════════════════════════

def _run_cluster_cohens_d(df: pd.DataFrame, suffix: str = ""):
    """
    Pairwise Cohen's d heatmap across all behavioral clusters for each ratio.
    Complements the ANOVA by showing WHICH cluster pairs differ and by how much,
    rather than just confirming that differences exist.
    suffix: appended to output filename.
    """
    import seaborn as sns

    cluster_col   = "CLUSTER_LABEL"
    cluster_order = sorted(df[cluster_col].dropna().unique())
    n_c           = len(cluster_order)
    ratio_keys    = list(RATIOS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.flatten()

    for ax, col in zip(axes, ratio_keys):
        meta = get_ratio_meta(col)

        # Build n_c × n_c Cohen's d matrix (row - col direction)
        d_matrix = np.full((n_c, n_c), np.nan)
        for i, c1 in enumerate(cluster_order):
            for j, c2 in enumerate(cluster_order):
                if i == j:
                    continue
                g1 = df[df[cluster_col] == c1][col].dropna()
                g2 = df[df[cluster_col] == c2][col].dropna()
                if len(g1) > 1 and len(g2) > 1:
                    d_matrix[i, j] = cohen_d(g1, g2)

        # Symmetric color scale, min ±0.5 so color always has meaning
        abs_max = float(np.nanmax(np.abs(d_matrix)))
        vmax    = max(abs_max, 0.5)

        # Annotation strings: "+1.23" or "—" on diagonal
        annot = np.empty((n_c, n_c), dtype=object)
        for i in range(n_c):
            for j in range(n_c):
                if i == j:
                    annot[i, j] = "—"
                elif np.isnan(d_matrix[i, j]):
                    annot[i, j] = "n/a"
                else:
                    annot[i, j] = f"{d_matrix[i, j]:+.2f}"

        # Plot with diagonal zeroed so it renders as neutral center color
        d_plot = pd.DataFrame(d_matrix, index=cluster_order, columns=cluster_order)
        np.fill_diagonal(d_plot.values, 0.0)

        sns.heatmap(
            d_plot,
            ax=ax,
            annot=annot,
            fmt="",
            annot_kws={"size": 9, "weight": "bold"},
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Cohen's d", "shrink": 0.5},
        )

        # Gray out diagonal cells so "—" is visually distinct from near-zero d
        for i in range(n_c):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                        color="#EEEEEE", zorder=3))
            ax.text(i + 0.5, i + 0.5, "—",
                    ha="center", va="center",
                    fontsize=10, color=PALETTE["gray"], zorder=4)

        ax.set_title(meta["label"], loc="left", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(
        "Pairwise Cohen's d Between Behavioral Clusters\n"
        "Red = row cluster has HIGHER ratio than column cluster.  "
        "Blue = row cluster has LOWER ratio.  "
        "|d| < 0.2 trivial · 0.2–0.5 small · 0.5–0.8 medium · > 0.8 large",
        fontsize=10, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, f"02f_cluster_cohens_d{suffix}.png")


def run_cluster_anova(clust_df: pd.DataFrame, quintile_eta_sq: dict,
                      suffix: str = ""):
    """
    Re-run the same hypothesis tests as Section 2b but using behavioral
    cluster groupings instead of income quintiles.

    Theoretically more meaningful than quintile-based grouping because
    clusters reflect actual behavioral profiles. Comparing the effect
    sizes (cluster η² vs quintile η²) tells us how much more variance
    behavioral grouping captures than income grouping.

    Note: ratios were used as inputs to the clustering, so significance
    is essentially guaranteed. The interpretive value is in the EFFECT
    SIZES and post-hoc pairwise comparisons, not the p-values.

    suffix: appended to output filenames; "_restricted" for the second pass.
    """
    label = "RESTRICTED — " if suffix else ""
    sep(f"SECTION 2f — {label}CLUSTER-BASED ANOVA")

    # Apply the same caps used in Section 2b for consistency
    cap = 5.0
    df  = clust_df.copy()
    for col in RATIOS:
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    cluster_col = "CLUSTER_LABEL"
    cluster_order = sorted(df[cluster_col].dropna().unique())

    print(f"\n  Clusters tested: {cluster_order}")
    print(f"  Note: ratios were used as clustering inputs — effect sizes,")
    print(f"  not p-values, are the interpretive output.\n")

    header = (f"  {'Ratio':<20} {'F':>10} {'df':>6} {'p-value':>10} "
              f"{'Cluster η²':>12} {'Quintile η²':>13} {'Difference':>12}")
    print(header)
    sep()

    cluster_eta_sq = {}
    cluster_anova  = {}

    for col in RATIOS:
        meta = get_ratio_meta(col)
        sub = df[df[cluster_col].notna() & df[col].notna()].copy()
        sub["w_norm"] = normalize_weights(sub["FINLWT21"])

        model  = smf.wls(f"{col} ~ C({cluster_col})",
                         data=sub, weights=sub["w_norm"])
        result = model.fit()

        f_stat = result.fvalue
        f_p    = result.f_pvalue
        df_num = int(result.df_model)
        eta_sq = eta_squared_from_wls(result)

        cluster_eta_sq[col] = eta_sq
        cluster_anova[col]  = result

        q_eta = quintile_eta_sq.get(col, np.nan)
        diff  = eta_sq - q_eta if not np.isnan(q_eta) else np.nan

        print(f"  {meta['short']:<20} {f_stat:>10.2f} {df_num:>6} {f_p:>10.4f} "
              f"{eta_sq:>12.4f} {q_eta:>13.4f} {diff:>+12.4f}")

    # ── Effect size comparison chart: cluster vs quintile ─────────────────
    ratio_labels = [RATIOS[c]["short"] for c in RATIOS]
    cluster_vals = [cluster_eta_sq[c] for c in RATIOS]
    quintile_vals = [quintile_eta_sq.get(c, 0) for c in RATIOS]

    x = np.arange(len(ratio_labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w/2, quintile_vals, w,
                    color=PALETTE["blue"], alpha=0.85,
                    edgecolor="white", label="Income Quintile grouping")
    bars2 = ax.bar(x + w/2, cluster_vals, w,
                    color=PALETTE["orange"], alpha=0.85,
                    edgecolor="white", label="Behavioral Cluster grouping")

    for bars, vals in [(bars1, quintile_vals), (bars2, cluster_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    for thresh, label, ls in [(0.01, "Small", ":"),
                               (0.06, "Medium", "--"),
                               (0.14, "Large", "-.")]:
        ax.axhline(thresh, color=PALETTE["gray"], lw=1.0,
                   ls=ls, alpha=0.55, label=f"η² {label} ({thresh:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels, fontsize=10)
    ax.set_yticks([])
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("ANOVA Effect Size Comparison: Income Quintile vs Behavioral Cluster\n"
                 "Larger bars = grouping captures more variance in the ratio",
                 loc="left")
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    source_note(ax, "Cluster η² is naturally inflated since ratios were "
                    "clustering inputs. Compare relative magnitudes.")
    fig.tight_layout()
    save_fig(fig, f"02f_cluster_anova_effect_sizes{suffix}.png")

    # ── Cluster group means chart ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, RATIOS):
        meta = get_ratio_meta(col)
        means = []
        cis   = []
        labels = []

        for clust in cluster_order:
            sub = df[(df[cluster_col] == clust) & df[col].notna()].copy()
            if len(sub) == 0:
                continue
            sub["w_norm"] = normalize_weights(sub["FINLWT21"])
            wm = np.average(sub[col], weights=sub["w_norm"])
            # Bootstrap CI
            boot_means = []
            for _ in range(300):
                idx  = np.random.choice(len(sub), len(sub), replace=True)
                bsub = sub.iloc[idx]
                boot_means.append(np.average(bsub[col], weights=bsub["w_norm"]))
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
            means.append(wm)
            cis.append((wm - ci_lo, ci_hi - wm))
            labels.append(clust)

        x_pos  = np.arange(len(labels))
        means  = np.array(means)
        yerr   = np.array(cis).T

        # Color by cluster (consistent palette across all 4 panels)
        cluster_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"],
                          PALETTE["red"], PALETTE["purple"], PALETTE["gray"]]
        bar_colors = [cluster_colors[i % len(cluster_colors)]
                      for i in range(len(labels))]

        bars = ax.bar(x_pos, means, color=bar_colors,
                      alpha=0.82, edgecolor="white", width=0.6)
        ax.errorbar(x_pos, means, yerr=yerr, fmt="none",
                    color=PALETTE["dark"], capsize=4,
                    capthick=1.2, linewidth=1.2)

        for i, val in enumerate(means):
            ax.text(i, val + max(means) * 0.03, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks([])
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(meta["label"], loc="left", fontsize=11)
        ax.annotate(f"η²={cluster_eta_sq[col]:.4f}",
                    xy=(0.97, 0.97), xycoords="axes fraction",
                    fontsize=9, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=PALETTE["light_gray"], alpha=0.9))

    fig.suptitle("Weighted Group Means by Behavioral Cluster\n"
                 "Error bars = bootstrapped 95% CI (300 iterations).",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"02f_cluster_group_means{suffix}.png")

    _run_cluster_cohens_d(df, suffix=suffix)

    return cluster_eta_sq


# ══════════════════════════════════════════════════════════════════════════════
# RESTRICTED RE-ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_restricted_analysis(df: pd.DataFrame,
                             clust_df: pd.DataFrame,
                             quintile_eta_sq: dict):
    """
    Re-run clustering and regression after removing minority/outlier cluster
    households identified in the initial pass.

    Which clusters to drop is controlled by RESTRICTED_EXCLUDE_CLUSTER_INDICES
    (0-based).  The remaining ~91% of households are re-clustered from scratch
    (new elbow/silhouette/k selection) and all five regression models are re-fit
    on the same restricted sample.

    All outputs carry a '_restricted' filename suffix so they coexist with the
    initial-pass figures in outputs/figures/.
    """
    if not RESTRICTED_EXCLUDE_CLUSTER_INDICES:
        return

    sep("RESTRICTED RE-ANALYSIS — MAIN CLUSTERS ONLY")

    # ── Identify minority-cluster rows ────────────────────────────────────
    excl_idx    = set(RESTRICTED_EXCLUDE_CLUSTER_INDICES)
    minority_mask = clust_df["CLUSTER"].isin(excl_idx)
    minority_idx  = clust_df[minority_mask].index

    n_excl  = minority_mask.sum()
    n_keep  = len(clust_df) - n_excl
    pct_keep = n_keep / len(clust_df) * 100

    excl_labels = [clust_df.loc[clust_df["CLUSTER"] == i, "CLUSTER_LABEL"].iloc[0]
                   if (clust_df["CLUSTER"] == i).any() else f"Cluster {i+1}"
                   for i in sorted(excl_idx)]

    print(f"\n  Excluding {n_excl:,} rows from: {excl_labels}")
    print(f"  Restricted sample: {n_keep:,} rows ({pct_keep:.1f}% of clustered sample)")

    # ── Build restricted DataFrames ───────────────────────────────────────
    # For clustering: use the already-cleaned clust_df (extreme/invalid already removed)
    restricted_clust_input = clust_df[~minority_mask].copy()

    # For regression: filter the full analysis df by the same row indices
    # (minority_idx came from clust_df which is a subset of df)
    restricted_df = df[~df.index.isin(minority_idx)].copy()

    # ── Re-run clustering on restricted sample ────────────────────────────
    restricted_clust_df, _ = run_clustering(restricted_clust_input,
                                             suffix="_restricted")

    # ── Re-run cluster ANOVA on new clusters ─────────────────────────────
    run_cluster_anova(restricted_clust_df, quintile_eta_sq,
                      suffix="_restricted")

    # ── Re-run all five regression models on restricted sample ────────────
    run_regression(restricted_df, suffix="_restricted")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2():
    sep("PHASE 2 — OBSERVATIONAL STATISTICAL ANALYSES")

    print("\n[00] Preparing data...")
    df = prepare_data()
    df = get_analysis_sample(df)

    n_extreme = df.get("FLAG_EXTREME_RATIO", pd.Series(dtype=int)).sum() \
                if "FLAG_EXTREME_RATIO" in df.columns else 0
    print(f"  Analysis sample: {len(df):,} rows")
    print(f"  Extreme ratio rows (>5.0): {n_extreme:,} "
          f"(retained in ANOVA, excluded in regression/clustering)")

    # Run all sections
    assumption_results = run_assumption_checks(df)
    anova_results      = run_anova(df)
    posthoc_results    = run_posthoc(df, anova_results)
    regression_results = run_regression(df)
    clust_df, labels   = run_clustering(df)

    # Extract quintile η² from ANOVA results for cluster comparison
    quintile_eta_sq = {col: anova_results["anova"][col]["eta_sq"]
                       for col in RATIOS if col in anova_results["anova"]}

    cluster_eta_sq = run_cluster_anova(clust_df, quintile_eta_sq)

    # ── Restricted re-analysis (removes minority clusters, re-clusters, re-regresses)
    if RESTRICTED_EXCLUDE_CLUSTER_INDICES:
        run_restricted_analysis(df, clust_df, quintile_eta_sq)

    sep("PHASE 2 COMPLETE")
    n_figs = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
    print(f"\n  {n_figs} figures saved to {OUTPUT_DIR}/")
    print("\n  Key findings to bring into the report:")
    print("  1. ANOVA effect sizes — quintile η² vs cluster η² comparison")
    print("  2. LOG_INCOME coefficient across all five expense models")
    print("     (negative = discipline; for RETIRE_RATIO positive = more saving)")
    print("  3. Cluster composition chart — which quintiles contain which clusters?")
    print("  4. Restricted re-analysis — do patterns hold in the 91% main sample?")
    print("  5. Q1 sensitivity — do regression findings hold with Q1 excluded?\n")


if __name__ == "__main__":
    run_phase2()