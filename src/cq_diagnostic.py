"""
00_cq_diagnostic.py
───────────────────
One-time diagnostic: cross-tab zero patterns in Current Quarter (CQ)
expenditure columns against interview metadata to determine whether
CQ = 0 reflects missing / unreported data or true zero spending.

Run ONCE from project root before finalizing 00_data_prep.py:
    python src/00_cq_diagnostic.py

No files are written. Console output only.

How to use the results
──────────────────────
Read the output section by section and answer these three questions:

  Q1 — SECTION 2: Is FLAG_CQ_MISSING concentrated in specific INTERI values?
       YES (e.g., 80%+ of zeros in INTERI 2 or 3) → missing data hypothesis
           confirmed. Use conditional averaging: PQ only where TOTEXPCQ = 0.
       NO (zeros dispersed evenly across INTERI 1-5) → zeros are likely
           structural (true zero spenders or BLS design). Averaging is safer.

  Q2 — SECTION 3: Is FLAG_CQ_MISSING concentrated in specific interview months?
       YES (e.g., March has far more zeros than January) → timing / publication
           lag is the driver. Confirms missing data hypothesis.
       NO (roughly equal across months) → weakens the timing hypothesis.

  Q3 — SECTION 4: Do CQ zeros appear in housing and grocery columns?
       YES (HOUSCQ = 0 or GROCERCQ = 0 in meaningful numbers) → almost
           certainly missing data. No household has $0 housing for a quarter.
       NO (zeros only in plausible categories like tobacco, education, alcohol)
           → zeros may be true. Averaging is more defensible.

All three questions pointing the same direction gives you a clean decision.
Mixed signals means you document the ambiguity and use the conservative rule
(treat TOTEXPCQ = 0 as missing) with a sensitivity note in the report.
"""

import os
import sys
import pandas as pd
import numpy as np

DATA_PATH = "data/fmli251.csv"

# Spending pairs: (PQ col, CQ col, base name, display label)
# Mirrors SPEND_PAIRS in 00_data_prep.py — keep in sync if categories change
SPEND_PAIRS = [
    ("TOTEXPPQ",  "TOTEXPCQ",  "TOTEXP",  "Total Expenditures"),
    ("HOUSPQ",    "HOUSCQ",    "HOUS",    "Housing"),
    ("SHELTPQ",   "SHELTCQ",   "SHELT",   "Shelter"),
    ("UTILPQ",    "UTILCQ",    "UTIL",    "Utilities"),
    ("GROCERPQ",  "GROCERCQ",  "GROCER",  "Food at Home"),
    ("FDAWAYPQ",  "FDAWAYCQ",  "FDAWAY",  "Food Away from Home"),
    ("TRANSPQ",   "TRANSCQ",   "TRANS",   "Transportation"),
    ("GASMOPQ",   "GASMOCQ",   "GASMO",   "Gas and Motor Oil"),
    ("HEALTHPQ",  "HEALTHCQ",  "HEALTH",  "Healthcare"),
    ("HLTHINPQ",  "HLTHINCQ",  "HLTHIN",  "Health Insurance"),
    ("ENTERTPQ",  "ENTERTCQ",  "ENTERT",  "Entertainment"),
    ("EDUCAPQ",   "EDUCACQ",   "EDUCA",   "Education"),
    ("APPARPQ",   "APPARCQ",   "APPAR",   "Apparel"),
    ("TOBACCPQ",  "TOBACCCQ",  "TOBACC",  "Tobacco"),
    ("RETPENPQ",  "RETPENCQ",  "RETPEN",  "Retirement Contributions"),
    ("CASHCOPQ",  "CASHCOCQ",  "CASHCO",  "Cash Contributions"),
    ("ALCBEVPQ",  "ALCBEVCQ",  "ALCBEV",  "Alcohol"),
    ("MISCPQ",    "MISCCQ",    "MISC",    "Miscellaneous"),
]

SEP  = "=" * 70
SEP2 = "-" * 70


def load_raw(path: str) -> pd.DataFrame:
    """Load only the columns needed for this diagnostic."""
    all_cols = (
        ["NEWID", "INTERI", "QINTRVMO", "QINTRVYR", "FINLWT21"] +
        [pq for pq, cq, base, label in SPEND_PAIRS] +
        [cq for pq, cq, base, label in SPEND_PAIRS]
    )
    available = pd.read_csv(path, nrows=0).columns.tolist()
    to_load   = [c for c in all_cols if c in available]
    missing   = [c for c in all_cols if c not in available]

    df = pd.read_csv(path, usecols=to_load, low_memory=False)

    if missing:
        print(f"\n  NOTE: {len(missing)} columns absent from file (will be skipped):")
        for c in missing:
            print(f"    - {c}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERALL CQ ZERO SUMMARY
# How many households have TOTEXPCQ = 0, and what share is that?
# This is the top-level signal before any cross-tabs.
# ══════════════════════════════════════════════════════════════════════════════

def section_1_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("SECTION 1 — OVERALL CQ ZERO SUMMARY")
    print(SEP)

    n_total = len(df)

    # Household-level CQ availability flag
    if "TOTEXPCQ" in df.columns:
        df["FLAG_CQ_MISSING"] = (df["TOTEXPCQ"] == 0).astype(int)
        n_missing = df["FLAG_CQ_MISSING"].sum()
        n_present = n_total - n_missing
        print(f"\n  Total consumer units:          {n_total:>6,}")
        print(f"  TOTEXPCQ > 0  (CQ available):  {n_present:>6,}  ({n_present/n_total:.1%})")
        print(f"  TOTEXPCQ = 0  (CQ missing?):   {n_missing:>6,}  ({n_missing/n_total:.1%})")
    else:
        print("\n  TOTEXPCQ not found in file — cannot compute FLAG_CQ_MISSING.")
        df["FLAG_CQ_MISSING"] = 0

    # Per-category CQ zero rates — the first signal on which categories are affected
    print(f"\n  CQ zero rate by spending category:")
    print(f"  {'Category':<30} {'CQ col':<14} {'N zeros':>8} {'% zeros':>9} {'PQ nonzero when CQ=0':>22}")
    print(f"  {SEP2}")

    for pq_col, cq_col, base, label in SPEND_PAIRS:
        if cq_col not in df.columns:
            continue
        n_cq_zero      = (df[cq_col] == 0).sum()
        pct_cq_zero    = n_cq_zero / n_total
        # Among rows where CQ = 0, how many have PQ > 0?
        # This is the key diagnostic: PQ>0 and CQ=0 is the suspicious pattern
        if pq_col in df.columns:
            n_pq_nonzero_cq_zero = ((df[cq_col] == 0) & (df[pq_col] > 0)).sum()
            pq_flag = f"{n_pq_nonzero_cq_zero:>10,} ({n_pq_nonzero_cq_zero/n_total:.1%})"
        else:
            pq_flag = "PQ col absent"

        print(f"  {label:<30} {cq_col:<14} {n_cq_zero:>8,} {pct_cq_zero:>8.1%}   {pq_flag}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CROSS-TAB: FLAG_CQ_MISSING BY INTERI
# INTERI = interview number in the household's 5-interview rotation.
# If zeros cluster in early interview numbers (2-3), that strongly supports
# the "not yet collected" hypothesis.
# INTERI values: 1=bounding, 2=Q1 reporting, 3=Q2, 4=Q3, 5=Q4
# ══════════════════════════════════════════════════════════════════════════════

def section_2_by_interi(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("SECTION 2 — CQ MISSING BY INTERVIEW NUMBER (INTERI)")
    print(SEP)
    print("""
  INTERI interpretation:
    1 = Bounding interview (no expenditure data collected)
    2 = First reporting quarter
    3 = Second reporting quarter
    4 = Third reporting quarter
    5 = Fourth reporting quarter

  WHAT TO LOOK FOR:
    Missing data hypothesis  → zeros concentrated in INTERI 2 or 3
    True zero hypothesis     → zeros spread evenly across INTERI 2-5
    INTERI 1 should always have CQ = 0 (by design — no data collected)
""")

    if "INTERI" not in df.columns:
        print("  INTERI column not found — skipping section.")
        return

    ct = pd.crosstab(
        df["INTERI"],
        df["FLAG_CQ_MISSING"],
        margins=True,
        margins_name="Total"
    )
    ct.columns = ["CQ Available (>0)", "CQ Missing (=0)", "Row Total"]
    ct["% Missing"] = (ct["CQ Missing (=0)"] / ct["Row Total"] * 100).round(1)

    print(f"\n  {ct.to_string()}")

    # Flag the dominant INTERI for missing CQ
    interi_miss = df[df["FLAG_CQ_MISSING"] == 1]["INTERI"].value_counts()
    print(f"\n  CQ-missing rows by INTERI (count):")
    for interi_val, count in interi_miss.items():
        pct = count / interi_miss.sum() * 100
        bar = "#" * int(pct / 2)
        print(f"    INTERI {interi_val}: {count:>5,}  ({pct:.1f}%)  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CROSS-TAB: FLAG_CQ_MISSING BY INTERVIEW MONTH
# If zeros are more prevalent in later months (March vs January), that
# supports a publication lag / data-not-yet-finalized explanation.
# ══════════════════════════════════════════════════════════════════════════════

def section_3_by_month(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("SECTION 3 — CQ MISSING BY INTERVIEW MONTH (QINTRVMO)")
    print(SEP)
    print("""
  WHAT TO LOOK FOR:
    More zeros in March than January → timing / publication lag confirmed
    Roughly equal across months      → timing is not the primary driver
""")

    if "QINTRVMO" not in df.columns:
        print("  QINTRVMO column not found — skipping section.")
        return

    month_map = {
        1: "January", 2: "February", 3: "March",
        4: "April",   5: "May",      6: "June",
        7: "July",    8: "August",   9: "September",
        10: "October",11: "November",12: "December",
    }
    df["month_label"] = df["QINTRVMO"].map(month_map)

    ct = pd.crosstab(
        df["month_label"],
        df["FLAG_CQ_MISSING"],
        margins=True,
        margins_name="Total"
    )
    ct.columns = ["CQ Available (>0)", "CQ Missing (=0)", "Row Total"]
    ct["% Missing"] = (ct["CQ Missing (=0)"] / ct["Row Total"] * 100).round(1)

    # Sort by calendar month order
    month_order = [month_map[m] for m in sorted(df["QINTRVMO"].unique())] + ["Total"]
    ct = ct.reindex([m for m in month_order if m in ct.index])

    print(f"\n  {ct.to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CQ ZERO IN HOUSING AND GROCERY (NEAR-IMPOSSIBLE TRUE ZEROS)
# Housing and grocery spending should never be $0 for a full quarter for any
# functioning household. Zeros here are almost certainly missing data.
# This section quantifies how many such cases exist.
# ══════════════════════════════════════════════════════════════════════════════

def section_4_implausible_zeros(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("SECTION 4 — IMPLAUSIBLE CQ ZEROS (HOUSING AND GROCERY)")
    print(SEP)
    print("""
  Housing and grocery CQ = 0 when PQ > 0 is the strongest signal of
  missing data. No functioning household has $0 shelter or food for
  an entire quarter. Any such case should be treated as unreported.
""")

    implausible_pairs = [
        ("HOUSPQ",   "HOUSCQ",   "Housing"),
        ("SHELTPQ",  "SHELTCQ",  "Shelter"),
        ("GROCERPQ", "GROCERCQ", "Food at Home"),
        ("TRANSPQ",  "TRANSCQ",  "Transportation"),
        ("UTILPQ",   "UTILCQ",   "Utilities"),
        ("HEALTHPQ", "HEALTHCQ", "Healthcare"),
    ]

    n_total = len(df)
    print(f"  {'Category':<25} {'PQ>0 & CQ=0':>12} {'% of total':>12} {'Verdict'}")
    print(f"  {'-'*65}")

    for pq_col, cq_col, label in implausible_pairs:
        if pq_col not in df.columns or cq_col not in df.columns:
            continue
        mask = (df[pq_col] > 0) & (df[cq_col] == 0)
        n    = mask.sum()
        pct  = n / n_total
        if pct > 0.05:
            verdict = "<-- STRONG missing data signal"
        elif pct > 0.01:
            verdict = "<-- Moderate signal — investigate"
        else:
            verdict = "Low / expected"
        print(f"  {label:<25} {n:>12,} {pct:>11.1%}  {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PLAUSIBLE CQ ZEROS (CATEGORIES WHERE TRUE ZERO IS COMMON)
# Some categories frequently have legitimate zero spend: tobacco, alcohol,
# education, retirement contributions. Zeros here do not indicate missing data.
# This section confirms whether the zero pattern is category-specific or global.
# ══════════════════════════════════════════════════════════════════════════════

def section_5_plausible_zeros(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("SECTION 5 — PLAUSIBLE CQ ZEROS (DISCRETIONARY CATEGORIES)")
    print(SEP)
    print("""
  These categories are commonly zero even for households with full data.
  Compare their CQ zero rate against their PQ zero rate.
  If CQ zero rate >> PQ zero rate, missing data is still a concern.
  If CQ zero rate ~ PQ zero rate, zeros here are likely structural.
""")

    discretionary_pairs = [
        ("TOBACCPQ",  "TOBACCCQ",  "Tobacco"),
        ("ALCBEVPQ",  "ALCBEVCQ",  "Alcohol"),
        ("EDUCAPQ",   "EDUCACQ",   "Education"),
        ("RETPENPQ",  "RETPENCQ",  "Retirement Contributions"),
        ("CASHCOPQ",  "CASHCOCQ",  "Cash Contributions"),
        ("ENTERTPQ",  "ENTERTCQ",  "Entertainment"),
        ("APPARPQ",   "APPARCQ",   "Apparel"),
    ]

    n_total = len(df)
    print(f"  {'Category':<30} {'PQ=0 rate':>10} {'CQ=0 rate':>10} {'Difference':>12} {'Signal'}")
    print(f"  {'-'*72}")

    for pq_col, cq_col, label in discretionary_pairs:
        if pq_col not in df.columns or cq_col not in df.columns:
            continue
        pq_zero_rate = (df[pq_col] == 0).mean()
        cq_zero_rate = (df[cq_col] == 0).mean()
        diff         = cq_zero_rate - pq_zero_rate
        if diff > 0.10:
            signal = "<-- CQ has materially more zeros"
        elif diff > 0.03:
            signal = "<-- Small excess — worth noting"
        else:
            signal = "Rates similar — likely structural"
        print(f"  {label:<30} {pq_zero_rate:>9.1%}  {cq_zero_rate:>9.1%}  {diff:>+11.1%}  {signal}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INTERI x MONTH INTERACTION
# The combination of interview number AND month tells the most complete story.
# A household in INTERI=2 interviewed in March has had the least time for
# their data to be finalized — they should show the highest missing rate.
# ══════════════════════════════════════════════════════════════════════════════

def section_6_interi_month_interaction(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("SECTION 6 — INTERI x INTERVIEW MONTH INTERACTION")
    print(SEP)
    print("""
  WHAT TO LOOK FOR:
    INTERI=2 + March having the highest % missing → strongest possible
    confirmation of the timing / publication lag hypothesis.
    Flat pattern across cells → timing is not the driver.
""")

    if "INTERI" not in df.columns or "QINTRVMO" not in df.columns:
        print("  INTERI or QINTRVMO not found — skipping section.")
        return

    interaction = df.groupby(["INTERI", "QINTRVMO"])["FLAG_CQ_MISSING"].agg(
        ["sum", "count"]
    ).reset_index()
    interaction["pct_missing"] = (
        interaction["sum"] / interaction["count"] * 100
    ).round(1)
    interaction.columns = ["INTERI", "Month", "N Missing", "N Total", "% Missing"]

    # Pivot for readability
    pivot = interaction.pivot(index="INTERI", columns="Month", values="% Missing")
    pivot.columns = [f"Month {int(c)}" for c in pivot.columns]
    pivot.index   = [f"INTERI {int(i)}" for i in pivot.index]

    print(f"\n  % of households with CQ missing, by INTERI and interview month:\n")
    print(f"  {pivot.to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DECISION GUIDE
# Prints a plain-language interpretation framework based on what the data shows
# ══════════════════════════════════════════════════════════════════════════════

def section_7_decision_guide():
    print(f"\n{SEP}")
    print("SECTION 7 — DECISION GUIDE")
    print(SEP)
    print("""
  After reviewing Sections 1-6, apply this decision logic to update
  the averaging strategy in 00_data_prep.py:

  SCENARIO A — Missing data confirmed
    Signals: Zeros cluster in INTERI 2-3, zeros in housing/grocery,
             CQ zero rate >> PQ zero rate for core categories,
             more zeros in March than January.
    Action:  Use TOTEXPCQ = 0 as a household-level missing data flag.
             For those households: AVG_Q = PQ (do not average).
             For all others: AVG_Q = (PQ + CQ) / 2.
             Add FLAG_CQ_MISSING = 1 for affected households.
             Document in report: N households (X%) used single-quarter
             estimate due to incomplete current quarter data at publication.

  SCENARIO B — True zeros confirmed
    Signals: Zeros evenly distributed across INTERI, zeros only in
             plausible discretionary categories, housing/grocery CQ
             zeros are rare, CQ and PQ zero rates are similar.
    Action:  Average PQ and CQ normally for all households.
             Category-level zeros are genuine and should be included.
             Document assumption of consistent quarterly spending.

  SCENARIO C — Mixed signals
    Signals: Some missing data indicators but not all pointing same way.
    Action:  Use Scenario A (conservative) as the primary method.
             Run the analysis both ways as a sensitivity check.
             Report both sets of results and note the difference.
             If estimates are materially similar, the choice is defensible
             either way — document and move on.

  Once you have decided, return to 00_data_prep.py and update
  the averaging block in Step 3 accordingly.
""")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostic():
    print(SEP)
    print("CQ ZERO PATTERN DIAGNOSTIC — BLS CE FMLI 2025 Q1")
    print("Run this script once before finalizing 00_data_prep.py")
    print(SEP)

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    df = load_raw(DATA_PATH)

    df = section_1_overall_summary(df)
    section_2_by_interi(df)
    section_3_by_month(df)
    section_4_implausible_zeros(df)
    section_5_plausible_zeros(df)
    section_6_interi_month_interaction(df)
    section_7_decision_guide()


if __name__ == "__main__":
    run_diagnostic()