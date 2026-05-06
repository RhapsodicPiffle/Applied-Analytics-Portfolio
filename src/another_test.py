import pandas as pd
import numpy as np
from data_prep import prepare_data, get_analysis_sample, SPEND_CATEGORIES, QUINTILE_LABELS

df = prepare_data()
adf = get_analysis_sample(df)

# ── Component breakdown ────────────────────────────────────────────────────
bases = list(SPEND_CATEGORIES.keys())
labels = list(SPEND_CATEGORIES.values())

rows = []
for base, label in zip(bases, labels):
    ann_col = f"{base}_ANN"
    if ann_col not in adf.columns:
        continue
    wtd_mean = np.average(adf[ann_col].fillna(0), weights=adf["FINLWT21"])
    rows.append({"Category": label, "Wtd Mean Annual ($)": wtd_mean})

summary = pd.DataFrame(rows)
total = summary["Wtd Mean Annual ($)"].sum()
summary["% of Total Spend"] = summary["Wtd Mean Annual ($)"] / total * 100
summary = summary.sort_values("Wtd Mean Annual ($)", ascending=False)
summary["Wtd Mean Annual ($)"] = summary["Wtd Mean Annual ($)"].map("${:,.0f}".format)
summary["% of Total Spend"] = summary["% of Total Spend"].map("{:.1f}%".format)
print(summary.to_string(index=False))

# ── Q1 expense ratio distribution ─────────────────────────────────────────
q1 = adf[adf["INCOME_QUINTILE"] == "Q1 (Bottom 20%)"]["EXPENSE_RATIO"].dropna()
print(f"\nQ1 expense ratio distribution:")
print(q1.describe(percentiles=[.25, .5, .75, .90, .95, .99]))
print(f"Q1 rows with ratio > 1.0: {(q1 > 1.0).sum()} ({(q1 > 1.0).mean():.1%})")
print(f"Q1 rows with ratio > 2.0: {(q1 > 2.0).sum()} ({(q1 > 2.0).mean():.1%})")
print(f"Q1 rows with ratio > 3.0: {(q1 > 3.0).sum()} ({(q1 > 3.0).mean():.1%})")

# ── Check transfer income proxies ──────────────────────────────────────────
print(f"\nWELFAREX in Q1 (non-zero rows):")
q1_welfare = adf[adf["INCOME_QUINTILE"] == "Q1 (Bottom 20%)"]
print(f"  Households with WELFAREX > 0: {(q1_welfare['WELFAREX'] > 0).sum()}")
print(f"  Mean WELFAREX where > 0: ${q1_welfare.loc[q1_welfare['WELFAREX'] > 0, 'WELFAREX'].mean():,.0f}")
print(f"\nFSSIX (Social Security) in Q1:")
print(f"  Households with FSSIX > 0: {(q1_welfare['FSSIX'] > 0).sum()}")
print(f"  Mean FSSIX where > 0: ${q1_welfare.loc[q1_welfare['FSSIX'] > 0, 'FSSIX'].mean():,.0f}")