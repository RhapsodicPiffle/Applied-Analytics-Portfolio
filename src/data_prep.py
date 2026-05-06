"""
00_data_prep.py
───────────────
Phase 0: Data Preparation
BLS Consumer Expenditure Survey — FMLI 2025 Q1 (fmli251.csv)

Responsibilities:
  - Select the analytical column set from 781 raw columns
  - Decode all categorical variables using confirmed code schemes
  - Derive all computed columns (annualized spend, ratios, shares, flags)
  - Apply and document all exclusion decisions
  - Validate output and print a data quality report
  - Expose prepare_data() for import by downstream scripts

Importable interface:
    from src.data_prep import prepare_data
    df = prepare_data()

Standalone execution saves outputs/analytical_dataset.csv:
    python src/00_data_prep.py

──────────────────────────────────────────────────────────────
VARIABLE NAMING CONVENTIONS
──────────────────────────────────────────────────────────────
Raw CE file suffixes:
  *PQ  = Prior Quarter expenditure
  *CQ  = Current Quarter expenditure

Derived column suffixes:
  {BASE}_AVG_Q  = average quarterly spend (see averaging logic below)
  {BASE}_ANN    = {BASE}_AVG_Q * 4  (annualized estimate)
  {BASE}_SHARE  = {BASE}_AVG_Q / TOTEXP_AVG_Q  (composition share)

──────────────────────────────────────────────────────────────
AVERAGING LOGIC (updated after 00_cq_diagnostic.py results)
──────────────────────────────────────────────────────────────
Diagnostic finding: CQ = 0 for ALL January households (QINTRVMO == 1),
regardless of interview number. February and March households have
complete CQ data (0% missing). This is a calendar boundary artifact:
January interviews occur at the very start of Q1 2025. Q1 had not
concluded when BLS published this file, so CQ data for January
respondents does not yet exist.

This is structural — not a household-level data quality issue.

Averaging rule applied:
  QINTRVMO in (2, 3)  → AVG_Q = (PQ + CQ) / 2   (two quarters available)
  QINTRVMO == 1       → AVG_Q = PQ               (prior quarter only)

FLAG_CQ_MISSING is set on QINTRVMO == 1 (the structural cause), not on
TOTEXPCQ == 0 (the symptom). They are equivalent in this file but the
causal variable is more defensible and portable to other CE file quarters.

Annualization assumption for January households:
  ANN = PQ * 4 assumes consistent quarterly spending. This is documented
  and applied uniformly. Households flagged with FLAG_CQ_MISSING = 1
  can be isolated for sensitivity analysis in downstream scripts.

──────────────────────────────────────────────────────────────
CODING SCHEME DECISIONS (confirmed against actual data values)
──────────────────────────────────────────────────────────────
  EDUC_REF   — modern scheme (codes 0, 10-16). Legacy codes 1-7 absent.
  OCCUCOD1/2 — modern single-digit scheme (codes 1-15). Legacy two-digit absent.
  STATE      — standard FIPS codes (1-56). CE-legacy codes 58-95 absent.
  All others — single unambiguous scheme confirmed in dictionary.

──────────────────────────────────────────────────────────────
EXCLUSION DECISIONS (flagged, not silently dropped)
──────────────────────────────────────────────────────────────
  FINCBTAX < 0  — 2 rows (net business losses). Excluded from ratio
                  calculations. Retained in descriptive counts.
  FINCBTAX = 0  — 304 rows (zero reported income). Excluded from ratio
                  calculations. Retained in descriptive counts.
  TOTEXPPQ = 0  — No rows found in audit. Flag added for safety.
  FDHOMEPQ/CQ   — Both are 0.0 for all rows in this file. Known CE file
                  variant artifact. Use GROCERPQ/CQ for food-at-home.
                  FDHOME columns retained but excluded from analysis.
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


# ── File paths ────────────────────────────────────────────────────────────────

RAW_DATA_PATH   = "data/fmli251.csv"
OUTPUT_CSV_PATH = "outputs/analytical_dataset.csv"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — COLUMN SELECTION
# ══════════════════════════════════════════════════════════════════════════════
#
# The raw FMLI file contains 781 columns. We select a curated analytical set.
# Both PQ (Prior Quarter) and CQ (Current Quarter) columns are loaded for all
# spending categories. The averaging logic in Step 3 handles the January
# CQ-missing case conditionally rather than dropping those households.
#
# Columns intentionally excluded:
#   *_ (trailing underscore)  — imputation flag variants
#   FINCBTX1-5, FSALARY1-5   — income sub-components by interview wave
#   FDHOMEPQ / FDHOMECQ       — confirmed all-zero artifact; use GROCERPQ/CQ
#   ETOTAL*, TTOTAL*          — CE-computed aggregates that double-count PQ/CQ

COLS_ID = [
    "NEWID",        # Unique consumer unit identifier
    "CUID",         # CU identifier across quarters (links diary and interview)
    "QINTRVMO",     # Interview month — KEY variable for CQ averaging decision
    "QINTRVYR",     # Interview year
    "FINLWT21",     # Final calibrated survey weight — required for all estimates
    "INTERI",       # Interview number in household's 5-interview rotation
]

# BRR replicate weights for variance estimation (Balanced Repeated Replication)
# All 44 needed for survey-aware standard errors in Phase 2
COLS_WEIGHTS = [f"WTREP{str(i).zfill(2)}" for i in range(1, 45)]

COLS_INCOME = [
    "FINCBTAX",     # Total family income before taxes, last 12 months (annual)
    "INCLASS2",     # BLS pre-computed income class (1-7); primary grouping variable
    "INC_RANK",     # BLS pre-computed income percentile rank (0.0-1.0)
    "FSALARYX",     # Wage and salary income, annual
    "FSSIX",        # Social Security income, annual
    "WELFAREX",     # Public assistance / welfare income, annual
    "OTHRINCX",     # Other income sources, annual
    "NO_EARNR",     # Number of earners in the consumer unit
    "PRINEARN",     # Member number of the principal earner
]

# Spending categories — both PQ (Prior Quarter) and CQ (Current Quarter).
# Format: (PQ column, CQ column, base name, display label)
# CQ columns for January households will be zero by design (see module docstring).
# The averaging logic in Step 3 handles this conditionally.
SPEND_PAIRS = [
    # Total
    ("TOTEXPPQ",  "TOTEXPCQ",  "TOTEXP",  "Total Expenditures"),
    ("TOTEX4PQ",  "TOTEX4CQ",  "TOTEX4",  "Total Excl. Gifts"),
    # Housing
    ("HOUSPQ",    "HOUSCQ",    "HOUS",    "Housing"),
    ("SHELTPQ",   "SHELTCQ",   "SHELT",   "Shelter"),
    ("UTILPQ",    "UTILCQ",    "UTIL",    "Utilities"),
    ("MRTINTPQ",  "MRTINTCQ",  "MRTINT",  "Mortgage Interest"),
    ("RNTXRPPQ",  "RNTXRPCQ",  "RNTXRP",  "Rent Paid"),
    # Food
    ("GROCERPQ",  "GROCERCQ",  "GROCER",  "Food at Home"),
    ("FDAWAYPQ",  "FDAWAYCQ",  "FDAWAY",  "Food Away from Home"),
    # Transportation
    ("TRANSPQ",   "TRANSCQ",   "TRANS",   "Transportation"),
    ("GASMOPQ",   "GASMOCQ",   "GASMO",   "Gas and Motor Oil"),
    ("VEHFINPQ",  "VEHFINCQ",  "VEHFIN",  "Vehicle Finance Charges"),
    ("PUBTRAPQ",  "PUBTRACQ",  "PUBTRA",  "Public Transportation"),
    # Healthcare
    ("HEALTHPQ",  "HEALTHCQ",  "HEALTH",  "Healthcare"),
    ("HLTHINPQ",  "HLTHINCQ",  "HLTHIN",  "Health Insurance"),
    ("MEDSRVPQ",  "MEDSRVCQ",  "MEDSRV",  "Medical Services"),
    ("PREDRGPQ",  "PREDRGCQ",  "PREDRG",  "Prescription Drugs"),
    # Other categories
    ("ENTERTPQ",  "ENTERTCQ",  "ENTERT",  "Entertainment"),
    ("EDUCAPQ",   "EDUCACQ",   "EDUCA",   "Education"),
    ("APPARPQ",   "APPARCQ",   "APPAR",   "Apparel"),
    ("TOBACCPQ",  "TOBACCCQ",  "TOBACC",  "Tobacco"),
    ("RETPENPQ",  "RETPENCQ",  "RETPEN",  "Retirement Contributions"),
    ("LIFINSPQ",  "LIFINSCQ",  "LIFINS",  "Life Insurance"),
    ("CASHCOPQ",  "CASHCOCQ",  "CASHCO",  "Cash Contributions"),
    ("PERSCAPQ",  "PERSCACQ",  "PERSCA",  "Personal Care"),
    ("READPQ",    "READCQ",    "READ",    "Reading"),
    ("ALCBEVPQ",  "ALCBEVCQ",  "ALCBEV",  "Alcohol"),
    ("MISCPQ",    "MISCCQ",    "MISC",    "Miscellaneous"),
]

COLS_SPEND_PQ = [pq  for pq, cq, base, label in SPEND_PAIRS]
COLS_SPEND_CQ = [cq  for pq, cq, base, label in SPEND_PAIRS]

COLS_DEMOG = [
    "AGE_REF",      # Age of reference person
    "AGE2",         # Age of spouse (NaN for single-person CUs)
    "SEX_REF",      # Sex of reference person
    "SEX2",         # Sex of spouse (NaN if no spouse)
    "REF_RACE",     # Race of reference person
    "RACE2",        # Race of spouse (NaN if no spouse)
    "HISP_REF",     # Hispanic/Latino indicator, reference person
    "HISP2",        # Hispanic/Latino indicator, spouse
    "EDUC_REF",     # Education of reference person (modern scheme, codes 0/10-16)
    "HIGH_EDU",     # Highest education level anywhere in the CU
    "OCCUCOD1",     # Occupation of reference person (modern scheme, codes 1-15)
    "OCCUCOD2",     # Occupation of spouse (NaN if no spouse)
    "MARITAL1",     # Marital status of reference person
    "FAM_TYPE",     # Family type / household composition structure
    "FAM_SIZE",     # Total number of CU members
    "PERSLT18",     # Number of members under age 18
    "PERSOT64",     # Number of members over age 64
]

COLS_GEO = [
    "CUTENURE",     # Housing tenure (own w/ mortgage / own free / rent / other)
    "BLS_URBN",     # Urban (1) or rural (2)
    "REGION",       # Census region (1=NE, 2=MW, 3=S, 4=W)
    "STATE",        # FIPS state code (confirmed standard FIPS)
    "POPSIZE",      # MSA population size class
    "DIVISION",     # Census division
    "SMSASTAT",     # Metropolitan statistical area status
    "BEDROOMQ",     # Number of bedrooms
    "BATHRMQ",      # Number of complete bathrooms
    "UNISTRQ",      # Housing unit type (detached, apartment, mobile home, etc.)
]

COLS_ASSETS = [
    "NUM_AUTO",     # Total owned vehicles
    "VEHQ",         # Total vehicles (owned + leased)
    "ST_HOUS",      # Value of owned home
    "CREDITB",      # Credit card balance
    "LIQUIDB",      # Liquid assets (checking, savings, money market)
    "IRAB",         # IRA / Keogh account balance
    "STOCKB",       # Stocks, bonds, mutual funds balance
]

ALL_COLS = (
    COLS_ID +
    COLS_WEIGHTS +
    COLS_INCOME +
    COLS_SPEND_PQ +
    COLS_SPEND_CQ +
    COLS_DEMOG +
    COLS_GEO +
    COLS_ASSETS
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DECODE MAPS
# Confirmed against actual unique values in the raw data
# ══════════════════════════════════════════════════════════════════════════════

INCLASS2_MAP = {
    1: "Under $15,000",
    2: "$15,000-$29,999",
    3: "$30,000-$49,999",
    4: "$50,000-$69,999",
    5: "$70,000-$99,999",
    6: "$100,000-$149,999",
    7: "$150,000 and over",
}

INCLASS2_SHORT = {
    1: "<$15K",
    2: "$15-30K",
    3: "$30-50K",
    4: "$50-70K",
    5: "$70-100K",
    6: "$100-150K",
    7: "$150K+",
}

# EDUC_REF / HIGH_EDU: confirmed modern scheme (codes 0, 10-16)
EDUC_MAP = {
    0:  "Never attended",
    10: "Grades 1-8",
    11: "High school, no diploma",
    12: "High school graduate",
    13: "Some college, no degree",
    14: "Associate's degree",
    15: "Bachelor's degree",
    16: "Graduate / Professional degree",
}
EDUC_ORDER = [0, 10, 11, 12, 13, 14, 15, 16]

# OCCUCOD1 / OCCUCOD2: confirmed modern single-digit scheme (codes 1-15)
OCCUCOD_MAP = {
    1:  "Manager / Administrator",
    2:  "Teacher",
    3:  "Professional",
    4:  "Administrative / Clerical",
    5:  "Sales, retail",
    6:  "Sales, business services",
    7:  "Technician",
    8:  "Protective service",
    9:  "Private household service",
    10: "Other service",
    11: "Machine / transportation operator",
    12: "Construction / mechanics",
    13: "Farming",
    14: "Forestry / fishing / groundskeeping",
    15: "Armed forces",
}

OCCUCOD_BROAD = {
    1:  "White collar",    2:  "White collar",    3:  "White collar",
    4:  "White collar",    5:  "Sales / service",  6:  "Sales / service",
    7:  "Sales / service", 8:  "Sales / service",  9:  "Sales / service",
    10: "Sales / service", 11: "Blue collar",      12: "Blue collar",
    13: "Blue collar",     14: "Blue collar",       15: "Military",
}

MARITAL_MAP = {
    1: "Married", 2: "Widowed", 3: "Divorced",
    4: "Separated", 5: "Never married",
}

FAM_TYPE_MAP = {
    1: "Married couple only",
    2: "Married, children <6",
    3: "Married, children 6-17",
    4: "Married, children >17",
    5: "Other married couple",
    6: "Single father",
    7: "Single mother",
    8: "Single consumer",
    9: "Other family",
}

FAM_TYPE_BROAD = {
    1: "Married couple",
    2: "Married with children",  3: "Married with children",
    4: "Married with children",  5: "Married couple",
    6: "Single parent",          7: "Single parent",
    8: "Single consumer",        9: "Other family",
}

CUTENURE_MAP = {
    1: "Owned with mortgage", 2: "Owned, no mortgage",
    4: "Renter", 5: "Student housing", 6: "Other tenure",
}
CUTENURE_OWNER = {1: "Owner", 2: "Owner", 4: "Renter", 5: "Other", 6: "Other"}

REGION_MAP  = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
URBAN_MAP   = {1: "Urban", 2: "Rural"}
SEX_MAP     = {1: "Male", 2: "Female"}

RACE_MAP = {
    1: "White", 2: "Black / African American", 3: "Native American",
    4: "Asian", 5: "Pacific Islander", 6: "Multi-race",
}

HISP_MAP    = {1: "Hispanic / Latino", 2: "Not Hispanic / Latino"}

POPSIZE_MAP = {
    1: "Not in MSA", 2: "MSA under 1.5M", 3: "MSA 1.5M-4M",
    4: "MSA 4M+",    5: "MSA size unclassified",
}

UNISTRQ_MAP = {
    1: "Single-family detached", 2: "Single-family attached",
    3: "Apartment (2-4 units)",  4: "Apartment (5+ units)",
    5: "Mobile home / trailer",  6: "Other",
}

STATE_MAP = {
    1:  "Alabama",          2:  "Alaska",           4:  "Arizona",
    5:  "Arkansas",         6:  "California",        8:  "Colorado",
    9:  "Connecticut",      10: "Delaware",          11: "District of Columbia",
    12: "Florida",          13: "Georgia",           15: "Hawaii",
    16: "Idaho",            17: "Illinois",          18: "Indiana",
    19: "Iowa",             20: "Kansas",            21: "Kentucky",
    22: "Louisiana",        23: "Maine",             24: "Maryland",
    25: "Massachusetts",    26: "Michigan",          27: "Minnesota",
    28: "Mississippi",      29: "Missouri",          30: "Montana",
    31: "Nebraska",         32: "Nevada",            33: "New Hampshire",
    34: "New Jersey",       35: "New Mexico",        36: "New York",
    37: "North Carolina",   38: "North Dakota",      39: "Ohio",
    40: "Oklahoma",         41: "Oregon",            42: "Pennsylvania",
    44: "Rhode Island",     45: "South Carolina",    46: "South Dakota",
    47: "Tennessee",        48: "Texas",             49: "Utah",
    50: "Vermont",          51: "Virginia",          53: "Washington",
    54: "West Virginia",    55: "Wisconsin",         56: "Wyoming",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPENDING CATEGORY REFERENCE
# Single source of truth used by all downstream scripts
# ══════════════════════════════════════════════════════════════════════════════

# Base name -> display label for the 13 primary composition categories
SPEND_CATEGORIES = {
    "HOUS":    "Housing",
    "GROCER":  "Food at Home",
    "FDAWAY":  "Food Away from Home",
    "TRANS":   "Transportation",
    "HEALTH":  "Healthcare",
    "ENTERT":  "Entertainment",
    "EDUCA":   "Education",
    "APPAR":   "Apparel",
    "RETPEN":  "Retirement Contributions",
    "CASHCO":  "Cash Contributions",
    "TOBACC":  "Tobacco",
    "ALCBEV":  "Alcohol",
    "MISC":    "Miscellaneous",
}

SPEND_NECESSITY = {
    "HOUS":    "Necessity",
    "GROCER":  "Necessity",
    "FDAWAY":  "Discretionary",
    "TRANS":   "Necessity",
    "HEALTH":  "Necessity",
    "ENTERT":  "Discretionary",
    "EDUCA":   "Discretionary",
    "APPAR":   "Discretionary",
    "RETPEN":  "Savings / Investment",
    "CASHCO":  "Discretionary",
    "TOBACC":  "Discretionary",
    "ALCBEV":  "Discretionary",
    "MISC":    "Discretionary",
}

# Ordered labels for the five weighted income quintiles.
# Q1 = bottom 20% of households by survey-weighted income distribution.
# Q5 = top 20%. Boundaries are data-driven, not fixed dollar thresholds.
QUINTILE_LABELS = ["Q1 (Bottom 20%)", "Q2", "Q3", "Q4", "Q5 (Top 20%)"]
QUINTILE_LABELS_SHORT = ["Q1", "Q2", "Q3", "Q4", "Q5"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WEIGHTED STATISTICAL HELPERS
# Module-level functions used in prepare_data() and importable by downstream
# scripts so implementations are defined once and never duplicated.
# ══════════════════════════════════════════════════════════════════════════════

def weighted_quantile(values, weights, q):
    """
    Weighted q-th quantile (q in [0, 1]).

    Sorts values ascending, accumulates weights, returns the value at which
    cumulative weight first crosses q * total_weight.

    Used to compute data-driven income quintile breakpoints so each quintile
    represents an equal share of the survey-weighted population.
    """
    mask  = values.notna()
    v     = values[mask].values
    w     = weights[mask].values
    idx   = np.argsort(v)
    v_s   = v[idx]
    w_s   = w[idx]
    cumw  = np.cumsum(w_s)
    return float(v_s[np.searchsorted(cumw, q * cumw[-1])])


def weighted_gini(values, weights):
    """
    Weighted Gini coefficient (result in [0, 1]).

    Constructs the Lorenz curve — cumulative population share vs. cumulative
    income share — then returns:  Gini = 1 - 2 * area_under_Lorenz_curve

    The line of perfect equality has area = 0.5, yielding Gini = 0.
    Perfect inequality (one unit earns everything) yields Gini approaching 1.

    Previous implementation had an algebraic error in the numerator that
    produced negative values. This version uses np.trapz on the explicit
    Lorenz curve, which is transparent and easy to verify.

    Expected range for U.S. household income: roughly 0.45 to 0.55.
    """
    mask = values.notna() & (values >= 0)
    v    = values[mask].values
    w    = weights[mask].values

    idx    = np.argsort(v)
    v_s    = v[idx]
    w_s    = w[idx]

    # Normalize weights to sum to 1 (convert to population share fractions)
    w_norm = w_s / w_s.sum()

    # Weighted mean income
    w_mean = float(np.sum(w_norm * v_s))
    if w_mean == 0:
        return 0.0

    # Lorenz curve points
    cum_pop    = np.cumsum(w_norm)
    cum_income = np.cumsum(w_norm * v_s) / w_mean

    # Normalize to [0, 1] — last value should already equal 1.0
    cum_income = cum_income / cum_income[-1]

    # Prepend origin so trapezoid integration starts at (0, 0)
    lorenz_x = np.concatenate([[0.0], cum_pop])
    lorenz_y = np.concatenate([[0.0], cum_income])

    # Gini = 1 - 2 * (area under Lorenz curve)
    area = float(np.trapz(lorenz_y, lorenz_x))
    return float(1.0 - 2.0 * area)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CORE PREPARATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw CE FMLI file, apply all preparation steps, return clean DataFrame.

    Steps:
      1. Load selected analytical columns
      2. Gracefully resolve any missing columns
      3. Decode all categorical variables
      4. Conditionally average PQ and CQ, then annualize
      5. Derive ratios from annualized values
      6. Derive composition shares from averaged quarterly values
      7. Create binary flags and exclusion flags
      8. Create ordinal and log-transformed columns for regression
      9. Validate output

    Returns
    -------
    pd.DataFrame
        All raw source columns preserved alongside derived columns.
        No rows dropped — exclusion decisions surfaced as FLAG_ columns.
    """

    # ── Step 1: Load ──────────────────────────────────────────────────────
    print("[00] Loading raw data...")
    raw_cols     = pd.read_csv(path, nrows=0).columns.tolist()
    cols_to_load = [c for c in ALL_COLS if c in raw_cols]
    cols_missing = [c for c in ALL_COLS if c not in raw_cols]

    if cols_missing:
        print(f"  WARNING: {len(cols_missing)} requested columns absent from raw file:")
        for c in cols_missing:
            print(f"    - {c}")

    df = pd.read_csv(path, usecols=cols_to_load, low_memory=False)
    print(f"  OK: Loaded {len(df):,} rows x {len(df.columns)} columns")

    # ── Step 2: Decode categoricals ───────────────────────────────────────
    print("[00] Decoding categorical variables...")

    df["inclass_label"]   = df["INCLASS2"].map(INCLASS2_MAP)
    df["inclass_short"]   = df["INCLASS2"].map(INCLASS2_SHORT)
    df["educ_label"]      = df["EDUC_REF"].map(EDUC_MAP)
    df["high_educ_label"] = df["HIGH_EDU"].map(EDUC_MAP)
    df["occucod1_label"]  = df["OCCUCOD1"].map(OCCUCOD_MAP)
    df["occucod1_broad"]  = df["OCCUCOD1"].map(OCCUCOD_BROAD)
    df["marital_label"]   = df["MARITAL1"].map(MARITAL_MAP)
    df["fam_type_label"]  = df["FAM_TYPE"].map(FAM_TYPE_MAP)
    df["fam_type_broad"]  = df["FAM_TYPE"].map(FAM_TYPE_BROAD)
    df["tenure_label"]    = df["CUTENURE"].map(CUTENURE_MAP)
    df["tenure_owner"]    = df["CUTENURE"].map(CUTENURE_OWNER)
    df["region_label"]    = df["REGION"].map(REGION_MAP)
    # 72 rows have REGION = NaN and STATE = NaN simultaneously.
    # Confirmed via diagnostic: these are not Puerto Rico (FIPS 72) —
    # STATE is fully suppressed, indicating BLS withheld geography to
    # prevent identification of households in small or sparse areas.
    # Label them explicitly so they appear in descriptive counts rather
    # than silently disappearing from grouped tables.
    df["region_label"] = df["region_label"].fillna("Suppressed")
    df["urban_label"]     = df["BLS_URBN"].map(URBAN_MAP)
    df["sex_ref_label"]   = df["SEX_REF"].map(SEX_MAP)
    df["race_ref_label"]  = df["REF_RACE"].map(RACE_MAP)
    df["hisp_ref_label"]  = df["HISP_REF"].map(HISP_MAP)
    df["state_label"]     = df["STATE"].map(STATE_MAP)
    df["popsize_label"]   = df["POPSIZE"].map(POPSIZE_MAP)

    for raw_col, label_col, decode_map in [
        ("OCCUCOD2", "occucod2_label", OCCUCOD_MAP),
        ("RACE2",    "race2_label",    RACE_MAP),
        ("SEX2",     "sex2_label",     SEX_MAP),
        ("UNISTRQ",  "unistrq_label",  UNISTRQ_MAP),
    ]:
        if raw_col in df.columns:
            df[label_col] = df[raw_col].map(decode_map)

    # ── Step 3: Conditionally average PQ and CQ, then annualize ──────────
    #
    # Diagnostic finding (00_cq_diagnostic.py):
    #   All January households (QINTRVMO == 1) have CQ = 0 across every
    #   spending category. February and March households have complete CQ data.
    #   Root cause: Q1 2025 had not concluded when January interviews were
    #   conducted and this file was published. This is a calendar boundary
    #   artifact, not a household-level data quality issue.
    #
    # Rule applied:
    #   QINTRVMO in (2, 3)  -> AVG_Q = (PQ + CQ) / 2
    #     Two observed quarters are available. Averaging reduces noise from
    #     one-time large purchases hitting a single quarter.
    #
    #   QINTRVMO == 1       -> AVG_Q = PQ
    #     Only prior quarter is available. CQ is structurally absent.
    #     ANN = PQ * 4 assumes consistent quarterly spending — documented.
    #
    # FLAG_CQ_MISSING is set on QINTRVMO == 1 (the structural cause), not
    # TOTEXPCQ == 0 (the symptom). Equivalent in this file but the causal
    # variable is more defensible and portable to other CE quarterly files.
    #
    # Fallback for absent CQ columns (file-level, not household-level):
    #   If a CQ column is missing from the file entirely, fall back to PQ
    #   for all households and log the exception.

    print("[00] Averaging PQ / CQ and annualizing...")

    cq_available = df["QINTRVMO"] != 1     # True for February and March
    df["FLAG_CQ_MISSING"] = (~cq_available).astype("Int8")

    n_jan = df["FLAG_CQ_MISSING"].sum()
    n_feb_mar = (~df["FLAG_CQ_MISSING"].astype(bool)).sum()
    print(f"  CQ averaging method breakdown:")
    print(f"    January  (PQ only, FLAG_CQ_MISSING=1): {n_jan:,} households ({n_jan/len(df):.1%})")
    print(f"    Feb/Mar  ((PQ + CQ) / 2):              {n_feb_mar:,} households ({n_feb_mar/len(df):.1%})")

    file_level_fallbacks = []

    for pq_col, cq_col, base, label in SPEND_PAIRS:
        avg_col = f"{base}_AVG_Q"
        ann_col = f"{base}_ANN"

        pq_present = pq_col in df.columns
        cq_present = cq_col in df.columns

        if not pq_present:
            # PQ absent — cannot proceed for this category
            # Already surfaced in cols_missing above; skip silently here
            continue

        if pq_present and cq_present:
            # Standard path: conditional on interview month
            df[avg_col] = np.where(
                cq_available,
                (df[pq_col] + df[cq_col]) / 2,   # Feb/Mar: average both quarters
                df[pq_col]                          # Jan: prior quarter only
            )
        else:
            # CQ column absent from file entirely (file-level gap, not Jan artifact)
            df[avg_col] = df[pq_col]
            file_level_fallbacks.append(
                (base, label, f"{cq_col} absent from file — using {pq_col} only")
            )

        df[ann_col] = df[avg_col] * 4

    if file_level_fallbacks:
        print(f"  WARNING: {len(file_level_fallbacks)} categories missing CQ column entirely:")
        for base, label, reason in file_level_fallbacks:
            print(f"    - {label}: {reason}")

    # ── Step 4: Derived ratios ─────────────────────────────────────────────
    #
    # All ratios compare annualized expenditure against annual income (FINCBTAX).
    # NaN where FINCBTAX <= 0 — mathematically undefined, not imputation.
    # FLAG_INCOME_INVALID identifies which rows are affected.

    print("[00] Deriving ratios...")
    income_valid = df["FINCBTAX"] > 0

    def safe_ratio(ann_series: pd.Series) -> np.ndarray:
        """Annualized expenditure / FINCBTAX; NaN where income <= 0."""
        return np.where(income_valid, ann_series / df["FINCBTAX"], np.nan)

    if "TOTEXP_ANN" in df.columns:
        df["EXPENSE_RATIO"] = safe_ratio(df["TOTEXP_ANN"])

    ratio_targets = {
        "HOUSING_RATIO":   "HOUS_ANN",
        "TRANSPORT_RATIO": "TRANS_ANN",
        "HEALTH_RATIO":    "HEALTH_ANN",
        "RETIRE_RATIO":    "RETPEN_ANN",
    }
    for ratio_col, ann_col in ratio_targets.items():
        if ann_col in df.columns:
            df[ratio_col] = safe_ratio(df[ann_col])

    # Combined food ratio (home + away)
    if "GROCER_ANN" in df.columns and "FDAWAY_ANN" in df.columns:
        df["FOOD_RATIO"] = safe_ratio(df["GROCER_ANN"] + df["FDAWAY_ANN"])

    # ── Lifestyle ratio — purely discretionary spending ────────────────────
    # Components: entertainment, alcohol, tobacco, apparel, food away from
    # home, and cash contributions. These six categories are defensibly
    # discretionary for all households regardless of family size, income
    # level, or geography. No household *needs* to spend on entertainment
    # or alcohol — spending here reflects behavioral choice, not structural
    # constraint. This makes LIFESTYLE_RATIO the cleanest behavioral signal
    # for the Income Fallacy hypothesis.
    #
    # Deliberately excluded from the lifestyle bucket:
    #   Housing, food at home, transportation, healthcare — all have a
    #   genuine necessity floor that varies by household composition.
    #   Retirement contributions — savings behavior, not consumption.
    #   Miscellaneous — too heterogeneous to classify confidently.
    #
    # LIFESTYLE_ANN = sum of the six discretionary category annuals.
    # LIFESTYLE_RATIO = LIFESTYLE_ANN / FINCBTAX (NaN where income <= 0).
    lifestyle_components = [
        "ENTERT_ANN",   # Entertainment
        "ALCBEV_ANN",   # Alcohol
        "TOBACC_ANN",   # Tobacco
        "APPAR_ANN",    # Apparel
        "FDAWAY_ANN",   # Food away from home
        "CASHCO_ANN",   # Cash contributions (charity, gifts)
    ]
    present_lifestyle = [c for c in lifestyle_components if c in df.columns]
    missing_lifestyle = [c for c in lifestyle_components if c not in df.columns]
    if missing_lifestyle:
        print(f"  WARNING: {len(missing_lifestyle)} lifestyle components absent "
              f"from file: {missing_lifestyle}")
    if present_lifestyle:
        df["LIFESTYLE_ANN"]   = df[present_lifestyle].fillna(0).sum(axis=1)
        df["LIFESTYLE_RATIO"] = safe_ratio(df["LIFESTYLE_ANN"])
        df["LIFESTYLE_SHARE"] = np.where(
            df["TOTEXP_AVG_Q"] > 0,
            df[[c.replace("_ANN", "_AVG_Q") for c in present_lifestyle
                if c.replace("_ANN", "_AVG_Q") in df.columns]].fillna(0).sum(axis=1)
            / df["TOTEXP_AVG_Q"],
            np.nan
        )
        print(f"  LIFESTYLE_RATIO derived from {len(present_lifestyle)} components: "
              f"{', '.join(c.replace('_ANN','') for c in present_lifestyle)}")

    # ── Four-bucket expense decomposition ──────────────────────────────────
    # These four buckets are mutually exclusive and collectively sum to
    # approximately TOTEXP_ANN. Designed so each bucket can be used as an
    # independent clustering variable without double-counting.
    #
    # HOUSING_RATIO — already computed above from HOUS_ANN
    #   Components: shelter, utilities, mortgage interest, rent, maintenance
    #
    # NECESSITY_RATIO — non-housing structural costs
    #   Components: food at home, transportation, healthcare, life insurance
    #   Life insurance is included here (not with retirement) because it is
    #   a household financial obligation, not a voluntary savings vehicle.
    #
    # DISCRETIONARY_RATIO — behavioral spending choices
    #   Components: food away from home, entertainment, apparel, alcohol,
    #   tobacco, education, cash contributions, personal care, reading, misc
    #   Education is included here — most CE respondents reporting education
    #   spending are paying for elective/post-secondary, not K-12 (tax-funded).
    #   This assignment is documented for the final report.
    #
    # RETIRE_RATIO — already computed above from RETPEN_ANN
    #   Components: retirement and pension contributions (includes FICA,
    #   mandatory pension deductions, and voluntary 401k/IRA contributions)
    #
    # Math: HOUSING + NECESSITY + DISCRETIONARY + RETIREMENT ≈ TOTAL EXPENSES
    # Any small residual is from the PQ/CQ averaging method.

    # ── Necessity bucket ───────────────────────────────────────────────────
    necessity_components = [
        "GROCER_ANN",   # Food at home
        "TRANS_ANN",    # Transportation
        "HEALTH_ANN",   # Healthcare
        "LIFINS_ANN",   # Life insurance
    ]
    present_necessity = [c for c in necessity_components if c in df.columns]
    missing_necessity = [c for c in necessity_components if c not in df.columns]
    if missing_necessity:
        print(f"  WARNING: {len(missing_necessity)} necessity components absent: "
              f"{missing_necessity}")
    if present_necessity:
        df["NECESSITY_ANN"]   = df[present_necessity].fillna(0).sum(axis=1)
        df["NECESSITY_RATIO"] = safe_ratio(df["NECESSITY_ANN"])
        if "TOTEXP_AVG_Q" in df.columns:
            avg_q_cols = [c.replace("_ANN", "_AVG_Q") for c in present_necessity
                          if c.replace("_ANN", "_AVG_Q") in df.columns]
            if avg_q_cols:
                df["NECESSITY_SHARE"] = np.where(
                    df["TOTEXP_AVG_Q"] > 0,
                    df[avg_q_cols].fillna(0).sum(axis=1) / df["TOTEXP_AVG_Q"],
                    np.nan
                )
        print(f"  NECESSITY_RATIO derived from {len(present_necessity)} components: "
              f"{', '.join(c.replace('_ANN','') for c in present_necessity)}")

    # ── Discretionary bucket ───────────────────────────────────────────────
    discretionary_components = [
        "FDAWAY_ANN",   # Food away from home
        "ENTERT_ANN",   # Entertainment
        "APPAR_ANN",    # Apparel
        "ALCBEV_ANN",   # Alcohol
        "TOBACC_ANN",   # Tobacco
        "EDUCA_ANN",    # Education (see note above)
        "CASHCO_ANN",   # Cash contributions (charity, gifts)
        "PERSCA_ANN",   # Personal care
        "READ_ANN",     # Reading
        "MISC_ANN",     # Miscellaneous
    ]
    present_discretionary = [c for c in discretionary_components if c in df.columns]
    missing_discretionary = [c for c in discretionary_components if c not in df.columns]
    if missing_discretionary:
        print(f"  WARNING: {len(missing_discretionary)} discretionary components absent: "
              f"{missing_discretionary}")
    if present_discretionary:
        df["DISCRETIONARY_ANN"]   = df[present_discretionary].fillna(0).sum(axis=1)
        df["DISCRETIONARY_RATIO"] = safe_ratio(df["DISCRETIONARY_ANN"])
        if "TOTEXP_AVG_Q" in df.columns:
            avg_q_cols = [c.replace("_ANN", "_AVG_Q") for c in present_discretionary
                          if c.replace("_ANN", "_AVG_Q") in df.columns]
            if avg_q_cols:
                df["DISCRETIONARY_SHARE"] = np.where(
                    df["TOTEXP_AVG_Q"] > 0,
                    df[avg_q_cols].fillna(0).sum(axis=1) / df["TOTEXP_AVG_Q"],
                    np.nan
                )
        print(f"  DISCRETIONARY_RATIO derived from {len(present_discretionary)} components: "
              f"{', '.join(c.replace('_ANN','') for c in present_discretionary)}")

    # ── Decomposition validation ───────────────────────────────────────────
    # Check that the four buckets approximately sum to total expenses
    if all(c in df.columns for c in ["HOUS_ANN", "NECESSITY_ANN",
                                      "DISCRETIONARY_ANN", "RETPEN_ANN",
                                      "TOTEXP_ANN"]):
        decomp_sum = (df["HOUS_ANN"] + df["NECESSITY_ANN"] +
                      df["DISCRETIONARY_ANN"] + df["RETPEN_ANN"])
        residual_pct = ((decomp_sum - df["TOTEXP_ANN"]).abs() /
                        df["TOTEXP_ANN"].clip(lower=1)).median() * 100
        print(f"  Decomposition residual (median): {residual_pct:.1f}% of total expenses")
        if residual_pct > 5:
            print(f"  WARNING: residual exceeds 5% — some categories may be missing")

    # ── Step 5: Spending composition shares ───────────────────────────────
    #
    # Share of average quarterly spend allocated to each category.
    # Denominator is TOTEXP_AVG_Q so shares sum to approximately 1.
    # NaN where TOTEXP_AVG_Q = 0.

    print("[00] Deriving spending composition shares...")
    if "TOTEXP_AVG_Q" in df.columns:
        spend_valid = df["TOTEXP_AVG_Q"] > 0
        for base in SPEND_CATEGORIES:
            avg_col   = f"{base}_AVG_Q"
            share_col = f"{base}_SHARE"
            if avg_col in df.columns:
                df[share_col] = np.where(
                    spend_valid,
                    df[avg_col] / df["TOTEXP_AVG_Q"],
                    np.nan
                )

        # Combined food share (home + away)
        if "GROCER_AVG_Q" in df.columns and "FDAWAY_AVG_Q" in df.columns:
            df["FOOD_TOTAL_SHARE"] = np.where(
                spend_valid,
                (df["GROCER_AVG_Q"] + df["FDAWAY_AVG_Q"]) / df["TOTEXP_AVG_Q"],
                np.nan
            )

    # ── Step 6: Binary and exclusion flags ────────────────────────────────
    print("[00] Creating flags...")

    # Household characteristic flags
    df["HOMEOWNER"]     = df["CUTENURE"].isin([1, 2]).astype("Int8")
    df["HAS_CHILDREN"]  = (df["PERSLT18"] > 0).astype("Int8")
    df["HAS_ELDERLY"]   = (df["PERSOT64"] > 0).astype("Int8")
    df["SINGLE_EARNER"] = (df["NO_EARNR"] == 1).astype("Int8")
    df["ZERO_EARNER"]   = (df["NO_EARNR"] == 0).astype("Int8")

    # Housing cost burden flags (HUD standard thresholds)
    if "HOUSING_RATIO" in df.columns:
        df["HOUSING_BURDENED"] = np.where(
            df["HOUSING_RATIO"].notna(),
            (df["HOUSING_RATIO"] > 0.30).astype("Int8"),
            pd.NA
        )
        df["HOUSING_SEVERE"] = np.where(
            df["HOUSING_RATIO"].notna(),
            (df["HOUSING_RATIO"] > 0.50).astype("Int8"),
            pd.NA
        )

    # Geography suppression flag
    # 72 households have REGION = NaN and STATE = NaN — BLS-suppressed for
    # privacy. These rows are valid for income, spending, and behavioral
    # analysis. Exclude with FLAG_REGION_UNKNOWN == 1 only when region is
    # a required variable (e.g., region-stratified tables). Retain in all
    # other analyses including income quintiles and clustering.
    df["FLAG_REGION_UNKNOWN"] = df["REGION"].isna().astype("Int8")

    # Income exclusion flags — no rows dropped here.
    # Downstream scripts apply via get_analysis_sample().
    df["FLAG_NEG_INCOME"]     = (df["FINCBTAX"] < 0).astype("Int8")
    df["FLAG_ZERO_INCOME"]    = (df["FINCBTAX"] == 0).astype("Int8")
    df["FLAG_INCOME_INVALID"] = (df["FINCBTAX"] <= 0).astype("Int8")

    if "TOTEXP_AVG_Q" in df.columns:
        df["FLAG_ZERO_SPEND"] = (df["TOTEXP_AVG_Q"] == 0).astype("Int8")

    if "EXPENSE_RATIO" in df.columns:
        df["FLAG_EXTREME_RATIO"] = np.where(
            df["EXPENSE_RATIO"].notna(),
            (df["EXPENSE_RATIO"] > 5.0).astype("Int8"),
            pd.NA
        )

    # ── Step 7: Variables for regression and grouping ────────────────────

    # Ordinal education encoding (ascending rank, reference = HS grad = 3)
    educ_int_map       = {code: rank for rank, code in enumerate(EDUC_ORDER)}
    df["EDUC_REF_ORD"] = df["EDUC_REF"].map(educ_int_map)

    # Log10 income — undefined for income <= 0
    df["LOG_INCOME"] = np.where(
        df["FINCBTAX"] > 0,
        np.log10(df["FINCBTAX"]),
        np.nan
    )

    # ── Weighted income quintiles ──────────────────────────────────────────
    # Quintile breakpoints are derived from the weighted income distribution
    # so that each quintile represents an equal share of the survey-weighted
    # population — not an equal number of rows.
    #
    # Only households with FINCBTAX > 0 are used to compute breakpoints.
    # The zero/negative income households receive INCOME_QUINTILE = NaN.
    #
    # Why weighted quintiles instead of BLS INCLASS2:
    #   INCLASS2 uses fixed dollar thresholds that don't adapt to the sample
    #   distribution. The $150K+ bucket is an uncapped catch-all. Weighted
    #   quintiles produce five groups of equal population weight with
    #   data-driven boundaries, making cross-quintile comparisons clean.
    income_pos = df.loc[df["FINCBTAX"] > 0, "FINCBTAX"]
    weight_pos = df.loc[df["FINCBTAX"] > 0, "FINLWT21"]
    breaks     = [weighted_quantile(income_pos, weight_pos, q)
                  for q in [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]]

    # Ensure strictly increasing breaks — floating point ties at extremes
    # can produce duplicate edges, which pd.cut rejects
    breaks = sorted(set(breaks))
    if len(breaks) < 6:
        raise ValueError(
            f"Weighted quintile breaks collapsed to {len(breaks)} unique values. "
            "Check FINCBTAX distribution."
        )

    df["INCOME_QUINTILE"] = pd.cut(
        df["FINCBTAX"],
        bins=breaks,
        labels=QUINTILE_LABELS,
        include_lowest=True,
    )
    # Zero/negative income rows land outside the bins — already NaN by default

    # Store the breakpoints as a module-level artifact for report reference
    df.attrs["quintile_breaks"] = breaks

    n_quintile_null = df["INCOME_QUINTILE"].isnull().sum()
    print(f"  Weighted quintile breakpoints: "
          f"{', '.join(f'${b:,.0f}' for b in breaks)}")
    print(f"  INCOME_QUINTILE nulls (income <= 0): {n_quintile_null}")

    # ── Step 8: Validate ──────────────────────────────────────────────────
    _validate(df)

    print(f"  OK: Preparation complete: {len(df):,} rows x {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _validate(df: pd.DataFrame) -> None:
    """
    Assert key invariants on the prepared dataset.
    Raises ValueError immediately so errors surface before any analysis runs.
    """
    print("[00] Validating...")
    errors = []

    # Row count must be preserved — no drops in prep
    if len(df) < 4500:
        errors.append(f"Row count {len(df):,} unexpectedly low (expected ~4,602).")

    # Survey weight is mandatory and must be strictly positive
    if "FINLWT21" not in df.columns:
        errors.append("FINLWT21 (survey weight) missing — all statistics invalid.")
    elif (df["FINLWT21"] <= 0).any():
        errors.append(f"{(df['FINLWT21'] <= 0).sum()} rows have FINLWT21 <= 0.")

    # INCLASS2 must be codes 1-7 only
    unexpected_cls = set(df["INCLASS2"].dropna().unique()) - {1, 2, 3, 4, 5, 6, 7}
    if unexpected_cls:
        errors.append(f"Unexpected INCLASS2 values: {unexpected_cls}")

    # EDUC_REF must use modern scheme only
    unexpected_educ = set(df["EDUC_REF"].dropna().unique()) - {0, 10, 11, 12, 13, 14, 15, 16}
    if unexpected_educ:
        errors.append(f"Unexpected EDUC_REF values (legacy codes?): {unexpected_educ}")

    # QINTRVMO must be 1, 2, or 3 for this file
    unexpected_mo = set(df["QINTRVMO"].dropna().unique()) - {1, 2, 3}
    if unexpected_mo:
        errors.append(f"Unexpected QINTRVMO values: {unexpected_mo}")

    # FLAG_CQ_MISSING must equal exactly the January count
    n_jan           = (df["QINTRVMO"] == 1).sum()
    n_flag_missing  = df["FLAG_CQ_MISSING"].sum()
    if n_jan != n_flag_missing:
        errors.append(
            f"FLAG_CQ_MISSING ({n_flag_missing}) does not match "
            f"January household count ({n_jan})."
        )

    # AVG_Q columns must be non-negative
    avg_cols = [c for c in df.columns if c.endswith("_AVG_Q")]
    for col in avg_cols:
        if (df[col].dropna() < 0).any():
            errors.append(f"Negative values in averaged column {col}.")

    # Expense ratio must be non-negative where defined
    if "EXPENSE_RATIO" in df.columns:
        if (df["EXPENSE_RATIO"].dropna() < 0).any():
            errors.append("Negative EXPENSE_RATIO values detected.")

    # Spending shares must be in [0, 1]
    share_cols = [c for c in df.columns if c.endswith("_SHARE")]
    for col in share_cols:
        valid = df[col].dropna()
        if (valid < 0).any() or (valid > 1.001).any():
            errors.append(f"Share column {col} has values outside [0, 1].")

    # Income exclusion flag counts must match audit findings
    if df["FLAG_NEG_INCOME"].sum() > 10:
        errors.append(f"FLAG_NEG_INCOME = {df['FLAG_NEG_INCOME'].sum()} (expected ~2).")
    if df["FLAG_ZERO_INCOME"].sum() > 500:
        errors.append(f"FLAG_ZERO_INCOME = {df['FLAG_ZERO_INCOME'].sum()} (expected ~304).")

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    print("  OK: All validation checks passed")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATA QUALITY CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_quality_report(df: pd.DataFrame) -> None:
    """Print structured data quality summary for development review."""
    total_w = df["FINLWT21"].sum()

    print("\n" + "=" * 65)
    print("DATA QUALITY REPORT — BLS CE FMLI 2025 Q1")
    print("=" * 65)

    print(f"\n  Consumer units (rows):             {len(df):,}")
    print(f"  Survey-weighted households:        {total_w:,.0f}")
    print(f"  Analytical columns:                {len(df.columns)}")
    print(f"  Interview months:                  {sorted(df['QINTRVMO'].unique())}")
    print(f"  Interview year:                    {df['QINTRVYR'].unique()[0]}")

    print("\n-- CQ Averaging Method Applied --------------------------------")
    n_jan     = df["FLAG_CQ_MISSING"].sum()
    n_feb_mar = len(df) - n_jan
    print(f"  January  (PQ only):       {n_jan:>6,}  ({n_jan/len(df):.1%})  FLAG_CQ_MISSING = 1")
    print(f"  Feb/Mar  (PQ + CQ) / 2:  {n_feb_mar:>6,}  ({n_feb_mar/len(df):.1%})  FLAG_CQ_MISSING = 0")
    print(f"  Root cause: Q1 2025 CQ data not yet published for January interviews.")

    print("\n-- Income -----------------------------------------------------")
    print(f"  Range:   ${df['FINCBTAX'].min():>12,.0f}  to  ${df['FINCBTAX'].max():,.0f}")
    print(f"  Median:  ${df['FINCBTAX'].median():>12,.0f}")
    print(f"  Mean:    ${df['FINCBTAX'].mean():>12,.0f}")
    print(f"  Negative income:          {df['FLAG_NEG_INCOME'].sum():>5}  rows  (FLAG_NEG_INCOME)")
    print(f"  Zero income:              {df['FLAG_ZERO_INCOME'].sum():>5}  rows  (FLAG_ZERO_INCOME)")
    print(f"  Invalid for ratios:       {df['FLAG_INCOME_INVALID'].sum():>5}  rows  excluded from ratio calcs")

    print("\n-- Averaged Quarterly Expenditures ----------------------------")
    if "TOTEXP_AVG_Q" in df.columns:
        print(f"  TOTEXP_AVG_Q range:  ${df['TOTEXP_AVG_Q'].min():,.0f}  to  ${df['TOTEXP_AVG_Q'].max():,.0f}")
        print(f"  TOTEXP_AVG_Q median: ${df['TOTEXP_AVG_Q'].median():,.0f}")
        if "FLAG_ZERO_SPEND" in df.columns:
            print(f"  Zero spend rows:     {df['FLAG_ZERO_SPEND'].sum()}")

    print("\n-- Key Derived Variables --------------------------------------")
    ratio_cols = {
        "EXPENSE_RATIO":       "Total expense ratio",
        "HOUSING_RATIO":       "Housing ratio",
        "NECESSITY_RATIO":     "Necessities ratio",
        "DISCRETIONARY_RATIO": "Discretionary ratio",
        "RETIRE_RATIO":        "Retirement ratio",
        "LIFESTYLE_RATIO":     "Lifestyle ratio (original 6-component)",
    }
    for col, label in ratio_cols.items():
        if col in df.columns:
            valid = df[col].dropna()
            print(f"  {label:<35} median={valid.median():.3f}  mean={valid.mean():.3f}  n={len(valid):,}")
    if "HOUSING_BURDENED" in df.columns:
        print(f"  Housing burdened (>30%):              {df['HOUSING_BURDENED'].sum():,} rows")
        print(f"  Severely burdened (>50%):             {df['HOUSING_SEVERE'].sum():,} rows")

    print("\n-- Derived Column Counts --------------------------------------")
    print(f"  AVG_Q columns:   {len([c for c in df.columns if c.endswith('_AVG_Q')])}")
    print(f"  ANN columns:     {len([c for c in df.columns if c.endswith('_ANN')])}")
    print(f"  Share columns:   {len([c for c in df.columns if c.endswith('_SHARE')])}")

    print("\n-- Decoded Categoricals Null Check ----------------------------")
    for col in ["inclass_label", "educ_label", "marital_label", "fam_type_label",
                "tenure_label", "region_label", "urban_label", "race_ref_label"]:
        if col in df.columns:
            n_null = df[col].isnull().sum()
            flag   = "  <-- INVESTIGATE" if n_null > 0 else ""
            print(f"  {col:<30} {n_null:>5} nulls{flag}")

    print("\n-- Geography Suppression --------------------------------------")
    n_suppressed = df["FLAG_REGION_UNKNOWN"].sum()
    print(f"  Rows with REGION and STATE suppressed: {n_suppressed}")
    print(f"  Cause: BLS privacy suppression (small/sparse area disclosure risk)")
    print(f"  Treatment: labeled 'Suppressed' in region_label, FLAG_REGION_UNKNOWN = 1")
    print(f"  Impact: retain in all analyses except region-stratified tables")

    print("\n-- FDHOMEPQ / FDHOMECQ Artifact -------------------------------")
    print("  Both columns are 0.0 for all rows in this file.")
    print("  Use GROCER_AVG_Q / GROCER_ANN for food-at-home analysis.")

    print("\n-- Survey Weight Distribution ---------------------------------")
    print(f"  Min:  {df['FINLWT21'].min():>10,.1f}")
    print(f"  Max:  {df['FINLWT21'].max():>10,.1f}")
    print(f"  Mean: {df['FINLWT21'].mean():>10,.1f}   Std: {df['FINLWT21'].std():,.1f}")
    print("\n" + "=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CONVENIENCE ACCESSORS FOR DOWNSTREAM SCRIPTS
# ══════════════════════════════════════════════════════════════════════════════

def get_analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the subset valid for ratio analyses.
    Excludes rows where FLAG_INCOME_INVALID = 1 (FINCBTAX <= 0).
    Use as the default analytical sample in Phases 1-3.
    Note: January households (FLAG_CQ_MISSING = 1) are retained here.
    Their annualized estimates are based on PQ only — documented and valid.
    To run sensitivity analysis on Feb/Mar only, additionally filter on
    FLAG_CQ_MISSING == 0.
    """
    return df[df["FLAG_INCOME_INVALID"] == 0].copy()


def get_feb_mar_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return February and March households only (both PQ and CQ available).
    Use for sensitivity analysis comparing two-quarter vs single-quarter
    households. Excludes income-invalid rows.
    """
    return df[
        (df["FLAG_INCOME_INVALID"] == 0) &
        (df["FLAG_CQ_MISSING"] == 0)
    ].copy()


def get_avg_q_col(base: str) -> str:
    """get_avg_q_col('HOUS') -> 'HOUS_AVG_Q'"""
    return f"{base}_AVG_Q"


def get_ann_col(base: str) -> str:
    """get_ann_col('HOUS') -> 'HOUS_ANN'"""
    return f"{base}_ANN"


def get_share_col(base: str) -> str:
    """get_share_col('HOUS') -> 'HOUS_SHARE'"""
    return f"{base}_SHARE"


def get_income_class_labels(short: bool = False) -> dict:
    """Return the INCLASS2 label map. Pass short=True for chart labels."""
    return INCLASS2_SHORT if short else INCLASS2_MAP


def get_spend_categories() -> dict:
    """Return {base_name: display_label} for all primary spending categories."""
    return SPEND_CATEGORIES.copy()


def get_quintile_labels(short=False) -> list:
    """
    Return ordered quintile labels for INCOME_QUINTILE column.
    Pass short=True for compact chart axis labels (Q1-Q5).
    Pass short=False (default) for full descriptive labels.
    """
    return QUINTILE_LABELS_SHORT if short else QUINTILE_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — STANDALONE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    df = prepare_data()
    print_quality_report(df)

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n  OK: Analytical dataset saved -> {OUTPUT_CSV_PATH}")
    print(f"      ({len(df):,} rows x {len(df.columns)} columns)\n")