"""
ACCT 3343 Analysis App: a Quantitative Value screen for SELL (short) and BUY (long) candidates.

SELL screen (margin of safety, textbook Part 2 + Ch. 9):
    earnings-quality and distress red flags: STA and SNOA (Ch. 3), Beneish PMAN (Ch. 3),
    Altman Z and the Campbell-Hilscher-Szilagyi PFD (Ch. 4), and short interest (Ch. 9).
BUY screen (quality at a bargain price, Ch. 2, 5, 6, 7, 11):
    price (EBIT/TEV, book-to-market) and quality (franchise power + financial strength, or GPA).

The two screens use different measures, so BUY is not SELL "flipped" (assignment rule 2).
Data: universe.csv, built in Google Colab with yfinance (build_universe.ipynb).
Only streamlit, pandas, numpy and plotly are used. Missing values stay missing.
"""
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Quant Value Screener", page_icon="📉", layout="wide")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "universe.csv"   # the CSV sits next to app.py in the repository

# Assignment rule 1: the 60-second pitch company cannot be one of the app's recommendations.
# Put its ticker here (Yahoo format, e.g. "COST" or "BRK-B") and it is removed from the universe.
EXCLUDE_TICKERS = ["DNUT"]   # Krispy Kreme, my 60-second pitch company

# Data errors found while checking the file. These rows are removed and listed on the Data coverage tab.
DATA_ERRORS = {
    "YPF": "Yahoo labels its statements as Argentine pesos, but the values are already in US dollars, so "
           "converting them shrank revenue to about $12 million.",
}

STATUS_COLORS = {"SELL": "#c0392b", "BUY": "#1f77b4", "BUY and SELL": "#e69f00", "Neither": "#c8ccd2"}


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def div(a, b):
    """a / b, but missing (NaN) when b is zero or missing."""
    return a / b.where(b != 0)


def absent_as_zero(s):
    """Yahoo leaves out lines that are zero (no preferred stock, no minority interest, no debt,
    no buybacks). Only these 'optional' lines are treated as 0 when absent. Core items such as
    revenue, total assets or operating cash flow are never filled: missing stays missing."""
    return s.fillna(0.0)


def norm_cdf(x):
    """Standard normal CDF (Excel NORMDIST(x, 0, 1, TRUE)), used to turn PROBM into PMAN."""
    return x.apply(lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) if pd.notna(v) else np.nan)


def winsorize(s, lo=0.05, hi=0.95):
    """Clip extreme values to the 5th / 95th percentile (as Campbell et al. do)."""
    return s.clip(s.quantile(lo), s.quantile(hi))


def pct_rank(s, groups=None, min_group=10):
    """Percentile rank 0-100 (100 = highest value). Missing values stay missing.
    With groups (e.g. sector), the rank is computed within each group; groups with fewer than
    min_group companies are too small for percentiles, so they use the universe-wide rank."""
    overall = s.rank(pct=True) * 100
    if groups is None:
        return overall
    within = s.groupby(groups).rank(pct=True) * 100
    small = groups.map(groups.value_counts()) < min_group
    return within.where(~small, overall)


def years_available(d):
    return [t for t in range(5) if f"total_assets_t{t}" in d.columns]


def ebit(d, t):
    """EBIT for year t; if Yahoo has no EBIT line, use operating income."""
    return d[f"ebit_t{t}"].fillna(d[f"operating_income_t{t}"])


def gross_profit(d, t):
    """Revenue - COGS (class formula); if COGS is missing, Yahoo's gross profit line."""
    return (d[f"revenue_t{t}"] - d[f"cogs_t{t}"]).fillna(d[f"gross_profit_t{t}"])


def total_debt(d, t):
    """Total debt; if absent, short-term + long-term debt; if those are absent too, 0 (no debt)."""
    parts = absent_as_zero(d[f"current_debt_t{t}"]) + absent_as_zero(d[f"long_term_debt_t{t}"])
    return d[f"total_debt_t{t}"].fillna(parts)


def depreciation(d, t):
    return d[f"da_cf_t{t}"].fillna(d[f"da_is_t{t}"])


def free_cash_flow(d, t):
    """FCF = operating cash flow - capital expenditures (Yahoo reports capex as a negative number)."""
    return d[f"free_cash_flow_t{t}"].fillna(d[f"operating_cash_flow_t{t}"] + d[f"capex_t{t}"])


# ---------------------------------------------------------------------------------------------
# Course measures. Each function takes the data and returns it with new columns.
# ---------------------------------------------------------------------------------------------
def add_price_measures(d):
    """Ch. 2 and 7: EBIT/TEV (Greenblatt's earnings yield) and book-to-market."""
    d = d.copy()
    cash, ca, cl = d["cash_t0"], d["current_assets_t0"], d["current_liabilities_t0"]
    # Greenblatt's excess cash: cash not needed to fund working capital (never below zero)
    d["excess_cash"] = (cash - (cl - (ca - cash)).clip(lower=0)).clip(lower=0)
    d["TEV"] = (d["market_cap"] + total_debt(d, 0) + absent_as_zero(d["preferred_equity_t0"])
                + absent_as_zero(d["minority_interest_t0"]) - d["excess_cash"])
    d["TEV"] = d["TEV"].where(d["TEV"] > 0)
    d["EBIT_TEV"] = div(ebit(d, 0), d["TEV"])
    d["BM"] = div(d["common_equity_t0"], d["market_cap"])
    return d


def add_gpa(d):
    """Ch. 2: GPA = (Revenue - COGS) / Total Assets."""
    d = d.copy()
    d["GPA"] = div(gross_profit(d, 0), d["total_assets_t0"])
    return d


def add_magic_roc(d):
    """Ch. 2: Greenblatt ROC = EBIT / (Net PPE + Current Assets - Current Liabilities)."""
    d = d.copy()
    capital = d["net_ppe_t0"] + d["current_assets_t0"] - d["current_liabilities_t0"]
    d["MF_ROC"] = div(ebit(d, 0), capital.where(capital > 0))
    return d


def add_sta(d):
    """Ch. 3 (simplified, as in class): STA = (Net Income - Cash Flow from Operations) / Total Assets."""
    d = d.copy()
    d["STA"] = div(d["net_income_t0"] - d["operating_cash_flow_t0"], d["total_assets_t0"])
    return d


def add_snoa(d):
    """Ch. 3: SNOA = (Operating Assets - Operating Liabilities) / Total Assets.
    OA = TA - cash;  OL = TA - ST debt - LT debt - minority interest - preferred - common equity."""
    d = d.copy()
    ta = d["total_assets_t0"]
    op_assets = ta - d["cash_t0"]
    op_liabs = (ta - absent_as_zero(d["current_debt_t0"]) - absent_as_zero(d["long_term_debt_t0"])
                - absent_as_zero(d["minority_interest_t0"]) - absent_as_zero(d["preferred_equity_t0"])
                - d["common_equity_t0"])
    d["SNOA"] = div(op_assets - op_liabs, ta)
    return d


def add_beneish(d):
    """Ch. 3: Beneish PROBM and PMAN = N(PROBM). Year t = latest fiscal year, t-1 = the year before.
    Indexes whose inputs are not reported (receivables, SG&A, depreciation, PPE, COGS) are set to
    1.0, a neutral value (the book does this for Enron's SGAI). The count is kept in PROBM_neutral
    and PROBM is left missing if more than two indexes had to be set to neutral."""
    d = d.copy()
    rev0, rev1 = d["revenue_t0"], d["revenue_t1"]
    ta0, ta1 = d["total_assets_t0"], d["total_assets_t1"]
    gm0, gm1 = div(gross_profit(d, 0), rev0), div(gross_profit(d, 1), rev1)
    dep0, dep1 = depreciation(d, 0), depreciation(d, 1)
    ppe0, ppe1 = d["net_ppe_t0"], d["net_ppe_t1"]
    ltd0, ltd1 = absent_as_zero(d["long_term_debt_t0"]), absent_as_zero(d["long_term_debt_t1"])

    raw = pd.DataFrame(index=d.index)
    raw["DSRI"] = div(div(d["receivables_t0"], rev0), div(d["receivables_t1"], rev1))
    raw["GMI"] = div(gm1, gm0.where(gm0 > 0))
    raw["AQI"] = div(1 - div(d["current_assets_t0"] + ppe0, ta0), 1 - div(d["current_assets_t1"] + ppe1, ta1))
    raw["SGI"] = div(rev0, rev1)
    raw["DEPI"] = div(div(dep1, dep1 + ppe1), div(dep0, dep0 + ppe0))
    raw["SGAI"] = div(div(d["sga_t0"], rev0), div(d["sga_t1"], rev1))
    raw["LVGI"] = div(div(d["current_liabilities_t0"] + ltd0, ta0), div(d["current_liabilities_t1"] + ltd1, ta1))
    # TATA (Sloan's total accruals, as in the book's checklist):
    # (change in CA - change in cash - (change in CL - change in current debt - change in taxes payable) - D&A) / TA
    d_ca = d["current_assets_t0"] - d["current_assets_t1"]
    d_cash = d["cash_t0"] - d["cash_t1"]
    d_cl = d["current_liabilities_t0"] - d["current_liabilities_t1"]
    d_cd = absent_as_zero(d["current_debt_t0"]) - absent_as_zero(d["current_debt_t1"])
    d_tax = absent_as_zero(d["income_tax_payable_t0"]) - absent_as_zero(d["income_tax_payable_t1"])
    raw["TATA"] = div(d_ca - d_cash - (d_cl - d_cd - d_tax) - dep0, ta0)

    optional = ["DSRI", "GMI", "AQI", "DEPI", "SGAI"]           # may be set to neutral 1.0
    raw = raw.replace([np.inf, -np.inf], np.nan)
    d["PROBM_neutral"] = raw[optional].isna().sum(axis=1)
    for c in optional:
        raw[c] = raw[c].fillna(1.0)
    for c in raw.columns:
        d[c] = raw[c]
    d["PROBM"] = (-4.84 + 0.92 * raw["DSRI"] + 0.528 * raw["GMI"] + 0.404 * raw["AQI"] + 0.892 * raw["SGI"]
                  + 0.115 * raw["DEPI"] - 0.172 * raw["SGAI"] + 4.679 * raw["TATA"] - 0.327 * raw["LVGI"])
    d["PROBM"] = d["PROBM"].where(d["PROBM_neutral"] <= 2)
    d["PMAN"] = norm_cdf(d["PROBM"])
    return d


def add_altman(d):
    """Ch. 4: Altman Z = 0.012 X1 + 0.014 X2 + 0.033 X3 + 0.006 X4 + 0.999 X5.
    Z_class enters X1-X4 as decimals, exactly as in the class example (EMP: Z = 1.89).
    Z_pct enters X1-X4 as percentages, as Altman (1968) specified; this equals
    1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 0.999 X5 with decimal ratios (EMP: Z = 2.77)."""
    d = d.copy()
    ta = d["total_assets_t0"]
    d["X1"] = div(d["current_assets_t0"] - d["current_liabilities_t0"], ta)     # working capital / TA
    d["X2"] = div(d["retained_earnings_t0"], ta)                                # retained earnings / TA
    d["X3"] = div(ebit(d, 0), ta)                                               # EBIT / TA
    d["X4"] = div(d["market_cap"], d["total_liabilities_t0"])                   # MV equity / BV liabilities
    d["X5"] = div(d["revenue_t0"], ta)                                          # sales / TA
    d["Z_class"] = 0.012 * d["X1"] + 0.014 * d["X2"] + 0.033 * d["X3"] + 0.006 * d["X4"] + 0.999 * d["X5"]
    d["Z_pct"] = (0.012 * 100 * d["X1"] + 0.014 * 100 * d["X2"] + 0.033 * 100 * d["X3"]
                  + 0.006 * 100 * d["X4"] + 0.999 * d["X5"])
    return d


def add_campbell_pfd(d):
    """Ch. 4: Campbell, Hilscher and Szilagyi probability of financial distress (the book's PFD).
    LPFD = -20.26 NIMTAAVG + 1.42 TLMTA - 7.13 EXRETAVG + 1.41 SIGMA - 0.045 RSIZE
           - 2.13 CASHMTA + 0.075 MB - 0.058 PRICE - 9.16;   PFD = 1 / (1 + e^-LPFD)
    Averages use the book's weights .5333, .2666, .1333, .0666 on the last four quarters."""
    d = d.copy()
    w = [0.5333, 0.2666, 0.1333, 0.0666]
    mta = d["total_liabilities_t0"] + d["market_cap"]            # market value of total assets
    nimta = [div(d[f"ni_q{i}"], mta) for i in range(4)]
    exret = [np.log1p(d[f"ret_q{i}"]) - np.log1p(d[f"mkt_ret_q{i}"]) for i in range(4)]
    d["NIMTAAVG"] = sum(wi * x for wi, x in zip(w, nimta))
    d["EXRETAVG"] = sum(wi * x for wi, x in zip(w, exret))
    d["TLMTA"] = div(d["total_liabilities_t0"], mta)
    d["CASHMTA"] = div(d["cash_t0"], mta)
    d["SIGMA"] = d["sigma_3m"]
    # S&P 500 total value approximated by the total market value of the companies in this file
    d["RSIZE"] = np.log(div(d["market_cap"], pd.Series(d["market_cap"].sum(), index=d.index)))
    be = d["common_equity_t0"]
    adj_book = be + 0.1 * (d["market_cap"] - be)
    d["MB"] = div(mta, adj_book.where(adj_book > 0))
    d["PRICE_CHS"] = np.log(d["price"].clip(upper=15).where(d["price"] > 0))
    inputs = ["NIMTAAVG", "TLMTA", "EXRETAVG", "SIGMA", "RSIZE", "CASHMTA", "MB", "PRICE_CHS"]
    x = {c: winsorize(d[c]) for c in inputs}
    d["LPFD"] = (-20.26 * x["NIMTAAVG"] + 1.42 * x["TLMTA"] - 7.13 * x["EXRETAVG"] + 1.41 * x["SIGMA"]
                 - 0.045 * x["RSIZE"] - 2.13 * x["CASHMTA"] + 0.075 * x["MB"] - 0.058 * x["PRICE_CHS"] - 9.16)
    d["PFD"] = 1 / (1 + np.exp(-d["LPFD"]))
    return d


def _geometric_average(frame, min_years=3):
    """Geometric average return over the available years (penalizes volatility, Ch. 5)."""
    g = 1 + frame
    ok = (g.notna().sum(axis=1) >= min_years) & ~(g <= 0).any(axis=1)
    return (np.exp(np.log(g.where(g > 0)).mean(axis=1)) - 1).where(ok)


def _margin_growth(gm, min_years=3):
    """Geometric annual growth of gross margin from the oldest to the newest available year.
    T = number of growth periods (years - 1), as in the class slides."""
    def one(row):
        v = row.dropna()
        v = v[v > 0]
        if len(v) < min_years:
            return np.nan
        newest, oldest = v.iloc[0], v.iloc[-1]
        periods = int(v.index[-1]) - int(v.index[0])
        return (newest / oldest) ** (1 / periods) - 1
    return gm.apply(one, axis=1)


def add_franchise(d):
    """Ch. 5 franchise power, using all fiscal years Yahoo provides (usually 4; the book uses 8):
    long-term geometric ROA and ROC, CFOA = sum of FCF / total assets, margin growth (MG)
    and margin stability (MS = average gross margin / its standard deviation)."""
    d = d.copy()
    yrs = years_available(d)
    roa = pd.DataFrame({t: div(d[f"net_income_cont_t{t}"].fillna(d[f"net_income_t{t}"]),
                               d[f"total_assets_t{t}"]) for t in yrs})
    capital = pd.DataFrame({t: total_debt(d, t) + d[f"common_equity_t{t}"] - d[f"cash_t{t}"] for t in yrs})
    roc = pd.DataFrame({t: div(ebit(d, t), capital[t].where(capital[t] > 0)) for t in yrs})
    gm = pd.DataFrame({t: div(gross_profit(d, t), d[f"revenue_t{t}"]) for t in yrs})
    fcf = pd.DataFrame({t: free_cash_flow(d, t) for t in yrs})

    d["n_years"] = roa.notna().sum(axis=1)
    d["LT_ROA"] = _geometric_average(roa)
    d["LT_ROC"] = _geometric_average(roc)
    d["CFOA"] = div(fcf.sum(axis=1, min_count=3), d["total_assets_t0"])
    d["MG"] = _margin_growth(gm)
    gm_std = gm.std(axis=1, ddof=1)
    d["MS"] = div(gm.mean(axis=1), gm_std.where(gm_std > 0)).where(gm.notna().sum(axis=1) >= 3)
    d["GM_t0"] = gm[0]
    return d


def add_financial_strength(d):
    """Ch. 6: the book's FS_SCORE, ten 0/1 signals (higher = stronger).
    Computed when at least 9 of 10 signals are available, rescaled to a 10-point basis."""
    d = d.copy()
    ta0, ta1, ta2 = d["total_assets_t0"], d["total_assets_t1"], d["total_assets_t2"]
    ni0 = d["net_income_cont_t0"].fillna(d["net_income_t0"])
    ni1 = d["net_income_cont_t1"].fillna(d["net_income_t1"])
    roa0, roa1 = div(ni0, ta0), div(ni1, ta1)
    fcfta0, fcfta1 = div(free_cash_flow(d, 0), ta0), div(free_cash_flow(d, 1), ta1)
    gm0, gm1 = div(gross_profit(d, 0), d["revenue_t0"]), div(gross_profit(d, 1), d["revenue_t1"])
    lever = div(absent_as_zero(d["long_term_debt_t1"]), ta1) - div(absent_as_zero(d["long_term_debt_t0"]), ta0)
    liquid = (div(d["current_assets_t0"], d["current_liabilities_t0"])
              - div(d["current_assets_t1"], d["current_liabilities_t1"]))
    neqiss = absent_as_zero(d["repurchase_t0"]).abs() - absent_as_zero(d["issuance_t0"]).abs()
    turn = div(d["revenue_t0"], ta1) - div(d["revenue_t1"], ta2)       # sales / beginning assets

    signals = {
        "FS_ROA": roa0 > 0, "FS_FCFTA": fcfta0 > 0, "FS_ACCRUAL": (fcfta0 - roa0) > 0,
        "FS_LEVER": lever > 0, "FS_LIQUID": liquid > 0, "FS_NEQISS": neqiss > 0,
        "FS_dROA": (roa0 - roa1) > 0, "FS_dFCFTA": (fcfta0 - fcfta1) > 0,
        "FS_dMARGIN": (gm0 - gm1) > 0, "FS_dTURN": turn > 0,
    }
    inputs = {"FS_ROA": roa0, "FS_FCFTA": fcfta0, "FS_ACCRUAL": fcfta0 - roa0, "FS_LEVER": lever,
              "FS_LIQUID": liquid, "FS_NEQISS": neqiss, "FS_dROA": roa0 - roa1,
              "FS_dFCFTA": fcfta0 - fcfta1, "FS_dMARGIN": gm0 - gm1, "FS_dTURN": turn}
    for k, sig in signals.items():
        d[k] = sig.astype(float).where(inputs[k].notna())
    cols = list(signals)
    n_avail = d[cols].notna().sum(axis=1)
    d["FS_available"] = n_avail
    d["FS_SCORE"] = (d[cols].sum(axis=1) * 10 / n_avail.where(n_avail > 0)).where(n_avail >= 9)
    return d


@st.cache_data(show_spinner="Loading data and calculating measures...")
def load_universe(path, exclude):
    raw = pd.read_csv(path)
    excluded = raw[raw["ticker"].isin(exclude)]["ticker"].tolist()
    d = raw[~raw["ticker"].isin(list(exclude) + list(DATA_ERRORS))].reset_index(drop=True).copy()
    d["sector"] = d["sector"].fillna("Unclassified")
    d["market_cap_b"] = d["market_cap"] / 1e9
    for f in (add_price_measures, add_gpa, add_magic_roc, add_sta, add_snoa, add_beneish,
              add_altman, add_campbell_pfd, add_franchise, add_financial_strength):
        d = f(d)
    d = d.replace([np.inf, -np.inf], np.nan)
    d["name"] = d["name"].fillna(d["ticker"])
    d["label"] = d["ticker"] + " | " + d["name"].astype(str)
    return d, excluded


# ---------------------------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------------------------
try:
    data, excluded_tickers = load_universe(str(DATA_FILE), tuple(EXCLUDE_TICKERS))
except FileNotFoundError:
    st.error("Could not find universe.csv. Upload it to the repository next to app.py.")
    st.stop()

as_of = str(data["data_pulled"].iloc[0]) if "data_pulled" in data.columns else "unknown"

# ---------------------------------------------------------------------------------------------
# Sidebar filters (every value below feeds the screens; moving one reruns the whole script)
# ---------------------------------------------------------------------------------------------
sb = st.sidebar
sb.title("Filters")

sb.header("1. Universe")
all_sectors = sorted(data["sector"].dropna().unique())
default_sectors = [s for s in all_sectors if s not in ("Real Estate", "Utilities")]
sectors = sb.multiselect("Sectors to include", all_sectors, default=default_sectors,
                         help="Financial Services were excluded when the data was built: accrual, SNOA and "
                              "Altman measures do not fit banks and insurers. Real Estate and Utilities are off "
                              "by default: REIT accounting distorts accrual measures, and regulated utilities "
                              "carry heavy assets and debt by design (87% of utilities score below Altman's 1.81).")
industry_pool = sorted(data.loc[data["sector"].isin(sectors), "industry"].dropna().unique())
industries = sb.multiselect("Industries (optional, for a within-industry analysis)", industry_pool, default=[])
cap_options = [0.3, 0.5, 1, 2, 5, 10, 25, 50, 100]
min_cap = sb.select_slider("Minimum market cap ($ billions)", options=cap_options, value=0.3)
us_only = sb.checkbox("US-domiciled companies only", value=False)
within_sector = sb.checkbox("Rank percentiles within each sector", value=False,
                            help="Off: percentiles compare each firm with the whole screened universe (the book's "
                                 "approach). On: each firm is compared only with firms in its own sector.")

sb.header("2. SELL rule (red flags)")
sb.caption("A company is a SELL candidate when it trips at least the required number of active red flags.")
use_sta = sb.checkbox("Accruals: STA (Ch. 3)", value=True)
sta_cut = sb.slider("Flag STA at or above percentile", 50, 99, 90, disabled=not use_sta)
use_snoa = sb.checkbox("Bloated balance sheet: SNOA (Ch. 3)", value=True)
snoa_cut = sb.slider("Flag SNOA at or above percentile", 50, 99, 90, disabled=not use_snoa)
use_pman = sb.checkbox("Manipulation: Beneish PMAN (Ch. 3)", value=True)
pman_cut = sb.slider("Flag PMAN at or above percentile", 50, 99, 90, disabled=not use_pman)
use_z = sb.checkbox("Distress: Altman Z (Ch. 4)", value=True)
z_cut = sb.slider("Flag Altman Z below", 0.0, 4.0, 1.81, 0.01, disabled=not use_z)
use_pfd = sb.checkbox("Distress: Campbell et al. PFD (Ch. 4)", value=True)
pfd_cut = sb.slider("Flag PFD at or above percentile", 50, 99, 90, disabled=not use_pfd)
use_si = sb.checkbox("Smart money: short interest (Ch. 9)", value=True)
si_cut = sb.slider("Flag short interest at or above (% of float)", 1.0, 30.0, 10.0, 0.5, disabled=not use_si)
n_active = sum([use_sta, use_snoa, use_pman, use_z, use_pfd, use_si])
if n_active == 0:
    sb.warning("Turn on at least one red flag.")
flags_needed = sb.slider("Red flags required for SELL", 1, max(n_active, 1), min(3, max(n_active, 1)))

sb.header("3. BUY rule (quality at a bargain)")
price_choice = sb.radio("Price measure", ["EBIT/TEV (book's winner, Ch. 7)", "Book-to-market (Ch. 2)",
                                          "Average of both"], index=0)
quality_choice = sb.radio("Quality measure", ["QUALITY = franchise power + financial strength (Ch. 5-6)",
                                              "Gross profitability, GPA (Ch. 2)", "Average of both"], index=0)
method = sb.radio("Ranking method", ["Combined rank: price rank + quality rank (Ch. 2)",
                                     "Value first, then quality (Ch. 11)"], index=0)
price_cut = sb.slider("Price percentile at least (100 = cheapest)", 0, 99, 80)
quality_cut = sb.slider("Quality percentile at least", 0, 99, 70)
fs_min = sb.slider("Financial strength score at least (0-10)", 0, 10, 6)
buy_excludes_sell = sb.checkbox("Margin of safety: never BUY a SELL candidate", value=True,
                                help="Book Ch. 11, step 1: remove likely manipulators and distressed firms before "
                                     "buying. This only removes names; the BUY ranking uses different measures.")
top_n = sb.slider("Rows to show in each table", 10, 100, 25, 5)

with sb.expander("Advanced"):
    altman_version = st.radio("Altman Z inputs", ["Altman's scaling: X1-X4 as percentages (EMP Z = 2.77)",
                                                  "Class example: X1-X4 as decimals (EMP Z = 1.89)"], index=0,
                              help="With decimals, Z is almost entirely X5 (asset turnover) and about 9 in 10 "
                                   "companies fall below 1.81. See Methodology.")
z_col = "Z_pct" if altman_version.startswith("Altman") else "Z_class"

# ---------------------------------------------------------------------------------------------
# Apply universe filters
# ---------------------------------------------------------------------------------------------
u = data[data["sector"].isin(sectors)]
if industries:
    u = u[u["industry"].isin(industries)]
u = u[u["market_cap_b"] >= min_cap]
if us_only:
    u = u[u["country"] == "United States"]
u = u.copy()
if len(u) < 20:
    st.warning("Fewer than 20 companies pass the universe filters; percentiles are not meaningful. Widen the filters.")
    if len(u) == 0:
        st.stop()
groups = u["sector"] if within_sector else None


def P(col):
    return pct_rank(u[col], groups)


# ---------------------------------------------------------------------------------------------
# SELL screen
# ---------------------------------------------------------------------------------------------
u["Z"] = u[z_col]
u["P_STA"], u["P_SNOA"], u["P_PMAN"], u["P_PFD"] = P("STA"), P("SNOA"), P("PMAN"), P("PFD")
u["P_Zrisk"] = pct_rank(-u["Z"], groups)                      # low Z = high distress risk
u["P_short"] = P("short_pct_float")
u["COMBOACCRUAL"] = u[["P_STA", "P_SNOA"]].mean(axis=1, skipna=False)

flag_defs = [  # (on?, column name, test, input column)
    (use_sta, "flag_STA", u["P_STA"] >= sta_cut, "P_STA"),
    (use_snoa, "flag_SNOA", u["P_SNOA"] >= snoa_cut, "P_SNOA"),
    (use_pman, "flag_PMAN", u["P_PMAN"] >= pman_cut, "P_PMAN"),
    (use_z, "flag_Z", u["Z"] < z_cut, "Z"),
    (use_pfd, "flag_PFD", u["P_PFD"] >= pfd_cut, "P_PFD"),
    (use_si, "flag_SHORT", u["short_pct_float"] * 100 >= si_cut, "short_pct_float"),
]
u["red_flags"] = 0
u["flags_checked"] = 0
risk_pcts = []
for on, name, test, src in flag_defs:
    u[name] = (test & u[src].notna()) if on else False
    if on:
        u["red_flags"] += u[name].astype(int)
        u["flags_checked"] += u[src].notna().astype(int)
        risk_pcts.append({"Z": "P_Zrisk", "short_pct_float": "P_short"}.get(src, src))
u["SELL"] = (u["red_flags"] >= flags_needed) if n_active else False
u["SELL_score"] = u[risk_pcts].mean(axis=1) if risk_pcts else np.nan
u["PFD_100"] = u["PFD"] * 100                                 # PFD in percent, for display

# ---------------------------------------------------------------------------------------------
# BUY screen
# ---------------------------------------------------------------------------------------------
u["P_EBIT_TEV"], u["P_BM"], u["P_GPA"] = P("EBIT_TEV"), P("BM"), P("GPA")
u["P_LT_ROA"], u["P_LT_ROC"], u["P_CFOA"] = P("LT_ROA"), P("LT_ROC"), P("CFOA")
u["P_MG"], u["P_MS"] = P("MG"), P("MS")
u["MM"] = u[["P_MG", "P_MS"]].max(axis=1)                     # max margin: best of growth or stability
fp_parts = u[["P_LT_ROA", "P_LT_ROC", "P_CFOA", "MM"]]
u["FP_raw"] = fp_parts.mean(axis=1).where(fp_parts.notna().sum(axis=1) >= 3)
u["P_FP"] = P("FP_raw")                                       # franchise power percentile
u["QUALITY"] = 0.5 * u["P_FP"] + 0.5 * (u["FS_SCORE"] * 10)   # book: .5 P_FP + .5 P_FS
u["P_QUALITY"] = P("QUALITY")

if price_choice.startswith("EBIT"):
    u["price_score"] = u["P_EBIT_TEV"]
elif price_choice.startswith("Book"):
    u["price_score"] = u["P_BM"]
else:
    u["price_score"] = u[["P_EBIT_TEV", "P_BM"]].mean(axis=1, skipna=False)
if quality_choice.startswith("QUALITY"):
    u["quality_score"] = u["P_QUALITY"]
elif quality_choice.startswith("Gross"):
    u["quality_score"] = u["P_GPA"]
else:
    u["quality_score"] = u[["P_QUALITY", "P_GPA"]].mean(axis=1, skipna=False)

u["BUY"] = ((u["price_score"] >= price_cut) & (u["quality_score"] >= quality_cut)
            & (u["FS_SCORE"] >= fs_min) & (ebit(u, 0) > 0))
if buy_excludes_sell:
    u["BUY"] = u["BUY"] & ~u["SELL"]

if method.startswith("Combined"):
    # Class method: rank on price, rank on quality, add the ranks, lowest total is best
    u["BUY_rank_value"] = (u["price_score"].rank(ascending=False) + u["quality_score"].rank(ascending=False))
else:
    # Book method: among the cheap names, the highest quality wins
    u["BUY_rank_value"] = u["quality_score"].rank(ascending=False)
u["Status"] = np.select([u["BUY"] & u["SELL"], u["SELL"], u["BUY"]], ["BUY and SELL", "SELL", "BUY"], "Neither")

sell = u[u["SELL"]].sort_values(["red_flags", "SELL_score"], ascending=[False, False])
buy = u[u["BUY"]].sort_values("BUY_rank_value")
sell = sell.assign(Rank=range(1, len(sell) + 1))
buy = buy.assign(Rank=range(1, len(buy) + 1))

# ---------------------------------------------------------------------------------------------
# Page header and headline numbers
# ---------------------------------------------------------------------------------------------
st.title("Quantitative Value Screener: who to SELL, who to BUY")
st.caption(f"{len(data):,} US-listed companies from Yahoo Finance (data pulled {as_of}). "
           "SELL = accounting and distress red flags. BUY = high quality at a low price. "
           "Move any filter on the left and every table, chart and count below updates.")
st.caption(f"Excluded under assignment rule 1 (my 60-second pitch company): Krispy Kreme ({', '.join(EXCLUDE_TICKERS)}).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Companies screened", f"{len(u):,}")
k2.metric("SELL candidates", f"{int(u['SELL'].sum()):,}", help=f"{u['SELL'].mean() * 100:.1f}% of screened companies")
k3.metric("BUY candidates", f"{int(u['BUY'].sum()):,}", help=f"{u['BUY'].mean() * 100:.1f}% of screened companies")
k4.metric("Red flags required", f"{flags_needed} of {n_active}")

# Display helpers ------------------------------------------------------------------------------
pct_col = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f")
num = lambda label, fmt="%.3f": st.column_config.NumberColumn(label, format=fmt)


def show_table(df, cols, config, key):
    out = df[list(cols)].rename(columns=cols)
    st.dataframe(out.head(top_n), column_config=config, hide_index=True, width="stretch")
    st.download_button(f"Download all {len(df)} rows (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"{key}_candidates.csv", mime="text/csv", key=f"dl_{key}")


tab_sell, tab_buy, tab_charts, tab_drill, tab_concl, tab_method, tab_cov = st.tabs(
    ["SELL candidates", "BUY candidates", "Charts", "Company drill-down", "Conclusions", "Methodology",
     "Data coverage"])

# ---------------------------------------------------------------------------------------------
# SELL tab
# ---------------------------------------------------------------------------------------------
with tab_sell:
    active = [n for on, n in zip([use_sta, use_snoa, use_pman, use_z, use_pfd, use_si],
                                 [f"STA in the top {100 - sta_cut}%", f"SNOA in the top {100 - snoa_cut}%",
                                  f"PMAN in the top {100 - pman_cut}%", f"Altman Z below {z_cut:.2f}",
                                  f"PFD in the top {100 - pfd_cut}%", f"short interest of {si_cut:.1f}%+ of float"])
              if on]
    st.subheader(f"{len(sell)} SELL candidates")
    st.markdown(f"Rule: at least **{flags_needed}** of these red flags: " + "; ".join(active) + ". "
                "Ranked by number of red flags, then by the average risk percentile across the active measures "
                "(SELL score, 100 = riskiest).")
    sell_cols = {"Rank": "Rank", "ticker": "Ticker", "name": "Company", "sector": "Sector",
                 "market_cap_b": "Mkt cap $B", "red_flags": "Red flags", "flags_checked": "Flags checked",
                 "SELL_score": "SELL score", "STA": "STA", "P_STA": "STA pct", "SNOA": "SNOA",
                 "P_SNOA": "SNOA pct", "PMAN": "PMAN", "P_PMAN": "PMAN pct", "Z": "Altman Z", "PFD_100": "PFD %",
                 "P_PFD": "PFD pct", "short_pct_float": "Short % float", "ret_12m": "12m return"}
    sell_cfg = {"Mkt cap $B": num("Mkt cap $B", "%.1f"), "SELL score": pct_col("SELL score"),
                "STA": num("STA"), "STA pct": pct_col("STA pct"), "SNOA": num("SNOA"),
                "SNOA pct": pct_col("SNOA pct"), "PMAN": st.column_config.NumberColumn("PMAN", format="percent"),
                "PMAN pct": pct_col("PMAN pct"), "Altman Z": num("Altman Z", "%.2f"),
                "PFD %": st.column_config.NumberColumn("PFD %", format="%.3f%%"), "PFD pct": pct_col("PFD pct"),
                "Short % float": st.column_config.NumberColumn("Short % float", format="percent"),
                "12m return": st.column_config.NumberColumn("12m return", format="percent")}
    if len(sell):
        show_table(sell, sell_cols, sell_cfg, "sell")
        flag_labels = {"flag_STA": "STA", "flag_SNOA": "SNOA", "flag_PMAN": "PMAN", "flag_Z": "Altman Z",
                       "flag_PFD": "PFD", "flag_SHORT": "Short interest"}
        counts = pd.DataFrame({"Red flag": [flag_labels[c] for on, c, _, _ in flag_defs if on],
                               "SELL candidates tripping it": [int(sell[c].sum()) for on, c, _, _ in flag_defs if on]})
        fig = px.bar(counts, x="SELL candidates tripping it", y="Red flag", orientation="h",
                     title="Which red flags drive the SELL list", color_discrete_sequence=[STATUS_COLORS["SELL"]])
        fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No company trips enough red flags. Lower 'Red flags required' or loosen a cutoff.")

# ---------------------------------------------------------------------------------------------
# BUY tab
# ---------------------------------------------------------------------------------------------
with tab_buy:
    st.subheader(f"{len(buy)} BUY candidates")
    st.markdown(f"Rule: price percentile of at least **{price_cut}** ({price_choice.split(' (')[0]}), quality "
                f"percentile of at least **{quality_cut}** ({quality_choice.split(' (')[0].split(' =')[0]}), "
                f"financial strength of at least **{fs_min}/10** and positive EBIT"
                + (", and not a SELL candidate" if buy_excludes_sell else "") + f". Ranked by: {method}.")
    buy_cols = {"Rank": "Rank", "ticker": "Ticker", "name": "Company", "sector": "Sector",
                "market_cap_b": "Mkt cap $B", "price_score": "Price pct", "quality_score": "Quality pct",
                "EBIT_TEV": "EBIT/TEV", "BM": "B/M", "GPA": "GPA", "P_FP": "Franchise pct",
                "FS_SCORE": "FS score", "LT_ROA": "LT ROA", "LT_ROC": "LT ROC", "CFOA": "CFOA",
                "MG": "Margin growth", "MS": "Margin stability", "MF_ROC": "Greenblatt ROC"}
    buy_cfg = {"Mkt cap $B": num("Mkt cap $B", "%.1f"), "Price pct": pct_col("Price pct"),
               "Quality pct": pct_col("Quality pct"), "EBIT/TEV": st.column_config.NumberColumn(format="percent"),
               "B/M": num("B/M", "%.2f"), "GPA": num("GPA", "%.2f"), "Franchise pct": pct_col("Franchise pct"),
               "FS score": num("FS score", "%.1f"), "LT ROA": st.column_config.NumberColumn(format="percent"),
               "LT ROC": st.column_config.NumberColumn(format="percent"), "CFOA": num("CFOA", "%.2f"),
               "Margin growth": st.column_config.NumberColumn(format="percent"),
               "Margin stability": num("Margin stability", "%.1f"),
               "Greenblatt ROC": st.column_config.NumberColumn(format="percent")}
    if len(buy):
        show_table(buy, buy_cols, buy_cfg, "buy")
    else:
        st.info("No company passes every BUY filter. Lower a percentile cutoff or the financial strength minimum.")

# ---------------------------------------------------------------------------------------------
# Charts tab
# ---------------------------------------------------------------------------------------------
with tab_charts:
    hover = {"ticker": True, "name": True, "sector": True, "Status": False}
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(u.dropna(subset=["COMBOACCRUAL", "P_PFD"]), x="COMBOACCRUAL", y="P_PFD", color="Status",
                         color_discrete_map=STATUS_COLORS, hover_data=hover, opacity=0.75,
                         category_orders={"Status": list(STATUS_COLORS)},
                         labels={"COMBOACCRUAL": "Accrual risk: average of STA and SNOA percentiles",
                                 "P_PFD": "Distress risk: PFD percentile"},
                         title="SELL map: accrual risk vs distress risk (top right = riskiest)")
        fig.update_layout(height=480, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.scatter(u.dropna(subset=["price_score", "quality_score"]), x="price_score", y="quality_score",
                         color="Status", color_discrete_map=STATUS_COLORS, hover_data=hover, opacity=0.75,
                         category_orders={"Status": list(STATUS_COLORS)},
                         labels={"price_score": "Price percentile (100 = cheapest)",
                                 "quality_score": "Quality percentile (100 = best)"},
                         title="BUY map: price vs quality (top right = cheap and high quality)")
        fig.add_vline(x=price_cut, line_dash="dash", line_color="gray")
        fig.add_hline(y=quality_cut, line_dash="dash", line_color="gray")
        fig.update_layout(height=480, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    by_sector = (u.groupby("sector").agg(Screened=("ticker", "size"), SELL=("SELL", "sum"), BUY=("BUY", "sum"))
                 .reset_index())
    long = by_sector.melt(id_vars=["sector", "Screened"], value_vars=["SELL", "BUY"], var_name="Status",
                          value_name="Companies")
    fig = px.bar(long, x="sector", y="Companies", color="Status", barmode="group", color_discrete_map=STATUS_COLORS,
                 hover_data=["Screened"], title="Candidates by sector")
    fig.update_layout(height=400, xaxis_title="", legend_title_text="")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------------------------
# Drill-down tab
# ---------------------------------------------------------------------------------------------
def fit(df):
    """Show a small table with all rows visible (no inner scrollbar)."""
    st.dataframe(df, hide_index=True, width="stretch", height=35 * (len(df) + 1) + 3)


def fmt(v, kind="num"):
    if pd.isna(v):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "rank":
        return f"{v:.0f}"
    return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"


with tab_drill:
    order = pd.concat([sell, buy, u[~u["SELL"] & ~u["BUY"]].sort_values("market_cap", ascending=False)])
    order = order[~order.index.duplicated()]
    pick = st.selectbox("Choose a company (SELL candidates first, then BUY, then everyone else)",
                        order["label"].tolist())
    r = u[u["label"] == pick].iloc[0]
    st.markdown(f"### {r['name']} ({r['ticker']}): **{r['Status']}**")
    st.caption(f"{r['sector']} / {r['industry']} · market cap \\${r['market_cap_b']:,.1f}B · "
               f"latest fiscal year ending {r.get('fy_end_t0', 'n/a')} · 12-month return {fmt(r['ret_12m'], 'pct')}")

    reasons = []
    for on, name, _, _ in flag_defs:
        if on and r[name]:
            reasons.append({"flag_STA": f"STA of {fmt(r['STA'])} is in the {r['P_STA']:.0f}th percentile: "
                                        "earnings are running well ahead of operating cash flow.",
                            "flag_SNOA": f"SNOA of {fmt(r['SNOA'])} ({r['P_SNOA']:.0f}th pct): operating assets have "
                                         "piled up relative to operating liabilities (bloated balance sheet).",
                            "flag_PMAN": f"Beneish PMAN of {fmt(r['PMAN'], 'pct')} ({r['P_PMAN']:.0f}th pct) signals "
                                         "a higher chance of earnings manipulation.",
                            "flag_Z": f"Altman Z of {r['Z']:.2f} is below the {z_cut:.2f} cutoff.",
                            "flag_PFD": f"Campbell et al. distress probability is in the {r['P_PFD']:.0f}th "
                                        "percentile (weak profits, high leverage or volatility, poor returns).",
                            "flag_SHORT": f"Short interest is {fmt(r['short_pct_float'], 'pct')} of the float: "
                                          "informed money is betting against it."}[name])
    st.markdown(f"**SELL side:** {int(r['red_flags'])} of {n_active} active red flags "
                f"({int(r['flags_checked'])} could be checked with the available data; {flags_needed} needed).")
    for x in reasons:
        st.markdown(f"- {x}")
    st.markdown(f"**BUY side:** price percentile {fmt(r['price_score'], 'rank')} (needs {price_cut}), quality "
                f"percentile {fmt(r['quality_score'], 'rank')} (needs {quality_cut}), financial strength "
                f"{fmt(r['FS_SCORE'], 'rank')}/10 (needs {fs_min}).")

    a, b = st.columns(2)
    with a:
        st.markdown("**Red-flag measures**")
        fit(pd.DataFrame({
            "Measure": ["STA (NI - CFO) / TA", "SNOA", "Beneish PROBM", "Beneish PMAN", "Altman Z (selected version)",
                        "Campbell et al. PFD", "Short interest (% float)"],
            "Value": [fmt(r["STA"]), fmt(r["SNOA"]), fmt(r["PROBM"]), fmt(r["PMAN"], "pct"), fmt(r["Z"]),
                      f"{r['PFD'] * 100:.3f}%" if pd.notna(r["PFD"]) else "n/a", fmt(r["short_pct_float"], "pct")],
            "Percentile": [fmt(r["P_STA"], "rank"), fmt(r["P_SNOA"], "rank"), fmt(r["P_PMAN"], "rank"),
                           fmt(r["P_PMAN"], "rank"), fmt(100 - r["P_Zrisk"], "rank") if pd.notna(r["P_Zrisk"]) else "n/a",
                           fmt(r["P_PFD"], "rank"), fmt(r["P_short"], "rank")],
        }))
        st.markdown("**Beneish components** (manipulators' average in Beneish 1999)")
        bench = {"DSRI": 1.465, "GMI": 1.193, "AQI": 1.254, "SGI": 1.607, "DEPI": 1.077, "SGAI": 1.041,
                 "LVGI": 1.111, "TATA": 0.031}
        fit(pd.DataFrame({"Index": list(bench), "Value": [fmt(r[k]) for k in bench],
                          "Manipulator avg": list(bench.values()),
                          "Above it?": ["yes" if pd.notna(r[k]) and r[k] > v else "" for k, v in bench.items()]}))
        if r["PROBM_neutral"] > 0:
            st.caption(f"{int(r['PROBM_neutral'])} Beneish index(es) set to a neutral 1.0 because an input is not reported.")
        st.markdown("**Altman components**")
        fit(pd.DataFrame({"Ratio": ["X1 working capital / TA", "X2 retained earnings / TA", "X3 EBIT / TA",
                                    "X4 market cap / liabilities", "X5 sales / TA"],
                          "Value": [fmt(r[k]) for k in ["X1", "X2", "X3", "X4", "X5"]]}))
    with b:
        st.markdown("**Quality and price measures**")
        fit(pd.DataFrame({
            "Measure": ["EBIT / TEV", "Book-to-market", "GPA", "Greenblatt ROC", f"Long-term ROA ({int(r['n_years'])} yrs, geometric)",
                        "Long-term ROC (geometric)", "CFOA (sum FCF / TA)", "Margin growth", "Margin stability",
                        "Franchise power", "QUALITY"],
            "Value": [fmt(r["EBIT_TEV"], "pct"), fmt(r["BM"]), fmt(r["GPA"]), fmt(r["MF_ROC"], "pct"),
                      fmt(r["LT_ROA"], "pct"), fmt(r["LT_ROC"], "pct"), fmt(r["CFOA"]), fmt(r["MG"], "pct"),
                      fmt(r["MS"]), "", fmt(r["QUALITY"], "rank")],
            "Percentile": [fmt(r["P_EBIT_TEV"], "rank"), fmt(r["P_BM"], "rank"), fmt(r["P_GPA"], "rank"), "",
                           fmt(r["P_LT_ROA"], "rank"), fmt(r["P_LT_ROC"], "rank"), fmt(r["P_CFOA"], "rank"),
                           fmt(r["P_MG"], "rank"), fmt(r["P_MS"], "rank"), fmt(r["P_FP"], "rank"),
                           fmt(r["P_QUALITY"], "rank")],
        }))
        st.markdown(f"**Financial strength: {fmt(r['FS_SCORE'], 'rank')}/10**")
        fs_names = {"FS_ROA": "ROA > 0", "FS_FCFTA": "FCF / TA > 0", "FS_ACCRUAL": "FCF / TA > ROA",
                    "FS_LEVER": "Leverage fell", "FS_LIQUID": "Current ratio rose", "FS_NEQISS": "Net buybacks",
                    "FS_dROA": "ROA improved", "FS_dFCFTA": "FCF / TA improved", "FS_dMARGIN": "Gross margin improved",
                    "FS_dTURN": "Asset turnover improved"}
        fit(pd.DataFrame({"Signal": list(fs_names.values()),
                          "Pass": ["✅" if r[k] == 1 else ("❌" if r[k] == 0 else "n/a") for k in fs_names]}))

    hist = []
    for t in years_available(u):
        if pd.notna(r.get(f"fy_end_t{t}")) and pd.notna(r[f"total_assets_t{t}"]):
            rev = r[f"revenue_t{t}"]
            gp = (rev - r[f"cogs_t{t}"]) if pd.notna(r[f"cogs_t{t}"]) else r[f"gross_profit_t{t}"]
            ni = r[f"net_income_cont_t{t}"] if pd.notna(r[f"net_income_cont_t{t}"]) else r[f"net_income_t{t}"]
            fcf = (r[f"free_cash_flow_t{t}"] if pd.notna(r[f"free_cash_flow_t{t}"])
                   else r[f"operating_cash_flow_t{t}"] + r[f"capex_t{t}"])
            ta = r[f"total_assets_t{t}"]
            hist += [{"Fiscal year end": r[f"fy_end_t{t}"], "Measure": "Gross margin", "Value": gp / rev if rev else np.nan},
                     {"Fiscal year end": r[f"fy_end_t{t}"], "Measure": "ROA", "Value": ni / ta},
                     {"Fiscal year end": r[f"fy_end_t{t}"], "Measure": "FCF / total assets", "Value": fcf / ta},
                     {"Fiscal year end": r[f"fy_end_t{t}"], "Measure": "Accruals: (NI - CFO) / TA",
                      "Value": (r[f"net_income_t{t}"] - r[f"operating_cash_flow_t{t}"]) / ta}]
    if hist:
        h = pd.DataFrame(hist).sort_values("Fiscal year end")
        h["Fiscal year end"] = "FY ending " + h["Fiscal year end"].astype(str).str[:7]
        fig = px.line(h, x="Fiscal year end", y="Value", color="Measure", markers=True,
                      title="Trends over the available fiscal years")
        fig.update_layout(height=380, yaxis_tickformat=".0%", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    if isinstance(r.get("summary"), str) and r["summary"]:
        with st.expander("Business description (Yahoo Finance)"):
            st.write(r["summary"] + ("..." if len(r["summary"]) >= 600 else ""))

# ---------------------------------------------------------------------------------------------
# Conclusions tab (written investment case)
# ---------------------------------------------------------------------------------------------
CONCLUSIONS = """
*Written for the default filter settings, using data pulled October 21, 2026. The screen finds candidates; the
case below adds what the numbers alone cannot show. Percentiles quoted are within the default universe of 876 firms.*

### SELL (short) candidates

**1. CoreWeave (CRWV): debt-funded growth with every distress signal lit.**
The screen ranks CoreWeave as the single most distressed company in the universe (PFD 100th percentile), with an
Altman Z of 0.26, far below 1.81. Beneish PMAN is 44% (97th percentile), driven by receivables growing almost three
times faster than sales (DSRI 2.84), and 19% of the float is sold short. The numbers behind the flags: revenue went
from $16 million in 2022 to $5.1 billion in 2025, but the growth was bought with borrowed money. Capital spending was
$10.3 billion in 2025, cumulative free cash flow over four years is about -$14.4 billion, and 2025's net loss was
$1.2 billion. In Q1 2026 alone CoreWeave spent $7.7 billion on capex, lost $740 million and recorded $536 million of
interest expense, with $50.8 billion of liabilities against $4.8 billion of equity. The bull case is a roughly $99 billion
backlog from a handful of AI customers, but interest is due whether or not that backlog converts on time. The stock
is already down 57% over 12 months, so this is a solvency short, not a valuation short.
*What would change my mind:* free cash flow turning positive as the backlog converts, or cheaper refinancing.

**2. Carvana (CVNA): the business is real, the earnings quality is not.**
Carvana trips four flags: STA in the 96th percentile (net income of $1.4 billion against operating cash flow of
$1.0 billion), PMAN of 38% (97th percentile, mostly a jump in the asset-quality index to 5.4), PFD in the 91st
percentile and 12.6% short interest. Beyond the score, 2025 operations were genuinely strong: retail units rose 43%
and operating income reached $1.9 billion. But after a $2.25 billion "other expense", pretax income was a loss of
about $0.9 billion, and net income only turned positive because of a $2.8 billion income tax benefit. Carvana's own
release says net income included a $621 million net non-cash benefit from releasing its tax valuation allowance and
recording a tax receivable agreement liability. That new deferred tax asset is what pushed AQI up. At a $67 billion
market cap, investors are paying for headline earnings that are larger than the cash earnings, which is exactly the
accrual gap Sloan found tends to reverse.
*What would change my mind:* operating cash flow catching up with reported earnings for several quarters.

**3. AeroVironment (AVAV): an acquisition that bloated the balance sheet.**
AeroVironment's SNOA is in the 97th percentile, PMAN is 43% (97th percentile), PFD is in the 95th percentile and
short interest is 12.8%. Revenue more than doubled to $2.0 billion in fiscal 2026 (year to April 2026) because of
the BlueHalo acquisition, yet gross margin fell from about 39% to 25%, the company lost $265 million and operating
cash flow was negative. Goodwill and intangibles are $3.4 billion of its $5.7 billion of assets (60%), and three
Beneish indexes sit above the manipulators' average (GMI 1.53, AQI 1.62, SGI 2.41). The first crack has already
appeared: a $151 million goodwill impairment in its Space unit after a Space Force stop-work order. The stock is down
68% over 12 months, but with that much goodwill left, further write-downs are the risk.

**Red flags I would not act on.** GATX ranks second, but it leases railcars: owning long-lived assets financed with
debt is its business model, so a high SNOA and a low Altman Z are structural (compare it with peers using the
Industries filter). Jersey Mike's (JMKE) went public in 2026 after Blackstone's buyout; buyout accounting makes
goodwill and intangibles 96% of its assets, which pushes SNOA and the Beneish asset-quality index to extremes, and
only four of six flags can be checked with its short history. QXO shows a PMAN of 100% because revenue jumped 120-fold
after it bought Beacon Roofing Supply (SGI of 120): that is an acquisition, not manipulation, and the Beneish model
was not built for transformational deals.

### BUY (long) candidates

**1. Adobe (ADBE): a franchise priced like a melting ice cube.**
Adobe's franchise power is in the 99.7th percentile: four years of free cash flow ($32 billion) exceed its total
assets (CFOA 1.09), long-term return on capital is 55% and gross margin is 89%. Yet it sits in the 89th percentile
for cheapness, with an EBIT/TEV of 8.4%, and passes 8 of 10 financial-strength signals with no red flags. The price
fell 25% over 12 months on fears that new AI design tools will commoditize creative software, plus CEO Shantanu
Narayen's announced departure in March 2026. Meanwhile revenue is still growing (Q1 fiscal 2026 up 12%) and
management is shrinking the share count hard: $11.3 billion of buybacks in fiscal 2025 and a new $25 billion
authorization. This is the textbook Quantitative Value setup: the market extrapolating bad news onto a high-quality
business. Note that on book-to-market (0.12) Adobe looks expensive, because buybacks shrink book equity; this is why
the book prefers EBIT/TEV (Ch. 7).
*Key risk:* AI disruption is real. I would watch subscription growth and AI product revenue.

**2. Cirrus Logic (CRUS): top-tier quality with one big caveat.**
Cirrus has the second-highest quality percentile in the screen (99.8th) and a perfect financial-strength score of
10/10, yet trades at an EBIT/TEV of 10.2% (93rd percentile) and a book-to-market of 0.38. It carries $134 million of
debt against $801 million of cash, earned record EPS in fiscal 2026 on a 52.8% gross margin, and bought back $280
million of stock. What the screen cannot see is concentration: one customer (Apple) supplied about 91% of fiscal 2026
revenue. Growth in PC chips and mixed-signal products is the diversification story, but I would hold a smaller
position than the score alone suggests.

**3. Mueller Industries (MLI): a debt-free compounder in an unglamorous industry.**
Mueller's quality is in the 97th percentile, built on four-year averages of 23% ROA and 59% ROC, and it trades at an
EBIT/TEV of 9.0% (90th percentile) with 8 of 10 financial-strength signals. 2025 was a record year (net income up 27%
to $765 million) and it ended the year with no debt, $1.39 billion of cash and a fivefold increase in buybacks to
$244 million. Two caveats: higher copper prices lifted selling prices while fourth-quarter volumes fell, and 2025
included about $67 million of one-time gains. The stock is up 25% over 12 months, so it is less of a contrarian
bargain than Adobe or Cirrus.

**Why the gold miners top the ranking but are not my picks.** Kinross, Gold Fields and AngloGold rank near the top
because record gold prices have inflated their margins: Kinross's gross margin grew 52% a year, but its margin
stability is only 2.1 (H&R Block's is 213). The book's Chapter 8 warns that peak-cycle earnings make cyclical stocks
look cheap; normalized for the gold price, they would not be.

### Robustness
Adobe, Cirrus Logic and Mueller stay near the top of the BUY list under the book's "value first, then quality"
method. CoreWeave and AeroVironment stay SELL candidates when percentiles are ranked within each sector. The themes
are consistent: the SELL list is dominated by debt-funded or acquisition-driven growth, and the BUY list by
cash-rich, buyback-heavy businesses that have been marked down.

*This is a class project, not investment advice.*
"""

with tab_concl:
    st.subheader("Investment conclusions")
    st.markdown(CONCLUSIONS.replace("$", "\\$"))   # escape $ so Streamlit does not read it as math
    c1, c2 = st.columns(2)
    c1.markdown("**Current top 5 SELL names (live)**")
    c1.dataframe(sell.head(5)[["ticker", "name", "red_flags", "SELL_score"]].rename(
        columns={"ticker": "Ticker", "name": "Company", "red_flags": "Red flags", "SELL_score": "SELL score"}),
        hide_index=True, width="stretch", column_config={"SELL score": num("SELL score", "%.0f")})
    c2.markdown("**Current top 5 BUY names (live)**")
    c2.dataframe(buy.head(5)[["ticker", "name", "price_score", "quality_score"]].rename(
        columns={"ticker": "Ticker", "name": "Company", "price_score": "Price pct", "quality_score": "Quality pct"}),
        hide_index=True, width="stretch",
        column_config={"Price pct": num("Price pct", "%.0f"), "Quality pct": num("Quality pct", "%.0f")})

# ---------------------------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------------------------
with tab_method:
    st.subheader("How the screens work")
    st.markdown(f"""
**Universe.** The {len(data):,} largest US-listed companies (NYSE and NASDAQ, market cap of at least \\$300M) from
Yahoo Finance's screener, excluding Financial Services. Data pulled with `yfinance` in Google Colab on {as_of}:
up to four annual statements, the last four quarters of net income, and two years of daily prices.
Statements reported in another currency are converted to US dollars. Duplicate share classes are dropped.

**Rule 2 (SELL is not BUY flipped).** The SELL screen uses accounting-quality and distress measures
(STA, SNOA, PMAN, Altman Z, PFD, short interest). The BUY screen ranks on completely different measures:
price (EBIT/TEV, B/M) and quality (franchise power, financial strength, GPA). The optional margin-of-safety
switch only removes SELL candidates from the BUY list, as the book's checklist does (Ch. 11, step 1).

**Percentiles.** Each measure is converted to a percentile (0 to 100) among the companies that pass the
universe filters, or within each sector if that box is ticked. Percentiles are recalculated on every change.
""")
    st.markdown("#### SELL measures (higher = worse, except Altman Z)")
    st.markdown(r"""
| Measure | Formula | Source |
|---|---|---|
| STA | (Net income − Cash flow from operations) / Total assets | Ch. 3, simplified version used in class |
| SNOA | (Operating assets − Operating liabilities) / Total assets; OA = TA − Cash; OL = TA − ST debt − LT debt − Minority interest − Preferred − Common equity | Ch. 3 |
| PROBM | −4.84 + 0.92 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI − 0.172 SGAI + 4.679 TATA − 0.327 LVGI | Ch. 3 (Beneish 1999) |
| PMAN | N(PROBM): standard normal CDF, the probability of manipulation | Ch. 3 |
| Altman Z | 0.012 X1 + 0.014 X2 + 0.033 X3 + 0.006 X4 + 0.999 X5; X1 = WC/TA, X2 = RE/TA, X3 = EBIT/TA, X4 = MV equity/Total liabilities, X5 = Sales/TA; cutoff 1.81 | Ch. 4 |
| PFD | 1 / (1 + e^−LPFD), LPFD = −20.26 NIMTAAVG + 1.42 TLMTA − 7.13 EXRETAVG + 1.41 SIGMA − 0.045 RSIZE − 2.13 CASHMTA + 0.075 MB − 0.058 PRICE − 9.16 | Ch. 4 (Campbell, Hilscher and Szilagyi) |
| Short interest | Shares sold short / Float | Ch. 9 |
""")
    st.markdown("#### BUY measures (higher = better)")
    st.markdown(r"""
| Measure | Formula | Source |
|---|---|---|
| EBIT/TEV | EBIT / (Market cap + Total debt + Preferred + Minority interest − Excess cash) | Ch. 2 and 7 |
| B/M | Common equity / Market cap | Ch. 2 |
| GPA | (Revenue − COGS) / Total assets | Ch. 2 |
| Greenblatt ROC | EBIT / (Net PPE + Current assets − Current liabilities) | Ch. 2 (shown for reference) |
| Long-term ROA, ROC | Geometric average of yearly ROA (NI before extraordinary items / TA) and ROC (EBIT / (Debt + Equity − Cash)) | Ch. 5 |
| CFOA | Sum of yearly free cash flow / Total assets (latest year) | Ch. 5 |
| Margin growth (MG) | (GM newest / GM oldest)^(1/T) − 1, T = number of growth periods | Ch. 5 |
| Margin stability (MS) | Average gross margin / Standard deviation of gross margin | Ch. 5 |
| Max margin (MM) | max(percentile of MG, percentile of MS) | Ch. 5 |
| Franchise power (FP) | Percentile of the average of the LT ROA, LT ROC, CFOA and MM percentiles | Ch. 5 / Ch. 11 checklist |
| FS score | Ten 0/1 signals: ROA > 0, FCF/TA > 0, FCF/TA > ROA, leverage fell, current ratio rose, net buybacks, ROA, FCF/TA, gross margin and asset turnover improved | Ch. 6 |
| QUALITY | 0.5 × FP percentile + 0.5 × FS score (as a percent) | Ch. 11 checklist |
""")
    st.markdown("""
#### Choices, approximations and limitations
* **Missing data stays missing.** A company is never flagged on a measure it lacks the inputs for; the
  SELL table shows how many flags could be checked. The only lines treated as zero when absent are those
  Yahoo omits when they are zero: preferred stock, minority interest, debt lines, taxes payable, buybacks and
  share issuance (the class SNOA notes also say "Preferred Stock Equity (0 if absent)").
* **Beneish.** Indexes whose inputs are not reported (receivables, SG&A, depreciation, PPE, COGS) are set to a
  neutral 1.0, as the book does for Enron's SGAI. PROBM is left missing when more than two indexes would need this.
  TATA uses Sloan's working-capital accrual definition from the book's checklist.
* **Altman Z scaling (cutoff 1.81).** The textbook formula has coefficients 0.012, 0.014, 0.033, 0.006 and 0.999.
  Altman (1968) applied the first four to ratios written as percentages (4.3 for 4.3%), which is the same as
  coefficients of 1.2, 1.4, 3.3 and 0.6 on decimal ratios. The class Empire example entered decimals (Z = 1.89); with
  percentages, Empire scores 2.77. With decimals, Z is almost entirely X5 (asset turnover): in this universe about
  9 in 10 companies, including Apple, Microsoft and NVIDIA, fall below 1.81, so the cutoff cannot separate distressed
  firms. With Altman's scaling about 1 in 5 do. The app therefore defaults to Altman's scaling with the 1.81 cutoff;
  the class-example version is one click away under *Advanced* in the sidebar.
* **Excess cash for TEV.** The book prints excess cash as "cash + current assets − current liabilities", which
  counts cash twice and can make TEV negative. The app uses Greenblatt's definition: cash minus any cash needed to
  cover a working-capital shortfall, never below zero.
* **History.** Yahoo provides four fiscal years, so long-term franchise measures use four years (three growth
  periods) instead of the book's eight. Firms with fewer than three years get no long-term measure.
* **PFD inputs.** Quarterly net income is scaled by the current market value of assets. Excess returns use four
  rolling 3-month windows against the S&P 500 Total Return index. The S&P 500's total value is approximated by the
  total market cap of this universe. Inputs are winsorized at the 5th and 95th percentiles, as Campbell et al. do.
* **Exclusions.** Financial Services (excluded when the data was built), Real Estate and Utilities (off by default,
  can be added back in the sidebar), the 60-second pitch company (Krispy Kreme, DNUT) and rows with known data errors
  (listed on the Data coverage tab). In within-sector mode, sectors with fewer than 10 companies use universe-wide ranks.
* **Not investment advice.** Yahoo data can contain errors, and screens identify candidates for research, not trades.
""")

# ---------------------------------------------------------------------------------------------
# Data coverage tab
# ---------------------------------------------------------------------------------------------
with tab_cov:
    st.subheader("How many screened companies have each measure")
    measures = {"STA": "STA", "SNOA": "SNOA", "PMAN": "PMAN", "Z": "Altman Z", "PFD": "PFD",
                "short_pct_float": "Short interest", "EBIT_TEV": "EBIT/TEV", "BM": "B/M", "GPA": "GPA",
                "LT_ROA": "Long-term ROA", "LT_ROC": "Long-term ROC", "CFOA": "CFOA", "MG": "Margin growth",
                "MS": "Margin stability", "P_FP": "Franchise power", "FS_SCORE": "FS score", "QUALITY": "QUALITY"}
    cov = pd.DataFrame({"Measure": list(measures.values()),
                        "Companies with a value": [int(u[c].notna().sum()) for c in measures],
                        "Share of screened": [u[c].notna().mean() for c in measures]})
    st.dataframe(cov, hide_index=True, width="stretch",
                 column_config={"Share of screened": st.column_config.ProgressColumn(
                     "Share of screened", min_value=0, max_value=1, format="percent")})
    st.caption("A company missing a measure is never flagged on it and is left out of that measure's percentiles.")
    st.markdown("**Rows removed because of data errors**")
    for t, why in DATA_ERRORS.items():
        st.markdown(f"- **{t}**: {why}")
    st.markdown(f"**Removed for assignment rule 1 (pitch company):** {', '.join(EXCLUDE_TICKERS)}"
                + ("" if excluded_tickers else " (not among the companies in the file, so nothing needed removing)"))
