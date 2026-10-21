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
import html
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

# Red pencil for SELL, ledger-blue ink for BUY (red and blue stay distinct for colour-blind readers)
STATUS_COLORS = {"SELL": "#D64550", "BUY": "#2A7AB8", "BUY and SELL": "#D99A2B", "Neither": "#A3ABB5"}


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
# Look and feel: IBM Plex type, red-pencil SELL / ledger-blue BUY, works in light and dark mode
# ---------------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp h1, .stApp h2, .stApp h3 {font-family: 'IBM Plex Sans Condensed', sans-serif; letter-spacing: -0.01em;}
.stMarkdown p, .stMarkdown li {font-family: 'IBM Plex Sans', sans-serif;}
.block-container {padding-top: 4.2rem; max-width: 1380px;}
.eyebrow {font-family: 'IBM Plex Mono', monospace; font-size: .72rem; letter-spacing: .14em;
          text-transform: uppercase; opacity: .65; margin: 0 0 .35rem 0;}
.hero-title {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; font-size: 2.8rem;
             line-height: 1.05; margin: 0;}
.hero-sub {font-family: 'IBM Plex Sans', sans-serif; font-size: 1.04rem; line-height: 1.5; opacity: .82;
           margin: .6rem 0 .3rem 0; max-width: 60rem;}
.hero-note {font-family: 'IBM Plex Sans', sans-serif; font-size: .8rem; opacity: .6; margin: 0 0 1.1rem 0;}
/* Ledger strip: the headline numbers, set like one row of a ledger */
.ledger {display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid rgba(128,128,128,.35);
         border-radius: 10px; overflow: hidden; margin: 0 0 .8rem 0;}
.ledger > div {padding: .75rem 1.1rem; border-right: 1px solid rgba(128,128,128,.25);}
.ledger > div:last-child {border-right: none;}
.ledger .k {font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .1em;
            text-transform: uppercase; opacity: .65;}
.ledger .v {font-family: 'IBM Plex Mono', monospace; font-size: 1.85rem; font-weight: 600; line-height: 1.25;}
.ledger .s {font-size: .78rem; opacity: .7;}
.ledger .sell .v {color: #D64550;}
.ledger .buy .v {color: #2A7AB8;}
/* Red-flag strip: one box per warning sign, like an auditor's tick marks */
.strip {display: inline-flex; gap: 3px; vertical-align: middle; flex-shrink: 0;}
.strip span {font-family: 'IBM Plex Mono', monospace; font-size: .58rem; width: 2.25rem; text-align: center;
             padding: 2px 0; border-radius: 3px; border: 1px solid rgba(128,128,128,.45); opacity: .6;}
.strip span.on {background: #D64550; border-color: #D64550; color: #fff; opacity: 1; font-weight: 600;}
.strip span.na {border-style: dashed; opacity: .35;}
.strip span.off {opacity: .18;}
/* Idea lists on the overview */
.panel-title {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; font-size: 1.2rem;
              margin: .3rem 0 .1rem 0;}
.panel-sub {font-family: 'IBM Plex Sans', sans-serif; font-size: .8rem; opacity: .65; margin-bottom: .4rem;}
.idea {display: flex; align-items: center; gap: .8rem; padding: .5rem .1rem;
       border-bottom: 1px solid rgba(128,128,128,.2);}
.idea .tk {font-family: 'IBM Plex Mono', monospace; font-weight: 600; min-width: 4.1rem;}
.idea .nm {flex: 1; min-width: 0; line-height: 1.25;}
.idea .nm b {font-weight: 500; display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.idea .why {font-size: .76rem; opacity: .68; display: block; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis;}
.figs {display: flex; gap: 1rem; flex-shrink: 0;}
.figs span {display: flex; flex-direction: column; align-items: flex-end; min-width: 3rem;}
.figs b {font-family: 'IBM Plex Mono', monospace; font-size: .95rem; font-weight: 600;}
.figs i {font-style: normal; font-size: .62rem; letter-spacing: .06em; text-transform: uppercase; opacity: .6;}
.tag {display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: .72rem; font-weight: 600;
      padding: 2px 10px; border-radius: 999px; color: #fff; vertical-align: middle;}
.co-head {display: flex; flex-wrap: wrap; align-items: center; gap: .7rem; margin: .4rem 0 .2rem 0;}
.co-name {font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; font-size: 1.6rem;}
.co-tk {font-family: 'IBM Plex Mono', monospace; opacity: .7;}
.footer {font-size: .75rem; opacity: .55; margin-top: 2.5rem; border-top: 1px solid rgba(128,128,128,.25);
         padding-top: .6rem;}
@media (max-width: 760px) {
  .hero-title {font-size: 2rem;}
  .ledger {grid-template-columns: repeat(2, 1fr);}
  .ledger > div:nth-child(2) {border-right: none;}
  .ledger > div:nth-child(-n+2) {border-bottom: 1px solid rgba(128,128,128,.25);}
  .strip span {width: 2.1rem;}
}
</style>
""", unsafe_allow_html=True)


def esc(text):
    """Make text safe to place inside HTML; '$' becomes an entity so Streamlit does not read it as math."""
    return html.escape(str(text)).replace("$", "&#36;")


# ---------------------------------------------------------------------------------------------
# Sidebar settings. Every widget has a key, so presets and the reset button can set them.
# Moving any setting reruns the whole script, which recalculates every screen below.
# ---------------------------------------------------------------------------------------------
all_sectors = sorted(data["sector"].dropna().unique())
PRICE_OPTS = ["EBIT/TEV (Ch. 7)", "Book-to-market (Ch. 2)", "Average of both"]
QUALITY_OPTS = ["QUALITY: franchise power + financial strength (Ch. 5-6)", "GPA (Ch. 2)", "Average of both"]
METHOD_OPTS = ["Combined rank: price rank + quality rank (Ch. 2)", "Value first, then quality (Ch. 11)"]
ALTMAN_OPTS = ["Altman's scaling: X1-X4 as percentages (EMP Z = 2.77)",
               "Class example: X1-X4 as decimals (EMP Z = 1.89)"]
DEFAULTS = {
    "sectors": [s for s in all_sectors if s not in ("Real Estate", "Utilities")], "industries": [],
    "min_cap": 0.3, "us_only": False, "within_sector": False,
    "use_sta": True, "sta_cut": 90, "use_snoa": True, "snoa_cut": 90, "use_pman": True, "pman_cut": 90,
    "use_z": True, "z_cut": 1.81, "use_pfd": True, "pfd_cut": 90, "use_si": True, "si_cut": 10.0,
    "flags_needed": 3, "price_choice": PRICE_OPTS[0], "quality_choice": QUALITY_OPTS[0], "method": METHOD_OPTS[0],
    "price_cut": 80, "quality_cut": 70, "fs_min": 6, "buy_excludes_sell": True, "top_n": 25,
    "altman_version": ALTMAN_OPTS[0],
}
PRESETS = {  # each preset only lists the settings it changes from the defaults
    "Balanced (default)": {},
    "Book checklist (Ch. 11)": {"sta_cut": 95, "snoa_cut": 95, "pman_cut": 95, "pfd_cut": 95, "use_z": False,
                                "use_si": False, "flags_needed": 1, "method": METHOD_OPTS[1], "price_cut": 90,
                                "quality_cut": 50, "fs_min": 0},
    "High conviction (fewer names)": {"flags_needed": 4, "price_cut": 90, "quality_cut": 80, "fs_min": 7},
    "Wide net (more names)": {"sta_cut": 85, "snoa_cut": 85, "pman_cut": 85, "pfd_cut": 85, "flags_needed": 2,
                              "price_cut": 70, "quality_cut": 60, "fs_min": 5},
}
PRESET_HELP = {
    "Balanced (default)": "The settings used for the written conclusions.",
    "Book checklist (Ch. 11)": "SELL = in the riskiest 5% on accruals, manipulation or distress. BUY = cheapest 10% "
                               "on EBIT/TEV, ranked by quality.",
    "High conviction (fewer names)": "Stricter on both sides: 4 red flags to SELL, cheapest 10% and top-20% quality to BUY.",
    "Wide net (more names)": "Looser cutoffs for a longer research list.",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
st.session_state.setdefault("preset", "Balanced (default)")


def apply_preset():
    for k, v in {**DEFAULTS, **PRESETS[st.session_state["preset"]]}.items():
        st.session_state[k] = v


def reset_all():
    st.session_state["preset"] = "Balanced (default)"
    apply_preset()


sb = st.sidebar
sb.markdown('<div class="eyebrow">Screen settings</div>', unsafe_allow_html=True)
sb.selectbox("Start from a preset", list(PRESETS), key="preset", on_change=apply_preset,
             help="Loads a ready-made set of cutoffs. You can still change anything afterwards.")
sb.caption(PRESET_HELP[st.session_state["preset"]])
sb.button("Reset all settings", on_click=reset_all, width="stretch")

sb.subheader("1 · Which companies")
sectors = sb.pills("Sectors", all_sectors, selection_mode="multi", key="sectors",
                   help="Financial Services were left out when the data was built: accrual, SNOA and Altman measures "
                        "do not fit banks and insurers. Real Estate and Utilities start switched off: REIT accounting "
                        "distorts accruals, and utilities carry heavy assets and debt by design.")
sectors = sectors or []
industry_pool = sorted(data.loc[data["sector"].isin(sectors), "industry"].dropna().unique())
st.session_state["industries"] = [i for i in st.session_state["industries"] if i in industry_pool]
industries = sb.multiselect("Narrow to industries (optional)", industry_pool, key="industries",
                            placeholder="All industries in the chosen sectors",
                            help="Pick one or more industries for a within-industry comparison.")
min_cap = sb.select_slider("Minimum market cap ($ billions)", options=[0.3, 0.5, 1, 2, 5, 10, 25, 50, 100],
                           key="min_cap")
us_only = sb.toggle("US-domiciled companies only", key="us_only")
within_sector = sb.toggle("Compare each firm with its own sector", key="within_sector",
                          help="Off: percentiles compare each company with every screened company (the book's approach). "
                               "On: each company is ranked only against its sector.")

sb.subheader("2 · What makes a SELL")
sb.caption("Each red flag marks the riskiest companies on one measure. A percentile of 90 flags the riskiest 10%.")
use_sta = sb.toggle("Accruals (STA, Ch. 3)", key="use_sta",
                    help="How much of reported profit is not backed by operating cash flow. High = earnings running "
                         "ahead of cash.")
sta_cut = sb.slider("STA flag at percentile", 50, 99, key="sta_cut", disabled=not use_sta)
use_snoa = sb.toggle("Bloated balance sheet (SNOA, Ch. 3)", key="use_snoa",
                     help="Operating assets piled up relative to operating liabilities. High = past earnings leaned on "
                          "accruals.")
snoa_cut = sb.slider("SNOA flag at percentile", 50, 99, key="snoa_cut", disabled=not use_snoa)
use_pman = sb.toggle("Manipulation score (Beneish PMAN, Ch. 3)", key="use_pman",
                     help="Beneish's probability that earnings are manipulated, built from eight ratios such as "
                          "receivables growth and falling margins.")
pman_cut = sb.slider("PMAN flag at percentile", 50, 99, key="pman_cut", disabled=not use_pman)
use_z = sb.toggle("Bankruptcy risk (Altman Z, Ch. 4)", key="use_z",
                  help="Altman's bankruptcy score. Below 1.81 is the distress zone.")
z_cut = sb.slider("Altman Z flag below", 0.0, 4.0, step=0.01, key="z_cut", disabled=not use_z)
use_pfd = sb.toggle("Distress probability (Campbell et al. PFD, Ch. 4)", key="use_pfd",
                    help="Probability of financial distress from profits, leverage, cash, volatility and recent stock "
                         "returns.")
pfd_cut = sb.slider("PFD flag at percentile", 50, 99, key="pfd_cut", disabled=not use_pfd)
use_si = sb.toggle("Short interest (Ch. 9)", key="use_si",
                   help="Share of the float sold short. Short sellers are often well-informed, so heavy shorting "
                        "corroborates the accounting flags.")
si_cut = sb.slider("Short interest flag at (% of float)", 1.0, 30.0, step=0.5, key="si_cut", disabled=not use_si)
n_active = sum([use_sta, use_snoa, use_pman, use_z, use_pfd, use_si])
if n_active == 0:
    sb.warning("Turn on at least one red flag to build a SELL list.")
st.session_state["flags_needed"] = min(st.session_state["flags_needed"], max(n_active, 1))
flags_needed = sb.slider("Red flags needed to SELL", 1, max(n_active, 1), key="flags_needed",
                         disabled=n_active <= 1)

sb.subheader("3 · What makes a BUY")
price_choice = sb.radio("Price measure (how cheap)", PRICE_OPTS, key="price_choice",
                        help="EBIT/TEV = operating earnings per dollar of enterprise value; the book's best performer.")
quality_choice = sb.radio("Quality measure (how good)", QUALITY_OPTS, key="quality_choice",
                          help="QUALITY blends long-term profitability, cash generation and margin strength with ten "
                               "financial-strength checks. GPA is gross profit over assets.")
method = sb.radio("Ranking method", METHOD_OPTS, key="method",
                  help="Combined rank adds the price and quality ranks (class method). Value first ranks the cheap "
                       "names by quality only (book method).")
price_cut = sb.slider("Price percentile at least (100 = cheapest)", 0, 99, key="price_cut")
quality_cut = sb.slider("Quality percentile at least", 0, 99, key="quality_cut")
fs_min = sb.slider("Financial strength at least (0 to 10)", 0, 10, key="fs_min")
buy_excludes_sell = sb.toggle("Never BUY a SELL candidate", key="buy_excludes_sell",
                              help="Book Ch. 11, step 1: remove likely manipulators and distressed firms before buying. "
                                   "This only removes names; the BUY ranking uses different measures.")

with sb.expander("More options"):
    top_n = st.slider("Rows shown in each list", 10, 100, step=5, key="top_n")
    altman_version = st.radio("Altman Z inputs", ALTMAN_OPTS, key="altman_version",
                              help="With decimals, Z is almost entirely asset turnover and about 9 in 10 companies "
                                   "fall below 1.81. See Methodology.")
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
if len(u) == 0:
    st.warning("No companies match these settings. Choose at least one sector, or lower the minimum market cap.")
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
FLAG_ABBR = {"flag_STA": "STA", "flag_SNOA": "SNOA", "flag_PMAN": "PMAN", "flag_Z": "Z", "flag_PFD": "PFD",
             "flag_SHORT": "SI"}
FLAG_WORDS = {"flag_STA": "accruals", "flag_SNOA": "bloated balance sheet", "flag_PMAN": "manipulation score",
              "flag_Z": "Altman Z", "flag_PFD": "distress probability", "flag_SHORT": "short interest"}
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
u["why_sell"] = u.apply(lambda r: ", ".join(FLAG_WORDS[n] for on, n, _, _ in flag_defs if on and r[n]), axis=1)

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
elif quality_choice.startswith("GPA"):
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
u["why_buy"] = ("EBIT/TEV " + (u["EBIT_TEV"] * 100).round(1).astype(str) + "% · quality "
                + u["quality_score"].round(0).astype("Int64").astype(str) + " pct · strength "
                + u["FS_SCORE"].round(0).astype("Int64").astype(str) + "/10")

sell = u[u["SELL"]].sort_values(["red_flags", "SELL_score"], ascending=[False, False])
buy = u[u["BUY"]].sort_values("BUY_rank_value")
sell = sell.assign(Rank=range(1, len(sell) + 1))
buy = buy.assign(Rank=range(1, len(buy) + 1))


# ---------------------------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------------------------
def flag_strip(r):
    """Six boxes, one per red flag: filled = tripped, outlined = clear, dashed = no data, faint = switched off."""
    cells = []
    for on, name, _, src in flag_defs:
        if not on:
            cls, state = "off", "switched off"
        elif pd.isna(r[src]):
            cls, state = "na", "no data"
        elif r[name]:
            cls, state = "on", "red flag"
        else:
            cls, state = "", "clear"
        cells.append(f'<span class="{cls}" title="{FLAG_WORDS[name]}: {state}">{FLAG_ABBR[name]}</span>')
    return f'<span class="strip">{"".join(cells)}</span>'


def tag(status):
    return f'<span class="tag" style="background:{STATUS_COLORS[status]}">{esc(status)}</span>'


def fmt(v, kind="num"):
    if pd.isna(v):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "rank":
        return f"{v:.0f}"
    return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"


def fit(df):
    """Show a small table with all rows visible (no inner scrollbar)."""
    st.dataframe(df, hide_index=True, width="stretch", height=35 * (len(df) + 1) + 3)


def style_fig(fig, height):
    fig.update_layout(height=height, font_family="IBM Plex Sans, Source Sans Pro, Helvetica, Arial, sans-serif",
                      legend_title_text="",
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


pct_col = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f")
num = lambda label, fmt_="%.3f": st.column_config.NumberColumn(label, format=fmt_)
pct_fmt = lambda label: st.column_config.NumberColumn(label, format="percent")


def flag_reasons(r):
    """Plain-language sentences for each red flag the company trips."""
    text = {"flag_STA": f"Accruals: STA of {fmt(r['STA'])} is in the {fmt(r['P_STA'], 'rank')}th percentile, so "
                        "earnings are running ahead of operating cash flow.",
            "flag_SNOA": f"Bloated balance sheet: SNOA of {fmt(r['SNOA'])} ({fmt(r['P_SNOA'], 'rank')}th percentile). "
                         "Operating assets have piled up relative to operating liabilities.",
            "flag_PMAN": f"Manipulation score: Beneish PMAN of {fmt(r['PMAN'], 'pct')} "
                         f"({fmt(r['P_PMAN'], 'rank')}th percentile).",
            "flag_Z": f"Bankruptcy risk: Altman Z of {fmt(r['Z'])} is below the {z_cut:.2f} cutoff.",
            "flag_PFD": f"Distress probability in the {fmt(r['P_PFD'], 'rank')}th percentile: weak profits, high "
                        "leverage or volatility, or poor recent returns.",
            "flag_SHORT": f"Short interest: {fmt(r['short_pct_float'], 'pct')} of the float is sold short."}
    return [text[n] for on, n, _, _ in flag_defs if on and r[n]]


def company_header(r):
    st.markdown(f'<div class="co-head"><span class="co-name">{esc(r["name"])}</span>'
                f'<span class="co-tk">{esc(r["ticker"])}</span>{tag(r["Status"])}{flag_strip(r)}</div>',
                unsafe_allow_html=True)
    st.caption(f"{r['sector']} · {r['industry']} · latest fiscal year ending {r.get('fy_end_t0', 'n/a')}")


def company_summary(r):
    """Why the company is (or is not) on each list, in plain language."""
    st.markdown(f"**SELL side:** {int(r['red_flags'])} of {n_active} red flags "
                f"({flags_needed} needed; {int(r['flags_checked'])} could be checked with the available data).")
    for x in flag_reasons(r):
        st.markdown(f"- {x}")
    st.markdown(f"**BUY side:** price percentile {fmt(r['price_score'], 'rank')} (needs {price_cut}), quality "
                f"percentile {fmt(r['quality_score'], 'rank')} (needs {quality_cut}), financial strength "
                f"{fmt(r['FS_SCORE'], 'rank')}/10 (needs {fs_min}).")


# ---------------------------------------------------------------------------------------------
# Page header: what this is, and the headline numbers as one ledger row
# ---------------------------------------------------------------------------------------------
st.markdown(f"""
<div class="eyebrow">ACCT 3343 · Analysis app · {len(data):,} US-listed companies · data pulled {esc(as_of)}</div>
<div class="hero-title">Quantitative Value Screener</div>
<div class="hero-sub">Which large companies look like shorts, which look like bargains, and why. The SELL list hunts
for accounting and distress red flags; the BUY list looks for high-quality businesses at low prices.
Change any setting on the left and every list, chart and count updates.</div>
<div class="hero-note">Krispy Kreme ({esc(", ".join(EXCLUDE_TICKERS))}), my 60-second pitch company, is excluded
under assignment rule 1.</div>
<div class="ledger">
  <div><div class="k">Companies screened</div><div class="v">{len(u):,}</div>
       <div class="s">of {len(data):,} in the data</div></div>
  <div class="sell"><div class="k">SELL candidates</div><div class="v">{len(sell):,}</div>
       <div class="s">{len(sell) / len(u) * 100:.1f}% of screened · {flags_needed} of {n_active} flags needed</div></div>
  <div class="buy"><div class="k">BUY candidates</div><div class="v">{len(buy):,}</div>
       <div class="s">{len(buy) / len(u) * 100:.1f}% of screened</div></div>
  <div><div class="k">BUY rule</div><div class="v" style="font-size:1.15rem;padding-top:.45rem">price ≥ {price_cut}
       · quality ≥ {quality_cut}</div><div class="s">strength ≥ {fs_min}/10 · percentiles, 100 = best</div></div>
</div>
""", unsafe_allow_html=True)

with st.expander("New here? How to use this app"):
    st.markdown("""
1. **Choose which companies to screen** in section 1 of the sidebar: sectors, industries, size.
2. **Decide what makes a SELL** in section 2: switch red flags on or off, set how strict each one is, and how many a
   company must trip.
3. **Decide what makes a BUY** in section 3: pick the price and quality measures and how high each must rank.

Click any row in the SELL or BUY list to see why that company is there. The **Company file** tab shows every number
behind a company. Not sure where to start? Pick a preset at the top of the sidebar; **Reset all settings** brings back
the defaults.

**Reading the red-flag strip:** each box is one warning sign
(STA = accruals, SNOA = bloated balance sheet, PMAN = manipulation score, Z = Altman Z, PFD = distress probability,
SI = short interest). Filled red means the company trips that flag; a dashed box means there was not enough data.
""")

tab_over, tab_sell, tab_buy, tab_co, tab_concl, tab_method, tab_cov = st.tabs(
    [":material/dashboard: Overview", ":material/trending_down: SELL list", ":material/trending_up: BUY list",
     ":material/search: Company file", ":material/edit_note: Conclusions", ":material/functions: Methodology",
     ":material/dataset: Data"])

# ---------------------------------------------------------------------------------------------
# Overview tab: the top ideas on each side, then the two maps
# ---------------------------------------------------------------------------------------------
with tab_over:
    left, right = st.columns(2, gap="large")
    with left:
        rows = "".join(
            f'<div class="idea"><span class="tk">{esc(r.ticker)}</span><span class="nm"><b>{esc(r["name"])}</b>'
            f'<span class="why">{esc(r.sector)} · {esc(f"${r.market_cap_b:,.1f}B")}</span></span>{flag_strip(r)}</div>'
            for _, r in sell.head(8).iterrows())
        st.markdown(f'<div class="panel-title" style="color:{STATUS_COLORS["SELL"]}">Top SELL ideas</div>'
                    f'<div class="panel-sub">Most red flags first, then the highest average risk percentile</div>'
                    f'{rows or "<p>No company trips enough red flags. Lower the number of flags needed.</p>"}',
                    unsafe_allow_html=True)
    with right:
        rows = "".join(
            f'<div class="idea"><span class="tk">{esc(r.ticker)}</span><span class="nm"><b>{esc(r["name"])}</b>'
            f'<span class="why">{esc(r.sector)} · {esc(f"${r.market_cap_b:,.1f}B")}</span></span>'
            f'<span class="figs"><span><b>{fmt(r.EBIT_TEV, "pct")}</b><i>EBIT/TEV</i></span>'
            f'<span><b>{fmt(r.quality_score, "rank")}</b><i>quality</i></span>'
            f'<span><b>{fmt(r.FS_SCORE, "rank")}/10</b><i>strength</i></span></span></div>'
            for _, r in buy.head(8).iterrows())
        st.markdown(f'<div class="panel-title" style="color:{STATUS_COLORS["BUY"]}">Top BUY ideas</div>'
                    f'<div class="panel-sub">Ranked by {esc(method.split(":")[0].split(" (")[0].lower())}</div>'
                    f'{rows or "<p>No company passes every BUY filter. Lower a percentile cutoff.</p>"}',
                    unsafe_allow_html=True)

    st.write("")
    hover = {"ticker": True, "name": True, "sector": True, "Status": False}
    symbols = {"SELL": "triangle-down", "BUY": "triangle-up", "BUY and SELL": "diamond", "Neither": "circle"}
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.scatter(u.dropna(subset=["COMBOACCRUAL", "P_PFD"]), x="COMBOACCRUAL", y="P_PFD", color="Status",
                         symbol="Status", symbol_map=symbols, color_discrete_map=STATUS_COLORS, hover_data=hover,
                         opacity=0.8, category_orders={"Status": list(STATUS_COLORS)},
                         labels={"COMBOACCRUAL": "Accrual risk (average of STA and SNOA percentiles)",
                                 "P_PFD": "Distress risk (PFD percentile)"},
                         title="SELL map: accrual risk vs distress risk<br><sup>Top right = riskiest</sup>")
        st.plotly_chart(style_fig(fig, 470), width="stretch")
    with c2:
        fig = px.scatter(u.dropna(subset=["price_score", "quality_score"]), x="price_score", y="quality_score",
                         color="Status", symbol="Status", symbol_map=symbols, color_discrete_map=STATUS_COLORS,
                         hover_data=hover, opacity=0.8, category_orders={"Status": list(STATUS_COLORS)},
                         labels={"price_score": "Price percentile (100 = cheapest)",
                                 "quality_score": "Quality percentile (100 = best)"},
                         title="BUY map: price vs quality<br><sup>Top right = cheap and high quality; dashed lines "
                               "are your cutoffs</sup>")
        fig.add_vline(x=price_cut, line_dash="dash", line_color="gray")
        fig.add_hline(y=quality_cut, line_dash="dash", line_color="gray")
        st.plotly_chart(style_fig(fig, 470), width="stretch")
    by_sector = (u.groupby("sector").agg(Screened=("ticker", "size"), SELL=("SELL", "sum"), BUY=("BUY", "sum"))
                 .reset_index())
    long = by_sector.melt(id_vars=["sector", "Screened"], value_vars=["SELL", "BUY"], var_name="Status",
                          value_name="Companies")
    fig = px.bar(long, x="sector", y="Companies", color="Status", barmode="group", color_discrete_map=STATUS_COLORS,
                 hover_data=["Screened"], title="Candidates by sector")
    fig.update_layout(xaxis_title="")
    st.plotly_chart(style_fig(fig, 380), width="stretch")


# ---------------------------------------------------------------------------------------------
# SELL and BUY lists: click a row to see why the company is there
# ---------------------------------------------------------------------------------------------
def list_table(df, cols, config, key):
    """Show the top rows with single-row selection; return the selected company (or None)."""
    out = df[list(cols)].rename(columns=cols)
    shown = out.head(top_n)
    event = st.dataframe(shown, column_config=config, hide_index=True, width="stretch", key=f"tbl_{key}",
                         on_select="rerun", selection_mode="single-row")
    st.download_button(f"Download all {len(df)} rows (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"{key}_candidates.csv", mime="text/csv", key=f"dl_{key}",
                       icon=":material/download:")
    picked = event.selection.rows
    if picked:
        ticker = shown.iloc[picked[0]]["Ticker"]
        return u[u["ticker"] == ticker].iloc[0]
    return None


def remember(r, source):
    """Send a clicked company to the Company file tab (only when the click is new)."""
    if r is not None and st.session_state.get(f"last_{source}") != r["label"]:
        st.session_state["company_pick"] = r["label"]
    st.session_state[f"last_{source}"] = None if r is None else r["label"]


with tab_sell:
    active = [txt for on, txt in zip([use_sta, use_snoa, use_pman, use_z, use_pfd, use_si],
                                     [f"STA in the top {100 - sta_cut}%", f"SNOA in the top {100 - snoa_cut}%",
                                      f"PMAN in the top {100 - pman_cut}%", f"Altman Z below {z_cut:.2f}",
                                      f"PFD in the top {100 - pfd_cut}%", f"short interest of {si_cut:.1f}% or more"])
              if on]
    st.subheader(f"{len(sell)} SELL candidates")
    st.markdown(f"A company makes this list when it trips at least **{flags_needed}** of these red flags: "
                + "; ".join(active) + ". The riskiest companies come first.")
    sell_cols = {"Rank": "Rank", "ticker": "Ticker", "name": "Company", "sector": "Sector",
                 "market_cap_b": "Market cap", "red_flags": "Red flags", "why_sell": "Why it's flagged",
                 "SELL_score": "Risk score", "STA": "STA", "SNOA": "SNOA", "PMAN": "PMAN", "Z": "Altman Z",
                 "PFD_100": "PFD", "short_pct_float": "Short interest", "ret_12m": "12-month return",
                 "flags_checked": "Flags checked"}
    sell_cfg = {"Market cap": num("Market cap", "$%.1fB"), "Risk score": pct_col("Risk score"),
                "Why it's flagged": st.column_config.TextColumn("Why it's flagged", width="large"),
                "STA": num("STA"), "SNOA": num("SNOA"), "PMAN": pct_fmt("PMAN"), "Altman Z": num("Altman Z", "%.2f"),
                "PFD": num("PFD", "%.3f%%"), "Short interest": pct_fmt("Short interest"),
                "12-month return": pct_fmt("12-month return")}
    if len(sell):
        st.caption("Click a row to see why that company is flagged.")
        r = list_table(sell, sell_cols, sell_cfg, "sell")
        remember(r, "sell")
        if r is not None:
            with st.container(border=True):
                company_header(r)
                company_summary(r)
                st.caption("Every number behind this company is in the Company file tab.")
        counts = pd.DataFrame({"Red flag": [FLAG_WORDS[c][0].upper() + FLAG_WORDS[c][1:] for on, c, _, _ in flag_defs if on],
                               "SELL candidates tripping it": [int(sell[c].sum()) for on, c, _, _ in flag_defs if on]})
        fig = px.bar(counts.sort_values("SELL candidates tripping it"), x="SELL candidates tripping it", y="Red flag",
                     orientation="h", title="Which red flags drive the SELL list",
                     color_discrete_sequence=[STATUS_COLORS["SELL"]])
        fig.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(style_fig(fig, 300), width="stretch")
    else:
        st.info("No company trips enough red flags. Lower 'Red flags needed to SELL' or loosen a cutoff.")

with tab_buy:
    st.subheader(f"{len(buy)} BUY candidates")
    st.markdown(f"A company makes this list when its price percentile is at least **{price_cut}** "
                f"({price_choice.split(' (')[0]}), its quality percentile is at least **{quality_cut}** "
                f"({quality_choice.split(':')[0].split(' (')[0]}), its financial strength is at least **{fs_min}/10** "
                "and its EBIT is positive" + (", and it is not a SELL candidate" if buy_excludes_sell else "")
                + f". Ranked by: {method}.")
    buy_cols = {"Rank": "Rank", "ticker": "Ticker", "name": "Company", "sector": "Sector",
                "market_cap_b": "Market cap", "price_score": "Price pct", "quality_score": "Quality pct",
                "EBIT_TEV": "EBIT/TEV", "BM": "B/M", "GPA": "GPA", "P_FP": "Franchise pct",
                "FS_SCORE": "Strength (0-10)", "LT_ROA": "Long-term ROA", "LT_ROC": "Long-term ROC", "CFOA": "CFOA",
                "MG": "Margin growth", "MS": "Margin stability", "MF_ROC": "Greenblatt ROC", "ret_12m": "12-month return"}
    buy_cfg = {"Market cap": num("Market cap", "$%.1fB"), "Price pct": pct_col("Price pct"),
               "Quality pct": pct_col("Quality pct"), "EBIT/TEV": pct_fmt("EBIT/TEV"), "B/M": num("B/M", "%.2f"),
               "GPA": num("GPA", "%.2f"), "Franchise pct": pct_col("Franchise pct"),
               "Strength (0-10)": num("Strength (0-10)", "%.0f"), "Long-term ROA": pct_fmt("Long-term ROA"),
               "Long-term ROC": pct_fmt("Long-term ROC"), "CFOA": num("CFOA", "%.2f"),
               "Margin growth": pct_fmt("Margin growth"), "Margin stability": num("Margin stability", "%.1f"),
               "Greenblatt ROC": pct_fmt("Greenblatt ROC"), "12-month return": pct_fmt("12-month return")}
    if len(buy):
        st.caption("Click a row to see why that company qualifies.")
        r = list_table(buy, buy_cols, buy_cfg, "buy")
        remember(r, "buy")
        if r is not None:
            with st.container(border=True):
                company_header(r)
                company_summary(r)
                st.caption("Every number behind this company is in the Company file tab.")
    else:
        st.info("No company passes every BUY filter. Lower a percentile cutoff or the financial strength minimum.")

# ---------------------------------------------------------------------------------------------
# Company file tab: every number behind one company
# ---------------------------------------------------------------------------------------------
with tab_co:
    order = pd.concat([sell, buy, u[~u["SELL"] & ~u["BUY"]].sort_values("market_cap", ascending=False)])
    order = order[~order.index.duplicated()]
    options = order["label"].tolist()
    if st.session_state.get("company_pick") not in options:
        st.session_state["company_pick"] = options[0]
    pick = st.selectbox("Find a company (type a ticker or name)", options, key="company_pick",
                        help="SELL candidates are listed first, then BUY candidates, then everyone else by size. "
                             "Clicking a row in either list also opens that company here.")
    r = u[u["label"] == pick].iloc[0]
    company_header(r)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Market cap", f"${r['market_cap_b']:,.1f}B")
    m2.metric("12-month return", fmt(r["ret_12m"], "pct"))
    m3.metric("Red flags", f"{int(r['red_flags'])} of {n_active}")
    m4.metric("Price percentile", fmt(r["price_score"], "rank"))
    m5.metric("Quality percentile", fmt(r["quality_score"], "rank"))
    company_summary(r)

    a, b = st.columns(2, gap="large")
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
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="")
        st.plotly_chart(style_fig(fig, 380), width="stretch")
    if isinstance(r.get("summary"), str) and r["summary"]:
        with st.expander("Business description (Yahoo Finance)"):
            st.write(r["summary"] + ("..." if len(r["summary"]) >= 600 else ""))


# ---------------------------------------------------------------------------------------------
# Conclusions tab (written investment case)
# ---------------------------------------------------------------------------------------------
CONCLUSIONS = """
*Written for the default settings, using data pulled October 21, 2026. The screen finds the candidates; below I
explain what the numbers do not show on their own. Percentiles are out of the 876 companies in the default screen.*

### SELL (short) candidates

**1. CoreWeave (CRWV): growth paid for with debt, and every distress flag is on.**
CoreWeave has the highest probability of financial distress (PFD) of all 876 companies (100th percentile). Its
Altman Z is 0.26, far below the 1.81 cutoff. Its Beneish probability of manipulation (PMAN) is 44%, in the 97th
percentile, mostly because the money customers owe it (receivables) grew almost three times faster than sales. And
19% of its shares are sold short.

Behind the flags: sales grew from $16 million in 2022 to $5.1 billion in 2025, but the company paid for that growth
with borrowed money. It spent $10.3 billion on equipment and data centres in 2025, its free cash flow over the last
four years adds up to about -$14.4 billion, and it lost $1.2 billion in 2025. In the first quarter of 2026 alone it
spent $7.7 billion on equipment, lost $740 million and had $536 million of interest expense. It has $50.8 billion of
total liabilities against only $4.8 billion of equity. The company points to about $99 billion of future contracts
from a few large AI customers, but the interest is due whether or not those contracts pay off on time. The stock is
already down 57% in 12 months. This is exactly the kind of falling knife the distress models are built to catch.

*What would change my mind:* free cash flow turning positive, or the company refinancing its debt at lower rates.

**2. Carvana (CVNA): the business is real, but the profit is not all cash.**
Carvana trips four red flags. Its STA (accruals) is in the 96th percentile: net income was $1.4 billion but cash flow
from operations was only $1.0 billion, so the bottom line is running ahead of the cash. Its PMAN is 38% (97th
percentile), its PFD is in the 91st percentile, and 12.6% of its shares are sold short.

Beyond the score: the business had a strong 2025. It sold 43% more cars and made $1.9 billion of operating income.
But after a $2.25 billion "other expense", it actually lost about $0.9 billion before taxes. It only showed a profit
because of a $2.8 billion tax benefit. Carvana's own report says $621 million of its net income came from a non-cash
tax item, not money coming in. (That tax item also added a large new asset to the balance sheet, which is why the
Beneish asset quality ratio jumped.) At a $67 billion market value, investors are paying for profits that are bigger
than the cash behind them. In class terms, the earnings rely heavily on accrual accounting, and Chapter 3 shows that
gap tends to reverse.

*What would change my mind:* cash flow from operations catching up to net income for several quarters.

**3. AeroVironment (AVAV): a big purchase that bloated the balance sheet.**
AeroVironment's SNOA (bloated balance sheet) is in the 97th percentile, its PMAN is 43% (97th percentile), its PFD is
in the 95th percentile, and 12.8% of its shares are sold short.

Sales more than doubled to $2.0 billion in fiscal 2026 (the year to April 2026) because it bought another company,
BlueHalo. But gross margin fell from about 39% to 25%, the company lost $265 million, and cash flow from operations
was negative. Goodwill and other intangible assets from the purchase now make up $3.4 billion of its $5.7 billion in
total assets (60%). Three Beneish ratios are above the average for known manipulators: falling gross margin, weaker
asset quality and very fast sales growth. It has already written down $151 million of goodwill in its Space business
after the Space Force issued a stop-work order. The stock is down 68% in 12 months, but with that much goodwill left,
more write-downs are the risk.

**False alarms: flags I would not act on.**
- **GATX** ranks second, but it leases railcars. Owning expensive equipment paid for with debt is its whole business,
  so a high SNOA and a low Altman Z are normal for it. Comparing it only with similar companies (using the Industries
  filter) is fairer.
- **Jersey Mike's (JMKE)** went public in 2026 after being bought by Blackstone. Because of how the buyout was
  recorded, goodwill and intangibles are 96% of its assets, which pushes SNOA and the Beneish asset quality ratio to
  extremes. It also only has enough history to check four of the six flags.
- **QXO** shows a 100% PMAN because its sales jumped 120 times after it bought Beacon Roofing Supply. That is an
  acquisition, not manipulation. The Beneish model was not built for a company that changes this much in one year.

The screen is a starting point, not the final answer: every flag needs a second look.

### BUY (long) candidates

**1. Adobe (ADBE): a wonderful company at a fair price.**
Buffett's line from Chapter 2 fits Adobe. Its franchise power is in the 99.7th percentile, with high and persistent
returns: four years of free cash flow ($32 billion) add up to more than its total assets (CFOA of 1.09), its
long-term return on capital is 55%, and its gross margin is 89%. It is also cheap: its EBIT/TEV is 8.4%, in the 89th
percentile. It passes 8 of the 10 financial strength checks and trips no red flags.

Why is it cheap? The stock fell 25% in 12 months because investors fear new AI design tools will take away its
customers, and because its CEO, Shantanu Narayen, announced in March 2026 that he is leaving. But sales are still
growing (up 12% in the first quarter of fiscal 2026), and Adobe is buying back a lot of its own stock: $11.3 billion
in fiscal 2025, with $25 billion more approved. I think the market is too pessimistic about a high-quality business.

One note: by book-to-market (0.12), Adobe looks expensive, because buybacks shrink book value. This is why the book
prefers EBIT/TEV as the price measure (Chapter 7).

*Main risk:* the AI threat is real. I would watch subscription growth and sales of its AI products.

**2. Cirrus Logic (CRUS): top quality, with one big risk.**
Cirrus has the second-highest quality percentile in the screen (99.8th) and a perfect financial strength score of
10/10. It is also cheap, with an EBIT/TEV of 10.2% (93rd percentile) and a book-to-market of 0.38. It has $134
million of debt and $801 million of cash, earned record earnings per share in fiscal 2026 with a 52.8% gross margin,
and bought back $280 million of stock.

What the screen cannot see: one customer, Apple, made up about 91% of its sales in fiscal 2026. If Apple switched
suppliers, Cirrus would be hurt badly. It is growing in PC chips and other products, but I would buy a smaller
position than the score alone suggests.

**3. Mueller Industries (MLI): no debt and high returns in a boring industry.**
Mueller's quality is in the 97th percentile, with four-year average returns of 23% on assets and 59% on capital. Its
EBIT/TEV is 9.0% (90th percentile) and it passes 8 of 10 financial strength checks. 2025 was its best year ever: net
income rose 27% to $765 million. It ended the year with no debt and $1.39 billion of cash, and it bought back five
times more stock than the year before ($244 million).

Two things to watch: higher copper prices let it raise its selling prices while the amount it sold fell late in the
year, and 2025 included about $67 million of one-time gains. The stock is also up 25% in 12 months, so it is less of a
bargain than Adobe or Cirrus.

**Why I am not buying the gold miners at the top of the list.**
Kinross, Gold Fields and AngloGold rank near the top because record gold prices have pushed up their margins.
Kinross's gross margin grew 52% a year, but its margin stability is only 2.1 (H&R Block's is 213). Their returns are
growing but not stable, which is not what a franchise looks like. Chapter 8 warns that peak earnings make cyclical
companies look cheap: if gold prices fall, so will their earnings.

### Do the picks hold up?
Adobe, Cirrus Logic and Mueller stay near the top of the BUY list when I switch to the book's "value first, then
quality" method. CoreWeave and AeroVironment stay on the SELL list when each company is compared only with its own
sector. The pattern makes sense: the SELL list is full of companies that grew with debt or big acquisitions, and the
BUY list is full of cash-rich companies buying back their own stock after their prices fell.

*This is a class project, not investment advice.*
"""

with tab_concl:
    st.subheader("Investment conclusions")
    st.markdown(CONCLUSIONS.replace("$", "\\$"))   # escape $ so Streamlit does not read it as math
    c1, c2 = st.columns(2)
    c1.markdown("**Top 5 SELL names at your current settings**")
    c1.dataframe(sell.head(5)[["ticker", "name", "red_flags", "SELL_score"]].rename(
        columns={"ticker": "Ticker", "name": "Company", "red_flags": "Red flags", "SELL_score": "SELL score"}),
        hide_index=True, width="stretch", column_config={"SELL score": num("SELL score", "%.0f")})
    c2.markdown("**Top 5 BUY names at your current settings**")
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
  the class-example version is one click away under *More options* in the sidebar.
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

st.markdown(f'<div class="footer">Data: Yahoo Finance via yfinance, pulled {esc(as_of)}. Formulas from Gray and Carlisle, '
            '<i>Quantitative Value</i>, and ACCT 3343 class notes. Built with Python and Streamlit. A class project, '
            'not investment advice.</div>', unsafe_allow_html=True)
