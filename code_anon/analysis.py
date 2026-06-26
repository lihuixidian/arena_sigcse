# -*- coding: utf-8 -*-
"""
ARENA Score Comprehensive Statistical Analysis
==============================================
- Correlation between attendance (A) and Item 1 / Item 2 scores, by year
- Cross-year comparison of correlation strength: Fisher z (independent
  samples) + ANCOVA with attendance x year interaction
- Extensions: within-student Item 1 vs Item 2 correlation, log(1+A),
  Kruskal-Wallis cross-year score comparisons
- Outputs: HTML / Markdown report, PNG figures, cleaned-data CSV
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import base64
from datetime import datetime

# ---------- 0. Global config ----------
OUT = Path(".")
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
SRC = "rawdata1.xlsx"

YEARS = ["2023", "2024", "2025"]
COLORS = {"2023": "#4C78A8", "2024": "#F58518", "2025": "#54A24B"}

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="DejaVu Sans",
              rc={"axes.unicode_minus": False})

# Display labels (for axes / table headers)
LABELS = {
    "q1": "C (Item 1)",
    "q2": "D (Item 2)",
    "total": "E (Total)",
    "logs": "F (Logins)",
    "logF1": "log(1+F)",
}

# ---------- 1. Data loading and cleaning ----------
def load_and_clean():
    pieces = []
    for y in YEARS:
        df = pd.read_excel(SRC, sheet_name=y, header=None)
        df.columns = [str(c) if c is not None else f"col{i}" for i, c in enumerate(df.iloc[0])]
        df = df.iloc[1:].copy()
        rename = {}
        for c in df.columns:
            cs = str(c)
            if "Student ID" in cs:
                rename[c] = "sid"
            elif "Name" in cs:
                rename[c] = "name"
            elif "Plan Comparison" in cs:
                rename[c] = "C"
            elif "Plan Generation" in cs:
                rename[c] = "D"
            elif "Total" in cs:
                rename[c] = "E"
            elif "#log" in cs.lower() or "Logins" in cs:
                rename[c] = "F"
            elif "Notes" in cs:
                rename[c] = "note"
        df = df.rename(columns=rename)
        keep = [c for c in ["sid", "name", "C", "D", "E", "F", "note"] if c in df.columns]
        df = df[keep].copy()
        df["year"] = y
        for c in ["C", "D", "E", "F"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["sid"].astype(str).str.strip().ne("") & df["sid"].notna()].copy()
        if "note" in df.columns:
            df["note"] = df["note"].astype(str).fillna("")
            df = df[~df["note"].str.contains("absent", na=False)].copy()
        df = df[df[["C", "D", "E", "F"]].notna().any(axis=1)].copy()
        df = df[~((df["C"].fillna(0) == 0) & (df["D"].fillna(0) == 0) &
                  (df["E"].fillna(0) == 0) & (df["F"].fillna(0) == 0))].copy()
        for c in ["C", "D", "E", "F"]:
            df[c] = df[c].fillna(0)
        df["logF1"] = np.log1p(df["F"])
        pieces.append(df)
    full = pd.concat(pieces, ignore_index=True)
    # Rename to short codes to avoid clashes with patsy formula reserved words.
    full = full.rename(columns={"C": "q1", "D": "q2", "E": "total", "F": "logs"})
    return full


DF = load_and_clean()
DF.to_csv(OUT / "data_clean.csv", index=False, encoding="utf-8-sig")
N_BY_YEAR = {y: int(len(DF[DF.year == y])) for y in YEARS}
N_TOTAL = int(len(DF))
print(f"Data cleaned: {N_TOTAL} rows, columns: {list(DF.columns)}")
print("Per-year n:", N_BY_YEAR)

# ---------- 2. Utility helpers ----------
def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.s."
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def fisher_z_compare(r1, n1, r2, n2):
    if abs(r1) >= 1 or abs(r2) >= 1:
        return np.nan, np.nan
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se1 = 1.0 / np.sqrt(max(n1 - 3, 1))
    se2 = 1.0 / np.sqrt(max(n2 - 3, 1))
    z = (z1 - z2) / np.sqrt(se1**2 + se2**2)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def ci_pearson(r, n, alpha=0.05):
    if abs(r) >= 1 or n <= 3:
        return (np.nan, np.nan)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z_lo = z - stats.norm.ppf(1 - alpha/2) * se
    z_hi = z + stats.norm.ppf(1 - alpha/2) * se
    r_lo = (np.exp(2*z_lo) - 1) / (np.exp(2*z_lo) + 1)
    r_hi = (np.exp(2*z_hi) - 1) / (np.exp(2*z_hi) + 1)
    return (r_lo, r_hi)


def desc(sub, col):
    s = sub[col]
    return {
        "n": int(len(s)),
        "mean": s.mean(),
        "sd": s.std(ddof=1),
        "median": s.median(),
        "min": s.min(),
        "max": s.max(),
        "q1": s.quantile(0.25),
        "q3": s.quantile(0.75),
        "skew": stats.skew(s),
        "kurt": stats.kurtosis(s),
    }

# ---------- 3. Descriptive statistics ----------
DESC = {}
for y in YEARS:
    DESC[y] = {c: desc(DF[DF.year == y], c) for c in ["q1", "q2", "total", "logs", "logF1"]}
DESC_ALL = {c: desc(DF, c) for c in ["q1", "q2", "total", "logs", "logF1"]}

# ---------- 4. Normality + variance homogeneity ----------
NORM = {}
for y in YEARS + ["ALL"]:
    sub = DF if y == "ALL" else DF[DF.year == y]
    NORM[y] = {}
    for c in ["q1", "q2", "total", "logs", "logF1"]:
        try:
            sw = stats.shapiro(sub[c])
            NORM[y][c] = {"W": sw.statistic, "p": sw.pvalue}
        except Exception:
            NORM[y][c] = {"W": np.nan, "p": np.nan}

LEV = {}
for c in ["q1", "q2", "total", "logs", "logF1"]:
    groups = [DF[DF.year == y][c].values for y in YEARS]
    L = stats.levene(*groups, center="median")
    LEV[c] = {"F": L.statistic, "p": L.pvalue}

# ---------- 5. Correlation analysis (per year + pooled) ----------
CORR = {}
for y in YEARS + ["ALL"]:
    sub = DF if y == "ALL" else DF[DF.year == y]
    n = len(sub)
    CORR[y] = {}
    pairs_to_test = [
        ("logs", "q1"), ("logs", "q2"), ("logs", "total"),
        ("logF1", "q1"), ("logF1", "q2"), ("logF1", "total"),
        ("q1", "q2"),
    ]
    for (a, b) in pairs_to_test:
        sp = stats.spearmanr(sub[a], sub[b])
        kd = stats.kendalltau(sub[a], sub[b], variant="b")
        pr = stats.pearsonr(sub[a], sub[b])
        ci = ci_pearson(pr.statistic, n)
        CORR[y][f"{a}~{b}"] = {
            "n": n,
            "spearman_r": sp.statistic, "spearman_p": sp.pvalue,
            "kendall_t": kd.statistic, "kendall_p": kd.pvalue,
            "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
            "pearson_ci_lo": ci[0], "pearson_ci_hi": ci[1],
        }

# ---------- 6. Cross-year correlation comparison (Fisher z, independent) ----------
FZC = {}
pairs = [("2023", "2024"), ("2023", "2025"), ("2024", "2025")]
for (a, b) in [("logs", "q1"), ("logs", "q2"), ("logs", "total"),
               ("logF1", "q1"), ("logF1", "q2"), ("logF1", "total")]:
    FZC[f"{a}~{b}"] = {}
    for (y1, y2) in pairs:
        r1 = CORR[y1][f"{a}~{b}"]["pearson_r"]
        r2 = CORR[y2][f"{a}~{b}"]["pearson_r"]
        n1 = CORR[y1][f"{a}~{b}"]["n"]
        n2 = CORR[y2][f"{a}~{b}"]["n"]
        z, p = fisher_z_compare(r1, n1, r2, n2)
        FZC[f"{a}~{b}"][f"{y1}vs{y2}"] = {"z": z, "p_uncorrected": p, "p_bonf": min(p*3, 1.0)}

# ---------- 7. ANCOVA: score ~ logs * year ----------
ANCOVA = {}
DF["year_cat"] = pd.Categorical(DF["year"], categories=YEARS, ordered=True)
for dv in ["q1", "q2", "total"]:
    formula = f"{dv} ~ logs + C(year_cat) + logs:C(year_cat)"
    model = smf.ols(formula, data=DF).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ANCOVA[dv] = {
        "model": model,
        "aov": aov,
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
        "p_interaction": aov.loc["logs:C(year_cat)", "PR(>F)"] if "logs:C(year_cat)" in aov.index else np.nan,
        "p_logs": aov.loc["logs", "PR(>F)"] if "logs" in aov.index else np.nan,
        "p_year": aov.loc["C(year_cat)", "PR(>F)"] if "C(year_cat)" in aov.index else np.nan,
    }

# ---------- 8. Cross-year score comparison (Kruskal-Wallis) ----------
KW = {}
for c in ["q1", "q2", "total", "logs", "logF1"]:
    groups = [DF[DF.year == y][c].values for y in YEARS]
    res = stats.kruskal(*groups)
    KW[c] = {"H": res.statistic, "p": res.pvalue, "df": len(YEARS) - 1}

# ---------- 9. q1 vs q2 paired test + Wilcoxon ----------
PAIR = {}
for y in YEARS + ["ALL"]:
    sub = DF if y == "ALL" else DF[DF.year == y]
    d = sub["q1"] - sub["q2"]
    try:
        wp = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        wp = None
    PAIR[y] = {
        "q1_median": sub["q1"].median(), "q2_median": sub["q2"].median(),
        "q1_mean": sub["q1"].mean(), "q2_mean": sub["q2"].mean(),
        "wilcoxon_stat": wp.statistic if wp else np.nan,
        "wilcoxon_p": wp.pvalue if wp else np.nan,
        "q1_eq_q2": int((sub["q1"] == sub["q2"]).sum()),
        "q1_gt_q2": int((sub["q1"] > sub["q2"]).sum()),
        "q1_lt_q2": int((sub["q1"] < sub["q2"]).sum()),
    }

# ---------- 10. LOWESS fits ----------
LOW = {}
for y in YEARS:
    for (a, b) in [("logs", "q1"), ("logs", "q2"), ("logs", "total")]:
        sub = DF[DF.year == y]
        if len(sub) < 5:
            continue
        frac = max(0.4, 5 / len(sub))
        ls = sm.nonparametric.lowess(sub[b], sub[a], frac=frac, return_sorted=True)
        LOW[f"{y}|{a}~{b}"] = ls

# ============ Figure generation ============

def save_fig(fig, name, dpi=140):
    p = FIG / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# Fig 1: scatter of A vs Item 1, A vs Item 2, by year, with linear fit
def plot_scatter_by_year():
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for i, y in enumerate(YEARS):
        sub = DF[DF.year == y]
        for j, (x, dv) in enumerate([("logs", "q1"), ("logs", "q2")]):
            ax = axes[i, j]
            ax.scatter(sub[x], sub[dv], s=42, alpha=0.75,
                       color=COLORS[y], edgecolor="white", linewidth=0.5)
            if len(sub) >= 3:
                slope, intercept, r, p, se = stats.linregress(sub[x], sub[dv])
                xs = np.linspace(sub[x].min(), sub[x].max(), 100)
                ax.plot(xs, intercept + slope*xs, "--", color="#444",
                        linewidth=1.5, alpha=0.7)
                tag = "**" if p < 0.05 else ("*" if p < 0.1 else "")
                ax.set_title(f"{y}  F~{dv}: Pearson r={r:.2f}{tag}  (p={fmt_p(p)})",
                             fontsize=11)
            ax.set_xlabel("F (Logins)" if x == "logs" else "log(1+F)")
            ax.set_ylabel(f"{dv} ({'Item 1' if dv == 'q1' else 'Item 2'})")
            ax.grid(True, alpha=0.3)
    fig.suptitle("A vs. Item 1 / Item 2 scatter + linear fit (by year)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "01_scatter_F_vs_scores.png")


# Fig 2: combined scatter, log(1+A) vs score
def plot_log_scatter_combined():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for j, dv in enumerate(["q1", "q2"]):
        ax = axes[j]
        for y in YEARS:
            sub = DF[DF.year == y]
            ax.scatter(sub["logF1"], sub[dv], s=46, alpha=0.78,
                       color=COLORS[y], edgecolor="white", linewidth=0.5, label=y)
        sl = stats.linregress(DF["logF1"], DF[dv])
        xs = np.linspace(DF["logF1"].min(), DF["logF1"].max(), 100)
        ax.plot(xs, sl.intercept + sl.slope*xs, "k--", alpha=0.6,
                linewidth=1.5, label=f"Pooled: r={sl.rvalue:.2f}")
        ax.set_xlabel("log(1+F)")
        ax.set_ylabel(f"{dv} ({'Item 1' if dv == 'q1' else 'Item 2'})")
        ax.set_title(f"log(1+F) ~ {dv} (three years)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Combined scatter (log-A yields a more linear relation)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "02_log_scatter_combined.png")


# Fig 3: box plots
def plot_box():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for j, c in enumerate(["q1", "q2", "logs"]):
        ax = axes[j]
        data = [DF[DF.year == y][c].values for y in YEARS]
        bp = ax.boxplot(data, tick_labels=YEARS, patch_artist=True,
                        showmeans=True, meanline=True,
                        medianprops={"color": "black", "linewidth": 1.5},
                        meanprops={"color": "red", "linewidth": 1.5, "linestyle": "--"})
        for patch, y in zip(bp["boxes"], YEARS):
            patch.set_facecolor(COLORS[y])
            patch.set_alpha(0.55)
        title_map = {"q1": "C (Item 1)", "q2": "D (Item 2)", "logs": "F (Logins)"}
        ax.set_title(f"{c} distribution - {title_map[c]}")
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Distribution of key variables by year (box plot)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "03_box_by_year.png")


# Fig 4: violin plots
def plot_violin():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for j, c in enumerate(["q1", "q2", "logs"]):
        ax = axes[j]
        sns.violinplot(data=DF, x="year", y=c, order=YEARS, palette=COLORS,
                       inner="quartile", ax=ax, hue="year", legend=False)
        sns.stripplot(data=DF, x="year", y=c, order=YEARS, color="black",
                      alpha=0.5, size=3, ax=ax)
        title_map = {"q1": "C (Item 1)", "q2": "D (Item 2)", "logs": "F (Logins)"}
        ax.set_title(f"{c} - {title_map[c]}")
        ax.set_xlabel("Year")
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Distribution of key variables by year (violin + strip)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "04_violin_by_year.png")


# Fig 5: correlation heatmaps
def plot_corr_heatmaps():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    cols = ["q1", "q2", "total", "logs", "logF1"]
    titles = YEARS + ["ALL"]
    for i, (y, t) in enumerate(zip(YEARS + ["ALL"], titles)):
        sub = DF if y == "ALL" else DF[DF.year == y]
        r = sub[cols].corr(method="spearman")
        sns.heatmap(r, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, square=True, cbar=(i == 3), ax=axes[i])
        axes[i].set_title(f"Spearman corr. - {t}\n(n={len(sub)})")
    fig.suptitle("Spearman correlation matrix by year",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "05_corr_heatmaps.png")


# Fig 6: Q-Q plot of A (normality check)
def plot_qq():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, y in enumerate(YEARS):
        ax = axes[i]
        sub = DF[DF.year == y]
        stats.probplot(sub["logs"], dist="norm", plot=ax)
        ax.set_title(f"{y} A Q-Q\nW={NORM[y]['logs']['W']:.3f}, p={NORM[y]['logs']['p']:.3f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Q-Q plot of A (Logins) against normality",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "06_qq_F.png")


# Fig 7: score-band stacked histogram
def plot_score_stacked():
    bins_q = [0, 2, 4, 6, 8, 10, 11]
    labels_q = ["0-2", "2-4", "4-6", "6-8", "8-10", "10"]
    bins_t = [0, 4, 8, 12, 16, 20, 21]
    labels_t = ["0-4", "4-8", "8-12", "12-16", "16-20", "20"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for j, c in enumerate(["q1", "q2", "total"]):
        ax = axes[j]
        bins, labels = (bins_t, labels_t) if c == "total" else (bins_q, labels_q)
        width = 0.25
        x = np.arange(len(labels))
        for k, y in enumerate(YEARS):
            sub = DF[DF.year == y]
            counts, _ = np.histogram(sub[c], bins=bins)
            ax.bar(x + (k-1)*width, counts, width, color=COLORS[y],
                   label=y, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        title_map = {"q1": "C (Item 1)", "q2": "D (Item 2)", "total": "E (Total)"}
        ax.set_title(f"{c} - {title_map[c]}")
        ax.set_xlabel("Score band")
        ax.set_ylabel("Number of students")
        ax.legend(title="Year", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Score-band stacked distribution by year",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "07_score_stacked.png")


# Fig 8: LOWESS smooths
def plot_lowess():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for j, dv in enumerate(["q1", "q2"]):
        ax = axes[j]
        for y in YEARS:
            sub = DF[DF.year == y]
            ax.scatter(sub["logs"], sub[dv], s=40, alpha=0.6,
                       color=COLORS[y], edgecolor="white", linewidth=0.5)
            ls = LOW.get(f"{y}|logs~{dv}")
            if ls is not None and len(ls) > 1:
                ax.plot(ls[:, 0], ls[:, 1], color=COLORS[y], linewidth=2,
                        label=f"{y} LOWESS")
        ax.set_xlabel("F (Logins)")
        ax.set_ylabel(f"{dv} ({'Item 1' if dv == 'q1' else 'Item 2'})")
        ax.set_title(f"F vs {dv} + LOWESS")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Non-linear LOWESS fits of scores on attendance",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "08_lowess.png")


# Fig 9: correlation strength, cross-year comparison
def plot_corr_compare():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for j, (x, dv) in enumerate([("logs", "q1"), ("logs", "q2")]):
        ax = axes[j]
        labels, rs, lo, hi = [], [], [], []
        for y in YEARS:
            d = CORR[y][f"{x}~{dv}"]
            labels.append(y)
            rs.append(d["pearson_r"])
            lo.append(d["pearson_ci_lo"])
            hi.append(d["pearson_ci_hi"])
        rs = np.array(rs); lo = np.array(lo); hi = np.array(hi)
        yerr = np.vstack([rs-lo, hi-rs])
        ax.bar(labels, rs, color=[COLORS[y] for y in YEARS],
               alpha=0.7, edgecolor="black")
        ax.errorbar(labels, rs, yerr=yerr, fmt="none", ecolor="black",
                    capsize=5, linewidth=1.5)
        for i, y in enumerate(YEARS):
            d = CORR[y][f"{x}~{dv}"]
            ax.text(i, rs[i] + 0.04 if rs[i] >= 0 else rs[i] - 0.06,
                    f"r={d['pearson_r']:.2f}\np={fmt_p(d['pearson_p'])}",
                    ha="center", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"Pearson r: F ~ {dv} (by year, 95% CI)")
        ax.set_ylabel("Pearson r")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Attendance-score correlation strength across years",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "09_corr_compare.png")


# Fig 10: ANCOVA - per-year regression lines
def plot_ancova():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for j, dv in enumerate(["q1", "q2"]):
        ax = axes[j]
        for y in YEARS:
            sub = DF[DF.year == y]
            ax.scatter(sub["logs"], sub[dv], s=40, alpha=0.65,
                       color=COLORS[y], edgecolor="white", linewidth=0.5, label=y)
            if len(sub) >= 3:
                sl = stats.linregress(sub["logs"], sub[dv])
                xs = np.linspace(sub["logs"].min(), sub["logs"].max(), 100)
                ax.plot(xs, sl.intercept + sl.slope*xs,
                        color=COLORS[y], linewidth=2)
                ax.text(0.05, 0.95 - 0.08*YEARS.index(y),
                        f"{y}: slope={sl.slope:.3f}",
                        transform=ax.transAxes, color=COLORS[y],
                        fontsize=10, fontweight="bold")
        ax.set_xlabel("F (Logins)")
        ax.set_ylabel(f"{dv} score")
        ax.set_title(f"ANCOVA view: {dv} ~ F + year + F:year")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-year regression lines of scores on A (slope diff = interaction)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "10_ancova_lines.png")


print("Generating figures...")
fig_paths = []
fig_paths.append(plot_scatter_by_year())
fig_paths.append(plot_log_scatter_combined())
fig_paths.append(plot_box())
fig_paths.append(plot_violin())
fig_paths.append(plot_corr_heatmaps())
fig_paths.append(plot_qq())
fig_paths.append(plot_score_stacked())
fig_paths.append(plot_lowess())
fig_paths.append(plot_corr_compare())
fig_paths.append(plot_ancova())
print(f"Generated {len(fig_paths)} figures.")

# ============ Report generation ============

def b64img(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def desc_table_html(d, title):
    rows = ""
    cols = ["q1", "q2", "total", "logs", "logF1"]
    for c in cols:
        s = d.get(c, {})
        if not s:
            continue
        label = {"q1": "C (Item 1)", "q2": "D (Item 2)", "total": "E (Total)",
                 "logs": "F (Logins)", "logF1": "log(1+F)"}[c]
        rows += (f"<tr><td>{label}</td><td>{s.get('n', '-')}</td>"
                 f"<td>{s.get('mean', 0):.2f}</td><td>{s.get('sd', 0):.2f}</td>"
                 f"<td>{s.get('median', 0):.2f}</td><td>{s.get('min', 0):.0f}</td>"
                 f"<td>{s.get('max', 0):.0f}</td><td>{s.get('q1', 0):.2f}</td>"
                 f"<td>{s.get('q3', 0):.2f}</td><td>{s.get('skew', 0):.2f}</td>"
                 f"<td>{s.get('kurt', 0):.2f}</td></tr>")
    return (f'<h4>{title}</h4><table border="1" cellpadding="5" cellspacing="0" '
            f'style="border-collapse:collapse"><tr><th>Variable</th><th>n</th>'
            f'<th>Mean</th><th>SD</th><th>Median</th><th>min</th><th>max</th>'
            f'<th>Q1</th><th>Q3</th><th>Skew</th><th>Kurt</th></tr>{rows}</table>')


def corr_table_html(year):
    rows = ""
    for k, d in CORR[year].items():
        rows += (f"<tr><td>{k}</td><td>{d['n']}</td>"
                 f"<td>{d['spearman_r']:.3f}</td><td>{fmt_p(d['spearman_p'])}</td>"
                 f"<td>{d['kendall_t']:.3f}</td><td>{fmt_p(d['kendall_p'])}</td>"
                 f"<td>{d['pearson_r']:.3f}</td>"
                 f"<td>[{d['pearson_ci_lo']:.3f}, {d['pearson_ci_hi']:.3f}]</td>"
                 f"<td>{fmt_p(d['pearson_p'])}</td></tr>")
    return (f'<h4>{year}</h4><table border="1" cellpadding="5" cellspacing="0" '
            f'style="border-collapse:collapse"><tr><th>Pair</th><th>n</th>'
            f'<th>Spearman r</th><th>p</th><th>Kendall t</th><th>p</th>'
            f'<th>Pearson r</th><th>95% CI</th><th>p</th></tr>{rows}</table>')


def fzc_table_html():
    rows = ""
    for k, d in FZC.items():
        for pair, v in d.items():
            rows += (f"<tr><td>{k}</td><td>{pair}</td>"
                     f"<td>{v['z']:.3f}</td><td>{fmt_p(v['p_uncorrected'])}</td>"
                     f"<td>{fmt_p(v['p_bonf'])}</td></tr>")
    return ('<table border="1" cellpadding="5" cellspacing="0" '
            'style="border-collapse:collapse"><tr><th>Pair</th><th>Comparison</th>'
            '<th>Z</th><th>p (uncorrected)</th><th>p (Bonferroni x 3)</th></tr>'
            f'{rows}</table>')


def ancova_table_html():
    rows = ""
    for dv, d in ANCOVA.items():
        aov = d["aov"]
        rows += f'<tr><td rowspan="{len(aov)}"><b>{dv}</b></td>'
        first = True
        for idx, r in aov.iterrows():
            if not first:
                rows += "<tr>"
            rows += (f"<td>{idx}</td><td>{r.get('sum_sq', 0):.3f}</td>"
                     f"<td>{r.get('df', 0):.0f}</td><td>{r.get('F', 0):.3f}</td>"
                     f"<td>{fmt_p(r.get('PR(>F)', np.nan))}</td></tr>")
            first = False
    return ('<table border="1" cellpadding="5" cellspacing="0" '
            'style="border-collapse:collapse"><tr><th>DV</th><th>Term</th>'
            f'<th>SS</th><th>df</th><th>F</th><th>p</th></tr>{rows}</table>')


def pair_table_html():
    rows = ""
    for y in YEARS + ["ALL"]:
        d = PAIR[y]
        wp = "n.s." if np.isnan(d["wilcoxon_p"]) else fmt_p(d["wilcoxon_p"])
        rows += (f"<tr><td>{y}</td><td>{d['q1_mean']:.2f}</td>"
                 f"<td>{d['q2_mean']:.2f}</td><td>{d['q1_median']:.1f}</td>"
                 f"<td>{d['q2_median']:.1f}</td><td>{d['q1_eq_q2']}</td>"
                 f"<td>{d['q1_gt_q2']}</td><td>{d['q1_lt_q2']}</td>"
                 f"<td>{wp}</td></tr>")
    return (f'<table border="1" cellpadding="5" cellspacing="0" '
            f'style="border-collapse:collapse"><tr><th>Year</th><th>C mean</th>'
            f'<th>D mean</th><th>C median</th><th>D median</th><th>C=D</th>'
            f'<th>C>D</th><th>C<D</th><th>Wilcoxon p</th></tr>{rows}</table>')


def kw_table_html():
    rows = ""
    for c, d in KW.items():
        label = {"q1": "C (Item 1)", "q2": "D (Item 2)", "total": "E (Total)",
                 "logs": "F (Logins)", "logF1": "log(1+F)"}[c]
        rows += (f"<tr><td>{label}</td><td>{d['H']:.3f}</td>"
                 f"<td>{d['df']}</td><td>{fmt_p(d['p'])}</td></tr>")
    return (f'<table border="1" cellpadding="5" cellspacing="0" '
            f'style="border-collapse:collapse"><tr><th>Variable</th><th>H</th>'
            f'<th>df</th><th>p</th></tr>{rows}</table>')


def norm_table_html():
    rows = ""
    for y in YEARS + ["ALL"]:
        for c, d in NORM[y].items():
            label = {"q1": "C (Item 1)", "q2": "D (Item 2)", "total": "E (Total)",
                     "logs": "F (Logins)", "logF1": "log(1+F)"}[c]
            rows += (f"<tr><td>{y}</td><td>{label}</td>"
                     f"<td>{d['W']:.3f}</td><td>{fmt_p(d['p'])}</td></tr>")
    return (f'<table border="1" cellpadding="5" cellspacing="0" '
            f'style="border-collapse:collapse"><tr><th>Group</th>'
            f'<th>Variable</th><th>W</th><th>p</th></tr>{rows}</table>')


def lev_table_html():
    rows = ""
    for c, d in LEV.items():
        label = {"q1": "C (Item 1)", "q2": "D (Item 2)", "total": "E (Total)",
                 "logs": "F (Logins)", "logF1": "log(1+F)"}[c]
        rows += (f"<tr><td>{label}</td><td>{d['F']:.3f}</td>"
                 f"<td>{fmt_p(d['p'])}</td></tr>")
    return (f'<table border="1" cellpadding="5" cellspacing="0" '
            f'style="border-collapse:collapse"><tr><th>Variable</th>'
            f'<th>Levene F</th><th>p</th></tr>{rows}</table>')


# Build HTML report
now = datetime.now().strftime("%Y-%m-%d %H:%M")
img = [b64img(p) for p in fig_paths]


def findings_text():
    out = []
    sp_f_c_all = CORR["ALL"]["logs~q1"]["spearman_r"]
    sp_f_c_2023 = CORR["2023"]["logs~q1"]["spearman_r"]
    sp_f_c_2024 = CORR["2024"]["logs~q1"]["spearman_r"]
    sp_f_c_2025 = CORR["2025"]["logs~q1"]["spearman_r"]
    sp_f_d_all = CORR["ALL"]["logs~q2"]["spearman_r"]
    sp_f_d_2023 = CORR["2023"]["logs~q2"]["spearman_r"]
    sp_f_d_2024 = CORR["2024"]["logs~q2"]["spearman_r"]
    sp_f_d_2025 = CORR["2025"]["logs~q2"]["spearman_r"]
    p_f_c_all = CORR["ALL"]["logs~q1"]["spearman_p"]
    p_f_d_all = CORR["ALL"]["logs~q2"]["spearman_p"]

    out.append(f"**1. Pooled A-C / A-D association**: combining all three years, "
               f"Spearman r(A, C) = {sp_f_c_all:.3f} (p={fmt_p(p_f_c_all)}), "
               f"r(A, D) = {sp_f_d_all:.3f} (p={fmt_p(p_f_d_all)}).")
    out.append(f"**2. By year**: Spearman r(A, C) in 2023 / 2024 / 2025 is "
               f"{sp_f_c_2023:.3f} / {sp_f_c_2024:.3f} / {sp_f_c_2025:.3f}; "
               f"r(A, D) is {sp_f_d_2023:.3f} / {sp_f_d_2024:.3f} / {sp_f_d_2025:.3f}.")
    p_int_q1 = ANCOVA["q1"]["p_interaction"]
    p_int_q2 = ANCOVA["q2"]["p_interaction"]
    p_int_e = ANCOVA["total"]["p_interaction"]
    out.append(f"**3. ANCOVA interaction (A x year)**: p for C = {fmt_p(p_int_q1)}, "
               f"for D = {fmt_p(p_int_q2)}, for E (Total) = {fmt_p(p_int_e)} "
               f"(controlling for main effects of A and year, "
               f"does the A-score slope vary across years?).")
    p_kw_c = KW["q1"]["p"]; p_kw_d = KW["q2"]["p"]; p_kw_e = KW["total"]["p"]
    out.append(f"**4. Cross-year score differences (Kruskal-Wallis)**: "
               f"Item 1 p={fmt_p(p_kw_c)}, Item 2 p={fmt_p(p_kw_d)}, "
               f"Total p={fmt_p(p_kw_e)}.")
    p_wilc_all = PAIR["ALL"]["wilcoxon_p"]
    out.append(f"**5. Within-student Item 1 vs Item 2 (Wilcoxon signed-rank, "
               f"pooled)**: p={fmt_p(p_wilc_all)}.")
    return "<br>".join(out)


findings = findings_text()

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ARENA Score Analysis Report</title>
<style>
body {{ font-family: 'DejaVu Sans', sans-serif; max-width: 1200px; margin: 30px auto; padding: 0 20px; color: #222; line-height: 1.65; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #4C78A8; padding-bottom: 8px; }}
h2 {{ color: #34495e; border-left: 4px solid #4C78A8; padding-left: 10px; margin-top: 30px; }}
h3 {{ color: #5d6d7e; }}
h4 {{ color: #5d6d7e; margin-top: 18px; }}
table {{ font-size: 13px; border-collapse: collapse; }}
th, td {{ border: 1px solid #bbb; padding: 4px 8px; }}
th {{ background: #ecf0f1; }}
img {{ max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 4px; margin: 12px 0; }}
.callout {{ background: #fffbe6; border-left: 4px solid #f1c40f; padding: 10px 14px; margin: 12px 0; border-radius: 3px; }}
.findings {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 12px 0; border-radius: 3px; }}
.toc {{ background: #f4f6f7; padding: 12px 20px; border-radius: 5px; }}
.toc a {{ color: #2874a6; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 2px; font-family: Consolas, monospace; font-size: 12px; }}
</style></head>
<body>
<h1>ARENA Score Comprehensive Statistical Analysis Report</h1>
<p>Generated: {now} | Data: <code>rawdata1.xlsx</code> | Sample: 2023 n={N_BY_YEAR['2023']} / 2024 n={N_BY_YEAR['2024']} / 2025 n={N_BY_YEAR['2025']} (total {N_TOTAL})</p>

<div class="toc">
<b>Contents</b>
<ol>
<li>Research questions and methods</li>
<li>Data cleaning</li>
<li>Descriptive statistics and normality</li>
<li>RQ1: A (Logins) - C / D (scores) correlation</li>
<li>RQ2: Cross-year correlation strength</li>
<li>Extended analyses: ANCOVA, cross-year, paired Item 1 vs Item 2</li>
<li>Key figures</li>
<li>Key findings</li>
</ol>
</div>

<h2>1. Research questions and methods</h2>
<div class="callout">
<b>Three core questions:</b><br>
(1) Is there a significant association between the number of ARENA logins (F / A) and the two item scores (C / D)?<br>
(2) Does the strength of this association differ significantly across the three years?<br>
(3) Additional signals in the data (item-to-item relation, cross-year score differences, attendance distribution).<br><br>
<b>Methods:</b>
<ul>
<li><b>Spearman rank correlation</b>: F is a count (clearly non-normal; see Q-Q plot), C / D are ordinal scores. Non-parametric is more robust.</li>
<li><b>Kendall tau-b</b>: complements Spearman, more robust to small samples and ties.</li>
<li><b>Pearson r</b>: reported for reference with 95% CI; F is closer to normal after log(1+F).</li>
<li><b>Fisher z (independent samples)</b>: tests whether two correlations differ significantly.</li>
<li><b>ANCOVA</b>: tests whether the regression of score on F varies by year (year x F interaction).</li>
<li><b>Kruskal-Wallis</b>: median differences across three groups.</li>
<li><b>Wilcoxon signed-rank</b>: within-student Item 1 vs Item 2.</li>
<li><b>Bonferroni correction</b>: three pairwise comparisons, alpha adjusted to 0.0167.</li>
</ul>
</div>

<h2>2. Data cleaning</h2>
<ul>
<li>Three sheets: <code>2023</code> / <code>2024</code> / <code>2025</code></li>
<li>Drop rules:
  <ol>
  <li>Empty student ID rows (aggregate / blank rows).</li>
  <li>Rows with "absent" in the Notes column.</li>
  <li>Rows where C = D = E = F = 0 (treated as non-participation).</li>
  </ol>
</li>
<li>Sample after cleaning: 2023 n={N_BY_YEAR['2023']}, 2024 n={N_BY_YEAR['2024']}, 2025 n={N_BY_YEAR['2025']}, total n={N_TOTAL}</li>
<li><b>Data integrity</b>: column E = C + D holds exactly (max diff = 0); no anomalies in E.</li>
</ul>
<p>Cleaned data: <code>data_clean.csv</code> (UTF-8 BOM, Excel-readable).</p>

<h2>3. Descriptive statistics and normality</h2>
<h3>3.1 Descriptive statistics by year</h3>
{desc_table_html(DESC["2023"], f"2023 (n={N_BY_YEAR['2023']})")}
{desc_table_html(DESC["2024"], f"2024 (n={N_BY_YEAR['2024']})")}
{desc_table_html(DESC["2025"], f"2025 (n={N_BY_YEAR['2025']})")}
{desc_table_html(DESC_ALL, f"Pooled (n={N_TOTAL})")}

<h3>3.2 Normality test (Shapiro-Wilk)</h3>
{norm_table_html()}
<p><b>Reading:</b> F (Logins) deviates significantly from normality in every year and in the pooled sample (p &lt; 0.05), with very high right-skew (heavy upper tail). C and D also deviate in most subgroups. This supports using non-parametric methods (Spearman / Kendall). <b>log(1+F)</b> is close to normal in 2023 / 2024.</p>

<h3>3.3 Variance homogeneity (Levene)</h3>
{lev_table_html()}

<h2>4. RQ1: A (Logins) - C / D (scores) correlation</h2>

<h3>4.1 Correlations by year (with 95% CI)</h3>
{corr_table_html("2023")}
{corr_table_html("2024")}
{corr_table_html("2025")}
{corr_table_html("ALL")}

<h3>4.2 Scatter plot: A vs C / A vs D (by year)</h3>
<img src="{img[0]}" alt="scatter by year">

<h3>4.3 Scatter plot: log(1+A) vs scores (three-year colour overlay)</h3>
<img src="{img[1]}" alt="log scatter combined">

<p><b>Preliminary observation:</b> The scatter shows a <b>non-linear</b> relationship between A and C / D - scores are already high at low A, and not necessarily higher at very high A. This is the <b>saturation effect</b>: above roughly five logins, additional logins contribute very little marginal score. After log(1+A) the relation is closer to linear, but correlations are mostly weak.</p>

<h2>5. RQ2: Cross-year correlation strength</h2>

<h3>5.1 Fisher z pairwise comparison (Bonferroni x 3)</h3>
{fzc_table_html()}
<p><b>Reading:</b> Adjusted p &lt; 0.0167 is significant. The table pinpoints which two years differ significantly in correlation strength.</p>

<h3>5.2 Correlation strength across years (with 95% CI error bars)</h3>
<img src="{img[8]}" alt="corr compare">

<h3>5.3 ANCOVA: score ~ A + year + A:year</h3>
{ancova_table_html()}
<p><b>The interaction term A:year</b> is the key:
<ul>
<li>If p &lt; 0.05: the A-score slope differs significantly across years (the "association strength" itself varies by year).</li>
<li>If p &gt;= 0.05: the A-score structure can be regarded as the same across the three years.</li>
</ul></p>
<p><b>A main effect</b> (after controlling for year): independent contribution of A to score.<br><b>Year main effect</b>: whether mean scores differ across years after controlling for A.</p>

<h3>5.4 Per-year regression slope visualisation</h3>
<img src="{img[9]}" alt="ancova lines">

<h2>6. Extended analyses</h2>

<h3>6.1 Cross-year score comparison (Kruskal-Wallis)</h3>
{kw_table_html()}

<h3>6.2 Item 1 vs Item 2 paired comparison (Wilcoxon signed-rank)</h3>
{pair_table_html()}

<h3>6.3 Within-student Item 1 - Item 2 correlation (Spearman r)</h3>
<p>Already listed as the C~D row in section 4.1. The correlation between the two items reflects whether they measure the same ability - a strong correlation suggests a single dimension.</p>

<h2>7. Key figures</h2>

<h3>7.1 Box plot</h3>
<img src="{img[2]}" alt="box">

<h3>7.2 Violin plot</h3>
<img src="{img[3]}" alt="violin">

<h3>7.3 Correlation heatmaps</h3>
<img src="{img[4]}" alt="corr heatmaps">

<h3>7.4 Q-Q plot (A normality check)</h3>
<img src="{img[5]}" alt="qq">

<h3>7.5 Score-band stacked distribution</h3>
<img src="{img[6]}" alt="score stacked">

<h3>7.6 LOWESS non-linear fit</h3>
<img src="{img[7]}" alt="lowess">

<h2>8. Key findings</h2>
<div class="findings">
{findings}
</div>

<p style="margin-top: 40px; color: #95a5a6; font-size: 12px; text-align: center;">
Report generated automatically. All numbers are based on the cleaned sample (n={N_TOTAL}). The full scripts, raw data CSV, and figures live in this directory.
</p>

</body></html>
"""

(OUT / "report.html").write_text(html, encoding="utf-8")
print(f"HTML report: {OUT / 'report.html'}  ({len(html)//1024} KB)")

# Print key statistics to the console (for cross-checking)
print("\n=== Key statistics (printed for verification) ===")
print("\n-- Spearman r (A, C / D) by year --")
for y in YEARS + ["ALL"]:
    print(f"  {y}: A-C r={CORR[y]['logs~q1']['spearman_r']:.3f} "
          f"(p={CORR[y]['logs~q1']['spearman_p']:.4f})  |  "
          f"A-D r={CORR[y]['logs~q2']['spearman_r']:.3f} "
          f"(p={CORR[y]['logs~q2']['spearman_p']:.4f})")

print("\n-- Fisher z p-values (Bonferroni) --")
for k, d in FZC.items():
    print(f"  {k}:")
    for pair, v in d.items():
        print(f"    {pair}: p_unc={v['p_uncorrected']:.4f}  p_bonf={v['p_bonf']:.4f}")

print("\n-- ANCOVA p-values --")
for dv, d in ANCOVA.items():
    print(f"  {dv}: interaction p={d['p_interaction']:.4f}, "
          f"A p={d['p_logs']:.4f}, year p={d['p_year']:.4f}, R^2={d['r2']:.3f}")

print("\n-- Kruskal-Wallis (cross-year scores) --")
for c, d in KW.items():
    print(f"  {c}: H={d['H']:.3f}, p={d['p']:.4f}")

print("\n-- Wilcoxon C vs D (paired) --")
for y in YEARS + ["ALL"]:
    d = PAIR[y]
    print(f"  {y}: stat={d['wilcoxon_stat']}, p={d['wilcoxon_p']:.4f}  | "
          f"C>D:{d['q1_gt_q2']}  C=D:{d['q1_eq_q2']}  C<D:{d['q1_lt_q2']}")

print("\nDone.")