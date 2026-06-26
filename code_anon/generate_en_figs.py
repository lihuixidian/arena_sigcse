# -*- coding: utf-8 -*-
"""
ARENA Score Analysis - English Figure Generation (v3 academic style)

- Reads data_clean.csv and regenerates 11 figures with English labels.
- Academic style: serif / Times font + muted palette + simplified borders.
- Variable mapping: C -> SQ1, D -> SQ2, E -> SQ1+SQ2, F -> A
  (italic, enlarged).
- File names carry the semantic titles (no figure-level suptitle).
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
from pathlib import Path

# ---------- Config ----------
OUT_FIG = Path("figs")
OUT_FIG.mkdir(parents=True, exist_ok=True)
DATA = "data_clean.csv"

YEARS = ["2023", "2024", "2025"]
# Academic-style muted palette (Nature / Cell inspired) - blue / brick red / green.
COLORS = {"2023": "#4C72B0", "2024": "#C44E52", "2025": "#55A868"}

# Academic-style global font: serif + Times.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "STIX Two Text", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="serif",
              rc={"axes.unicode_minus": False,
                  "axes.linewidth": 0.8,
                  "font.family": "serif",
                  "font.serif": ["Times New Roman", "Liberation Serif", "STIX Two Text", "DejaVu Serif"]})

# Display rules for variable names (italic + enlarged).  Font size is
# controlled separately via the matplotlib text fontsize argument.
VAR_LABELS = {
    "q1":    r"$\mathit{SQ}_{1}$",
    "q2":    r"$\mathit{SQ}_{2}$",
    "total": r"$\mathit{SQ}_{1}{+}\mathit{SQ}_{2}$",
    "logs":  r"$\mathit{A}$",
    "logF1": r"$\log(1{+}\mathit{A})$",
}

# Default font sizes (slightly larger so italic labels stay readable).
LABEL_FS = 14
TICK_FS  = 11
TITLE_FS = 12

# ---------- Read data ----------
DF = pd.read_csv(DATA, dtype={"year": str})
DF["year"] = DF["year"].astype(str)
print(f"Loaded {len(DF)} rows; per-year: {dict(DF.year.value_counts())}")
N = {y: int((DF.year == y).sum()) for y in YEARS}

# ---------- Helpers ----------
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
    zl = z - stats.norm.ppf(1 - alpha/2) * se
    zh = z + stats.norm.ppf(1 - alpha/2) * se
    return ((np.exp(2*zl) - 1) / (np.exp(2*zl) + 1), (np.exp(2*zh) - 1) / (np.exp(2*zh) + 1))


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


# Pre-compute
DESC = {y: {c: desc(DF[DF.year == y], c) for c in ["q1", "q2", "total", "logs", "logF1"]} for y in YEARS}
DESC_ALL = {c: desc(DF, c) for c in ["q1", "q2", "total", "logs", "logF1"]}

CORR = {}
for y in YEARS + ["ALL"]:
    sub = DF if y == "ALL" else DF[DF.year == y]
    n = len(sub)
    CORR[y] = {}
    for (a, b) in [("logs", "q1"), ("logs", "q2"), ("logs", "total"),
                   ("logF1", "q1"), ("logF1", "q2"), ("logF1", "total"),
                   ("q1", "q2")]:
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

NORM = {}
for y in YEARS + ["ALL"]:
    sub = DF if y == "ALL" else DF[DF.year == y]
    NORM[y] = {}
    for c in ["q1", "q2", "total", "logs", "logF1"]:
        sw = stats.shapiro(sub[c])
        NORM[y][c] = {"W": sw.statistic, "p": sw.pvalue}

LOW = {}
for y in YEARS:
    for (a, b) in [("logs", "q1"), ("logs", "q2"), ("logs", "total")]:
        sub = DF[DF.year == y]
        if len(sub) < 5:
            continue
        frac = max(0.4, 5/len(sub))
        ls = sm.nonparametric.lowess(sub[b], sub[a], frac=frac, return_sorted=True)
        LOW[f"{y}|{a}~{b}"] = ls


def save(fig, name, dpi=150):
    p = OUT_FIG / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")
    return p


def style_axes(ax):
    """Apply a unified academic style: drop top/right spines, dotted
    horizontal grid lines, consistent font sizes."""
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis='both', labelsize=TICK_FS, length=4, width=0.8)
    ax.grid(True, axis='y', alpha=0.45, linestyle=':', linewidth=0.6)
    ax.set_axisbelow(True)

# ===================================================================
# Fig 01: Scatter A vs SQ1, A vs SQ2  (by year, with linear fit)
# ===================================================================
def fig01_scatter_by_year():
    fig, axes = plt.subplots(3, 2, figsize=(11, 12.5))
    for i, y in enumerate(YEARS):
        sub = DF[DF.year == y]
        for j, (x, dv) in enumerate([("logs", "q1"), ("logs", "q2")]):
            ax = axes[i, j]
            ax.scatter(sub[x], sub[dv], s=42, alpha=0.78, color=COLORS[y],
                       edgecolor="white", linewidth=0.5)
            if len(sub) >= 3:
                slope, intercept, r, p, se = stats.linregress(sub[x], sub[dv])
                xs = np.linspace(sub[x].min(), sub[x].max(), 100)
                ax.plot(xs, intercept + slope*xs, "--", color="#444",
                        linewidth=1.5, alpha=0.75)
                tag = "**" if p < 0.05 else ("*" if p < 0.1 else "")
                ax.set_title(f"{y}:  {VAR_LABELS[x]}  vs.  {VAR_LABELS[dv]}   "
                             f"$r={r:.2f}${tag},  $p={fmt_p(p)}$   ($n={len(sub)}$)",
                             fontsize=TITLE_FS)
            ax.set_xlabel(VAR_LABELS[x], fontsize=LABEL_FS)
            ax.set_ylabel(VAR_LABELS[dv], fontsize=LABEL_FS)
            style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig01_scatter_plots_of_attendance_vs_item_scores_by_year_with_linear_fit.png")

# ===================================================================
# Fig 02: Combined log(1+A) scatter
# ===================================================================
def fig02_log_scatter_combined():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for j, dv in enumerate(["q1", "q2"]):
        ax = axes[j]
        for y in YEARS:
            sub = DF[DF.year == y]
            ax.scatter(sub["logF1"], sub[dv], s=48, alpha=0.78, color=COLORS[y],
                       edgecolor="white", linewidth=0.5, label=y)
        sl = stats.linregress(DF["logF1"], DF[dv])
        xs = np.linspace(DF["logF1"].min(), DF["logF1"].max(), 100)
        ax.plot(xs, sl.intercept + sl.slope*xs, "k--", alpha=0.6, linewidth=1.5,
                label=f"Pooled: $r={sl.rvalue:.2f}$")
        ax.set_xlabel(VAR_LABELS["logF1"], fontsize=LABEL_FS)
        ax.set_ylabel(VAR_LABELS[dv], fontsize=LABEL_FS)
        ax.set_title(f"{VAR_LABELS['logF1']}  vs.  {VAR_LABELS[dv]}  (three years combined)",
                     fontsize=TITLE_FS)
        ax.legend(loc="best", fontsize=9, frameon=True)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig02_pooled_scatter_plot_log_transformed_F_for_linearity.png")

# ===================================================================
# Fig 03: Box plot of SQ1, SQ2, A by year
# ===================================================================
def fig03_box_by_year():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for j, c in enumerate(["q1", "q2", "logs"]):
        ax = axes[j]
        data = [DF[DF.year == y][c].values for y in YEARS]
        bp = ax.boxplot(data, tick_labels=YEARS, patch_artist=True,
                        showmeans=True, meanline=True,
                        medianprops={"color": "black", "linewidth": 1.3},
                        meanprops={"color": "black", "linewidth": 1.0, "linestyle": "--"},
                        whiskerprops={"color": "black", "linewidth": 1.0},
                        capprops={"color": "black", "linewidth": 1.0},
                        boxprops={"linewidth": 1.0, "edgecolor": "black"},
                        flierprops={"marker": "o", "markerfacecolor": "none",
                                    "markeredgecolor": "black", "markersize": 4.5,
                                    "markeredgewidth": 0.8})
        for patch, y in zip(bp["boxes"], YEARS):
            patch.set_facecolor(COLORS[y])
            patch.set_alpha(0.30)
        ax.set_title(f"Distribution of  {VAR_LABELS[c]}", fontsize=TITLE_FS)
        ax.set_ylabel(VAR_LABELS[c], fontsize=LABEL_FS)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig03_box_plots_of_key_variables_by_year.png")

# ===================================================================
# Fig 04: Violin + strip plot
# ===================================================================
def fig04_violin_by_year():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for j, c in enumerate(["q1", "q2", "logs"]):
        ax = axes[j]
        sns.violinplot(data=DF, x="year", y=c, order=YEARS, palette=COLORS,
                       inner="quartile", ax=ax, hue="year", legend=False, linewidth=1)
        sns.stripplot(data=DF, x="year", y=c, order=YEARS, color="black",
                      alpha=0.55, size=3, ax=ax)
        ax.set_title(f"Density + raw data:  {VAR_LABELS[c]}", fontsize=TITLE_FS)
        ax.set_xlabel("Year", fontsize=LABEL_FS)
        ax.set_ylabel(VAR_LABELS[c], fontsize=LABEL_FS)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig04_violin_plots_with_overlaid_raw_data_points.png")

# ===================================================================
# Fig 05: Correlation heatmaps (Spearman)
# ===================================================================
def fig05_corr_heatmaps():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    cols = ["q1", "q2", "total", "logs", "logF1"]
    display = [VAR_LABELS[c] for c in cols]
    for i, (y, t) in enumerate(zip(YEARS + ["ALL"], YEARS + ["Pooled"])):
        sub = DF if y == "ALL" else DF[DF.year == y]
        r = sub[cols].corr(method="spearman")
        r.columns = display
        r.index = display
        sns.heatmap(r, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, square=True, cbar=(i == 3), ax=axes[i],
                    annot_kws={"size": 9})
        axes[i].set_title(f"Spearman  -  {t}  ($n={len(sub)}$)", fontsize=TITLE_FS)
    fig.tight_layout()
    return save(fig, "fig05_spearman_correlation_matrices_by_year_and_pooled.png")

# ===================================================================
# Fig 06: Q-Q plot of A (normality check)
# ===================================================================
def fig06_qq_F():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, y in enumerate(YEARS):
        ax = axes[i]
        sub = DF[DF.year == y]
        stats.probplot(sub["logs"], dist="norm", plot=ax)
        w = NORM[y]['logs']['W']
        p = NORM[y]['logs']['p']
        ax.set_title(f"{y}: Shapiro $W={w:.3f}$,  $p={p:.3f}$", fontsize=TITLE_FS)
        ax.set_xlabel("Theoretical normal quantiles", fontsize=LABEL_FS)
        ax.set_ylabel(f"Sample quantiles ({VAR_LABELS['logs']})", fontsize=LABEL_FS)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig06_qq_plots_of_A_attendance_against_normal_distribution.png")

# ===================================================================
# Fig 07: Score band stacked histogram
# ===================================================================
def fig07_score_stacked():
    bins_q = [0, 2, 4, 6, 8, 10, 11]
    labels_q = ["0-2", "2-4", "4-6", "6-8", "8-10", "10"]
    bins_t = [0, 4, 8, 12, 16, 20, 21]
    labels_t = ["0-4", "4-8", "8-12", "12-16", "16-20", "20"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for j, c in enumerate(["q1", "q2", "total"]):
        ax = axes[j]
        bins, labels = (bins_t, labels_t) if c == "total" else (bins_q, labels_q)
        width = 0.25
        x = np.arange(len(labels))
        for k, y in enumerate(YEARS):
            sub = DF[DF.year == y]
            counts, _ = np.histogram(sub[c], bins=bins)
            ax.bar(x + (k-1)*width, counts, width, color=COLORS[y], label=y,
                   alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Score distribution:  {VAR_LABELS[c]}", fontsize=TITLE_FS)
        ax.set_xlabel("Score band", fontsize=LABEL_FS)
        ax.set_ylabel("Number of students", fontsize=LABEL_FS)
        ax.legend(title="Year", fontsize=9)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig07_stacked_score_band_distributions_by_year.png")

# ===================================================================
# Fig 08: LOWESS fits of SQ1, SQ2 on A
# ===================================================================
def fig08_lowess():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for j, dv in enumerate(["q1", "q2"]):
        ax = axes[j]
        for y in YEARS:
            sub = DF[DF.year == y]
            ax.scatter(sub["logs"], sub[dv], s=42, alpha=0.62, color=COLORS[y],
                       edgecolor="white", linewidth=0.5)
            ls = LOW.get(f"{y}|logs~{dv}")
            if ls is not None and len(ls) > 1:
                ax.plot(ls[:, 0], ls[:, 1], color=COLORS[y], linewidth=2,
                        label=f"{y} LOWESS")
        ax.set_xlabel(VAR_LABELS["logs"], fontsize=LABEL_FS)
        ax.set_ylabel(VAR_LABELS[dv], fontsize=LABEL_FS)
        ax.set_title(f"{VAR_LABELS['logs']}  vs.  {VAR_LABELS[dv]}  -  non-parametric (LOWESS) fit",
                     fontsize=TITLE_FS)
        ax.legend(loc="best", fontsize=9)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig08_nonparametric_LOWESS_fits_of_scores_on_attendance.png")

# ===================================================================
# Fig 09: Pearson r comparison with 95% CI (the core figure)
# ===================================================================
def fig09_corr_compare():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
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
        ax.bar(labels, rs, color=[COLORS[y] for y in YEARS], alpha=0.7,
               edgecolor="black", linewidth=1)
        ax.errorbar(labels, rs, yerr=yerr, fmt="none", ecolor="black",
                    capsize=5, linewidth=1.5)
        for i, y in enumerate(YEARS):
            d = CORR[y][f"{x}~{dv}"]
            ax.text(i, rs[i] + 0.04 if rs[i] >= 0 else rs[i] - 0.07,
                    f"$r={d['pearson_r']:.2f}$\n$p={fmt_p(d['pearson_p'])}$",
                    ha="center", fontsize=9.5)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"Pearson  $r$:  {VAR_LABELS[x]}  vs.  {VAR_LABELS[dv]}  (by year, 95% CI)",
                     fontsize=TITLE_FS)
        ax.set_ylabel("Pearson  $r$", fontsize=LABEL_FS)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig09_attendance_score_correlation_strength_year_by_year_comparison.png")

# ===================================================================
# Fig 10: ANCOVA - per-year regression lines
# ===================================================================
def fig10_ancova_lines():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for j, dv in enumerate(["q1", "q2"]):
        ax = axes[j]
        for y in YEARS:
            sub = DF[DF.year == y]
            ax.scatter(sub["logs"], sub[dv], s=42, alpha=0.68, color=COLORS[y],
                       edgecolor="white", linewidth=0.5, label=y)
            if len(sub) >= 3:
                sl = stats.linregress(sub["logs"], sub[dv])
                xs = np.linspace(sub["logs"].min(), sub["logs"].max(), 100)
                ax.plot(xs, sl.intercept + sl.slope*xs,
                        color=COLORS[y], linewidth=2)
                ax.text(0.05, 0.95 - 0.08*YEARS.index(y),
                        f"{y}: slope = {sl.slope:.3f}",
                        transform=ax.transAxes, color=COLORS[y],
                        fontsize=10, fontweight="bold")
        ax.set_xlabel(VAR_LABELS["logs"], fontsize=LABEL_FS)
        ax.set_ylabel(VAR_LABELS[dv], fontsize=LABEL_FS)
        ax.set_title(f"ANCOVA view:  {VAR_LABELS[dv]}  $\\sim$ {VAR_LABELS['logs']} $+$ year $+$ {VAR_LABELS['logs']}$\\!:\\!$year",
                     fontsize=TITLE_FS)
        ax.legend(loc="best", fontsize=9)
        style_axes(ax)
    fig.tight_layout()
    return save(fig, "fig10_per_year_regression_lines_of_scores_on_A_slope_differences_equals_interaction.png")

# ===================================================================
# Fig 11: ARENA policy regime x A distribution
# ===================================================================
def fig11_arena_policy_f_distribution():
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8),
                              gridspec_kw={"height_ratios": [1.0, 4.5, 0.9],
                                           "hspace": 0.40, "wspace": 0.25})

    policies = [
        ("2023", "Optional (informed only)\nSelf-directed exploration"),
        ("2024", "Mandatory + use-count $\\to$ grade\nCredit-seeking behaviour"),
        ("2025", "Mandatory + binary (used/not)\nThreshold behaviour"),
    ]
    summary_stats = []
    for y in YEARS:
        s = DF[DF.year == y]["logs"]
        summary_stats.append((y, len(s), s.median(), s.mean(), s.max()))

    for i, ((y, policy_text), (yy, n, med, mean, mx)) in enumerate(zip(policies, summary_stats)):
        # Top: policy callout
        ax_top = axes[0, i]
        ax_top.text(0.5, 0.5, policy_text, ha='center', va='center',
                    fontsize=10.5, transform=ax_top.transAxes,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor=COLORS[y],
                              alpha=0.18, edgecolor=COLORS[y], linewidth=1.5))
        ax_top.axis('off')

        # Middle: violin + strip
        ax = axes[1, i]
        sub = DF[DF.year == y]
        sns.violinplot(data=sub, y='logs', color=COLORS[y],
                       inner='quartile', ax=ax, linewidth=1)
        sns.stripplot(data=sub, y='logs', color='black', alpha=0.55, size=3.5,
                      ax=ax, jitter=0.05)
        ax.set_title(f"{y}  ($n={n}$)", fontsize=TITLE_FS)
        ax.set_ylabel(VAR_LABELS["logs"] if i == 0 else "", fontsize=LABEL_FS)
        ax.set_xlabel("")
        style_axes(ax)

        # Bottom: summary statistics
        ax_bot = axes[2, i]
        ax_bot.text(0.5, 0.5,
                    f"median $= {med:.1f}$   .   mean $= {mean:.1f}$   .   max $= {mx:.0f}$",
                    ha='center', va='center', fontsize=10,
                    transform=ax_bot.transAxes, family="monospace")
        ax_bot.axis('off')

    fig.tight_layout()
    # File name carries the figure-level title (no suptitle at top of figure).
    return save(fig, "fig11_ARENA_grading_policy_regime_and_resulting_A_distribution_by_year.png")


# ===================================================================
# Run all
# ===================================================================
print("Generating English academic-style figures...")
fig01_scatter_by_year()
fig02_log_scatter_combined()
fig03_box_by_year()
fig04_violin_by_year()
fig05_corr_heatmaps()
fig06_qq_F()
fig07_score_stacked()
fig08_lowess()
fig09_corr_compare()
fig10_ancova_lines()
fig11_arena_policy_f_distribution()
print("All 11 figures regenerated with academic style + italic labels (C->SQ1, D->SQ2, E->SQ1+SQ2, F->A).")