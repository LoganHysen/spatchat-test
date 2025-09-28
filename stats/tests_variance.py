# stats/tests_variance.py
# Variance / k-group tests: ANOVA, Kruskal–Wallis, Levene’s, Tukey HSD, Dunn’s
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from core_utils import ordered_groups
from plot_helpers import (
    box_plot,
    violin_plot,
    fig_to_np,
)

# Optional Dunn’s (exact) via scikit-posthocs
try:
    import scikit_posthocs as sp
except Exception:
    sp = None

def run_anova(df: pd.DataFrame, value: str, group: str) -> Tuple[str, List[np.ndarray]]:
    gorder = ordered_groups(df, group)
    data = [df[df[group].astype(str) == g][value].dropna().astype(float).values for g in gorder]
    if len(data) < 2:
        raise ValueError("ANOVA requires at least two groups.")
    F, p = stats.f_oneway(*data)
    txt = f"One-way ANOVA on {value} by {group}\nGroups: {', '.join(gorder)}\nF = {F:.4g}, p = {p:.5g}"
    img_box = box_plot(df, value, group)
    img_vio = violin_plot(df, value, group)
    return txt, [img_box, img_vio]


def kruskal_wallis(df: pd.DataFrame, value: str, group: str) -> Tuple[str, List[np.ndarray]]:
    gorder = ordered_groups(df, group)
    data = [df[df[group].astype(str) == g][value].dropna().astype(float).values for g in gorder]
    if len(data) < 2:
        raise ValueError("Kruskal–Wallis requires at least two groups.")
    H, p = stats.kruskal(*data)
    n = sum(len(d) for d in data)
    k = len(data)
    eps2 = (H - k + 1) / (n - k) if n > k else np.nan  # effect size epsilon-squared
    txt = f"Kruskal–Wallis on {value} by {group}\nGroups: {', '.join(gorder)}\nH = {H:.4g}, p = {p:.5g}, ε² = {eps2:.3g}"

    # Matplotlib (keeps house style set globally)
    fig_box, ax = plt.subplots(figsize=(6, 3))
    ax.boxplot(data, tick_labels=gorder)
    ax.set_title(f"Boxplot of {value} by {group}")
    img_box = fig_to_np(fig_box)

    fig_vio, ax = plt.subplots(figsize=(6, 3))
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(gorder) + 1))
    ax.set_xticklabels(gorder)
    ax.set_title(f"Violin of {value} by {group}")
    img_vio = fig_to_np(fig_vio)
    return txt, [img_box, img_vio]


def levene_test(df: pd.DataFrame, value: str, group: str, center: str = "median") -> Tuple[str, List[np.ndarray]]:
    gorder = ordered_groups(df, group)
    data = [df[df[group].astype(str) == g][value].dropna().astype(float).values for g in gorder]
    if len(data) < 2:
        raise ValueError("Levene’s test requires at least two groups.")
    W, p = stats.levene(*data, center=center)
    txt = f"Levene’s test for equal variances on {value} by {group}\ncenter={center}, W = {W:.4g}, p = {p:.5g}"
    return txt, []


def tukey_hsd(df: pd.DataFrame, value: str, group: str) -> Tuple[str, List[np.ndarray]]:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    d = df[[value, group]].dropna()
    d = d.astype({value: float})
    res = pairwise_tukeyhsd(endog=d[value].values, groups=d[group].astype(str).values, alpha=0.05)
    summ = res.summary()
    data = summ.data[1:]  # skip header row
    cols = [c.strip() for c in summ.data[0]]
    tukey_df = pd.DataFrame(data, columns=cols)
    tukey_df.to_csv("outputs/tukey_hsd_results.csv", index=False)

    # Means + 95% t-CI per group (for preview plot)
    gorder = ordered_groups(d, group)
    means, ci_lo, ci_hi = [], [], []
    for g in gorder:
        vals = d[d[group].astype(str) == g][value].values
        n = len(vals)
        m = np.mean(vals) if n else np.nan
        sd = np.std(vals, ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n else np.nan
        tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
        lo, hi = (m - tcrit * se, m + tcrit * se) if (np.isfinite(se) and np.isfinite(tcrit)) else (np.nan, np.nan)
        means.append(m); ci_lo.append(lo); ci_hi.append(hi)

    x = np.arange(len(gorder))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(ci_lo), np.array(ci_hi) - np.array(means)],
        fmt="o",
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(gorder)
    ax.set_title("Tukey HSD (group means with 95% t-CIs)")
    img = fig_to_np(fig)

    txt = "Tukey HSD pairwise comparisons (α=0.05)\n" + tukey_df.to_string(index=False)
    return txt, [img]


def _holm_adjust(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni step-down adjustment (monotone, family-wise error control)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    # step-down: compare smallest p with α/m, next with α/(m-1), ...
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        adj_val = min(1.0, pvals[idx] * factor)
        running_max = max(running_max, adj_val)
        adj[idx] = running_max  # enforce monotonicity
    return adj.tolist()


def dunn_posthoc(df: pd.DataFrame, value: str, group: str, p_adjust: str = "holm") -> Tuple[str, List[np.ndarray]]:
    d = df[[value, group]].dropna()
    d = d.astype({value: float})
    if sp is not None:
        ptab = sp.posthoc_dunn(a=d, val_col=value, group_col=group, p_adjust=p_adjust)
        ptab.to_csv(f"outputs/dunn_{p_adjust}_pvalues.csv")
        txt = f"Dunn’s post-hoc (p-adjust={p_adjust}) after Kruskal–Wallis\n" + ptab.round(4).to_string()
        return txt, []

    # Fallback: pairwise Mann–Whitney + Holm adjust
    from itertools import combinations
    levels = ordered_groups(d, group)
    pairs, raw_p = [], []
    for a, b in combinations(levels, 2):
        va = d[d[group].astype(str) == a][value].values
        vb = d[d[group].astype(str) == b][value].values
        if len(va) == 0 or len(vb) == 0:
            p = np.nan
        else:
            p = stats.mannwhitneyu(va, vb, alternative="two-sided").pvalue
        pairs.append((a, b))
        raw_p.append(p)

    finite_idx = [i for i, p in enumerate(raw_p) if np.isfinite(p)]
    adj_p = [np.nan] * len(raw_p)
    if finite_idx:
        finite_vals = [raw_p[i] for i in finite_idx]
        adj_vals = _holm_adjust(finite_vals)
        for i, v in zip(finite_idx, adj_vals):
            adj_p[i] = v

    rows = [{"group1": a, "group2": b, "p_raw": p, "p_adj_holm": pa} for (a, b), p, pa in zip(pairs, raw_p, adj_p)]
    out = pd.DataFrame(rows)
    out.to_csv("outputs/dunn_fallback_mwu_holm.csv", index=False)
    txt = (
        "Dunn’s post-hoc (approx via pairwise Mann–Whitney + Holm; install scikit-posthocs for exact Dunn)\n"
        + out.round(4).to_string(index=False)
    )
    return txt, []
