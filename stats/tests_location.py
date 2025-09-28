# stats/tests_location.py
# Location tests & related effects: t-test, Mann–Whitney, Wilcoxon signed-rank, point-biserial
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# expected helper modules in project root (provided in subsequent snippets)
from core_utils import ordered_groups, is_integer_like
from plot_helpers import (
    bar_with_error_plot,
    box_plot,
    violin_plot,
    fig_to_np,
)

# ---------- t-test helpers ----------
def ttest_summary(
    a: np.ndarray, b: np.ndarray, g1: str, g2: str, equal_var: bool = False, paired: bool = False
) -> str:
    if paired:
        stat, p = stats.ttest_rel(a, b, nan_policy="omit")
        test_name = "Paired t-test"
        df_est = len(a) - 1 if len(a) == len(b) else np.nan
    else:
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
        test_name = "Student t-test" if equal_var else "Welch t-test"
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        na, nb = len(a), len(b)
        df_est = (va / na + vb / nb) ** 2 / (
            (va**2) / ((na**2) * (na - 1)) + (vb**2) / ((nb**2) * (nb - 1))
        )

    mean_a, mean_b = a.mean(), b.mean()
    sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
    diff = mean_a - mean_b
    se = np.sqrt(sd_a**2 / len(a) + sd_b**2 / len(b))
    try:
        tcrit = stats.t.ppf(0.975, df=df_est)
        ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
    except Exception:
        ci_low = ci_high = np.nan

    # Cohen's d (pooled) + Hedges' g
    sp = np.sqrt(((len(a) - 1) * sd_a**2 + (len(b) - 1) * sd_b**2) / (len(a) + len(b) - 2)) if len(a) + len(b) - 2 > 0 else np.nan
    d = diff / sp if sp and sp > 0 else np.nan
    J = 1 - (3 / (4 * (len(a) + len(b) - 2) - 1)) if (len(a) + len(b) - 2) > 1 else 1.0
    g = d * J if np.isfinite(d) else np.nan

    lines = [
        "t-test",
        test_name,
        f"Groups: {g1} (n={len(a)}, mean={mean_a:.3g}, sd={sd_a:.3g}) vs {g2} (n={len(b)}, mean={mean_b:.3g}, sd={sd_b:.3g})",
        f"Mean difference = {diff:.3g}, 95% CI [{ci_low:.4g}, {ci_high:.4g}]",
        f"t({df_est:.2f}) = {stat:.3g}, p = {p:.5g}",
        f"Effect size: Cohen's d = {d:.3g}, Hedges' g = {g:.3g}",
    ]
    return "\n".join(lines)


def run_ttest(
    df: pd.DataFrame, value: str, group: str, paired: bool = False, equal_var: bool = False
) -> Tuple[str, List[np.ndarray]]:
    gorder = ordered_groups(df, group)
    if len(gorder) != 2:
        raise ValueError(f"t-test requires exactly 2 groups; {group} has {len(gorder)} levels.")

    a = df[df[group].astype(str) == gorder[0]][value].dropna().astype(float).values
    b = df[df[group].astype(str) == gorder[1]][value].dropna().astype(float).values

    txt = ttest_summary(a, b, gorder[0], gorder[1], equal_var=equal_var, paired=paired)

    # Figures
    img_bar = bar_with_error_plot(df, value, group, error="sem", gorder=gorder)
    img_box = box_plot(df, value, group)
    img_vio = violin_plot(df, value, group)
    return txt, [img_bar, img_box, img_vio]


# ---------- Mann–Whitney U ----------
def mann_whitney(df: pd.DataFrame, value: str, group: str) -> Tuple[str, List[np.ndarray]]:
    gorder = ordered_groups(df, group)
    if len(gorder) != 2:
        raise ValueError(f"Mann–Whitney U requires exactly 2 groups; {group} has {len(gorder)} levels.")

    a = df[df[group].astype(str) == gorder[0]][value].dropna().astype(float).values
    b = df[df[group].astype(str) == gorder[1]][value].dropna().astype(float).values

    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    A = U / (n1 * n2) if n1 > 0 and n2 > 0 else np.nan  # common-language effect size
    r_rb = 2 * A - 1 if np.isfinite(A) else np.nan       # rank-biserial correlation

    txt = (
        f"Mann–Whitney U on {value} by {group}\n"
        f"Groups: {gorder[0]} (n={n1}) vs {gorder[1]} (n={n2})\n"
        f"U = {U:.4g}, p = {p:.5g}, rank-biserial r = {r_rb:.3g} (A = {A:.3g})"
    )

    img_bar = bar_with_error_plot(df, value, group, error="sem", gorder=gorder)

    # Matplotlib box/violin (house style should be set globally in app)
    fig_box, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot([a, b], tick_labels=[gorder[0], gorder[1]])
    ax.set_title(f"Boxplot of {value} by {group}")
    img_box = fig_to_np(fig_box)

    fig_vio, ax = plt.subplots(figsize=(5, 3))
    ax.violinplot([a, b], showmeans=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([gorder[0], gorder[1]])
    ax.set_title(f"Violin of {value} by {group}")
    img_vio = fig_to_np(fig_vio)

    return txt, [img_bar, img_box, img_vio]


# ---------- Wilcoxon signed-rank (paired) ----------
def wilcoxon_signed(df: pd.DataFrame, a: str, b: str) -> Tuple[str, List[np.ndarray]]:
    s = df[[a, b]].dropna().astype(float)
    if len(s) < 3:
        raise ValueError("Wilcoxon signed-rank needs at least 3 paired observations.")
    stat, p = stats.wilcoxon(s[a], s[b], zero_method="wilcox", alternative="two-sided")
    txt = f"Wilcoxon signed-rank on paired columns {a} vs {b}\nn={len(s)}, W = {stat:.4g}, p = {p:.5g}"

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(s[a], s[b])
    lim = [min(s[a].min(), s[b].min()), max(s[a].max(), s[b].max())]
    ax.plot(lim, lim, linestyle=":")
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title("Paired scatter (y=x reference)")
    img = fig_to_np(fig)
    return txt, [img]


# ---------- Point-biserial correlation ----------
def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 3 or np.isclose(abs(r), 1.0):
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return lo, hi


def point_biserial(df: pd.DataFrame, value: str, group: str) -> Tuple[str, List[np.ndarray]]:
    g = df[group]
    if pd.api.types.is_numeric_dtype(g):
        uniq = pd.unique(g.dropna())
        if len(uniq) != 2 or not is_integer_like(g):
            raise ValueError(f"{group} must be binary for point-biserial.")
        gb = g.astype(float)
    else:
        levels = pd.unique(g.dropna())
        if len(levels) != 2:
            raise ValueError(f"{group} must have exactly two levels for point-biserial.")
        mapping = {str(levels[0]): 0.0, str(levels[1]): 1.0}
        gb = g.astype(str).map(mapping)

    d = pd.DataFrame({value: df[value], "_gb": gb}).dropna()
    if len(d) < 3:
        raise ValueError("Not enough data for point-biserial.")
    try:
        r, p = stats.pointbiserialr(d["_gb"].values, d[value].astype(float).values)
    except Exception:
        r, p = stats.pearsonr(d["_gb"].values, d[value].astype(float).values)
    ci = _pearson_ci(r, len(d))

    lines = [f"Point-biserial correlation between {value} and {group}", f"n={len(d)}, r = {r:.4g}, p = {p:.5g}"]
    if not np.isnan(ci[0]):
        lines.append(f"95% CI for r: [{ci[0]:.3g}, {ci[1]:.3g}]")

    gorder = ordered_groups(df, group)
    img_bar = bar_with_error_plot(df, value, group, error="sem", gorder=gorder)
    img_box = box_plot(df, value, group)
    return "\n".join(lines), [img_bar, img_box]
