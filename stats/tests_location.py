# stats/tests_location.py
# Two-group & paired location tests: t-test (ind/paired), Mann–Whitney U, Wilcoxon signed-rank, Point-biserial
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from core_utils import ordered_groups
from plot_helpers import (
    bar_with_error_plot,
    box_plot,
    violin_plot,
    fig_to_np,
)

# Optional visual overlay for significance on bar plots
try:
    from viz_annotations import annotate_ttest_on_bar as _annotate_bar
except Exception:
    _annotate_bar = None


# ---------- Effect sizes & helpers ----------
def _cohen_d_independent_from_arrays(a: np.ndarray, b: np.ndarray, equal_var: bool = False) -> Tuple[float, float]:
    """
    Returns (Cohen's d, Hedges' g) for two independent samples.
    If equal_var=True, uses pooled SD; else average of group variances.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mean_a, mean_b = np.nanmean(a), np.nanmean(b)
    va, vb = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
    na, nb = np.isfinite(a).sum(), np.isfinite(b).sum()
    diff = mean_a - mean_b

    if equal_var:
        denom_df = max(1, (na + nb - 2))
        sp2 = ((na - 1) * va + (nb - 1) * vb) / denom_df if denom_df > 0 else np.nan
        sd = np.sqrt(sp2) if sp2 > 0 else np.nan
    else:
        sd = np.sqrt((va + vb) / 2.0) if (va > 0 and vb > 0) else np.nan

    d = diff / sd if (sd and sd > 0) else np.nan
    J = 1 - (3 / (4 * (na + nb - 2) - 1)) if (na + nb - 2) > 1 else 1.0
    g = d * J if np.isfinite(d) else np.nan
    return float(d), float(g)


# ---------- t-test ----------
def ttest_summary(a: np.ndarray, b: np.ndarray, g1: str, g2: str, equal_var=False, paired=False) -> str:
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

    # Cohen's d (pooled) + Hedges' g for independent-samples; paired uses dz
    if not paired:
        d, g = _cohen_d_independent_from_arrays(a, b, equal_var=equal_var)
    else:
        diffs = a - b
        sd_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else np.nan
        d = (np.mean(diffs) / sd_diff) if (sd_diff and sd_diff > 0) else np.nan
        J = 1 - (3 / (4 * (len(diffs) - 1) - 1)) if (len(diffs) - 1) > 1 else 1.0
        g = d * J if np.isfinite(d) else np.nan

    lines = [
        "t-test",
        test_name,
        f"Groups: {g1} (n={len(a)}, mean={mean_a:.3g}, sd={sd_a:.3g}) vs {g2} (n={len(b)}, mean={mean_b:.3g}, sd={sd_b:.3g})",
        f"Mean difference = {diff:.3g}, 95% CI [{ci_low:.4g}, {ci_high:.4g}]",
        f"t({df_est:.2f}) = {float(stat):.3g}, p = {float(p):.5g}",
        f"Effect size: Cohen's d = {float(d):.3g}, Hedges' g = {float(g):.3g}",
    ]
    return "\n".join(lines)


def run_ttest(
    df: pd.DataFrame, value: str, group: str, paired: bool = False, equal_var: bool = False
) -> Tuple[str, List[np.ndarray]]:
    """
    Compute two-sample t-test (Welch by default) or paired t-test.
    Returns (text, [bar±SEM, box, violin]) — signature unchanged.
    If viz_annotations.py is present, the first image is auto-annotated with p-value.
    """
    gorder = ordered_groups(df, group)
    if len(gorder) != 2:
        raise ValueError(f"t-test requires exactly 2 groups; {group} has {len(gorder)} levels.")

    g1, g2 = gorder[0], gorder[1]
    a = df[df[group].astype(str) == g1][value].dropna().astype(float).values
    b = df[df[group].astype(str) == g2][value].dropna().astype(float).values

    # Base visuals
    img_bar = bar_with_error_plot(df, value, group, error="sem", gorder=gorder)
    img_box = box_plot(df, value, group)
    img_vio = violin_plot(df, value, group)

    # Stats + (optional) annotated overlay for the bar plot
    txt = ttest_summary(a, b, g1, g2, equal_var=equal_var, paired=paired)

    # Recompute p here to overlay (ttest_summary does not return it)
    try:
        if paired:
            stat, p = stats.ttest_rel(a, b, nan_policy="omit")
            note = "Paired t-test"
        else:
            stat, p = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
            note = "Student t-test" if equal_var else "Welch t-test"
        if _annotate_bar is not None and np.isfinite(p):
            img_bar = _annotate_bar(df, value, group, g1, g2, float(p), note=note)
    except Exception:
        # If anything goes wrong, silently keep the un-annotated bar
        pass

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
    r_rb = 2 * A - 1 if np.isfinite(A) else np.nan      # rank-biserial correlation

    txt = [
        f"Mann–Whitney U on {value} by {group}",
        f"Groups: {gorder[0]} (n={n1}) vs {gorder[1]} (n={n2})",
        f"U = {float(U):.4g}, p = {float(p):.5g}, rank-biserial r = {float(r_rb):.3g} (A = {float(A):.3g})",
    ]

    # Visuals
    img_bar = bar_with_error_plot(df, value, group, error="sem", gorder=gorder)

    # Matplotlib-only box/violin (to mirror original look)
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

    return "\n".join(txt), [img_bar, img_box, img_vio]


# ---------- Wilcoxon signed-rank (paired) ----------
def wilcoxon_signed(df: pd.DataFrame, a: str, b: str) -> Tuple[str, List[np.ndarray]]:
    s = df[[a, b]].dropna().astype(float)
    if len(s) < 3:
        raise ValueError("Wilcoxon signed-rank needs at least 3 paired observations.")
    stat, p = stats.wilcoxon(s[a], s[b], zero_method="wilcox", alternative="two-sided")
    txt = f"Wilcoxon signed-rank on paired columns {a} vs {b}\nn={len(s)}, W = {float(stat):.4g}, p = {float(p):.5g}"

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
def point_biserial(df: pd.DataFrame, value: str, group: str) -> Tuple[str, List[np.ndarray]]:
    g = df[group]
    if pd.api.types.is_numeric_dtype(g):
        uniq = pd.unique(g.dropna())
        if len(uniq) != 2:
            raise ValueError(f"{group} must have exactly two distinct values for point-biserial.")
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

    # Fisher z CI for r
    if len(d) > 3 and np.isfinite(r) and not np.isclose(abs(r), 1.0):
        z = np.arctanh(r)
        se = 1 / np.sqrt(len(d) - 3)
        zcrit = stats.norm.ppf(0.975)
        ci = (np.tanh(z - zcrit * se), np.tanh(z + zcrit * se))
    else:
        ci = (np.nan, np.nan)

    lines = [
        f"Point-biserial correlation between {value} and {group}",
        f"n={len(d)}, r = {float(r):.4g}, p = {float(p):.5g}",
    ]
    if np.all(np.isfinite(ci)):
        lines.append(f"95% CI for r: [{ci[0]:.3g}, {ci[1]:.3g}]")

    # Visuals: bar ± SEM and boxplot
    gorder = ordered_groups(df, group)
    img_bar = bar_with_error_plot(df, value, group, error="sem", gorder=gorder)
    img_box = box_plot(df, value, group)

    return "\n".join(lines), [img_bar, img_box]
