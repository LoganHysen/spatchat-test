# stats/associations.py
# Categorical associations: Chi-square & Fisher’s exact + stacked proportion plot
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from plot_helpers import fig_to_np  # root helper (provided separately)

def chisq_test(df: pd.DataFrame, row: str, col: str, exact: bool = False) -> Tuple[str, List[np.ndarray]]:
    """
    Chi-square test of independence (general r×c) or Fisher's exact for 2×2 if exact=True.
    Returns (text, [image]) where image is a stacked column-proportion bar chart.
    """
    tab = pd.crosstab(df[row], df[col], dropna=True)
    if tab.empty or tab.shape[0] < 1 or tab.shape[1] < 1:
        raise ValueError("Contingency table is empty after dropping NA.")

    n = float(tab.values.sum())
    chi2 = np.nan
    if tab.shape == (2, 2) and exact:
        # Fisher for 2×2
        _, p = stats.fisher_exact(tab.values)
        method = "Fisher's exact (2×2)"
    else:
        chi2, p, dof, _ = stats.chi2_contingency(tab.values)
        method = f"Chi-square test (dof={int(dof)})"

    # Cramér's V (effect size) for r×c, defined when chi2 computed
    r, c = tab.shape
    if r > 1 and c > 1 and np.isfinite(chi2):
        V = np.sqrt(chi2 / (n * (min(r - 1, c - 1))))
    else:
        V = np.nan

    lines = [f"{method} on {row} × {col}", f"Table (n={int(n)}):", str(tab)]
    if np.isfinite(chi2):
        lines.append(f"χ² = {chi2:.4g}, p = {p:.5g}, Cramér's V = {V:.3g}")
    else:
        lines.append(f"p = {p:.5g} (Fisher)")

    # Stacked bar of column-wise proportions
    fig, ax = plt.subplots(figsize=(6, 3))
    cols = list(tab.columns)
    ind = np.arange(len(cols))
    bottom = np.zeros(len(cols), dtype=float)

    for level in tab.index:
        vals = tab.loc[level].values.astype(float)
        col_sums = tab.sum(axis=0).values.astype(float)
        # avoid divide-by-zero: if a column sum is 0, keep 0 proportion
        props = np.divide(vals, np.where(col_sums == 0, 1.0, col_sums), where=col_sums != 0)
        ax.bar(ind, props, bottom=bottom, label=str(level))
        bottom += props

    ax.set_xticks(ind)
    ax.set_xticklabels(cols)
    ax.set_ylabel("Proportion within column")
    ax.set_title(f"{row} proportions within {col}")
    ax.legend(fontsize=8, ncols=min(3, len(tab.index)))
    img = fig_to_np(fig)

    return "\n".join(lines), [img]
