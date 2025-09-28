# stats/correlations.py
# Correlations & associations: pairwise correlation, correlation matrix, partial correlation
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

from plot_helpers import scatter_with_reg, fig_to_np

# --- helpers (local) ---
def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher z CI for Pearson r."""
    if n <= 3 or not np.isfinite(r) or np.isclose(abs(r), 1.0):
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return lo, hi


def corr_pair(df: pd.DataFrame, x: str, y: str, method: str = "pearson") -> Tuple[str, List[np.ndarray]]:
    s = df[[x, y]].dropna()
    if len(s) < 2:
        raise ValueError("Not enough paired data for correlation.")

    method = (method or "pearson").lower()
    if method == "spearman":
        r, p = stats.spearmanr(s[x], s[y])
        ci = (np.nan, np.nan)
        title = "Spearman scatter"
    else:
        r, p = stats.pearsonr(s[x], s[y])
        ci = _pearson_ci(r, len(s))
        title = "Pearson scatter"

    lines = [f"{method.title()} correlation between {x} and {y}",
             f"n={len(s)}, r = {float(r):.4g}, p = {float(p):.5g}"]
    if np.all(np.isfinite(ci)):
        lines.append(f"95% CI for r: [{ci[0]:.3g}, {ci[1]:.3g}]")

    img = scatter_with_reg(s, x=x, y=y, ci=0.95, title=title)
    return "\n".join(lines), [img]


def corr_matrix_plot(
    df: pd.DataFrame, cols: Optional[List[str]] = None, method: str = "pearson"
) -> Tuple[str, List[np.ndarray]]:
    # choose numeric columns if none provided
    if cols is None or len(cols) < 2:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        raise ValueError("Need at least two numeric columns for a correlation matrix.")

    method = (method or "pearson").lower()
    mat = df[cols].corr(method=method)
    mat.to_csv("outputs/corr_matrix_{}.csv".format(method), index=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title(f"{method.title()} correlation matrix")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    img = fig_to_np(fig)

    txt = f"{method.title()} correlation matrix on: {', '.join(cols)}\n\n{mat.round(3).to_string()}"
    return txt, [img]


def partial_corr(
    df: pd.DataFrame, x: str, y: str, controls: List[str], method: str = "pearson"
) -> Tuple[str, List[np.ndarray]]:
    cols = [x, y] + list(controls or [])
    d = df[cols].dropna()
    if len(d) < (len(controls) + 3):
        raise ValueError("Not enough rows for partial correlation (need > controls + 2).")

    method = (method or "pearson").lower()
    if method == "spearman":
        for c in cols:
            d[c] = d[c].rank(method="average")

    Xc = sm.add_constant(d[controls]) if controls else sm.add_constant(pd.DataFrame(index=d.index))
    rx = sm.OLS(d[x].astype(float), Xc).fit().resid
    ry = sm.OLS(d[y].astype(float), Xc).fit().resid
    r, p = stats.pearsonr(rx, ry)
    n = len(d)
    k = len(controls)
    ci = _pearson_ci(r, n - k)  # use effective df

    lines = [
        f"Partial correlation ({method.title()}) between {x} and {y} controlling for {', '.join(controls)}" if controls
        else f"Partial correlation ({method.title()}) between {x} and {y} (no controls)",
        f"n={n}, k={k} controls, r = {r:.4g}, p = {p:.5g}",
    ]
    if np.all(np.isfinite(ci)):
        lines.append(f"95% CI for r: [{ci[0]:.3g}, {ci[1]:.3g}]")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(rx, ry)
    m, b = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 100)
    ax.plot(xs, m * xs + b)
    ax.set_xlabel(f"{x} residuals")
    ax.set_ylabel(f"{y} residuals")
    ax.set_title("Partial correlation (residuals)")
    img = fig_to_np(fig)

    return "\n".join(lines), [img]
