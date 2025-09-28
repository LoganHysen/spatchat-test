# stats/descriptives.py
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

from core_utils import usable_numeric_cols, is_integer_like
from plot_helpers import (
    fig_to_np,
    box_plot as _box_plot_primitive,
    violin_plot as _violin_plot_primitive,
)

__all__ = [
    "quick_summary",
    "plot_hist",
    "plot_box",
    "plot_violin",
    "check_normality",
]

# ----------------------------
# Internal helpers
# ----------------------------
def _summ_numeric(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) == 0:
        return "n=0"
    return f"n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, min={s.min():.4g}, max={s.max():.4g}"

def _summ_categorical(series: pd.Series, top_n: int = 5) -> str:
    vc = series.dropna().astype(str).value_counts()
    if vc.empty:
        return "n=0"
    head = vc.head(top_n)
    return f"n={len(series)}, unique={series.nunique(dropna=True)}, top={', '.join([f'{k} ({v})' for k, v in head.items()])}"

def _group_order(df: pd.DataFrame, by: str) -> List[str]:
    sby = df[by]
    if pd.api.types.is_categorical_dtype(sby):
        return [str(x) for x in sby.cat.categories]
    seen: List[str] = []
    for v in sby.dropna():
        sv = str(v)
        if sv not in seen:
            seen.append(sv)
    return sorted(seen, key=lambda x: (x.lower(), x))

# ----------------------------
# Main public API
# ----------------------------
def quick_summary(df: pd.DataFrame, col: str, by: Optional[str] = None) -> str:
    """
    Summarize numeric columns with mean/sd/min/max.
    Summarize non-numeric columns with unique count + top values.
    If `col` is '*', 'data', 'dataset', 'everything': summarize ALL columns.
    If a single `col` is provided: summarize only that column.
    Supports optional grouping via `by`.
    """
    special_all = {"*", "data", "dataset", "everything"}
    lines: List[str] = []

    # ----------------------------
    # Determine columns to summarize
    # ----------------------------
    if str(col).strip().lower() in special_all:
        cols = list(df.columns)
    else:
        if col not in df.columns:
            return f"Column '{col}' not found."
        cols = [col]

    # Grouping order
    group_order: List[str] = []
    if by is not None:
        if by not in df.columns:
            return f"Group column '{by}' not found."
        if df[by].nunique(dropna=True) < 2:
            return f"Group column '{by}' has <2 levels."
        group_order = _group_order(df, by)

    # ----------------------------
    # Build summary
    # ----------------------------
    if by is None:
        lines.append("Dataset summary (no grouping):")
        for c in cols:
            series = df[c]
            if pd.api.types.is_numeric_dtype(series) or is_integer_like(series):
                lines.append(f"- {c}: {_summ_numeric(series)}")
            else:
                lines.append(f"- {c}: {_summ_categorical(series)}")
    else:
        lines.append(f"Dataset summary by {by}:")
        for g in group_order:
            subset = df[df[by].astype(str) == g]
            lines.append(f"- {by} = {g}:")
            for c in cols:
                series = subset[c]
                if pd.api.types.is_numeric_dtype(series) or is_integer_like(series):
                    lines.append(f"  • {c}: {_summ_numeric(series)}")
                else:
                    lines.append(f"  • {c}: {_summ_categorical(series)}")

    return "\n".join(lines)

# ----------------------------
# Plotting helpers
# ----------------------------
def plot_hist(df: pd.DataFrame, col: str, bins: int = 30) -> Tuple[str, np.ndarray]:
    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(s, bins=bins, ax=ax)
    ax.set_title(f"Histogram of {col}")
    fig.tight_layout()
    return f"Histogram {col}", fig_to_np(fig)

def plot_box(df: pd.DataFrame, value: str, group: str) -> Tuple[str, np.ndarray]:
    img = _box_plot_primitive(df, value, group)
    return f"Boxplot {value}~{group}", img

def plot_violin(df: pd.DataFrame, value: str, group: str) -> Tuple[str, np.ndarray]:
    img = _violin_plot_primitive(df, value, group)
    return f"Violin {value}~{group}", img

def check_normality(df: pd.DataFrame, col: str) -> Tuple[str, np.ndarray]:
    series = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
    if 3 <= len(series) <= 5000:
        W, p = stats.shapiro(series)
        msg = f"Shapiro–Wilk normality on {col} (n={len(series)}): W={W:.4g}, p={p:.5g}"
    else:
        msg = f"Shapiro–Wilk skipped for {col} (requires 3–5000 values; n={len(series)})."
    sm.qqplot(series, line="45", fit=True)
    return msg, fig_to_np(plt.gcf())