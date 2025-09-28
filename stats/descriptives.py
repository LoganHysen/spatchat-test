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
    fig_to_np,           # image rendering helper
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

# ---------- internal util ----------
def _summ_one(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) == 0:
        return "n=0"
    return f"n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, min={s.min():.4g}, max={s.max():.4g}"

# ---------- summaries ----------
def quick_summary(df: pd.DataFrame, col: str, by: Optional[str] = None) -> str:
    """
    Flexible summary:
      - If col in {'*','data','dataset','everything'}: summarize ALL numeric columns
        (optionally grouped by 'by').
      - If specific column:
          * numeric-like -> numeric summary (optionally by group)
          * non-numeric   -> top categories (optionally by group)
    Graceful coercion; no exceptions on type conversion.
    """
    special_all = {"*", "data", "dataset", "everything"}

    # Validate grouping, if provided
    if by is not None:
        if by not in df.columns:
            return f"Group column '{by}' not found."
        if df[by].nunique(dropna=True) < 2:
            return f"Group column '{by}' has <2 levels."

    # ----- ALL NUMERIC COLUMNS -----
    if str(col).strip().lower() in special_all:
        nums = usable_numeric_cols(df)
        if not nums:
            return "No numeric columns found to summarize."
        lines: List[str] = []
        if by is None:
            lines.append("Summary of all numeric columns (no grouping):")
            for c in nums:
                lines.append(f"- {c}: {_summ_one(df[c])}")
        else:
            lines.append(f"Summary of all numeric columns by {by}:")
            sby = df[by]
            if pd.api.types.is_categorical_dtype(sby):
                order = [str(x) for x in sby.cat.categories]
            else:
                order = [str(v) for v in df[by].dropna().unique().tolist()]
                try:
                    order = [x for _, x in sorted(zip([float(v) for v in order], order), key=lambda t: t[0])]
                except Exception:
                    order = sorted(order, key=lambda x: (x.lower(), x))
            for g in order:
                lines.append(f"- {by} = {g}:")
                block = df[df[by].astype(str) == g]
                for c in nums:
                    lines.append(f"  • {c}: {_summ_one(block[c])}")
        return "\n".join(lines)

    # ----- SINGLE COLUMN -----
    if col not in df.columns:
        return f"Column '{col}' not found."

    series = df[col]
    is_numlike = pd.api.types.is_numeric_dtype(series) or is_integer_like(series)

    if by is None:
        if is_numlike:
            return f"Summary of {col}: {_summ_one(series)}"
        vc = series.dropna().astype(str).value_counts().head(10)
        if vc.empty:
            return f"Summary of {col}: n=0"
        return f"Top categories of {col} (overall):\n" + "\n".join([f"- {k}: {int(v)}" for k, v in vc.items()])

    # With grouping
    sby = df[by]
    if pd.api.types.is_categorical_dtype(sby):
        order = [str(x) for x in sby.cat.categories]
    else:
        order = [str(v) for v in df[by].dropna().unique().tolist()]
        try:
            order = [x for _, x in sorted(zip([float(v) for v in order], order), key=lambda t: t[0])]
        except Exception:
            order = sorted(order, key=lambda x: (x.lower(), x))

    if is_numlike:
        lines = [f"Summary of {col} by {by}:"]
        for g in order:
            s = df[df[by].astype(str) == g][col]
            lines.append(f"- {g}: {_summ_one(s)}")
        return "\n".join(lines)

    lines = [f"Top categories of {col} by {by}:"]
    for g in order:
        vc = df[df[by].astype(str) == g][col].dropna().astype(str).value_counts().head(10)
        if vc.empty:
            lines.append(f"- {g}: n=0")
        else:
            lines.append(f"- {g}: " + ", ".join([f"{k} ({int(v)})" for k, v in vc.items()]))
    return "\n".join(lines)

# ---------- plotting helpers expected by stats.__init__ ----------
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
    # Shapiro only reliable for 3–5000
    if 3 <= len(series) <= 5000:
        W, p = stats.shapiro(series)
        msg = f"Shapiro–Wilk normality on {col} (n={len(series)}): W={W:.4g}, p={p:.5g}"
    else:
        msg = f"Shapiro–Wilk skipped for {col} (requires 3–5000 values; n={len(series)})."
    sm.qqplot(series, line="45", fit=True)
    return msg, fig_to_np(plt.gcf())
