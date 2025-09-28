# stats/descriptives.py
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Local helpers
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

def _resolve_col_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    """Return the actual column name in df that matches `name` (case-insensitive), else None."""
    if name in df.columns:
        return name
    lname = str(name).strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == lname:
            return c
    return None

def _is_numericish(series: pd.Series, frac_threshold: float = 0.95) -> bool:
    """
    Treat a column as numeric if:
      - it's already a numeric dtype, OR
      - coercing to numeric yields >= frac_threshold finite values among non-null.
    """
    if pd.api.types.is_numeric_dtype(series):
        return True
    s = pd.to_numeric(series.dropna().astype(str), errors="coerce")
    if s.empty:
        return False
    good = np.isfinite(s).mean()
    return bool(good >= frac_threshold)

def _summ_numeric(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) == 0:
        return "n=0"
    return (
        f"n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, "
        f"median={np.median(s):.4g}, min={s.min():.4g}, max={s.max():.4g}"
    )

def _summ_categorical(series: pd.Series, top_n: int = 5) -> str:
    # Keep original values (strings), but count after dropping NaN
    s = series.dropna()
    if s.empty:
        return "n=0"
    vc = s.astype(str).value_counts()
    head = vc.head(top_n)
    parts = [f"{k} ({int(v)})" for k, v in head.items()]
    uniq = s.nunique(dropna=True)
    if uniq <= top_n:
        # Show all unique with counts
        return f"n={len(s)}, unique={uniq}, top=" + ", ".join(parts)
    else:
        return f"n={len(s)}, unique={uniq}, top=" + ", ".join(parts)

def _group_order(df: pd.DataFrame, by: str) -> List[str]:
    sby = df[by]
    if pd.api.types.is_categorical_dtype(sby):
        return [str(x) for x in sby.cat.categories]
    # Preserve first-seen order, then stable sort by lowercased string for determinism
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
    Summarize numeric columns with mean/sd/median/min/max.
    Summarize non-numeric columns with unique count + top values.

    If `col` ∈ {'*','data','dataset','everything'} → summarize ALL columns.
    Optional `by` groups the summary and prints both numeric and categorical blocks in each group.
    """
    special_all = {"*", "data", "dataset", "everything"}

    # Resolve grouping column
    by_resolved = None
    if by is not None:
        by_resolved = _resolve_col_case_insensitive(df, by)
        if by_resolved is None:
            print(f"[DEBUG] Group column '{by}' not found among {list(df.columns)}")
            return f"Group column '{by}' not found."
        if df[by_resolved].nunique(dropna=True) < 2:
            print(f"[DEBUG] Group column '{by_resolved}' has <2 levels")
            return f"Group column '{by_resolved}' has <2 levels."

    # Columns to summarize
    if str(col).strip().lower() in special_all:
        cols = [c for c in df.columns if c != by_resolved]
    else:
        c = _resolve_col_case_insensitive(df, col)
        if c is None:
            print(f"[DEBUG] Column '{col}' not found among {list(df.columns)}")
            return f"Column '{col}' not found."
        cols = [c]

    numeric_cols = [c for c in cols if _is_numericish(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]

    # Debug output
    print(f"[DEBUG] quick_summary called with col={col}, by={by}")
    print(f"[DEBUG] Columns to summarize: {cols}")
    print(f"[DEBUG] Numeric columns detected: {numeric_cols}")
    print(f"[DEBUG] Categorical columns detected: {categorical_cols}")

    lines: List[str] = []

    if by_resolved is None:
        lines.append("Dataset summary (no grouping):")
        if numeric_cols:
            lines.append("- Numeric:")
            for c in numeric_cols:
                lines.append(f"  • {c}: {_summ_numeric(df[c])}")
        if categorical_cols:
            lines.append("- Categorical:")
            for c in categorical_cols:
                lines.append(f"  • {c}: {_summ_categorical(df[c])}")
    else:
        lines.append(f"Dataset summary by {by_resolved}:")
        group_levels = _group_order(df, by_resolved)
        print(f"[DEBUG] Group levels detected: {group_levels}")
        for g in group_levels:
            subset = df[df[by_resolved].astype(str) == g]
            if numeric_cols:
                lines.append(f"- {by_resolved} = {g} (numeric):")
                for c in numeric_cols:
                    lines.append(f"  • {c}: {_summ_numeric(subset[c])}")
            if categorical_cols:
                lines.append(f"- {by_resolved} = {g} (categorical):")
                for c in categorical_cols:
                    lines.append(f"  • {c}: {_summ_categorical(subset[c])}")

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