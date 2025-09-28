# stats/recommendations.py
from __future__ import annotations

from typing import List, Tuple, Optional
import re
import numpy as np
import pandas as pd

__all__ = ["recommend_text_and_examples", "quick_summary"]

# ------------------------------------------------------------
# Numeric detection / coercion
# ------------------------------------------------------------

# Helpful name hints: treat these as numeric if they have at least a few numeric values
_NUMERIC_NAME_HINT = re.compile(
    r"(?:^|[_\W])(lat|latitude|lon|long|longitude|alt|altitude|elev|elevation|dist|distance|speed|depth|height|weight|x|y)(?:[_\W]|$)",
    re.IGNORECASE,
)

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Coerce a series to numeric safely; non-convertibles become NaN.
    Does not raise on strings like 'F24'—those simply become NaN.
    """
    return pd.to_numeric(s, errors="coerce")


def _is_numeric_by_dtype(s: pd.Series) -> bool:
    # Treat bool as categorical, not numeric, for summaries by group
    return (pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)) and not pd.api.types.is_bool_dtype(s)


def _is_effectively_numeric(s: pd.Series, colname: str) -> bool:
    """
    Decide if a column should be summarized as numeric.

    Rules (global, not per-group):
      - If dtype is numeric (int/float), accept.
      - Else coerce; consider numeric if:
          * there are at least 3 numeric values, AND
          * numeric values are >= 10% of non-null entries, OR
          * column name hints numeric AND numeric values >= 3.
    """
    if _is_numeric_by_dtype(s):
        return True

    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return False

    coerced = _coerce_numeric_series(s_nonnull)
    n_numeric = int(coerced.notna().sum())
    frac_numeric = n_numeric / max(1, len(s_nonnull))
    name_hint = bool(_NUMERIC_NAME_HINT.search(str(colname)))

    if name_hint and n_numeric >= 3:
        return True
    return (n_numeric >= 3) and (frac_numeric >= 0.10)


def _split_numeric_categorical_global(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Classify columns once on the whole dataset (not per group).
    This avoids losing columns in small groups.
    """
    num_cols, cat_cols = [], []
    for c in df.columns:
        s = df[c]
        if _is_effectively_numeric(s, c):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


# ------------------------------------------------------------
# Render helpers
# ------------------------------------------------------------

def _fmt_float(x: float) -> str:
    # Compact numeric format for readability
    try:
        return f"{float(x):.3g}"
    except Exception:
        return "nan"


def _render_numeric_block(df: pd.DataFrame, cols: List[str], group_label: Optional[str]) -> List[str]:
    """
    Render numeric summaries for given columns within df (which may be a group slice).
    Always lists every numeric column; if a group has 0 numeric entries for a column,
    we still show 'n=0'.
    """
    lines: List[str] = []
    if not cols:
        return lines
    if group_label is None:
        lines.append("- Numeric:")
    else:
        lines.append(f"- {group_label} (numeric):")

    for c in cols:
        vals = _coerce_numeric_series(df[c]).dropna()
        if vals.empty:
            stat = "n=0"
        else:
            n = int(vals.shape[0])
            mean = _fmt_float(np.nanmean(vals))
            sd = _fmt_float(np.nanstd(vals, ddof=1)) if n > 1 else "0"
            med = _fmt_float(np.nanmedian(vals))
            vmin = _fmt_float(np.nanmin(vals))
            vmax = _fmt_float(np.nanmax(vals))
            stat = f"n={n}, mean={mean}, sd={sd}, median={med}, min={vmin}, max={vmax}"
        lines.append(f"  • {c}: {stat}")
    return lines


def _render_categorical_block(df: pd.DataFrame, cols: List[str], group_label: Optional[str], top_k: int) -> List[str]:
    """
    Render categorical summaries. Always reports n and unique for each column.
    Shows up to top_k most frequent values with counts; if all unique, the 'top' list is omitted.
    """
    lines: List[str] = []
    if not cols:
        return lines
    if group_label is None:
        lines.append("- Categorical:")
    else:
        lines.append(f"- {group_label} (categorical):")

    for c in cols:
        s = df[c]
        # Keep original NaNs for counts; convert to string just for value listing
        n_non_null = int(s.notna().sum())
        vc = s.astype(str).replace({"nan": np.nan}).value_counts(dropna=True)
        uniq = int(vc.shape[0])

        if uniq == 0:
            lines.append(f"  • {c}: n=0, unique=0")
            continue

        # Build top list if there is any repetition; for all-unique, skip top list
        head = vc.head(min(top_k, uniq))
        repeated_exist = any(v > 1 for v in head.values)
        if repeated_exist:
            top_str = ", ".join([f"{k} ({int(v)})" for k, v in head.items()])
            lines.append(f"  • {c}: n={n_non_null}, unique={uniq}, top={top_str}")
        else:
            lines.append(f"  • {c}: n={n_non_null}, unique={uniq}")
    return lines


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def recommend_text_and_examples(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Lightweight guidance + examples based on observed schema.
    """
    num_cols, cat_cols = _split_numeric_categorical_global(df)
    bullets = []
    if num_cols and cat_cols:
        bullets.append("Compare numeric outcomes across groups (t-test / ANOVA / Kruskal; Tukey / Dunn for post-hoc).")
    if num_cols:
        bullets.append("Explore correlations (Pearson/Spearman) or model with OLS/GLM.")
        bullets.append("Plot histogram/box/violin/bar; check normality with a QQ plot.")
    if cat_cols:
        bullets.append("Test association between categorical variables (Chi-square / Fisher's exact).")
    if not bullets:
        bullets.append("Upload a dataset or ask for help choosing an analysis.")

    tech = "- " + "\n- ".join(bullets)

    ex = []
    if num_cols and cat_cols:
        ex.append(f"ttest {num_cols[0]} by {cat_cols[0]}")
    if len(cat_cols) >= 2:
        ex.append(f"chi-square {cat_cols[0]} vs {cat_cols[1]}")
    if len(num_cols) >= 2:
        ex.append(f"correlation {num_cols[0]} vs {num_cols[1]} (spearman)")
    if num_cols:
        ex.append(f"histogram {num_cols[0]} (bins=40)")
    examples = "• " + "\n• ".join(ex) if ex else "• what can I do with my data?"
    return tech, examples


def quick_summary(
    df: pd.DataFrame,
    col: Optional[str] = "data",
    by: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Summarize dataset or a column.

    - Numeric columns: n, mean, sd, median, min, max
    - Non-numeric columns: n, unique, top values with counts (up to top_k).
      If all values are unique, 'top' list is omitted but n/unique are still shown.

    If `by` is provided, summaries are emitted per level of `by`, using a global
    column classification so groups don't "lose" columns.
    """
    # Figure out whether we’re summarizing a single column or the entire dataset
    summarize_all = (col is None) or (str(col).strip().lower() in {"data", "dataset", "all", "*"})
    if summarize_all:
        numeric_cols, categorical_cols = _split_numeric_categorical_global(df)
    else:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        # Classify single column
        if _is_effectively_numeric(df[col], col):
            numeric_cols, categorical_cols = [col], []
        else:
            numeric_cols, categorical_cols = [], [col]

    # Ungrouped summary
    if by is None:
        title = f"Dataset summary ({'all columns' if summarize_all else col}):"
        parts: List[str] = [title]
        parts.extend(_render_numeric_block(df, numeric_cols, group_label=None))
        parts.extend(_render_categorical_block(df, categorical_cols, group_label=None, top_k=top_k))
        return "\n".join(parts)

    # Grouped summary
    if by not in df.columns:
        raise ValueError(f"Group column '{by}' not found.")

    title = f"Dataset summary by {by}:"
    parts: List[str] = [title]
    # groupby with dropna=False so NaN group (if any) is visible
    for level, gdf in df.groupby(by, dropna=False):
        group_label = f"{by} = {level}"
        parts.extend(_render_numeric_block(gdf, numeric_cols, group_label=group_label))
        parts.extend(_render_categorical_block(gdf, categorical_cols, group_label=group_label, top_k=top_k))
    return "\n".join(parts)