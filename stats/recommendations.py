# stats/recommendations.py
from __future__ import annotations

from typing import List, Tuple, Optional
import re
import numpy as np
import pandas as pd


# ---------- Numeric detection & coercion ----------

_NUMERIC_NAME_HINT = re.compile(
    r"(?:^|[_\W])(lat|lon|long|alt|elev|elevation|dist|distance|speed|x|y)(?:[_\W]|$)",
    re.IGNORECASE,
)

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric; non-convertibles -> NaN."""
    return pd.to_numeric(s, errors="coerce")


def _is_effectively_numeric(s: pd.Series, colname: str) -> bool:
    """
    Treat a column as numeric if, after coercion:
      - there are at least 3 numeric values, AND
      - numeric values are at least 10% of the non-null entries
    Also: if the name looks numeric-ish (lat/lon/alt/dist/speed/x/y) and there are ≥3
    numeric values, accept it regardless of the 10% rule.
    """
    s_orig_nonnull = s.dropna()
    if s_orig_nonnull.empty:
        return False

    coerced = _coerce_numeric_series(s_orig_nonnull)
    n_numeric = int(coerced.notna().sum())
    frac_numeric = n_numeric / max(1, len(s_orig_nonnull))

    name_hint = bool(_NUMERIC_NAME_HINT.search(str(colname)))
    if name_hint and n_numeric >= 3:
        return True

    return (n_numeric >= 3) and (frac_numeric >= 0.10)


def _split_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in df.columns:
        if _is_effectively_numeric(df[c], c):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


# ---------- Rendering helpers ----------

def _render_numeric_block(
    df: pd.DataFrame, cols: List[str], group_label: Optional[str] = None
) -> List[str]:
    lines: List[str] = []
    if cols:
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
            mean = float(np.nanmean(vals))
            sd = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
            med = float(np.nanmedian(vals))
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            stat = f"n={n}, mean={mean:.3g}, sd={sd:.3g}, median={med:.3g}, min={vmin:.3g}, max={vmax:.3g}"
        bullet = "  • " if group_label is not None else "  • "
        lines.append(f"{bullet}{c}: {stat}")
    return lines


def _render_categorical_block(
    df: pd.DataFrame, cols: List[str], group_label: Optional[str] = None, top_k: int = 5
) -> List[str]:
    lines: List[str] = []
    if cols:
        if group_label is None:
            lines.append("- Categorical:")
        else:
            lines.append(f"- {group_label} (categorical):")
    for c in cols:
        s = df[c].astype(str)
        s = s.replace({"nan": np.nan})
        n = int(s.notna().sum())
        vc = s.value_counts(dropna=True)
        uniq = int(vc.shape[0])
        head = vc.head(min(top_k, uniq))
        if head.empty:
            top_str = "—"
        else:
            top_str = ", ".join([f"{k} ({int(v)})" for k, v in head.items()])
        lines.append(f"  • {c}: n={n}, unique={uniq}{', top=' if uniq else ''}{top_str if uniq else ''}")
    return lines


# ---------- Public API ----------

def recommend_text_and_examples(df: pd.DataFrame) -> Tuple[str, str]:
    num_cols, cat_cols = _split_numeric_categorical(df)
    tips = []
    if num_cols and cat_cols:
        tips.append("Compare numeric outcomes across groups (t-test/ANOVA/Kruskal; Tukey/Dunn for post-hoc).")
    if num_cols:
        tips.append("Explore correlations (Pearson/Spearman) or model with OLS/GLM.")
        tips.append("Plot histogram/box/violin/bar; check normality.")
    if cat_cols:
        tips.append("Test association between categorical variables with Chi-square or Fisher's exact.")
    if not tips:
        tips.append("Upload a dataset or ask for help choosing an analysis.")
    tech = "- " + "\n- ".join(tips)

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
    Summarize the dataset or a column.
      • Numeric: n, mean, sd, median, min, max
      • Categorical: n, unique, top values (with counts)
    If `by` is provided, summaries are computed per group.
    """
    # Resolve "all columns" mode
    summarize_all = (col is None) or (str(col).strip().lower() in {"data", "dataset", "all", "*"})
    if summarize_all:
        num_cols, cat_cols = _split_numeric_categorical(df)
    else:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if _is_effectively_numeric(df[col], col):
            num_cols, cat_cols = [col], []
        else:
            num_cols, cat_cols = [], [col]

    lines: List[str] = []
    # Ungrouped
    if by is None:
        title = f"Dataset summary ({'all columns' if summarize_all else col}):"
        lines.append(title)
        lines.extend(_render_numeric_block(df, num_cols, group_label=None))
        lines.extend(_render_categorical_block(df, cat_cols, group_label=None, top_k=top_k))
        return "\n".join(lines)

    # Grouped
    if by not in df.columns:
        raise ValueError(f"Group column '{by}' not found.")

    title = f"Dataset summary by {by}:"
    lines.append(title)
    for level, gdf in df.groupby(by, dropna=False):
        group_label = f"{by} = {level}"
        lines.extend(_render_numeric_block(gdf, num_cols, group_label=group_label))
        lines.extend(_render_categorical_block(gdf, cat_cols, group_label=group_label, top_k=top_k))
    return "\n".join(lines)