# stats/recommendations.py
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


# ========== Helper detection/coercion ==========

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Try to coerce a series to numeric while preserving NaNs for non-convertibles.
    """
    return pd.to_numeric(s, errors="coerce")


def _is_effectively_numeric(s: pd.Series) -> bool:
    """
    A column is treated as numeric if, after coercion, at least half of
    the non-null entries are numeric and there is >1 distinct numeric value.
    """
    coerced = _coerce_numeric_series(s)
    nn = coerced.notna()
    if nn.sum() == 0:
        return False
    frac_numeric = nn.mean()
    if frac_numeric < 0.5:
        return False
    return coerced[nn].nunique(dropna=True) > 1


def _split_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in df.columns:
        s = df[c]
        if _is_effectively_numeric(s):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


# ========== Render helpers ==========

def _render_numeric_block(df: pd.DataFrame, cols: List[str], group_label: Optional[str] = None) -> List[str]:
    lines: List[str] = []
    if group_label is not None:
        lines.append(f"- {group_label}:")
    for c in cols:
        s = _coerce_numeric_series(df[c])
        # dropna for stats
        vals = s.dropna()
        if vals.empty:
            stat = "n=0"
        else:
            n = int(vals.shape[0])
            mean = np.nanmean(vals)
            sd = np.nanstd(vals, ddof=1) if n > 1 else 0.0
            med = np.nanmedian(vals)
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            stat = f"n={n}, mean={mean:.3g}, sd={sd:.3g}, median={med:.3g}, min={vmin:.3g}, max={vmax:.3g}"
        bullet_prefix = "  • " if group_label is not None else "- "
        lines.append(f"{bullet_prefix}{c}: {stat}")
    return lines


def _render_categorical_block(df: pd.DataFrame, cols: List[str], group_label: Optional[str] = None, top_k: int = 5) -> List[str]:
    lines: List[str] = []
    if group_label is not None:
        # Add a header line only once for the categorical portion if needed
        pass
    for c in cols:
        s = df[c].astype(str)
        # treat 'nan' string (from astype) as NaN for counts reporting
        s_clean = s.replace({"nan": np.nan})
        n = int(s_clean.notna().sum())
        vc = s_clean.value_counts(dropna=True)
        uniq = int(vc.shape[0])
        top_parts = []
        take = min(top_k, uniq)
        for k, v in vc.head(take).items():
            top_parts.append(f"{k} ({int(v)})")
        top_str = ", ".join(top_parts) if top_parts else "—"
        bullet_prefix = "  • " if group_label is not None else "- "
        lines.append(f"{bullet_prefix}{c}: n={n}, unique={uniq}{', top=' if take else ''}{top_str if take else ''}")
    return lines


# ========== Public API ==========

def recommend_text_and_examples(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Lightweight guidance string + examples based on observed schema.
    """
    num_cols, cat_cols = _split_numeric_categorical(df)
    tips = []
    if num_cols and cat_cols:
        tips.append("You can compare numeric outcomes across groups (t-test/ANOVA/Kruskal; Tukey/Dunn for post-hoc).")
    if num_cols:
        tips.append("Explore correlations (Pearson/Spearman) or model with OLS/GLM.")
        tips.append("Visualize with histogram/box/violin/bar and check normality.")
    if cat_cols:
        tips.append("Test association between two categorical columns with Chi-square/Fisher.")
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


def quick_summary(df: pd.DataFrame, col: Optional[str] = "data", by: Optional[str] = None, top_k: int = 5) -> str:
    """
    Summarize dataset or a single column.
      - If `by` is provided: summarize within each level of `by`.
      - Numeric columns: n, mean, sd, median, min, max.
      - Non-numeric columns: n, unique, top values with counts.
    The `col="data"` (or None) means summarize ALL columns.
    """
    # Normalize flags
    summarize_all = (col is None) or (str(col).strip().lower() in {"data", "dataset", "all", "*"})
    header_name = by if by else (col if not summarize_all else "data")

    # Column sets
    if summarize_all:
        num_cols, cat_cols = _split_numeric_categorical(df)
    else:
        # single column path
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if _is_effectively_numeric(df[col]):
            num_cols, cat_cols = [col], []
        else:
            num_cols, cat_cols = [], [col]

    lines: List[str] = []
    if by is None:
        title = f"Dataset summary ({'all columns' if summarize_all else col}):"
        lines.append(title)
        # Numeric block
        if num_cols:
            lines.append("- Numeric:")
            lines.extend(_render_numeric_block(df, num_cols, group_label=None))
        # Categorical block
        if cat_cols:
            lines.append("- Categorical:")
            lines.extend(_render_categorical_block(df, cat_cols, group_label=None, top_k=top_k))
        return "\n".join(lines)

    # Grouped summary
    if by not in df.columns:
        raise ValueError(f"Group column '{by}' not found.")

    title = f"Dataset summary by {by}:"
    lines.append(title)
    # group with dropna=False to show NaN group if present
    for level, gdf in df.groupby(by, dropna=False):
        group_label = f"{by} = {level}"
        # Numeric within group
        if num_cols:
            lines.extend(_render_numeric_block(gdf, num_cols, group_label=group_label))
        # Categorical within group
        if cat_cols:
            lines.extend(_render_categorical_block(gdf, cat_cols, group_label=group_label, top_k=top_k))
    return "\n".join(lines)