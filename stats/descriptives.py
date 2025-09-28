# stats/descriptives.py
from typing import Optional, List
import numpy as np
import pandas as pd
from core_utils import usable_numeric_cols, is_integer_like  # already in your repo

def _summ_one(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) == 0:
        return "n=0"
    return f"n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, min={s.min():.4g}, max={s.max():.4g}"

def quick_summary(df: pd.DataFrame, col: str, by: Optional[str] = None) -> str:
    """
    Flexible summary:
      - If col == '*' or col in {'data','dataset','everything'}: summarize ALL numeric columns
        (optionally by a grouping column).
      - If a specific column is given:
          * numeric -> numeric summary (optionally by group)
          * non-numeric -> show top categories (optionally by group)
    Never raises on type conversion; falls back gracefully.
    """
    special_all = { "*", "data", "dataset", "everything" }

    # Grouping column sanity
    if by is not None:
        if by not in df.columns:
            return f"Group column '{by}' not found."
        if df[by].nunique(dropna=True) < 2:
            return f"Group column '{by}' has <2 levels."

    # ----- ALL NUMERIC COLUMNS PATH -----
    if str(col).strip().lower() in special_all:
        nums = usable_numeric_cols(df)
        if not nums:
            return "No numeric columns found to summarize."
        lines: List[str] = []
        if by is None:
            lines.append(f"Summary of all numeric columns (no grouping):")
            for c in nums:
                lines.append(f"- {c}: {_summ_one(df[c])}")
        else:
            lines.append(f"Summary of all numeric columns by {by}:")
            order = []
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

    # ----- SINGLE COLUMN PATH -----
    if col not in df.columns:
        return f"Column '{col}' not found."

    series = df[col]
    # Numeric-like?
    is_numeric = pd.api.types.is_numeric_dtype(series) or is_integer_like(series)

    if by is None:
        if is_numeric:
            return f"Summary of {col}: {_summ_one(series)}"
        # non-numeric overall: show top categories
        vc = series.dropna().astype(str).value_counts().head(10)
        if vc.empty:
            return f"Summary of {col}: n=0"
        return f"Top categories of {col} (overall):\n" + "\n".join([f"- {k}: {int(v)}" for k, v in vc.items()])

    # With grouping
    order = []
    sby = df[by]
    if pd.api.types.is_categorical_dtype(sby):
        order = [str(x) for x in sby.cat.categories]
    else:
        order = [str(v) for v in df[by].dropna().unique().tolist()]
        try:
            order = [x for _, x in sorted(zip([float(v) for v in order], order), key=lambda t: t[0])]
        except Exception:
            order = sorted(order, key=lambda x: (x.lower(), x))

    if is_numeric:
        lines = [f"Summary of {col} by {by}:"]
        for g in order:
            s = df[df[by].astype(str) == g][col]
            lines.append(f"- {g}: {_summ_one(s)}")
        return "\n".join(lines)

    # non-numeric by group: top categories per group
    lines = [f"Top categories of {col} by {by}:"]
    for g in order:
        vc = (
            df[df[by].astype(str) == g][col]
            .dropna().astype(str).value_counts().head(10)
        )
        if vc.empty:
            lines.append(f"- {g}: n=0")
        else:
            lines.append(f"- {g}: " + ", ".join([f"{k} ({int(v)})" for k, v in vc.items()]))
    return "\n".join(lines)
