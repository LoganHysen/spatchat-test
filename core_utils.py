# core_utils.py

from typing import List, Dict
import re
import numpy as np
import pandas as pd

# --------------------------
# ID-like heuristics
# --------------------------
_ID_NAME_PAT = re.compile(r"(?:^|[_\W])(id|identifier|index|subject|animal|record|row)(?:[_\W]|$)", re.I)

def is_id_like(df: pd.DataFrame, col: str) -> bool:
    """Heuristic to flag columns that behave like row/subject IDs (avoid as numeric outcomes)."""
    name = str(col).strip().lower()
    if _ID_NAME_PAT.search(name) or name.endswith("_id"):
        return True
    s = df[col].dropna()
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return False
    as_float = s.astype(float)
    if not np.all(np.isfinite(as_float)):
        return False
    frac = np.abs(as_float - np.round(as_float))
    if (frac > 1e-9).mean() > 0.05:
        return False
    u = np.sort(pd.unique(np.round(as_float).astype(int)))
    if len(u) < 5:
        return False
    diffs = np.diff(u)
    if len(diffs) == 0:
        return False
    return (diffs == 1).mean() >= 0.9


def is_integer_like(series: pd.Series) -> bool:
    """Numeric series that are nearly integers (useful to detect encoded categories)."""
    s = series.dropna()
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return False
    frac = np.abs(s.astype(float) - np.round(s.astype(float)))
    return (frac > 1e-9).mean() <= 0.05


# --------------------------
# Schema helpers
# --------------------------
def infer_schema(df: pd.DataFrame) -> Dict:
    """Lightweight schema inference for recommendations and sensible defaults."""
    n, p = df.shape
    numeric_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical: List[str] = []

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            uniq = df[c].nunique(dropna=True)
            if uniq <= max(20, int(0.2 * max(n, 1))):
                categorical.append(c)

    cat_thresh = min(5, max(2, int(0.1 * max(n, 1))))
    for c in numeric_all:
        uniq = df[c].nunique(dropna=True)
        if is_integer_like(df[c]) and uniq <= cat_thresh:
            categorical.append(c)

    for c in ["group", "sex", "treatment", "class", "category"]:
        if c in df.columns and c not in categorical:
            categorical.append(c)

    binary = []
    for c in df.columns:
        u = df[c].nunique(dropna=True)
        if u == 2 and (not pd.api.types.is_numeric_dtype(df[c]) or is_integer_like(df[c])):
            binary.append(c)

    return {
        "n_rows": n,
        "n_cols": p,
        "numeric_all": numeric_all,
        "categorical": list(dict.fromkeys(categorical)),
        "binary": binary,
    }


def usable_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Numeric columns excluding ID-like fields; stable priority ordering."""
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    safe = [c for c in nums if not is_id_like(df, c)]
    priority = ["score", "height", "weight", "age"]
    ordered = [c for c in priority if c in safe]
    ordered += [c for c in safe if c not in ordered]
    if not ordered and nums:
        ordered = nums
    return ordered


# --------------------------
# Group ordering
# --------------------------
def ordered_groups(df: pd.DataFrame, group_col: str) -> List[str]:
    """
    Deterministic, human-friendly group ordering:
    - If categorical dtype, preserve category order.
    - If levels are numeric-like, sort numerically.
    - Else, case-insensitive alphabetical.
    """
    s = df[group_col]
    if pd.api.types.is_categorical_dtype(s):
        return [str(x) for x in s.cat.categories]
    vals = [str(v) for v in s.dropna().unique().tolist()]
    try:
        vals_float = [float(v) for v in vals]
        order = [x for _, x in sorted(zip(vals_float, vals), key=lambda t: t[0])]
        return order
    except Exception:
        return sorted(vals, key=lambda x: (str(x).lower(), str(x)))
