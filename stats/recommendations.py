# stats/recommendations.py
from typing import List, Tuple, Optional, Dict
import re
import numpy as np
import pandas as pd

from core_utils import usable_numeric_cols, is_integer_like

__all__ = ["recommend_text_and_examples"]


# ----------------------------
# Lightweight schema helpers
# ----------------------------
_ID_NAME_PAT = re.compile(r"(?:^|[_\W])(id|identifier|index|subject|animal|record|row)(?:[_\W]|$)", re.I)

def _is_id_like(df: pd.DataFrame, col: str) -> bool:
    name = str(col).strip().lower()
    if _ID_NAME_PAT.search(name) or name.endswith("_id"):
        return True
    s = df[col].dropna()
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return False
    as_float = s.astype(float)
    if not np.all(np.isfinite(as_float)):
        return False
    # mostly integers and mostly consecutive → likely an ID
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


def _infer_schema(df: pd.DataFrame) -> Dict:
    n, p = df.shape
    numeric_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # categorical if non-numeric and low cardinality
    categorical: List[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            uniq = df[c].nunique(dropna=True)
            if uniq <= max(20, int(0.2 * max(n, 1))):
                categorical.append(c)

    # include integer-like low-card numeric as categorical (e.g., 0/1; 1..5 Likert)
    cat_thresh = min(5, max(2, int(0.1 * max(n, 1))))
    for c in numeric_all:
        uniq = df[c].nunique(dropna=True)
        if is_integer_like(df[c]) and uniq <= cat_thresh:
            categorical.append(c)

    # common group-like names
    for c in ["group", "sex", "treatment", "class", "category"]:
        if c in df.columns and c not in categorical:
            categorical.append(c)

    # binary detection
    binary = []
    for c in df.columns:
        u = df[c].nunique(dropna=True)
        if u == 2 and (not pd.api.types.is_numeric_dtype(df[c]) or is_integer_like(df[c])):
            binary.append(c)

    # drop obvious IDs from “numeric candidates”
    numeric_safe = [c for c in numeric_all if not _is_id_like(df, c)]
    if not numeric_safe and numeric_all:
        numeric_safe = numeric_all

    return {
        "n_rows": n,
        "n_cols": p,
        "numeric_all": numeric_all,
        "numeric_safe": numeric_safe,
        "categorical": list(dict.fromkeys(categorical)),  # preserve order, dedupe
        "binary": binary,
    }


# ----------------------------
# Public API
# ----------------------------
def recommend_text_and_examples(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Returns:
      tech_text: human-readable overview + suggested analyses
      examples_text: short “say it like this” prompts for the chat
    """
    info = _infer_schema(df)
    n, p = info["n_rows"], info["n_cols"]
    nums_all = info["numeric_all"]
    nums_safe = info["numeric_safe"]
    cats = info["categorical"]
    bins = info["binary"]

    # pick a primary numeric outcome
    y_num: Optional[str] = nums_safe[0] if nums_safe else (nums_all[0] if nums_all else None)

    def _valid_group(col: str, outcome: Optional[str]) -> bool:
        if outcome and col == outcome:
            return False
        u = df[col].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[col]):
            return u >= 2 and is_integer_like(df[col])
        else:
            return u >= 2

    g_bin = next((c for c in bins if _valid_group(c, y_num)), None)
    g_multi = next((c for c in cats if df[c].nunique(dropna=True) >= 3 and _valid_group(c, y_num)), None)

    # Overview
    overview_lines = [f"Dataset overview: n={n} rows, p={p} columns."]
    if nums_all:
        preview_nums = ", ".join(map(str, nums_all[:8])) + ("…" if len(nums_all) > 8 else "")
        overview_lines.append(f"- Numeric columns ({len(nums_all)}): {preview_nums}")
    if cats:
        preview_cats = ", ".join(map(str, cats[:8])) + ("…" if len(cats) > 8 else "")
        overview_lines.append(f"- Categorical/low-cardinality ({len(cats)}): {preview_cats}")

    # Recommendations
    rec_lines = ["Here are some analysis ideas you can run next:"]
    if y_num:
        rec_lines.append(f"• See a quick summary of **{y_num}** (mean, sd, min/max).")
        rec_lines.append(f"• **Histogram** to see the shape of **{y_num}**.")
    if y_num and g_bin:
        rec_lines.append(f"• Compare average **{y_num}** between the two groups in **{g_bin}** (t-test).")
        rec_lines.append(f"• **Bar chart with error bars** for **{y_num}** by **{g_bin}**.")
        rec_lines.append(f"• **Mann–Whitney U** (rank-sum) for **{y_num}** by **{g_bin}** (nonparametric).")
        rec_lines.append(f"• **Point-biserial** correlation of **{y_num}** with **{g_bin}**.")
        rec_lines.append(f"• **Levene’s test** for equal variances across **{g_bin}**.")
    if y_num and g_multi:
        rec_lines.append(f"• Compare **{y_num}** across levels of **{g_multi}** (one-way ANOVA).")
        rec_lines.append(f"• **Post-hoc Tukey HSD** for pairwise differences.")
        rec_lines.append(f"• **Kruskal–Wallis** across **{g_multi}** (nonparametric).")
        rec_lines.append(f"• **Post-hoc Dunn’s test** with p-adjustment.")
        rec_lines.append(f"• **Box/violin plots** of **{y_num}** across **{g_multi}**.")
    if len(nums_safe) >= 2:
        rec_lines.append(f"• **Correlation**: Pearson or Spearman between **{nums_safe[0]}** and **{nums_safe[1]}**.")
    if len(nums_safe) >= 3:
        rec_lines.append(f"• **Correlation heatmap** for numeric columns.")
        rec_lines.append(f"• **Partial correlation** between two variables controlling for others.")
        rec_lines.append(f"• See how one value predicts another (linear regression), e.g., **{nums_safe[0]} ~ {nums_safe[1]}**.")
    if len(cats) >= 2:
        rec_lines.append("• **Chi-square test** of association between two categorical columns.")
    rec_lines.append("• **Normality check** with a QQ plot.")
    if g_bin:
        rec_lines.append("• **Power analysis** for a t-test to estimate sample size.")
    if g_multi:
        rec_lines.append("• **Power analysis** for one-way ANOVA to estimate sample size.")

    tech_text = "\n".join(overview_lines + [""] + rec_lines)

    # Examples
    ex = ["Here are some things you can say:"]
    if y_num:
        ex.append(f'• "What is the average {y_num}?"')
        ex.append(f'• "Show a histogram of {y_num}."')
    if y_num and g_bin:
        ex.append(f'• "I want to do a t-test on {y_num} by {g_bin}."')
        ex.append(f'• "Mann-Whitney U test on {y_num} by {g_bin}."')
        ex.append(f'• "Point-biserial correlation {y_num} by {g_bin}."')
        ex.append(f'• "Levene\'s test for {y_num} by {g_bin}."')
    if y_num and g_multi:
        ex.append(f'• "Run a one-way ANOVA of {y_num} by {g_multi}."')
        ex.append(f'• "Tukey HSD for {y_num} by {g_multi}."')
        ex.append(f'• "Kruskal-Wallis on {y_num} by {g_multi}."')
        ex.append(f'• "Dunn\'s post-hoc for {y_num} by {g_multi}."')
        ex.append(f'• "Show box and violin plots for {y_num} by {g_multi}."')
    if len(nums_safe) >= 2:
        ex.append(f'• "Pearson correlation between {nums_safe[0]} and {nums_safe[1]}."')
        ex.append(f'• "Spearman correlation {nums_safe[0]} vs {nums_safe[1]}."')
        ex.append('• "Correlation heatmap of numeric columns."')
        ex.append('• "Partial correlation height and score controlling for age, weight."')
        ex.append(f'• "Fit a linear regression: {nums_safe[0]} ~ {nums_safe[1]}."')
    if len(cats) >= 2:
        ex.append('• "Chi-square test of sex by group."')
    ex.append('• "Check normality of the main outcome and show a QQ plot."')
    if g_bin:
        ex.append('• "Power analysis for a t-test with 80% power and effect size 0.5."')
    if g_multi:
        ex.append('• "Power analysis for a one-way ANOVA with 3 groups and 80% power."')

    examples_text = "\n".join(ex)
    return tech_text, examples_text