# stats/recommendations.py
# Dataset-aware guidance & quick summaries used by the chat handlers.
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from core_utils import infer_schema, usable_numeric_cols, ordered_groups, is_integer_like

def quick_summary(df: pd.DataFrame, col: str, by: Optional[str] = None) -> str:
    """
    Text summary of a numeric column, optionally stratified by a grouping column.
    Matches the original signature/behavior used by the UI.
    """
    if by and by in df.columns and by != col and df[by].nunique(dropna=True) > 1:
        lines = [f"Summary of {col} by {by}:"]
        gorder = ordered_groups(df, by)
        for g in gorder:
            s = df[df[by].astype(str) == g][col].dropna().astype(float)
            if len(s) == 0:
                lines.append(f"- {g}: n=0")
            else:
                lines.append(
                    f"- {g}: n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, "
                    f"min={s.min():.4g}, max={s.max():.4g}"
                )
        return "\n".join(lines)
    else:
        s = df[col].dropna().astype(float)
        return (
            f"Summary of {col}: n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, "
            f"min={s.min():.4g}, max={s.max():.4g}"
        )


def recommend_text_and_examples(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Generate a short, dataset-aware technical overview and a compact list of
    example prompts, preserving the tone and structure used in the app.
    """
    info = infer_schema(df)
    n, p = info["n_rows"], info["n_cols"]
    nums_all = info["numeric_all"]
    cats = info["categorical"]
    bins = info["binary"]
    nums_safe = usable_numeric_cols(df)
    y_num = nums_safe[0] if nums_safe else (nums_all[0] if nums_all else None)

    def valid_group(col: str, outcome: Optional[str]) -> bool:
        if outcome and col == outcome:
            return False
        u = df[col].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[col]):
            return u >= 2 and is_integer_like(df[col])
        else:
            return u >= 2

    g_bin = next((c for c in bins if valid_group(c, y_num)), None)
    g_multi = next((c for c in cats if df[c].nunique(dropna=True) >= 3 and valid_group(c, y_num)), None)

    overview_lines = [f"Dataset overview: n={n} rows, p={p} columns."]
    if nums_all:
        overview_lines.append(
            f"- Numeric columns ({len(nums_all)}): "
            f"{', '.join(map(str, nums_all[:8]))}{'…' if len(nums_all) > 8 else ''}"
        )
    if cats:
        overview_lines.append(
            f"- Categorical/low-cardinality ({len(cats)}): "
            f"{', '.join(map(str, cats[:8]))}{'…' if len(cats) > 8 else ''}"
        )

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

    ex = ["Here are some things you can say:"]
    if y_num:
        ex.append(f'• "What is the average {y_num}?"')
        ex.append(f'• "Show a histogram of {y_num}."')
    if y_num and g_bin:
        ex.append(f'• "I want to do a t-test on {y_num} by {g_bin}."')
        ex.append(f'• "Mann-Whitney U test on {y_num} by {g_bin}."')
        ex.append(f'• "Point-biserial correlation {y_num} by {g_bin}."')
        ex.append(f"• \"Levene's test for {y_num} by {g_bin}.\"")
    if y_num and g_multi:
        ex.append(f'• "Run a one-way ANOVA of {y_num} by {g_multi}."')
        ex.append(f'• "Tukey HSD for {y_num} by {g_multi}."')
        ex.append(f'• "Kruskal-Wallis on {y_num} by {g_multi}."')
        ex.append("• \"Dunn's post-hoc for {y_num} by {g_multi}.\"")
        ex.append(f'• "Show box and violin plots for {y_num} by {g_multi}."')
    if len(nums_safe) >= 2:
        ex.append(f'• "Pearson correlation between {nums_safe[0]} and {nums_safe[1]}."')
        ex.append(f'• "Spearman correlation {nums_safe[0]} vs {nums_safe[1]}."')
        ex.append('• "Correlation heatmap of numeric columns."')
        ex.append('• "Partial correlation height and score controlling for age, weight."')
        ex.append(f'• "Fit a linear regression: {nums_safe[0]} ~ {nums_safe[1]}."')
    if len(cats) >= 2:
        ex.append('• "Chi-square test of sex by group."')
    ex.append('• "Check normality of score (QQ plot)."')
    if g_bin:
        ex.append('• "Power analysis for a t-test with 80% power and effect size 0.5."')
    if g_multi:
        ex.append('• "Power analysis for a one-way ANOVA with 3 groups and 80% power."')

    return "\n".join(overview_lines + [""] + rec_lines), "\n".join(ex)
