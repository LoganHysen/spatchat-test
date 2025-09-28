# stats/descriptives.py
# Descriptive plots & distribution checks: hist, box, violin, QQ/normality
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

from plot_helpers import fig_to_np, box_plot, violin_plot  # root helpers (provided separately)

def plot_hist(df: pd.DataFrame, col: str, bins: int = 30) -> Tuple[str, np.ndarray]:
    """
    Histogram for a numeric column. Returns (title, image_array).
    Matches existing signature/behavior used by the UI.
    """
    s = df[col].dropna().astype(float)
    if len(s) == 0:
        raise ValueError(f"No numeric data found in column '{col}'.")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(s, bins=int(bins or 30), ax=ax)
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return f"Histogram {col}", fig_to_np(fig)


def plot_box(df: pd.DataFrame, value: str, group: str) -> Tuple[str, np.ndarray]:
    """
    Boxplot of value by group. Returns (title, image_array).
    Delegates to shared plotting helper for consistent style.
    """
    img = box_plot(df, value, group)
    return f"Boxplot {value}~{group}", img


def plot_violin(df: pd.DataFrame, value: str, group: str) -> Tuple[str, np.ndarray]:
    """
    Violin plot of value by group. Returns (title, image_array).
    Delegates to shared plotting helper for consistent style.
    """
    img = violin_plot(df, value, group)
    return f"Violin {value}~{group}", img


def check_normality(df: pd.DataFrame, col: str) -> Tuple[str, np.ndarray]:
    """
    Shapiro–Wilk (n in [3, 5000]) + QQ plot for a numeric column.
    Returns (message_text, image_array).
    """
    series = df[col].dropna().astype(float)
    n = len(series)
    if n == 0:
        raise ValueError(f"No numeric data found in column '{col}'.")
    if 3 <= n <= 5000:
        W, p = stats.shapiro(series)
        msg = f"Shapiro–Wilk normality on {col} (n={n}): W={W:.4g}, p={p:.5g}"
    else:
        msg = f"Shapiro–Wilk skipped for {col} (requires 3–5000 values). n={n}"

    sm.qqplot(series, line="45", fit=True)
    img = fig_to_np(plt.gcf())
    return msg, img
