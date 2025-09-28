# plot_helpers.py
# Shared plotting helpers kept at project root (alongside app.py).
# Used by modules in stats/. Centralizes style + small wrappers that return NumPy images.

from typing import Optional, List, Tuple
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------- HOUSE STYLE ----------
def set_house_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "legend.frameon": False,
        }
    )

set_house_style()


# ---------- CORE IMAGE UTILS ----------
def fig_to_np(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib figure to an RGB numpy array and close the figure."""
    buf = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    img = Image.open(buf).convert("RGB")
    return np.array(img)


def _render(fig: plt.Figure) -> np.ndarray:
    try:
        return fig_to_np(fig)
    finally:
        plt.close(fig)


def _sb_errorbar_kwargs() -> dict:
    """Handle Seaborn API differences (errorbar=None in 0.12+, ci=None in 0.11)."""
    try:
        major, minor = [int(p) for p in sns.__version__.split(".")[:2]]
        return {"errorbar": None} if (major, minor) >= (0, 12) else {"ci": None}
    except Exception:
        return {"ci": None}


# ---------- GROUP ORDERING (local fallback) ----------
def _ordered_groups(df: pd.DataFrame, group_col: str) -> List[str]:
    """Local copy used by plotting helpers (stats modules may import core_utils.ordered_groups)."""
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


# ---------- PLOTTING PRIMITIVES ----------
def bar_with_error_plot(
    df: pd.DataFrame, value: str, group: str, error: str = "sem", gorder: Optional[List[str]] = None
) -> np.ndarray:
    """
    Seaborn-styled bar of group means ± error (SEM, SD, or CI95).
    Returns NumPy image array.
    """
    if gorder is None:
        gorder = _ordered_groups(df, group)

    d = df[[value, group]].dropna().copy()
    d[group] = d[group].astype(str)

    desc = d.groupby(group)[value].agg(["mean", "std", "count"]).reset_index()
    desc.rename(columns={group: "group"}, inplace=True)
    desc["group"] = desc["group"].astype(str)
    desc = desc.set_index("group").reindex(gorder).reset_index()

    if error.lower() == "sd":
        yerr = desc["std"].to_numpy()
    elif error.lower() in {"ci95", "ci", "ci_95"}:
        se = desc["std"] / np.sqrt(desc["count"].clip(lower=1))
        tcrit = stats.t.ppf(0.975, df=(desc["count"] - 1).clip(lower=1))
        yerr = (se * tcrit).to_numpy()
    else:
        # default sem
        yerr = (desc["std"] / np.sqrt(desc["count"].clip(lower=1))).to_numpy()

    x_pos = np.arange(len(desc))
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=desc, x="group", y="mean", order=gorder, ax=ax, **_sb_errorbar_kwargs())
    ax.errorbar(x=x_pos, y=desc["mean"], yerr=yerr, fmt="none", capsize=4, lw=1.5)
    ax.set_xlabel(group)
    ax.set_ylabel(value)
    ax.set_title(f"Bar ± {error.upper()} of {value} by {group}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gorder)
    fig.tight_layout()
    return _render(fig)


def box_plot(df: pd.DataFrame, value: str, group: str) -> np.ndarray:
    gorder = _ordered_groups(df, group)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(data=df, x=group, y=value, order=gorder, ax=ax)
    ax.set_title(f"Boxplot of {value} by {group}")
    fig.tight_layout()
    return _render(fig)


def violin_plot(df: pd.DataFrame, value: str, group: str) -> np.ndarray:
    gorder = _ordered_groups(df, group)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.violinplot(data=df, x=group, y=value, order=gorder, inner="quartile", cut=0, ax=ax)
    ax.set_title(f"Violin of {value} by {group}")
    fig.tight_layout()
    return _render(fig)


def scatter_with_reg(df: pd.DataFrame, x: str, y: str, ci: float = 0.95, title: Optional[str] = None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 3))
    # Use seaborn regplot to get CI shading; pass integer percent
    sns.regplot(data=df, x=x, y=y, ci=int(ci * 100), ax=ax)
    ax.set_title(title or f"{y} vs {x} with linear fit")
    fig.tight_layout()
    return _render(fig)


def point_ci_plot_from_summary(
    summary_df: pd.DataFrame,
    x: str,
    y: str,
    ci_low: str,
    ci_high: str,
    title: str,
    ylabel: Optional[str] = None,
) -> np.ndarray:
    """Point plot with custom CI whiskers at integer tick positions from a precomputed summary table."""
    cats = list(summary_df[x].astype(str))
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.pointplot(data=summary_df, x=x, y=y, order=cats, join=False, ax=ax, **_sb_errorbar_kwargs())
    # custom CI whiskers at integer tick positions
    for i, r in enumerate(summary_df.itertuples()):
        ylo, yhi = getattr(r, ci_low), getattr(r, ci_high)
        ax.vlines(i, ylo, yhi, lw=1.5)
        ax.plot([i - 0.05, i + 0.05], [ylo, ylo], lw=1.5)
        ax.plot([i - 0.05, i + 0.05], [yhi, yhi], lw=1.5)
    ax.set_title(title)
    ax.set_xlabel(x)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    fig.tight_layout()
    return _render(fig)
