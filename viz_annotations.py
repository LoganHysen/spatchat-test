# viz_annotations.py

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from plot_helpers import bar_with_error_plot, fig_to_np  # root helpers

def _p_to_stars(p: float) -> str:
    """Convert a p-value to significance stars."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p >= 0.05:
        return "ns"
    if p >= 0.01:
        return "*"
    if p >= 0.001:
        return "**"
    return "***"

def annotate_ttest_on_bar(
    df,
    value: str,
    group: str,
    g1: str,
    g2: str,
    p: float,
    note: Optional[str] = None,
) -> np.ndarray:
    """
    Overlays a simple p-value + stars on top of a bar±SEM image.
    Returns a NumPy RGB image array. Purely visual; does not recompute stats.

    Parameters
    ----------
    df : pd.DataFrame
    value : numeric outcome column
    group : grouping column (must contain levels g1 and g2)
    g1, g2 : two group labels (order determines left/right)
    p : p-value to display (e.g., from a t-test)
    note : optional extra text to append, e.g. 'Welch t-test'
    """
    # Underlay: the existing bar ± SEM image for the two groups
    underlay = bar_with_error_plot(df, value, group, error="sem", gorder=[g1, g2])
    stars = _p_to_stars(float(p) if p is not None else np.nan)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(underlay)
    ax.axis("off")

    label = f"p = {p:.3g} ({stars})" if np.isfinite(p) else f"p = n/a ({stars})"
    if note:
        label = f"{label} — {note}"

    # Draw the label near the top center; tweak y if needed
    ax.text(
        0.5,
        0.06,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.75),
    )
    fig.tight_layout()
    return fig_to_np(fig)
