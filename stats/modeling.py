# stats/modeling.py
# Regression models: OLS and GLM (Gaussian/Binomial/Poisson/Gamma)
from typing import List, Tuple
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from plot_helpers import fig_to_np  # root helper (provided separately)

def run_ols(df: pd.DataFrame, formula: str) -> Tuple[str, List[np.ndarray]]:
    """
    Fit OLS using a Patsy-style formula (e.g., 'y ~ x1 + x2').
    Returns (text_summary, [images]) with optional scatter+fit (if single numeric predictor),
    residuals vs fitted, and QQ plot.
    """
    model = smf.ols(formula, data=df).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    coef.to_csv("outputs/ols_coef.csv", index=False)

    images: List[np.ndarray] = []

    # If single numeric predictor, add scatter + fitted line
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    terms = [t.strip() for t in re.split(r"\+", rhs) if t.strip()]
    if len(terms) == 1 and terms[0] in df.columns and pd.api.types.is_numeric_dtype(df[terms[0]]):
        x = df[terms[0]].astype(float)
        y = df[lhs].astype(float)
        order = np.argsort(x.values)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(x, y)
        ax.plot(x.values[order], model.fittedvalues.values[order])
        ax.set_xlabel(terms[0])
        ax.set_ylabel(lhs)
        ax.set_title("Scatter + OLS fit")
        images.append(fig_to_np(fig))

    # Residuals vs Fitted
    fig_r, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(model.fittedvalues, model.resid)
    ax.axhline(0, linestyle=":")
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    images.append(fig_to_np(fig_r))

    # QQ plot
    sm.qqplot(model.resid, line="45", fit=True)
    images.append(fig_to_np(plt.gcf()))

    return str(model.summary()), images


def run_glm(df: pd.DataFrame, formula: str, family: str) -> Tuple[str, List[np.ndarray]]:
    """
    Fit GLM with family in {'gaussian','binomial','poisson','gamma'}.
    Returns (text_summary, [images]) with residuals vs fitted and QQ of deviance residuals.
    """
    fam_map = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
    }
    fam = fam_map.get((family or "gaussian").lower(), sm.families.Gaussian())
    model = smf.glm(formula, data=df, family=fam).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    coef.to_csv("outputs/glm_coef.csv", index=False)

    images: List[np.ndarray] = []

    # Residuals vs Fitted (deviance residuals)
    fig_r, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(model.fittedvalues, model.resid_deviance)
    ax.axhline(0, linestyle=":")
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals (deviance)")
    ax.set_title(f"Residuals vs Fitted (GLM: {family})")
    images.append(fig_to_np(fig_r))

    # QQ of deviance residuals
    sm.qqplot(model.resid_deviance, line="45", fit=True)
    images.append(fig_to_np(plt.gcf()))

    return str(model.summary()), images
