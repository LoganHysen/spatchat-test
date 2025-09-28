# stats/power.py
# Power & sample size calculators: two-sample t-test (independent) and one-way ANOVA
from typing import Optional
import numpy as np
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

def power_ttest_ind(
    effect_size: Optional[float],
    alpha: float,
    power: Optional[float],
    ratio: float,
    solve_for: str,
) -> str:
    """
    Wrapper around statsmodels TTestIndPower with stable text outputs.
    solve_for ∈ {'n_total','power','effect_size'}.
    - If solve_for == 'n_total', 'power' is interpreted as desired power; returns per-group sizes and total.
    - If solve_for == 'power', 'power' arg is actually nobs1 (sample size in group1); returns achieved power.
    - If solve_for == 'effect_size', 'power' arg is nobs1; returns implied Cohen's d.
    """
    tool = TTestIndPower()
    solve_for = (solve_for or "n_total").lower()

    if solve_for == "n_total":
        n1 = tool.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=ratio, alternative="two-sided")
        n2 = n1 * ratio
        return (
            "Required total sample size (two-sample t-test): "
            f"group1 ≈ {int(np.ceil(n1))}, group2 ≈ {int(np.ceil(n2))} "
            f"(total ≈ {int(np.ceil(n1 + n2))})"
        )

    if solve_for == "power":
        pw = tool.solve_power(effect_size=effect_size, nobs1=power, alpha=alpha, ratio=ratio, alternative="two-sided")
        return f"Achieved power ≈ {pw:.3f}"

    if solve_for == "effect_size":
        es = tool.solve_power(effect_size=None, nobs1=power, alpha=alpha, ratio=ratio, alternative="two-sided")
        return f"Implied effect size d ≈ {float(es):.3f}"

    return "Unknown solve_for for t-test power."


def power_anova_oneway(
    effect_size: Optional[float],
    k_groups: int,
    alpha: float,
    power: Optional[float],
    solve_for: str,
) -> str:
    """
    Wrapper around statsmodels FTestAnovaPower with stable text outputs.
    solve_for ∈ {'n_per_group','power','effect_size'}.
    - If solve_for == 'n_per_group', uses desired 'power' to return total and per-group sizes.
    - If solve_for == 'power', 'power' arg is actually total nobs; returns achieved power.
    - If solve_for == 'effect_size', 'power' arg is total nobs; returns implied Cohen's f.
    """
    tool = FTestAnovaPower()
    solve_for = (solve_for or "n_per_group").lower()

    if solve_for == "n_per_group":
        n_total = tool.solve_power(effect_size=effect_size, k_groups=int(k_groups), alpha=alpha, power=power)
        n_per_group = int(np.ceil(n_total / max(int(k_groups), 1)))
        n_total = int(np.ceil(n_total))
        return (
            f"Required sample size (one-way ANOVA): total ≈ {n_total}, "
            f"per group ≈ {n_per_group} (k={k_groups}, f={effect_size}, α={alpha}, power={power})"
        )

    if solve_for == "power":
        pw = tool.solve_power(effect_size=effect_size, k_groups=int(k_groups), alpha=alpha, nobs=power)
        return f"Achieved power ≈ {pw:.3f}"

    if solve_for == "effect_size":
        es = tool.solve_power(effect_size=None, k_groups=int(k_groups), alpha=alpha, nobs=power)
        return f"Implied effect size f ≈ {float(es):.3f}"

    return "Unknown solve_for for ANOVA power."
