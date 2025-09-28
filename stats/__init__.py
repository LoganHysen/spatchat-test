# stats/__init__.py
# Re-exports to keep existing imports/handlers working after modularization.

# Two-group / paired tests
from .tests_location import (
    ttest_summary,
    run_ttest,
    mann_whitney,
    wilcoxon_signed,
    point_biserial,
)

# Variance / k-group tests & posthocs
from .tests_variance import (
    run_anova,
    kruskal_wallis,
    levene_test,
    tukey_hsd,
    dunn_posthoc,
)

# Correlations
from .correlations import (
    corr_pair,
    corr_matrix_plot,
    partial_corr,
)

# Categorical associations
from .associations import chisq_test

# Modeling
from .modeling import run_ols, run_glm

# Descriptives & distribution checks
from .descriptives import (
    quick_summary,     # <- single source of truth lives in descriptives.py
    plot_hist,
    plot_box,
    plot_violin,
    check_normality,
)

# Power
from .power import (
    power_ttest_ind,
    power_anova_oneway,
)

# Recommendations (schema-driven suggestions for next analyses)
from .recommendations import recommend_text_and_examples

__all__ = [
    # tests_location
    "ttest_summary",
    "run_ttest",
    "mann_whitney",
    "wilcoxon_signed",
    "point_biserial",
    # tests_variance
    "run_anova",
    "kruskal_wallis",
    "levene_test",
    "tukey_hsd",
    "dunn_posthoc",
    # correlations
    "corr_pair",
    "corr_matrix_plot",
    "partial_corr",
    # associations
    "chisq_test",
    # modeling
    "run_ols",
    "run_glm",
    # descriptives
    "quick_summary",
    "plot_hist",
    "plot_box",
    "plot_violin",
    "check_normality",
    # power
    "power_ttest_ind",
    "power_anova_oneway",
    # recommendations
    "recommend_text_and_examples",
]