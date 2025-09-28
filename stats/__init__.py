# stats/__init__.py

# Location / two-group / paired tests
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

# Descriptives & distribution checks (NO quick_summary export here)
from .descriptives import (
    plot_hist,
    plot_box,
    plot_violin,
    check_normality,
)

# Power
from .power import power_ttest_ind, power_anova_oneway

# >>> Recommendations & dataset summary (canonical)
# Make SURE quick_summary comes ONLY from recommendations
from .recommendations import quick_summary, recommend_text_and_examples

__all__ = [
    # tests_location
    "ttest_summary", "run_ttest", "mann_whitney", "wilcoxon_signed", "point_biserial",
    # tests_variance
    "run_anova", "kruskal_wallis", "levene_test", "tukey_hsd", "dunn_posthoc",
    # correlations
    "corr_pair", "corr_matrix_plot", "partial_corr",
    # associations
    "chisq_test",
    # modeling
    "run_ols", "run_glm",
    # descriptives (plots/checks only)
    "plot_hist", "plot_box", "plot_violin", "check_normality",
    # power
    "power_ttest_ind", "power_anova_oneway",
    # recommendations & summary
    "quick_summary", "recommend_text_and_examples",
]