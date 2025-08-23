import os
import io
import re
import json
import zipfile
import shutil
import base64
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

# Together client is optional; app works without it
try:
    from together import Together
except Exception:
    Together = None  # type: ignore

print("Starting SpatChat: Stats Room (chat-first)")

# ---------- GLOBAL STATE ----------
cached_df: Optional[pd.DataFrame] = None
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# ---------- ENV & LLM CONFIG ----------
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
SPATCHAT_LLM_ENABLED = os.getenv("SPATCHAT_LLM_ENABLED", "1") == "1"
SPATCHAT_LLM_MODEL = os.getenv("SPATCHAT_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
LLM_COOLDOWN_SEC = float(os.getenv("SPATCHAT_LLM_COOLDOWN_SEC", "2.0"))
LLM_MAX_RETRIES = int(os.getenv("SPATCHAT_LLM_MAX_RETRIES", "2"))
SPATCHAT_HUMANIZE_RECS = os.getenv("SPATCHAT_HUMANIZE_RECS", "1") == "1"
SPATCHAT_HUMANIZE_RESULTS = os.getenv("SPATCHAT_HUMANIZE_RESULTS", "1") == "1"

_llm_client = None
_last_llm_time = 0.0

def _get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    if Together is None or not TOGETHER_API_KEY:
        return None
    _llm_client = Together(api_key=TOGETHER_API_KEY)
    return _llm_client

# ---------- PROMPTS ----------
SYSTEM_PROMPT = """
Return JSON only for tool calls:
{"tool":"stats","action":"ttest|anova|ols|glm|plot|check|power|recommend","args":{...}}
Args:
- ttest {"value":"Y","group":"G","paired":false,"equal_var":false}
- anova {"value":"Y","group":"G"}
- ols {"formula":"y ~ x1 + x2"}
- glm {"formula":"y ~ x1 + x2","family":"gaussian|binomial|poisson|gamma"}
- plot {"hist":{"col":"c","bins":30}|"box":{"value":"Y","group":"G"}|"violin":{"value":"Y","group":"G"}|"bar":{"value":"Y","group":"G","error":"sem|sd|ci95"}}
- check {"normality":{"col":"Y"}|"homogeneity":{"value":"Y","group":"G"}}
- power {"ttest_ind":{...}} | {"anova_oneway":{...}}
- recommend {}
If the user asks “what can I do with my data?” or similar, use action="recommend".
Use minimal defaults. No prose unless not a tool call (≤3 sentences).
""".strip()

FALLBACK_PROMPT = "You are SpatChat, a concise statistics tutor. If not a tool call, answer helpfully in ≤3 sentences."

# ---------- UTILITIES ----------
def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def save_png(png: bytes, fname: str) -> str:
    fp = os.path.join(outputs_dir, fname)
    with open(fp, "wb") as f:
        f.write(png)
    return fp

def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def categorical_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

def _is_binary_series(s: pd.Series) -> bool:
    s = s.dropna()
    if s.empty:
        return False
    uniq = pd.unique(s)
    norm = set()
    for v in uniq:
        if isinstance(v, (int, np.integer, bool)):
            norm.add(int(v))
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            norm.add(int(v))
        elif isinstance(v, str) and v.strip() in {"0", "1"}:
            norm.add(int(v.strip()))
        else:
            return False
    return norm in [{0, 1}, {1, 0}] and len(uniq) == 2

def _is_count_series(s: pd.Series) -> bool:
    s = s.dropna()
    if s.empty:
        return False
    if (s < 0).any():
        return False
    return np.all(np.floor(s.astype(float)) == s.astype(float))

# ---------- ID-like detection & usable numeric ----------
ID_NAME_PAT = re.compile(r"(?:^|[_\W])(id|identifier|index|subject|animal|record|row)(?:[_\W]|$)", re.I)

def is_id_like(df: pd.DataFrame, col: str) -> bool:
    """Heuristics: name pattern, very high uniqueness, or sequential integers."""
    name = str(col).strip().lower()
    if ID_NAME_PAT.search(name) or name.endswith("_id"):
        return True
    s = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(s):
        return False
    n = len(s)
    if n == 0:
        return False
    nunq = s.nunique()
    if nunq / n >= 0.8:
        return True
    s_int = s.astype(float)
    if np.all(np.floor(s_int) == s_int):
        u = np.sort(pd.unique(s_int))
        if len(u) > 4:
            if int(u[-1] - u[0] + 1) == len(u):
                return True
    return False

def usable_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Numeric columns excluding those that look like identifiers (for auto defaults)."""
    return [c for c in numeric_cols(df) if not is_id_like(df, c)]

# ---------- RECOMMENDATIONS ----------
def infer_schema(df: pd.DataFrame) -> Dict[str, List[str] or int]:
    nums_all = numeric_cols(df)
    nums = usable_numeric_cols(df)
    cat_like = []
    for c in df.columns:
        if c in nums_all:
            nunq = df[c].nunique(dropna=True)
            if 2 <= nunq <= 10:
                cat_like.append(c)
        else:
            cat_like.append(c)
    seen = set()
    cats = [c for c in cat_like if not (c in seen or seen.add(c))]
    bins = [c for c in cats if df[c].nunique(dropna=True) == 2]
    y_bin = [c for c in df.columns if (pd.api.types.is_bool_dtype(df[c]) or _is_binary_series(df[c]))]
    y_count = [c for c in nums if _is_count_series(df[c]) and df[c].nunique(dropna=True) > 2]
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "numeric": nums_all,
        "numeric_usable": nums,
        "categorical": cats,
        "binary_groups": bins,
        "y_bin": y_bin,
        "y_count": y_count,
    }

def recommend_analyses_text(df: pd.DataFrame) -> str:
    info = infer_schema(df)
    n, p = info["n_rows"], info["n_cols"]
    nums_all = info["numeric"]
    nums = info["numeric_usable"]
    cats = info["categorical"]
    bins = info["binary_groups"]
    y_bin, y_count = info["y_bin"], info["y_count"]

    lines = []
    lines.append(f"Dataset overview: n={n} rows, p={p} columns.")
    if nums_all:
        lines.append(f"- Numeric columns ({len(nums_all)}): {', '.join(map(str, nums_all[:8]))}{'…' if len(nums_all)>8 else ''}")
    if cats:
        lines.append(f"- Categorical/low-cardinality ({len(cats)}): {', '.join(map(str, cats[:8]))}{'…' if len(cats)>8 else ''}")

    recs = []
    if bins and nums:
        g = bins[0]; y = next((c for c in nums if c != g), nums[0])
        recs.append(f"• t-test: compare means of {y} across {g} (2 groups).\nTry: ttest value={y} group={g}")
    multi_group = [c for c in cats if df[c].nunique(dropna=True) >= 3]
    if multi_group and nums:
        g = multi_group[0]; y = next((c for c in nums if c != g), nums[0])
        recs.append(f"• One-way ANOVA: {y} by {g} (≥3 groups).\nTry: anova value={y} group={g}")
    if len(nums) >= 2:
        y = nums[0]; x = nums[1]
        recs.append(f"• Regression (OLS): {y} with {x}.\nTry: ols {y} ~ {x}")
    if y_bin and len(nums) >= 1:
        y = y_bin[0]
        x = nums[0] if nums and nums[0] != y else (nums[1] if len(nums) > 1 else None)
        if x:
            recs.append(f"• Logistic GLM for binary {y}.\nTry: glm {y} ~ {x} family=binomial")
    if y_count and len(nums) >= 1:
        y = y_count[0]
        x = nums[0] if nums and nums[0] != y else (nums[1] if len(nums) > 1 else None)
        if x:
            recs.append(f"• Poisson GLM for counts {y}.\nTry: glm {y} ~ {x} family=poisson")
    if nums:
        y = nums[0]
        recs.append(f"• Normality check: check normality col={y}; histogram: plot hist col={y}")
    if cats and nums:
        y = nums[0]; g = cats[0] if cats[0] != y else (cats[1] if len(cats)>1 else cats[0])
        recs.append(f"• Group visuals: plot box value={y} group={g} or plot violin value={y} group={g}")
    if bins and nums:
        y = nums[0]; g = bins[0]
        recs.append(f"• Bar ± error: plot bar value={y} group={g} error=sem|sd|ci95")
    if bins and nums:
        recs.append("• Power (t-test): power ttest_ind effect_size=0.5 power=0.8")
    if multi_group:
        recs.append("• Power (ANOVA): power anova_oneway effect_size=0.25 k_groups=3 power=0.8")
    if not recs:
        recs.append("• Tell me your goal (e.g., “compare two groups on height” or “predict score from features”).")
    lines.append("\nRecommended next steps:\n" + "\n".join(recs))
    return "\n".join(lines)

def recommend_examples_natural(df: pd.DataFrame) -> str:
    info = infer_schema(df)
    nums = info["numeric_usable"]
    cats = info["categorical"]
    bins = info["binary_groups"]
    y_bin = info["y_bin"]
    y_count = info["y_count"]

    lines = []
    lines.append("Here are some things you can say:")
    if bins and nums:
        g = bins[0]; y = next((c for c in nums if c != g), nums[0])
        lines.append(f'• "I want to do a t-test on {y} by {g}."')
    multi_group = [c for c in cats if df[c].nunique(dropna=True) >= 3]
    if multi_group and nums:
        g = multi_group[0]; y = next((c for c in nums if c != g), nums[0])
        lines.append(f'• "Run a one-way ANOVA of {y} by {g}."')
    if len(nums) >= 2:
        y = nums[0]; x = nums[1]
        lines.append(f'• "Fit a linear regression: {y} ~ {x}."')
    if y_bin and len(nums) >= 1:
        y = y_bin[0]
        x = nums[0] if nums and nums[0] != y else (nums[1] if len(nums) > 1 else None)
        if x:
            lines.append(f'• "Do a logistic regression: {y} ~ {x}."')
    if y_count and len(nums) >= 1:
        y = y_count[0]
        x = nums[0] if nums and nums[0] != y else (nums[1] if len(nums) > 1 else None)
        if x:
            lines.append(f'• "Fit a Poisson regression for {y} using {x}."')
    if nums:
        y = nums[0]
        lines.append(f'• "Check normality of {y} and show a histogram."')
    if cats and nums:
        y = nums[0]; g = cats[0] if cats[0] != y else (cats[1] if len(cats) > 1 else cats[0])
        lines.append(f'• "Show box and violin plots for {y} by {g}."')
    if bins and nums:
        y = nums[0]; g = bins[0]
        lines.append(f'• "Make a bar chart of {y} by {g} with 95% CI error bars."')
    if bins and nums:
        lines.append('• "Power analysis for a t-test with 80% power and effect size 0.5."')
    if multi_group:
        lines.append('• "Power analysis for a one-way ANOVA with 3 groups and 80% power."')
    if len(lines) == 1:
        lines.append('• "Suggest some analyses based on my columns."')
    return "\n".join(lines)

# ---------- LLM HUMANIZERS ----------
def humanize_recommendations(tech_text: str, df: pd.DataFrame) -> str:
    if not (SPATCHAT_LLM_ENABLED and SPATCHAT_HUMANIZE_RECS):
        return tech_text
    client = _get_llm_client()
    if client is None:
        return tech_text
    schema = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns][:20],
        "numeric_usable": usable_numeric_cols(df)[:10],
        "categorical": categorical_cols(df)[:10],
        "example_rows": df.head(2).to_dict(orient="records"),
    }
    system = (
        "You are a helpful data analyst. Rewrite a technical recommendation list for a general audience. "
        "Be clear and friendly, ~150–220 words. For each suggested analysis, say what question it answers "
        "in plain language and when to use it. Include 1 short example prompt in quotes when helpful. "
        "Avoid jargon; do not invent columns; prefer manipulable numeric columns over IDs."
    )
    user = f"Dataset (truncated JSON):\n{json.dumps(schema)}\n\nTechnical recommendations:\n{tech_text}"
    global _last_llm_time
    since = time.time() - _last_llm_time
    if since < LLM_COOLDOWN_SEC:
        time.sleep(LLM_COOLDOWN_SEC - since)
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=SPATCHAT_LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3,
            ).choices[0].message.content
            _last_llm_time = time.time()
            return resp.strip()
        except Exception as e:
            if "429" in str(e) and attempt < LLM_MAX_RETRIES:
                attempt += 1
                time.sleep(1.0 * attempt)
                continue
            return tech_text

def humanize_result(kind: str, details: str) -> str:
    canned = {
        "t-test": "Compares the average of one numeric variable between two groups (e.g., males vs females). Useful when groups are independent; Welch’s version is robust to unequal variances.",
        "ANOVA": "Tests whether at least one group mean differs when you have 3+ groups. Follow up with plots or pairwise comparisons to see where differences are.",
        "OLS": "Fits a straight-line relationship between a numeric outcome and one or more predictors, showing how much the outcome changes per unit change in each predictor.",
        "GLM": "A flexible regression that matches the type of outcome (e.g., yes/no with logistic, counts with Poisson) to get better estimates and valid uncertainty.",
    }
    if not (SPATCHAT_LLM_ENABLED and SPATCHAT_HUMANIZE_RESULTS):
        return canned.get(kind, "")
    client = _get_llm_client()
    if client is None:
        return canned.get(kind, "")
    system = ("Write a 1–2 sentence, non-jargony explanation of why this analysis is appropriate and what it tells you. "
              "Do NOT restate the numeric results; describe the purpose. Keep it under 45 words.")
    user = f"Analysis kind: {kind}\nContext snapshot:\n{details}"
    global _last_llm_time
    since = time.time() - _last_llm_time
    if since < LLM_COOLDOWN_SEC:
        time.sleep(LLM_COOLDOWN_SEC - since)
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=SPATCHAT_LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.2,
            ).choices[0].message.content
            _last_llm_time = time.time()
            return resp.strip()
        except Exception as e:
            if "429" in str(e) and attempt < LLM_MAX_RETRIES:
                attempt += 1
                time.sleep(1.0 * attempt)
                continue
            return canned.get(kind, "")

# ---------- ANALYSES ----------
@dataclass
class ModelOutput:
    kind: str
    summary_text: str
    table: Optional[pd.DataFrame] = None
    plots: Optional[List[Tuple[str, Optional[bytes]]]] = None

def _ttest_summary(a: np.ndarray, b: np.ndarray, paired: bool, equal_var: bool, g1: str, g2: str) -> str:
    alpha = 0.05
    mean1, mean2 = a.mean(), b.mean()
    n1, n2 = len(a), len(b)
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    diff = mean1 - mean2
    if paired:
        d = a - b
        n = len(d); df = n - 1
        md = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
        t = md / se; p = 2 * stats.t.sf(abs(t), df)
        tcrit = stats.t.ppf(1 - alpha/2, df)
        ci_low, ci_high = md - tcrit * se, md + tcrit * se
        dz = md / sd if sd > 0 else np.nan
        return (f"Paired t-test\nGroups: {g1}, {g2}\n"
                f"n = {n}\nMean diff = {md:.4g}, 95% CI [{ci_low:.4g}, {ci_high:.4g}]\n"
                f"t({df}) = {t:.4g}, p = {p:.4g}\nEffect size: Cohen's dz = {dz:.4g}")
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
        se = np.sqrt(sp2 * (1/n1 + 1/n2))
        t = diff / se; p = 2 * stats.t.sf(abs(t), df)
        tcrit = stats.t.ppf(1 - alpha/2, df)
        ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
        d = diff / np.sqrt(sp2) if sp2 > 0 else np.nan
    else:
        se = np.sqrt(var1 / n1 + var2 / n2)
        df = (var1/n1 + var2/n2)**2 / ((var1**2)/(n1**2*(n1-1)) + (var2**2)/(n2**2*(n2-1)))
        t = diff / se; p = 2 * stats.t.sf(abs(t), df)
        tcrit = stats.t.ppf(1 - alpha/2, df)
        ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
        sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        d = diff / np.sqrt(sp2) if sp2 > 0 else np.nan
    g = d * (1 - 3 / (4*(n1 + n2) - 9)) if (n1 + n2) > 3 else np.nan
    return (f"{'Student' if equal_var else 'Welch'} t-test\n"
            f"Groups: {g1} (n={n1}, mean={mean1:.4g}, sd={np.sqrt(var1):.4g}) vs {g2} (n={n2}, mean={mean2:.4g}, sd={np.sqrt(var2):.4g})\n"
            f"Mean difference = {diff:.4g}, 95% CI [{ci_low:.4g}, {ci_high:.4g}]\n"
            f"t({df:.2f}) = {t:.4g}, p = {p:.4g}\n"
            f"Effect size: Cohen's d = {d:.4g}, Hedges' g = {g:.4g}")

def run_ttest(df: pd.DataFrame, value: str, group: str, paired=False, equal_var=False, explicit_value=False) -> ModelOutput:
    order_raw = list(pd.unique(df[group].dropna()))
    if len(order_raw) != 2:
        raise gr.Error("t-test requires a grouping column with exactly 2 levels.")
    g1, g2 = order_raw[0], order_raw[1]

    a = df.loc[df[group] == g1, value].dropna().astype(float).values
    b = df.loc[df[group] == g2, value].dropna().astype(float).values

    # Box
    fig_box, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot([a, b], tick_labels=[str(g1), str(g2)])
    ax.set_title("Boxplot by Group")
    png_box = fig_to_png(fig_box)

    # Violin
    fig_vio, ax = plt.subplots(figsize=(5, 3))
    ax.violinplot([a, b], showmeans=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels([str(g1), str(g2)])
    ax.set_title("Violin by Group")
    png_vio = fig_to_png(fig_vio)

    # Bar ± SEM — consistent order
    dfg = pd.DataFrame({group: [g1] * len(a) + [g2] * len(b), value: np.concatenate([a, b])})
    stats_tbl = dfg.groupby(group, sort=False)[value].agg(['mean', 'std', 'count']).reset_index()
    labels = [str(g1), str(g2)]
    stats_tbl[group] = stats_tbl[group].astype(str)
    stats_tbl = stats_tbl.set_index(group).reindex(labels).reset_index()
    means = stats_tbl['mean'].values
    sds   = stats_tbl['std'].values
    ns    = stats_tbl['count'].values.astype(float)
    sem   = sds / np.sqrt(np.where(ns > 0, ns, np.nan))
    sem   = np.nan_to_num(sem, nan=0.0)
    x = np.arange(len(labels))
    fig_bar, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, means, yerr=sem, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f"Mean {value}")
    ax.set_title(f"{value} by {group} (error: SEM)")
    png_bar = fig_to_png(fig_bar)

    table = pd.DataFrame({
        "group": labels,
        "n": [len(a), len(b)],
        "mean": [a.mean(), b.mean()],
        "sd": [a.std(ddof=1), b.std(ddof=1)]
    })

    summary = _ttest_summary(a, b, paired=paired, equal_var=equal_var, g1=str(g1), g2=str(g2))
    caution = ""
    if is_id_like(df, value) and explicit_value:
        caution = f"\n\nCaution: `{value}` looks like an identifier column; t-tests on IDs can be hard to interpret."
    return ModelOutput("t-test", summary + caution, table, [("Bar±SEM", png_bar), ("Boxplot", png_box), ("Violin", png_vio)])

def run_anova(df: pd.DataFrame, value: str, group: str, explicit_value=False) -> ModelOutput:
    order_raw = list(pd.unique(df[group].dropna()))
    if len(order_raw) < 2:
        raise gr.Error("ANOVA requires a grouping factor with at least 2 levels.")
    labels = [str(x) for x in order_raw]
    groups_arrays = [df.loc[df[group] == lvl, value].dropna().astype(float).values for lvl in order_raw]
    F, p = stats.f_oneway(*groups_arrays)
    ns = [len(g) for g in groups_arrays]
    means = [g.mean() for g in groups_arrays]
    overall = np.concatenate(groups_arrays).mean()
    SSB = sum(n * (m - overall) ** 2 for n, m in zip(ns, means))
    SSW = sum(((g - m) ** 2).sum() for g, m in zip(groups_arrays, means))
    SST = SSB + SSW
    k = len(groups_arrays); N = sum(ns)
    df1, df2 = k - 1, N - k
    eta2 = SSB / SST if SST > 0 else np.nan

    fig_box, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot(groups_arrays, tick_labels=labels)
    ax.set_title("Boxplot by Group")
    png_box = fig_to_png(fig_box)

    fig_vio, ax = plt.subplots(figsize=(5, 3))
    ax.violinplot(groups_arrays, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
    ax.set_title("Violin by Group")
    png_vio = fig_to_png(fig_vio)

    means_table = pd.DataFrame({"group": labels, "n": ns, f"mean_{value}": means})
    caution = ""
    if is_id_like(df, value) and explicit_value:
        caution = f"\n\nCaution: `{value}` looks like an identifier; ANOVA on IDs can be hard to interpret."
    summary = (
        f"One-way ANOVA\n"
        f"Groups: {', '.join(labels)}\n"
        f"F({df1}, {df2}) = {F:.4g}, p = {p:.4g}, η² = {eta2:.4g}\n"
        f"Group means (n): " + ", ".join([f"{lab}={m:.4g} (n={n})" for lab, m, n in zip(labels, means, ns)])
    ) + caution
    return ModelOutput("ANOVA", summary, means_table, [("Boxplot", png_box), ("Violin", png_vio)])

# ---------- PLOTS & CHECKS ----------
def plot_hist(df: pd.DataFrame, col: str, bins: int = 30) -> Tuple[str, bytes]:
    series = df[col].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(series, bins=bins)
    ax.set_title(f"Histogram of {col}")
    return f"Histogram {col}", fig_to_png(fig)

def plot_box(df: pd.DataFrame, value: str, group: str, order: Optional[List[str]] = None) -> Tuple[str, bytes]:
    order_raw = list(pd.unique(df[group].dropna())) if order is None else order
    arrays = [df.loc[df[group] == lvl, value].dropna().astype(float).values for lvl in order_raw]
    labels = [str(x) for x in order_raw]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot(arrays, tick_labels=labels)
    ax.set_title(f"Boxplot of {value} by {group}")
    return f"Boxplot {value}~{group}", fig_to_png(fig)

def plot_violin(df: pd.DataFrame, value: str, group: str, order: Optional[List[str]] = None) -> Tuple[str, bytes]:
    order_raw = list(pd.unique(df[group].dropna())) if order is None else order
    arrays = [df.loc[df[group] == lvl, value].dropna().astype(float).values for lvl in order_raw]
    labels = [str(x) for x in order_raw]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.violinplot(arrays, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
    ax.set_title(f"Violin of {value} by {group}")
    return f"Violin {value}~{group}", fig_to_png(fig)

def plot_bar_with_error(df: pd.DataFrame, value: str, group: str, error: str = "sem", order: Optional[List[str]] = None) -> Tuple[str, bytes]:
    dfg = df[[group, value]].dropna()
    dfg[value] = dfg[value].astype(float)
    stats_tbl = dfg.groupby(group, sort=False)[value].agg(['mean', 'std', 'count']).reset_index()
    if order is None:
        order = [str(x) for x in pd.unique(dfg[group])]
    stats_tbl[group] = stats_tbl[group].astype(str)
    stats_tbl = stats_tbl.set_index(group).reindex(order).reset_index()
    means = stats_tbl['mean'].values
    sds   = stats_tbl['std'].values
    ns    = stats_tbl['count'].values.astype(float)
    if error.lower() == "sd":
        yerr = sds; err_label = "SD"
    elif error.lower() == "ci95":
        sem = sds / np.sqrt(np.where(ns > 0, ns, np.nan))
        tcrit = stats.t.ppf(0.975, np.maximum(ns - 1, 1))
        yerr = sem * tcrit; err_label = "95% CI"
    else:
        yerr = sds / np.sqrt(np.where(ns > 0, ns, np.nan)); err_label = "SEM"
    yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)
    labels = order
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, means, yerr=yerr, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f"Mean {value}")
    ax.set_title(f"{value} by {group} (error: {err_label})")
    return f"Bar {value}~{group} ({err_label})", fig_to_png(fig)

def check_normality(df: pd.DataFrame, col: str) -> Tuple[str, Optional[bytes]]:
    series = df[col].dropna().astype(float)
    stat, p = stats.shapiro(series) if len(series) <= 5000 else (np.nan, np.nan)
    msg = f"Shapiro–Wilk (n={len(series)}): W={stat:.4f}, p={p:.4g}" if not np.isnan(stat) else "Shapiro skipped (n>5000)."
    sm.qqplot(series, line="45", fit=True)
    png = fig_to_png(plt.gcf())
    return msg, png

def check_homogeneity(df: pd.DataFrame, value: str, group: str) -> str:
    arrays = [g[value].dropna().values for _, g in df[[group, value]].groupby(group, sort=False)]
    stat, p = stats.levene(*arrays)
    return f"Levene test for equal variances: W={stat:.4f}, p={p:.4g}"

# ---------- FILE/ZIP ----------
def write_report(sections: List[Tuple[str, str, Optional[bytes]]]) -> str:
    parts = ["<h1>SpatChat – Stats Report</h1>"]
    for title, html_text, png in sections:
        parts.append(f"<h2>{title}</h2>\n<pre style='white-space:pre-wrap'>{html_text}</pre>")
        if png is not None:
            b64 = base64.b64encode(png).decode("ascii")
            parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto' />")
    html = "\n".join(parts).encode("utf-8")
    fp = os.path.join(outputs_dir, "stats_report.html")
    with open(fp, "wb") as f:
        f.write(html)
    return fp

def save_zip() -> str:
    archive = os.path.join(outputs_dir, "spatchat_stats_results.zip")
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(outputs_dir):
            for fn in files:
                if fn.endswith('.zip'):
                    continue
                z.write(os.path.join(root, fn), arcname=fn)
    return archive

def clear_outputs():
    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
    os.makedirs(outputs_dir, exist_ok=True)

# ---------- LLM + PARSERS ----------
def ask_llm(chat_history, user_input):
    if not SPATCHAT_LLM_ENABLED:
        return None, ("LLM disabled. Use explicit commands like "
                      "`ttest value=height group=sex` or enable SPATCHAT_LLM_ENABLED=1.")
    client = _get_llm_client()
    if client is None:
        return None, ("TOGETHER_API_KEY not set. You can still run explicit commands, e.g. "
                      "`ttest value=score group=sex`, `anova value=height group=site`.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history + [{"role": "user", "content": user_input}]
    global _last_llm_time
    since = time.time() - _last_llm_time
    if since < LLM_COOLDOWN_SEC:
        time.sleep(LLM_COOLDOWN_SEC - since)
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=SPATCHAT_LLM_MODEL, messages=messages, temperature=0.0
            ).choices[0].message.content
            _last_llm_time = time.time()
            try:
                call = json.loads(resp)
                return call, resp
            except Exception:
                conv = client.chat.completions.create(
                    model=SPATCHAT_LLM_MODEL,
                    messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
                    temperature=0.7
                ).choices[0].message.content
                _last_llm_time = time.time()
                return None, conv
        except Exception as e:
            if "429" in str(e) and attempt < LLM_MAX_RETRIES:
                attempt += 1
                time.sleep(1.0 * attempt)
                continue
            return None, f"(LLM unavailable: {e}) You can run explicit commands like `ttest value=height group=sex`."

def _find_first_col_mention(text: str, df: pd.DataFrame) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    cols_lower = {c.lower(): c for c in df.columns.astype(str)}
    for t in tokens:
        if t.lower() in cols_lower:
            return cols_lower[t.lower()]
    t = text.strip()
    if t.lower() in cols_lower:
        return cols_lower[t.lower()]
    return None

def parse_explicit(user_message: str) -> Optional[Tuple[str, dict]]:
    s = user_message.lower()
    grp_hint = None
    m = re.search(r"\bby\s+([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
    if m: grp_hint = m.group(1)

    # Recommend
    if re.search(r"\b(what can i do|how (should|to) (i )?analy[sz]e|what (analys(e|es)|tests?) should i do|recommend(ation)?s|help analy[sz]e|suggest (analy|tests?))\b", s):
        return ("recommend", {})

    # Histogram
    if re.search(r"\bhist(?:ogram)?\b", s):
        m = re.search(r"(?:col(?:umn)?|of|on|for)\s*=?\s*([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        col = m.group(1) if m else None
        bins_m = re.search(r"bins\s*=\s*(\d+)", s)
        bins = int(bins_m.group(1)) if bins_m else 30
        return ("plot", {"hist": {"col": col, "bins": bins}})

    # Box & Violin
    if re.search(r"\bbox(?:plot)?\b", s):
        v = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message)
        g = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message)
        if not v:
            v = re.search(r"(?:box(?:plot)?\s+(?:of|on|for)\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        if not g:
            g = re.search(r"(?:by\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        return ("plot", {"box": {"value": v and v.group(1), "group": g.group(1) if g else grp_hint}})

    if re.search(r"\bviolin\b", s):
        v = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message)
        g = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message)
        if not v:
            v = re.search(r"(?:violin\s+(?:of|on|for)\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        if not g:
            g = re.search(r"(?:by\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        return ("plot", {"violin": {"value": v and v.group(1), "group": g.group(1) if g else grp_hint}})

    # Bar
    if re.search(r"\bbar\s*(plot|chart)?\b", s):
        value = None; group = grp_hint; error = None
        m = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message);  value = m and m.group(1)
        m = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message);  group = m.group(1) if m else (group if grp_hint else None)
        m = re.search(r"error\s*=\s*(sem|sd|ci95)", s);               error = m and m.group(1)
        # also support "bar of X by Y"
        if value is None:
            m = re.search(r"\bbar(?:\s*(?:plot|chart))?\s+(?:of|on|for)\s+([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
            value = m and m.group(1)
        return ("plot", {"bar": {"value": value, "group": group, "error": error or "sem"}})

    # t-test
    if re.search(r"\bt[-\s]?test\b|\bttest\b", s):
        v = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message)
        if not v:
            v = re.search(r"(?:t[-\s]?test|ttest)\s+(?:of|on|for)\s+([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        g = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message)
        if not g and grp_hint:
            g_val = grp_hint
        else:
            g_val = g and g.group(1)
        return ("ttest", {"value": v and v.group(1), "group": g_val,
                          "_explicit_value": bool(v), "_explicit_group": bool(g or grp_hint)})

    # ANOVA
    if re.search(r"\banova\b", s):
        v = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message)
        if not v:
            v = re.search(r"(?:anova)\s+(?:of|on|for)\s+([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        g = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message)
        if not g and grp_hint:
            g_val = grp_hint
        else:
            g_val = g and g.group(1)
        return ("anova", {"value": v and v.group(1), "group": g_val,
                          "_explicit_value": bool(v), "_explicit_group": bool(g or grp_hint)})

    # OLS / GLM plain text formulas are passed through by LLM or manual
    return None

# ---------- CLARIFICATION HELPERS ----------
def resolve_pending(pending: Dict, user_message: str, df: pd.DataFrame) -> Tuple[Optional[Tuple[str, dict]], Optional[str], Dict]:
    action = pending.get("action")
    wait = pending.get("await")
    group = pending.get("group")
    value = pending.get("value")
    v_match = re.search(r"value\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", user_message)
    g_match = re.search(r"group\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", user_message)
    if v_match: value = v_match.group(1)
    if g_match: group = g_match.group(1)

    if wait == "hist_col":
        col = _find_first_col_mention(user_message, df)
        if not col or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            return None, "Please specify a numeric column for the histogram, e.g., `col=height`.", pending
        bins = int(pending.get("bins", 30))
        return ("plot", {"hist": {"col": col, "bins": bins}}), None, {}

    if wait in ("ttest_group", "anova_group"):
        if not group:
            group = _find_first_col_mention(user_message, df)
        if not group or group not in df.columns:
            return None, "I couldn't find that group column. Please reply with an existing column name, e.g., `group=sex`.", pending
        nunq = df[group].nunique(dropna=True)
        if action == "ttest" and nunq != 2:
            return None, f"`{group}` has {nunq} levels; t-test needs exactly 2. Pick a different group (e.g., `group=sex`).", pending
        if action == "anova" and nunq < 2:
            return None, f"`{group}` has <2 levels for ANOVA. Choose another factor.", pending
        pending["group"] = group
        nums = [c for c in usable_numeric_cols(df) if c != group]
        if value and value in df.columns and pd.api.types.is_numeric_dtype(df[value]):
            return (action, {"value": value, "group": group, "_explicit_value": True}), None, {}
        if not nums:
            return None, "I couldn't find a numeric outcome column different from the group. Which numeric column should I compare?", {"action": action, "await": f"{action}_value", "group": group}
        opts = ", ".join(nums[:12]) + ("…" if len(nums) > 12 else "")
        return None, f"Which numeric outcome should I test? Candidates: {opts}", {"action": action, "await": f"{action}_value", "group": group}

    if wait in ("ttest_value", "anova_value"):
        if not value:
            value = _find_first_col_mention(user_message, df)
        if not value or value not in df.columns or not pd.api.types.is_numeric_dtype(df[value]):
            return None, "Please specify an existing numeric column, e.g., `value=height`.", pending
        if not group:
            cats = [c for c in categorical_cols(df) if df[c].nunique(dropna=True) >= 2]
            if not cats:
                return None, "I couldn't find a valid grouping column. Reply with `group=<column>`.", {"action": action, "await": f"{action}_group", "value": value}
            opts = ", ".join(cats[:12]) + ("…" if len(cats) > 12 else "")
            return None, f"Which grouping column? Candidates: {opts}", {"action": action, "await": f"{action}_group", "value": value}
        nunq = df[group].nunique(dropna=True)
        if action == "ttest" and nunq != 2:
            return None, f"`{group}` has {nunq} levels; t-test needs exactly 2. Pick a different group (e.g., `group=sex`).", {"action": action, "await": f"{action}_group", "value": value}
        return (action, {"value": value, "group": group, "_explicit_value": True}), None, {}
    return None, None, pending

# ---------- CHAT HANDLERS ----------
def handle_upload(file):
    global cached_df
    clear_outputs()
    try:
        df = pd.read_csv(file)
        cached_df = df
        cols = ", ".join(df.columns.astype(str))
        preview = df.head(200)
        return (
            [{"role": "assistant", "content": f"CSV uploaded. Columns detected: {cols}. Ask me for t-test, ANOVA, OLS/GLM, hist/box/violin/bar, normality checks, power, or say **'what can I do with my data?'**"}],
            gr.update(value=preview, visible=True),
            gr.update(visible=False),
            [], 0,
            {}
        )
    except Exception as e:
        return [{"role": "assistant", "content": f"Failed to read CSV: {e}"}], gr.update(visible=False), gr.update(visible=False), [], 0, {}

def handle_chat(chat_history, user_message, pending_state):
    global cached_df
    chat_history = list(chat_history)
    pending = dict(pending_state) if isinstance(pending_state, dict) else {}

    if cached_df is None:
        chat_history.append({"role": "assistant", "content": "Please upload a CSV first."})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, pending

    df = cached_df.copy()

    # --- Fresh command override: if this message is explicit, drop pending
    explicit = parse_explicit(user_message)
    if explicit:
        pending = {}  # reset any old state

    # If not explicit and we had pending, continue that flow
    if not explicit and pending.get("action"):
        resolved, clarify_msg, new_pending = resolve_pending(pending, user_message, df)
        if clarify_msg:
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": clarify_msg})
            return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, new_pending
        if resolved:
            action, args = resolved
            explicit = (action, args)
        else:
            explicit = None

    tool = None; llm_output = None
    if not explicit:
        tool, llm_output = ask_llm(chat_history, user_message)
        if tool and isinstance(tool, dict):
            if tool.get("tool") == "stats":
                # already normalized
                pass

    # --- Decide action/args
    if explicit:
        action = explicit[0]; args = explicit[1]; tool_ok = True
    elif tool and tool.get("tool") == "stats":
        action = tool.get("action"); args = tool.get("args", {}); tool_ok = True
    else:
        tool_ok = False

    # --- Execute
    if tool_ok:
        sections: List[Tuple[str, str, Optional[bytes]]] = []
        plot_filepaths: List[str] = []
        preview_path: Optional[str] = None

        try:
            if action == "ttest":
                group = args.get("group")
                value = args.get("value")
                explicit_value = bool(args.get("_explicit_value", False))

                # Validate group: needs exactly 2 levels
                if not group or group not in df.columns or df[group].nunique(dropna=True) != 2:
                    bins = [c for c in categorical_cols(df) if df[c].nunique(dropna=True) == 2]
                    if not bins:
                        msg = "I don't see a binary grouping column (2 levels). Tell me which column is your group, e.g., `group=sex`."
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"ttest","await":"ttest_group","value":value}
                    if len(bins) > 1 and not group:
                        opts = ", ".join(bins[:12]) + ("…" if len(bins)>12 else "")
                        msg = f"Which column should be the 2-level group? Candidates: {opts}\nTry: `ttest value=<numeric_col> group=<one_of_above>`"
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"ttest","await":"ttest_group","value":value}
                    group = bins[0] if not group else group

                # Choose outcome
                candidates = [c for c in usable_numeric_cols(df) if c != group]
                if not value or value not in df.columns or not pd.api.types.is_numeric_dtype(df[value]):
                    if len(candidates) == 0:
                        msg = "I couldn't find a numeric outcome column different from the group. Which numeric column should I compare?"
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"ttest","await":"ttest_value","group":group}
                    if len(candidates) > 1 and not value:
                        opts = ", ".join(candidates[:12]) + ("…" if len(candidates)>12 else "")
                        msg = f"Which numeric outcome should I test? Candidates: {opts}\nExample: `ttest value={candidates[0]} group={group}`"
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"ttest","await":"ttest_value","group":group}
                    value = candidates[0] if not value else value

                out = run_ttest(df, value=value, group=group,
                                paired=bool(args.get("paired", False)),
                                equal_var=bool(args.get("equal_var", False)),
                                explicit_value=explicit_value)
                if out.table is not None:
                    out.table.to_csv(os.path.join(outputs_dir, "ttest_table.csv"), index=False)
                for title, png in (out.plots or []):
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').replace('±','pm').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                sections.append((out.kind, out.summary_text, None))
                new_pending = {}

            elif action == "anova":
                group = args.get("group")
                value = args.get("value")
                explicit_value = bool(args.get("_explicit_value", False))

                if not group or group not in df.columns or df[group].nunique(dropna=True) < 2:
                    cats = [c for c in categorical_cols(df) if df[c].nunique(dropna=True) >= 2]
                    if not cats:
                        msg = "I couldn't find a valid grouping factor (≥2 levels). Reply with `group=<column>`."
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"anova","await":"anova_group","value":value}
                    if len(cats) > 1 and not group:
                        opts = ", ".join(cats[:12]) + ("…" if len(cats)>12 else "")
                        msg = f"Which grouping factor (≥2 levels)? Candidates: {opts}\nTry: `anova value=<numeric_col> group=<factor>`"
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"anova","await":"anova_group","value":value}
                    group = cats[0] if not group else group

                nums = [c for c in usable_numeric_cols(df) if c != group]
                if not value or value not in df.columns or not pd.api.types.is_numeric_dtype(df[value]):
                    if len(nums) == 0:
                        msg = "I couldn't find a numeric outcome column different from the group. Which numeric column should I analyze?"
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"anova","await":"anova_value","group":group}
                    if len(nums) > 1 and not value:
                        opts = ", ".join(nums[:12]) + ("…" if len(nums)>12 else "")
                        msg = f"Which numeric outcome for ANOVA? Candidates: {opts}\nExample: `anova value={nums[0]} group={group}`"
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, {"action":"anova","await":"anova_value","group":group}
                    value = nums[0] if not value else value

                out = run_anova(df, value=value, group=group, explicit_value=explicit_value)
                for title, png in (out.plots or []):
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                sections.append((out.kind, out.summary_text, None))
                new_pending = {}

            elif action == "ols":
                out = smf.ols(args.get("formula"), data=df).fit()
                coef = out.summary2().tables[1].reset_index().rename(columns={"index":"term"})
                # simple visuals
                lhs, rhs = [s.strip() for s in args.get("formula").split("~", 1)]
                terms = [t.strip() for t in re.split(r"\+|:|\*", rhs) if t.strip()]
                fig_fit: Optional[bytes] = None
                if len(terms) == 1 and terms[0] in df.columns and pd.api.types.is_numeric_dtype(df[terms[0]]):
                    x = df[terms[0]]; y = df[lhs]; fig, ax = plt.subplots(figsize=(5,3))
                    ax.scatter(x, y)
                    order = np.argsort(x.values)
                    ax.plot(x.values[order], out.fittedvalues.values[order])
                    ax.set_xlabel(terms[0]); ax.set_ylabel(lhs); ax.set_title("Scatter + OLS fit")
                    fig_fit = fig_to_png(fig)
                fig_r, ax = plt.subplots(figsize=(5,3))
                ax.scatter(out.fittedvalues, out.resid); ax.axhline(0, linestyle=":")
                ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted")
                png_r = fig_to_png(fig_r)
                sm.qqplot(out.resid, line="45", fit=True)
                png_qq = fig_to_png(plt.gcf())
                header = (f"OLS Regression\nn = {int(out.nobs)}, df_model = {out.df_model:.0f}, df_resid = {out.df_resid:.0f}\n"
                          f"R² = {out.rsquared:.4g}, adj. R² = {out.rsquared_adj:.4g}\nF = {out.fvalue:.4g}, p(F) = {out.f_pvalue:.4g}")
                sections.append(("OLS", header + "\n\n" + str(out.summary()), None))
                for title, png in [("Scatter+Fit", fig_fit), ("Residuals", png_r), ("QQ", png_qq)]:
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                new_pending = {}

            elif action == "glm":
                fam_map = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(),
                           "poisson": sm.families.Poisson(), "gamma": sm.families.Gamma()}
                fam = fam_map.get(args.get("family", "gaussian"), sm.families.Gaussian())
                out = smf.glm(args.get("formula"), data=df, family=fam).fit()
                coef = out.summary2().tables[1].reset_index().rename(columns={"index":"term"})
                fig_r, ax = plt.subplots(figsize=(5,3))
                ax.scatter(out.fittedvalues, out.resid_deviance); ax.axhline(0, linestyle=":")
                ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted (GLM)")
                png_r = fig_to_png(fig_r)
                sm.qqplot(out.resid_deviance, line="45", fit=True)
                png_qq = fig_to_png(plt.gcf())
                header = (f"GLM ({args.get('family','gaussian')})\n"
                          f"n = {int(out.nobs)}, df_model = {out.df_model:.0f}, df_resid = {out.df_resid:.0f}\n"
                          f"AIC = {out.aic:.4g}, BIC = {out.bic:.4g}")
                sections.append((f"GLM ({args.get('family','gaussian')})", header + "\n\n" + str(out.summary()), None))
                for title, png in [("Residuals", png_r), ("QQ", png_qq)]:
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                new_pending = {}

            elif action == "plot":
                if "hist" in args:
                    p = args["hist"]
                    col = p.get("col")
                    if not col or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                        nums = usable_numeric_cols(df)
                        opts = ", ".join(nums[:12]) + ("…" if len(nums) > 12 else "")
                        msg = f"Which numeric column for the histogram? Candidates: {opts}"
                        chat_history.extend([{"role":"user","content":user_message},{"role":"assistant","content":msg}])
                        return (chat_history, gr.update(value=None), gr.update(value=None, visible=False),
                                [], 0, {"action":"plot","await":"hist_col","bins": int(p.get("bins",30))})
                    title, png = plot_hist(df, col, int(p.get("bins", 30)))
                    if png:
                        fp = save_png(png, f"plot_hist_{col}.png"); plot_filepaths.append(fp); preview_path = fp
                    sections.append((title, "", png))

                if "box" in args:
                    p = args["box"]; title, png = plot_box(df, p.get("value"), p.get("group"))
                    if png:
                        fp = save_png(png, f"plot_box_{p.get('value')}_{p.get('group')}.png")
                        plot_filepaths.append(fp); preview_path = fp
                    sections.append((title, "", png))

                if "violin" in args:
                    p = args["violin"]; title, png = plot_violin(df, p.get("value"), p.get("group"))
                    if png:
                        fp = save_png(png, f"plot_violin_{p.get('value')}_{p.get('group')}.png")
                        plot_filepaths.append(fp); preview_path = fp
                    sections.append((title, "", png))

                if "bar" in args:
                    p = args["bar"]
                    order = [str(x) for x in pd.unique(df[p.get("group")].dropna())] if p.get("group") in df.columns else None
                    title, png = plot_bar_with_error(df, p.get("value"), p.get("group"), p.get("error", "sem"), order=order)
                    if png:
                        fp = save_png(png, f"plot_bar_{p.get('value')}_{p.get('group')}_{p.get('error','sem')}.png")
                        plot_filepaths.append(fp); preview_path = fp
                    sections.append((title, f"Bar chart of mean {p.get('value')} by {p.get('group')} with {(p.get('error','sem')).upper()} error bars.", png))
                new_pending = {}

            elif action == "check":
                if "normality" in args:
                    p = args["normality"]; msg, png = check_normality(df, p.get("col"))
                    if png:
                        fp = save_png(png, f"plot_qq_{p.get('col')}.png")
                        plot_filepaths.append(fp); preview_path = fp
                    sections.append(("Normality", msg, png))
                if "homogeneity" in args:
                    p = args["homogeneity"]; msg = check_homogeneity(df, p.get("value"), p.get("group"))
                    sections.append(("Homogeneity of variances", msg, None))
                new_pending = {}

            elif action == "power":
                if "ttest_ind" in args:
                    p = args["ttest_ind"]
                    msg = TTestIndPower().solve_power(
                        effect_size=p.get("effect_size"),
                        power=p.get("power"),
                        alpha=float(p.get("alpha", 0.05)),
                        ratio=float(p.get("ratio", 1.0)),
                        alternative="two-sided"
                    )
                    sections.append(("Power – t-test (ind)", f"Required total sample size ≈ {int(np.ceil((1+float(p.get('ratio',1.0)))*msg))}", None))
                if "anova_oneway" in args:
                    p = args["anova_oneway"]
                    n = FTestAnovaPower().solve_power(
                        effect_size=p.get("effect_size"),
                        k_groups=int(p.get("k_groups", 3)),
                        alpha=float(p.get("alpha", 0.05)),
                        power=p.get("power")
                    )
                    sections.append(("Power – ANOVA (one-way)", f"Required n per group ≈ {int(np.ceil(n))}", None))
                new_pending = {}

            elif action == "recommend":
                tech = recommend_analyses_text(df)
                friendly = humanize_recommendations(tech, df)
                examples = recommend_examples_natural(df)
                sections.append(("Recommendations", friendly, None))
                sections.append(("Say it like this", examples, None))
                new_pending = {}

            else:
                sections.append(("Note", f"Unknown action: {action}", None))
                new_pending = {}

        except Exception as e:
            sections.append(("Error", str(e), None))
            new_pending = {}

        chat_summary = "\n\n".join([f"## {title}\n{txt}" for (title, txt, _) in sections if txt])
        _ = write_report(sections)
        zip_fp = save_zip()
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": chat_summary or "Done. See preview and use Download Results."})
        if not plot_filepaths:
            return chat_history, gr.update(value=None), gr.update(value=zip_fp, visible=True), [], 0, new_pending
        return chat_history, gr.update(value=preview_path), gr.update(value=zip_fp, visible=True), plot_filepaths, len(plot_filepaths) - 1, new_pending

    # Not a tool call → natural reply
    if llm_output:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": llm_output})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, pending
    else:
        chat_history.append({"role": "assistant", "content": "Try: `ttest value=height group=sex` (auto Bar±SEM, Box, Violin), `anova value=score group=treatment`, `ols score ~ weight`, `plot hist col=height`, or ask **“what can I do with my data?”**"})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, pending

# ---------- PLOT NAV ----------
def show_prev(plots: List[str], idx: int):
    if not plots:
        return gr.update(value=None), 0
    new_idx = (idx - 1) % len(plots)
    return gr.update(value=plots[new_idx]), new_idx

def show_next(plots: List[str], idx: int):
    if not plots:
        return gr.update(value=None), 0
    new_idx = (idx + 1) % len(plots)
    return gr.update(value=plots[new_idx]), new_idx

# ---------- UI ----------
with gr.Blocks(title="SpatChat: Stats Room") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML("""
    <style>
    #logo-img img { height: 90px; margin: 10px 50px 10px 10px; border-radius: 6px; }
    </style>
    """)
    gr.Markdown("## 📊 SpatChat: Stats Room {stats}")
    gr.HTML("""
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=stats" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">📋 Copy Share Link</button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=stats" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=stats" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """)
    gr.Markdown("""
        <div style="font-size: 14px;">
        © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
        If you use Spatchat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Stats Room.</i>
        </div>
    """)

    plots_state = gr.State([])   # list of plot filepaths
    plot_index  = gr.State(0)    # current index
    pending_state = gr.State({}) # for clarifications

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role":"assistant","content":"Welcome! Upload a CSV, then ask: t-test (auto Bar±SEM, Box, Violin), ANOVA, OLS/GLM, hist/box/violin/bar, normality, power — or say **“what can I do with my data?”**"}]
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder='e.g., "I want a t-test on height by sex" | "plot histogram of height" | "ols score ~ weight"', lines=1)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        with gr.Column(scale=3):
            with gr.Row():
                preview_plot = gr.Image(label="Preview (navigate plots)", type="filepath")
            with gr.Row():
                prev_btn = gr.Button("◀ Prev plot", variant="secondary")
                next_btn = gr.Button("Next plot ▶", variant="secondary")
            data_preview = gr.Dataframe(label="Data Preview (first 200 rows)", interactive=False, wrap=True)
            download_btn = gr.DownloadButton("📥 Download Results", value=None, visible=False)

    file_input.change(
        handle_upload,
        inputs=file_input,
        outputs=[chatbot, data_preview, download_btn, plots_state, plot_index, pending_state]
    )
    user_input.submit(
        handle_chat,
        inputs=[chatbot, user_input, pending_state],
        outputs=[chatbot, preview_plot, download_btn, plots_state, plot_index, pending_state]
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

    prev_btn.click(show_prev, inputs=[plots_state, plot_index], outputs=[preview_plot, plot_index])
    next_btn.click(show_next, inputs=[plots_state, plot_index], outputs=[preview_plot, plot_index])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
