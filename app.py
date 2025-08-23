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

# By default we DO NOT humanize with LLM to avoid surprises
SPATCHAT_HUMANIZE_RECS = os.getenv("SPATCHAT_HUMANIZE_RECS", "0") == "1"
SPATCHAT_HUMANIZE_RESULTS = os.getenv("SPATCHAT_HUMANIZE_RESULTS", "0") == "1"

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

# ---------- PROMPTS (kept minimal) ----------
SYSTEM_PROMPT = """
Return JSON only for tool calls:
{"tool":"stats","action":"ttest|anova|ols|glm|plot|check|power|recommend|desc","args":{...}}
Args (non-exhaustive):
- ttest {"value":"Y","group":"G","paired":false,"equal_var":false}
- anova {"value":"Y","group":"G"}
- ols {"formula":"y ~ x1 + x2"}
- glm {"formula":"y ~ x1 + x2","family":"gaussian|binomial|poisson|gamma"}
- plot {"hist":{"col":"c","bins":30}|"box":{"value":"Y","group":"G"}|"violin":{"value":"Y","group":"G"}|"bar":{"value":"Y","group":"G","error":"sem|sd|ci95"}}
- check {"normality":{"col":"Y"}|"homogeneity":{"value":"Y","group":"G"}}
- power {"ttest_ind":{...}} | {"anova_oneway":{...}}
- recommend {}
- desc {"op":"mean|median|sd|min|max|range|mode|freq|summary","col":"Y","group":"G?"}
Only use prose (≤3 sentences) when not a tool call.
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

# ---------- ID-like detection ----------
ID_NAME_PAT = re.compile(r"(?:^|[_\W])(id|identifier|index|subject|animal|record|row)(?:[_\W]|$)", re.I)

def is_id_like(df: pd.DataFrame, col: str) -> bool:
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
        if len(u) > 4 and int(u[-1] - u[0] + 1) == len(u):
            return True
    return False

def usable_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in numeric_cols(df) if not is_id_like(df, c)]

# ---------- RECOMMENDATIONS (deterministic, no LLM) ----------
def infer_schema(df: pd.DataFrame) -> Dict[str, List[str] or int]:
    nums_all = numeric_cols(df)
    nums = usable_numeric_cols(df)
    cats = []
    for c in df.columns:
        if c in nums_all:
            nunq = df[c].nunique(dropna=True)
            if 2 <= nunq <= 10:
                cats.append(c)
        else:
            cats.append(c)
    seen = set()
    cats = [c for c in cats if not (c in seen or seen.add(c))]
    bins = [c for c in cats if df[c].nunique(dropna=True) == 2]
    y_bin = [c for c in df.columns if (pd.api.types.is_bool_dtype(df[c]) or _is_binary_series(df[c]))]
    y_count = [c for c in nums if _is_count_series(df[c]) and df[c].nunique(dropna=True) > 2]
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "numeric_all": nums_all,
        "numeric_usable": nums,
        "categorical": cats,
        "binary_groups": bins,
        "y_bin": y_bin,
        "y_count": y_count,
    }

def recommend_text_and_examples(df: pd.DataFrame) -> Tuple[str, str]:
    info = infer_schema(df)
    n, p = info["n_rows"], info["n_cols"]
    nums = info["numeric_usable"]
    nums_all = info["numeric_all"]
    cats = info["categorical"]
    bins = info["binary_groups"]
    y_bin, y_count = info["y_bin"], info["y_count"]

    overview_lines = [f"Dataset overview: n={n} rows, p={p} columns."]
    if nums_all:
        overview_lines.append(f"- Numeric columns ({len(nums_all)}): {', '.join(map(str, nums_all[:8]))}{'…' if len(nums_all)>8 else ''}")
    if cats:
        overview_lines.append(f"- Categorical/low-cardinality ({len(cats)}): {', '.join(map(str, cats[:8]))}{'…' if len(cats)>8 else ''}")

    # Choose sensible defaults that are NOT ID-like
    y_num = nums[0] if nums else (nums_all[0] if nums_all else None)
    g_bin = bins[0] if bins else (cats[0] if cats else None)
    g_multi = next((c for c in cats if df[c].nunique(dropna=True) >= 3), (cats[0] if cats else None))

    rec_lines = ["Here are some analysis ideas you can run next:"]
    if y_num:
        rec_lines.append(f"• **Quick summary** of {y_num} (mean, sd, min/max).")
        rec_lines.append(f"• **Histogram** to see the shape of {y_num}.")
    if y_num and g_bin:
        rec_lines.append(f"• **t-test** to compare average {y_num} between the two groups in {g_bin}.")
    if y_num and g_multi and g_multi != g_bin:
        rec_lines.append(f"• **One-way ANOVA** to compare {y_num} across levels of {g_multi}.")
    if len(nums) >= 2:
        rec_lines.append(f"• **Linear regression (OLS)** to see how one variable predicts another (e.g., {nums[0]} ~ {nums[1]}).")
    if y_bin and len(nums) >= 1:
        x = nums[0] if nums[0] != y_bin[0] else (nums[1] if len(nums) > 1 else None)
        if x:
            rec_lines.append(f"• **Logistic regression** if your outcome is yes/no (e.g., {y_bin[0]} ~ {x}).")
    if y_count and len(nums) >= 1:
        x = nums[0] if nums[0] != y_count[0] else (nums[1] if len(nums) > 1 else None)
        if x:
            rec_lines.append(f"• **Poisson regression** if your outcome is a count (e.g., {y_count[0]} ~ {x}).")
    if y_num:
        rec_lines.append(f"• **Normality check** of {y_num} with a QQ plot.")
    if g_bin and y_num:
        rec_lines.append(f"• **Bar chart with error bars** to compare mean {y_num} by {g_bin}.")
    if g_multi and y_num:
        rec_lines.append(f"• **Box/violin plots** to visualize {y_num} across {g_multi}.")
    rec_lines.append("• **Power analysis** to see how many samples you need to detect a meaningful difference.")

    # Natural-language example prompts users can literally copy
    ex_lines = ["Here are some things you can say:"]
    if y_num:
        ex_lines.append(f'• "What is the average {y_num}?"')
        ex_lines.append(f'• "Show a histogram of {y_num}."')
    if y_num and g_bin:
        ex_lines.append(f'• "I want to do a t-test on {y_num} by {g_bin}."')
        ex_lines.append(f'• "Make a bar chart of {y_num} by {g_bin} with 95% CI."')
    if y_num and g_multi and g_multi != g_bin:
        ex_lines.append(f'• "Run a one-way ANOVA of {y_num} by {g_multi}."')
        ex_lines.append(f'• "Show box and violin plots for {y_num} by {g_multi}."')
    if len(nums) >= 2:
        ex_lines.append(f'• "Fit a linear regression: {nums[0]} ~ {nums[1]}."')
    if y_bin and len(nums) >= 1:
        x = nums[0] if nums[0] != y_bin[0] else (nums[1] if len(nums) > 1 else None)
        if x:
            ex_lines.append(f'• "Do a logistic regression: {y_bin[0]} ~ {x}."')
    if y_count and len(nums) >= 1:
        x = nums[0] if nums[0] != y_count[0] else (nums[1] if len(nums) > 1 else None)
        if x:
            ex_lines.append(f'• "Fit a Poisson regression for {y_count[0]} using {x}."')
    if y_num:
        ex_lines.append(f'• "Check normality of {y_num} and show a QQ plot."')
    ex_lines.append('• "Power analysis for a t-test with 80% power and effect size 0.5."')
    ex_lines.append('• "Power analysis for a one-way ANOVA with 3 groups and 80% power."')

    tech = "\n".join(overview_lines + [""] + rec_lines)
    examples = "\n".join(ex_lines)
    return tech, examples

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

    # Box (consistent order)
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

    # Bar ± SEM
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

# ---------- DESCRIPTIVES ----------
def format_table_for_chat(df: pd.DataFrame, max_rows: int = 30) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
    # Avoid dependency on tabulate; use plain text
    return df.to_string(index=False)

def run_desc(df: pd.DataFrame, op: str, col: Optional[str], group: Optional[str]) -> ModelOutput:
    op = op.lower()
    # Summary of dataset or a single column
    if op in {"summary", "describe"}:
        if col and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna().astype(float)
                res = pd.DataFrame({
                    "count": [len(s)],
                    "mean": [s.mean()],
                    "sd": [s.std(ddof=1)],
                    "min": [s.min()],
                    "25%": [np.percentile(s,25)],
                    "50% (median)": [np.percentile(s,50)],
                    "75%": [np.percentile(s,75)],
                    "max": [s.max()],
                })
                text = f"Summary of `{col}`:\n\n" + format_table_for_chat(res)
                return ModelOutput("Summary", text)
            else:
                vc = df[col].astype(str).value_counts(dropna=True).reset_index()
                vc.columns = [col, "count"]
                text = f"Frequencies for `{col}`:\n\n" + format_table_for_chat(vc)
                return ModelOutput("Summary", text)
        else:
            nums = usable_numeric_cols(df)
            if nums:
                desc = df[nums].describe().T.reset_index().rename(columns={"index":"column"})
                text = "Summary of numeric columns:\n\n" + format_table_for_chat(desc)
            else:
                text = "No numeric columns found for summary."
            return ModelOutput("Summary", text)

    # Single metric (optionally by group)
    if not col or col not in df.columns:
        raise gr.Error("Please specify a valid column for this descriptive request.")

    if group and group in df.columns:
        if op in {"mean","average","avg"}:
            out = df.groupby(group)[col].mean(numeric_only=True).reset_index(name="mean")
            txt = f"Mean of `{col}` by `{group}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)
        if op in {"median"}:
            out = df.groupby(group)[col].median(numeric_only=True).reset_index(name="median")
            txt = f"Median of `{col}` by `{group}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)
        if op in {"sd","stdev","std","standard deviation"}:
            out = df.groupby(group)[col].std(ddof=1, numeric_only=True).reset_index(name="sd")
            txt = f"Standard deviation of `{col}` by `{group}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)
        if op in {"min"}:
            out = df.groupby(group)[col].min(numeric_only=True).reset_index(name="min")
            txt = f"Minimum of `{col}` by `{group}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)
        if op in {"max"}:
            out = df.groupby(group)[col].max(numeric_only=True).reset_index(name="max")
            txt = f"Maximum of `{col}` by `{group}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)
        if op in {"range"}:
            tmp = df.groupby(group)[col].agg(["min","max"]).reset_index()
            tmp["range"] = tmp["max"] - tmp["min"]
            txt = f"Range of `{col}` by `{group}`:\n\n" + format_table_for_chat(tmp[[group,"min","max","range"]])
            return ModelOutput("Descriptives", txt)
        if op in {"mode","modes"}:
            # compute per group top mode
            parts = []
            for g, gdf in df.groupby(group):
                s = gdf[col].dropna()
                m = s.mode()
                if m.empty:
                    parts.append({"group": g, "mode": None, "count": 0})
                else:
                    top = m.iloc[0]
                    parts.append({"group": g, "mode": top, "count": (s==top).sum()})
            out = pd.DataFrame(parts)
            txt = f"Mode of `{col}` by `{group}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)
        if op in {"freq","frequency","frequencies"}:
            if pd.api.types.is_numeric_dtype(df[col]):
                # bin numeric into 10 bins per group
                parts = []
                for g, gdf in df.groupby(group):
                    s = gdf[col].dropna().astype(float)
                    if len(s)==0:
                        continue
                    bins = np.histogram_bin_edges(s, bins="auto")
                    counts, edges = np.histogram(s, bins=bins)
                    for i, cnt in enumerate(counts):
                        parts.append({"group":g, "bin":[edges[i], edges[i+1]], "count":int(cnt)})
                out = pd.DataFrame(parts)
            else:
                out = df.groupby([group, col]).size().reset_index(name="count")
            txt = f"Frequencies of `{col}` by `{group}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)
    else:
        # No grouping
        if op in {"mean","average","avg"}:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            txt = f"Mean of `{col}` = {s.mean():.4g} (n={len(s)})"
            return ModelOutput("Descriptives", txt)
        if op in {"median"}:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            txt = f"Median of `{col}` = {np.median(s):.4g} (n={len(s)})"
            return ModelOutput("Descriptives", txt)
        if op in {"sd","stdev","std","standard deviation"}:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            txt = f"Standard deviation of `{col}` = {s.std(ddof=1):.4g} (n={len(s)})"
            return ModelOutput("Descriptives", txt)
        if op in {"min"}:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            txt = f"Minimum of `{col}` = {s.min():.4g}"
            return ModelOutput("Descriptives", txt)
        if op in {"max"}:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            txt = f"Maximum of `{col}` = {s.max():.4g}"
            return ModelOutput("Descriptives", txt)
        if op in {"range"}:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            txt = f"Range of `{col}` = {s.max()-s.min():.4g} (min={s.min():.4g}, max={s.max():.4g})"
            return ModelOutput("Descriptives", txt)
        if op in {"mode","modes"}:
            s = df[col].dropna()
            m = s.mode()
            if m.empty:
                txt = f"No mode for `{col}`."
            else:
                top = m.iloc[0]
                txt = f"Mode of `{col}` = {top} (count={(s==top).sum()})"
            return ModelOutput("Descriptives", txt)
        if op in {"freq","frequency","frequencies"}:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna().astype(float)
                bins = np.histogram_bin_edges(s, bins="auto")
                counts, edges = np.histogram(s, bins=bins)
                out = pd.DataFrame({"bin_start":edges[:-1], "bin_end":edges[1:], "count":counts})
                txt = f"Frequencies (binned) for `{col}`:\n\n" + format_table_for_chat(out)
            else:
                out = df[col].astype(str).value_counts(dropna=True).reset_index()
                out.columns = [col, "count"]
                txt = f"Frequencies for `{col}`:\n\n" + format_table_for_chat(out)
            return ModelOutput("Descriptives", txt)

    raise gr.Error("I didn't recognize that descriptive request. Try: 'What's the average height?' or 'Median score by sex'.")

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
                      "`ttest value=height group=sex` or ask natural questions like `What is the average height?`.")
    client = _get_llm_client()
    if client is None:
        return None, ("TOGETHER_API_KEY not set. You can still run explicit commands (e.g. "
                      "`ttest value=score group=sex`) and natural queries like `mean height by sex`.")
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

def _extract_by_group(text: str) -> Optional[str]:
    m = re.search(r"\bby\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.I)
    return m.group(1) if m else None

def parse_explicit(user_message: str) -> Optional[Tuple[str, dict]]:
    s = user_message.lower()

    # Recommend
    if re.search(r"\b(what can i do|how (should|to) (i )?analy[sz]e|what (analys(e|es)|tests?) should i do|recommend(ation)?s|help analy[sz]e|suggest (analy|tests?))\b", s):
        return ("recommend", {})

    # -------- Descriptive stats (highest priority so they don't fall into OLS) --------
    # mean / average
    if re.search(r"\b(mean|average|avg)\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "mean", "col": col, "group": group})
    # median
    if re.search(r"\bmedian\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "median", "col": col, "group": group})
    # sd / stdev / standard deviation
    if re.search(r"\b(sd|stdev|std|standard\s+deviation)\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "sd", "col": col, "group": group})
    # min / max / range
    if re.search(r"\bmin(imum)?\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "min", "col": col, "group": group})
    if re.search(r"\bmax(imum)?\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "max", "col": col, "group": group})
    if re.search(r"\brange\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "range", "col": col, "group": group})
    # mode
    if re.search(r"\bmode(s)?\b", s) and not re.search(r"\bmodel\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "mode", "col": col, "group": group})
    # freq / distribution
    if re.search(r"\b(freq(uenc(y|ies))?|distribution)\b", s) or re.search(r"\bcount(s)?\s+by\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        group = _extract_by_group(user_message)
        return ("desc", {"op": "freq", "col": col, "group": group})
    # summary / describe
    if re.search(r"\b(summary|describe|descriptive statistics|basic stats)\b", s):
        col = _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        return ("desc", {"op": "summary", "col": col, "group": None})

    # -------- Plots --------
    if re.search(r"\bhist(?:ogram)?\b", s):
        m = re.search(r"(?:col(?:umn)?|of|on|for)\s*=?\s*([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        col = m.group(1) if m else _find_first_col_mention(user_message, cached_df) if cached_df is not None else None
        bins_m = re.search(r"bins\s*=\s*(\d+)", s)
        bins = int(bins_m.group(1)) if bins_m else 30
        return ("plot", {"hist": {"col": col, "bins": bins}})
    if re.search(r"\bbox(?:plot)?\b", s):
        v = re.search(r"(?:box(?:plot)?\s+(?:of|on|for)\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        g = re.search(r"(?:by\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        return ("plot", {"box": {"value": (v and v.group(1)) or _find_first_col_mention(user_message, cached_df), "group": g and g.group(1)}})
    if re.search(r"\bviolin\b", s):
        v = re.search(r"(?:violin\s+(?:of|on|for)\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        g = re.search(r"(?:by\s+)([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        return ("plot", {"violin": {"value": (v and v.group(1)) or _find_first_col_mention(user_message, cached_df), "group": g and g.group(1)}})
    if re.search(r"\bbar\s*(plot|chart)?\b", s):
        value = None; group = _extract_by_group(user_message); error = None
        m = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message);  value = m and m.group(1)
        m = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message);  group = m.group(1) if m else group
        m = re.search(r"error\s*=\s*(sem|sd|ci95)", s);               error = m and m.group(1)
        if value is None:
            m = re.search(r"\bbar(?:\s*(?:plot|chart))?\s+(?:of|on|for)\s+([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
            value = m and m.group(1)
        return ("plot", {"bar": {"value": value or _find_first_col_mention(user_message, cached_df), "group": group, "error": error or "sem"}})

    # -------- Inference tests/models --------
    if re.search(r"\bt[-\s]?test\b|\bttest\b", s):
        v = re.search(r"(?:t[-\s]?test|ttest)\s+(?:of|on|for)\s+([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        g = _extract_by_group(user_message)
        return ("ttest", {"value": (v and v.group(1)) or _find_first_col_mention(user_message, cached_df), "group": g,
                          "_explicit_value": bool(v), "_explicit_group": bool(g)})
    if re.search(r"\banova\b", s):
        v = re.search(r"(?:anova)\s+(?:of|on|for)\s+([A-Za-z_][A-Za-z0-9_]*)", user_message, re.I)
        g = _extract_by_group(user_message)
        return ("anova", {"value": (v and v.group(1)) or _find_first_col_mention(user_message, cached_df), "group": g,
                          "_explicit_value": bool(v), "_explicit_group": bool(g)})

    # OLS/GLM left to LLM or explicit syntax (e.g., "ols y ~ x")
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
            [{"role": "assistant", "content": f"CSV uploaded. Columns detected: {cols}. Ask me for mean/median/sd, t-test, ANOVA, OLS/GLM, hist/box/violin/bar, normality, power — or say **'what can I do with my data?'**"}],
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

    # Fresh explicit command overrides any pending state
    explicit = parse_explicit(user_message)
    if explicit:
        pending = {}

    # If no explicit and we had pending, continue that flow
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

    # Decide action
    if explicit:
        action = explicit[0]; args = explicit[1]; tool_ok = True
    elif tool and tool.get("tool") == "stats":
        action = tool.get("action"); args = tool.get("args", {}); tool_ok = True
    else:
        tool_ok = False

    if tool_ok:
        sections: List[Tuple[str, str, Optional[bytes]]] = []
        plot_filepaths: List[str] = []
        preview_path: Optional[str] = None

        try:
            if action == "desc":
                op = args.get("op")
                col = args.get("col")
                group = args.get("group")
                # Auto-pick sensible defaults when missing
                if op in {"mean","median","sd","stdev","std","standard deviation","min","max","range","mode","freq","frequency","frequencies"} and (not col or col not in df.columns):
                    nums = usable_numeric_cols(df)
                    col = nums[0] if nums else numeric_cols(df)[0] if numeric_cols(df) else None
                out = run_desc(df, op, col, group)
                sections.append((out.kind, out.summary_text, None))
                new_pending = {}

            elif action == "ttest":
                group = args.get("group")
                value = args.get("value")
                explicit_value = bool(args.get("_explicit_value", False))

                # Group must be 2 levels
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
                        msg = f"Which numeric outcome should I test? Candidates: {opts}\nExample: `I want a t-test on {candidates[0]} by {group}`"
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
                        msg = f"Which numeric outcome for ANOVA? Candidates: {opts}\nExample: `Run ANOVA of {nums[0]} by {group}`"
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
                model = smf.ols(args.get("formula"), data=df).fit()
                lhs, rhs = [s.strip() for s in args.get("formula").split("~", 1)]
                terms = [t.strip() for t in re.split(r"\+|:|\*", rhs) if t.strip()]
                fig_fit: Optional[bytes] = None
                if len(terms) == 1 and terms[0] in df.columns and pd.api.types.is_numeric_dtype(df[terms[0]]):
                    x = df[terms[0]]; y = df[lhs]; fig, ax = plt.subplots(figsize=(5,3))
                    ax.scatter(x, y)
                    order = np.argsort(x.values)
                    ax.plot(x.values[order], model.fittedvalues.values[order])
                    ax.set_xlabel(terms[0]); ax.set_ylabel(lhs); ax.set_title("Scatter + OLS fit")
                    fig_fit = fig_to_png(fig)
                fig_r, ax = plt.subplots(figsize=(5,3))
                ax.scatter(model.fittedvalues, model.resid); ax.axhline(0, linestyle=":")
                ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted")
                png_r = fig_to_png(fig_r)
                sm.qqplot(model.resid, line="45", fit=True)
                png_qq = fig_to_png(plt.gcf())
                header = (f"OLS Regression\nn = {int(model.nobs)}, df_model = {model.df_model:.0f}, df_resid = {model.df_resid:.0f}\n"
                          f"R² = {model.rsquared:.4g}, adj. R² = {model.rsquared_adj:.4g}\nF = {model.fvalue:.4g}, p(F) = {model.f_pvalue:.4g}")
                sections.append(("OLS", header + "\n\n" + str(model.summary()), None))
                for title, png in [("Scatter+Fit", fig_fit), ("Residuals", png_r), ("QQ", png_qq)]:
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                new_pending = {}

            elif action == "glm":
                fam_map = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(),
                           "poisson": sm.families.Poisson(), "gamma": sm.families.Gamma()}
                fam = fam_map.get(args.get("family", "gaussian"), sm.families.Gaussian())
                model = smf.glm(args.get("formula"), data=df, family=fam).fit()
                fig_r, ax = plt.subplots(figsize=(5,3))
                ax.scatter(model.fittedvalues, model.resid_deviance); ax.axhline(0, linestyle=":")
                ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted (GLM)")
                png_r = fig_to_png(fig_r)
                sm.qqplot(model.resid_deviance, line="45", fit=True)
                png_qq = fig_to_png(plt.gcf())
                header = (f"GLM ({args.get('family','gaussian')})\n"
                          f"n = {int(model.nobs)}, df_model = {model.df_model:.0f}, df_resid = {model.df_resid:.0f}\n"
                          f"AIC = {model.aic:.4g}, BIC = {model.bic:.4g}")
                sections.append((f"GLM ({args.get('family','gaussian')})", header + "\n\n" + str(model.summary()), None))
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
                    tool = TTestIndPower()
                    ratio = float(p.get("ratio", 1.0))
                    es = p.get("effect_size", 0.5)
                    pw = p.get("power", 0.8)
                    alpha = float(p.get("alpha", 0.05))
                    n1 = tool.solve_power(effect_size=es, power=pw, alpha=alpha, ratio=ratio, alternative="two-sided")
                    total = int(np.ceil((1+ratio)*n1))
                    msg = f"Required total sample size (two-sample t-test): ≈ {total} (group1 ≈ {int(np.ceil(n1))}, group2 ≈ {int(np.ceil(n1*ratio))})"
                    sections.append(("Power – t-test (ind)", msg, None))
                if "anova_oneway" in args:
                    p = args["anova_oneway"]
                    tool = FTestAnovaPower()
                    es = p.get("effect_size", 0.25)
                    k = int(p.get("k_groups", 3))
                    pw = p.get("power", 0.8)
                    alpha = float(p.get("alpha", 0.05))
                    n = tool.solve_power(effect_size=es, k_groups=k, alpha=alpha, power=pw)
                    msg = f"Required n per group (one-way ANOVA): ≈ {int(np.ceil(n))}"
                    sections.append(("Power – ANOVA (one-way)", msg, None))
                new_pending = {}

            elif action == "recommend":
                tech, examples = recommend_text_and_examples(df)
                sections.append(("Recommendations", tech, None))
                sections.append(("Say it like this", examples, None))
                new_pending = {}

            else:
                sections.append(("Note", f"Unknown action: {action}", None))
                new_pending = {}

        except Exception as e:
            sections.append(("Error", str(e), None))
            new_pending = {}

        chat_body = "\n\n".join([f"## {title}\n{txt}" for (title, txt, _) in sections if txt])
        _ = write_report(sections)
        zip_fp = save_zip()
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": chat_body or "Done. See preview and use Download Results."})
        if not plot_filepaths:
            return chat_history, gr.update(value=None), gr.update(value=zip_fp, visible=True), [], 0, new_pending
        return chat_history, gr.update(value=preview_path), gr.update(value=zip_fp, visible=True), plot_filepaths, len(plot_filepaths) - 1, new_pending

    # Not a tool call → natural fallback
    if llm_output:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": llm_output})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0, pending
    else:
        chat_history.append({"role": "assistant", "content": "Try: “What is the average height?”, “t-test on score by sex”, “Run ANOVA of height by group”, “plot histogram of weight”, or ask **“what can I do with my data?”**"})
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
                value=[{"role":"assistant","content":"Welcome! Upload a CSV, then ask: mean/median/sd, t-test (auto Bar±SEM, Box, Violin), ANOVA, OLS/GLM, histogram/box/violin/bar, normality, power — or say **“what can I do with my data?”**"}]
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder='e.g., "What is the average height?" | "t-test on score by sex" | "plot histogram of weight" | "ols score ~ weight"', lines=1)
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
