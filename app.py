import os
import io
import re
import json
import zipfile
import shutil
import base64
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

# Optional Together client (lazy init below so missing key won't crash import)
try:
    from together import Together
except Exception:
    Together = None  # type: ignore

print("Starting SpatChat: Stats Room (chat-first)")

# ---------- GLOBAL STATE ----------
cached_df: Optional[pd.DataFrame] = None
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# ---------- LLM SETUP ----------
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
_llm_client = None  # lazy

SYSTEM_PROMPT = """
You are SpatChat, an expert statistics assistant for basic analyses.
When the user asks for an analysis, respond ONLY in compact JSON using this schema:
{"tool":"stats","action":"ttest|anova|ols|glm|plot|check|power","args":{...}}

Actions and args:
- ttest: {"value":"colY","group":"colG", "paired": false, "equal_var": false}
- anova: {"value":"colY","group":"colG"}
- ols: {"formula":"y ~ x1 + x2"}
- glm: {"formula":"y ~ x1 + x2", "family":"gaussian|binomial|poisson|gamma"}
- plot: one of {"hist":{"col":"c","bins":30}, "box":{"value":"y","group":"g"}, "violin":{"value":"y","group":"g"}, "bar":{"value":"y","group":"g","error":"sem|sd|ci95"}}
- check: {"normality":{"col":"y"}} or {"homogeneity":{"value":"y","group":"g"}}
- power: one of
   {"ttest_ind": {"effect_size": 0.5, "alpha": 0.05, "power": 0.8, "ratio": 1.0, "solve_for":"n_total|power|effect_size"}}
   {"anova_oneway": {"effect_size": 0.25, "k_groups": 3, "alpha": 0.05, "power": 0.8, "solve_for":"n_per_group|power|effect_size"}}
If the user mentions a formula like y ~ x + z, emit action="ols" (or glm if they say logistic/poisson).
If something is unclear, make a best guess and include minimal defaults.
NEVER include explanatory prose; JSON ONLY for tool calls.
For general questions (not a tool call), answer in 1–3 sentences of plain text.
"""

FALLBACK_PROMPT = "You are SpatChat, a concise statistics tutor. If you can't map to a tool call, answer naturally in <=3 sentences."

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

def add_to_report(parts: List[str], title: str, html_fragment: str):
    parts.append(f"<h2>{title}</h2>\n{html_fragment}")

def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def categorical_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

def guess_ttest_args(df: pd.DataFrame, value: Optional[str], group: Optional[str]) -> Tuple[str, str]:
    g = group
    if g is None:
        for cand in ["sex", "group", "am"]:
            if cand in df.columns:
                g = cand; break
        if g is None:
            for cand in df.columns:
                if len(df[cand].dropna().unique()) == 2:
                    g = cand; break
    if g is None:
        raise gr.Error("Couldn't infer group column for t-test. Specify group=<column>.")
    v = value
    if v is None:
        if "score" in df.columns and pd.api.types.is_numeric_dtype(df["score"]):
            v = "score"
        else:
            for cand in numeric_cols(df):
                if cand != g:
                    v = cand; break
    if v is None:
        raise gr.Error("Couldn't infer numeric value column for t-test. Specify value=<column>.")
    return v, g

def guess_anova_args(df: pd.DataFrame, value: Optional[str], group: Optional[str]) -> Tuple[str, str]:
    g = group
    if g is None:
        for cand in ["group", "treatment"]:
            if cand in df.columns:
                g = cand; break
        if g is None:
            for cand in categorical_cols(df):
                if len(df[cand].dropna().unique()) >= 2:
                    g = cand; break
    if g is None:
        raise gr.Error("Couldn't infer group factor for ANOVA. Specify group=<column>.")
    v = value
    if v is None:
        if "score" in df.columns and pd.api.types.is_numeric_dtype(df["score"]):
            v = "score"
        else:
            for cand in numeric_cols(df):
                if cand != g:
                    v = cand; break
    if v is None:
        raise gr.Error("Couldn't infer numeric value for ANOVA. Specify value=<column>.")
    return v, g

def guess_bar_args(df: pd.DataFrame, value: Optional[str], group: Optional[str]) -> Tuple[str, str]:
    return guess_anova_args(df, value, group)

# ---------- ANALYSES (R-like summaries) ----------

@dataclass
class ModelOutput:
    kind: str
    summary_text: str
    table: Optional[pd.DataFrame] = None
    plots: Optional[List[Tuple[str, Optional[bytes]]]] = None  # (title, png_bytes)

def _ttest_summary(a: np.ndarray, b: np.ndarray, paired: bool, equal_var: bool, g1: str, g2: str) -> str:
    alpha = 0.05
    mean1, mean2 = a.mean(), b.mean()
    n1, n2 = len(a), len(b)
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
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
    diff = mean1 - mean2
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

def run_ttest(df: pd.DataFrame, value: str, group: str, paired=False, equal_var=False) -> ModelOutput:
    gvals = df[group].dropna().unique().tolist()
    if len(gvals) != 2:
        raise gr.Error("t-test requires exactly two groups in the group column.")
    a = df[df[group] == gvals[0]][value].dropna().astype(float).values
    b = df[df[group] == gvals[1]][value].dropna().astype(float).values
    # run test (primary numbers come from SciPy; summary adds CI/df/effect sizes)
    _ = stats.ttest_rel(a, b, nan_policy="omit") if paired else stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")

    # --- Plots: Bar±SEM + Box + Violin (auto for t-test) ---
    # Box
    fig_box, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot([a, b], tick_labels=[str(gvals[0]), str(gvals[1])]); ax.set_title("Boxplot by Group")
    png_box = fig_to_png(fig_box)
    # Violin
    fig_vio, ax = plt.subplots(figsize=(5, 3))
    ax.violinplot([a, b], showmeans=True); ax.set_xticks([1, 2]); ax.set_xticklabels([str(gvals[0]), str(gvals[1])]); ax.set_title("Violin by Group")
    png_vio = fig_to_png(fig_vio)
    # Bar ± SEM
    dfg = pd.DataFrame({group: [gvals[0]] * len(a) + [gvals[1]] * len(b),
                        value: np.concatenate([a, b])})
    stats_tbl = dfg.groupby(group)[value].agg(['mean', 'std', 'count']).reset_index()
    means = stats_tbl['mean'].values; sds = stats_tbl['std'].values; ns = stats_tbl['count'].values.astype(float)
    sem = sds / np.sqrt(np.where(ns > 0, ns, np.nan)); sem = np.nan_to_num(sem, nan=0.0)
    labels = stats_tbl[group].astype(str).tolist(); x = np.arange(len(labels))
    fig_bar, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, means, yerr=sem, capsize=6); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f"Mean {value}"); ax.set_title(f"{value} by {group} (error: SEM)")
    png_bar = fig_to_png(fig_bar)

    table = pd.DataFrame({
        "group": [str(gvals[0]), str(gvals[1])],
        "n": [len(a), len(b)],
        "mean": [a.mean(), b.mean()],
        "sd": [a.std(ddof=1), b.std(ddof=1)]
    })
    summary = _ttest_summary(a, b, paired=paired, equal_var=equal_var, g1=str(gvals[0]), g2=str(gvals[1]))
    return ModelOutput("t-test", summary, table, [("Bar±SEM", png_bar), ("Boxplot", png_box), ("Violin", png_vio)])

def run_anova(df: pd.DataFrame, value: str, group: str) -> ModelOutput:
    groups = [g[value].dropna().astype(float).values for _, g in df[[group, value]].dropna().groupby(group)]
    labels = [str(k) for k in df[group].dropna().unique().tolist()]
    if len(groups) < 2: raise gr.Error("ANOVA requires at least two groups.")
    F, p = stats.f_oneway(*groups)
    ns = [len(g) for g in groups]; means = [g.mean() for g in groups]
    overall = np.concatenate(groups).mean()
    SSB = sum(n*(m-overall)**2 for n,m in zip(ns,means)); SSW = sum(((g-m)**2).sum() for g,m in zip(groups,means)); SST = SSB+SSW
    k = len(groups); N = sum(ns); df1, df2 = k-1, N-k; eta2 = SSB/SST if SST>0 else np.nan
    # Box
    fig_box, ax = plt.subplots(figsize=(5, 3)); df.boxplot(column=value, by=group, ax=ax)
    ax.set_title("Boxplot by Group"); ax.figure.suptitle(""); png_box = fig_to_png(fig_box)
    # Violin
    fig_vio, ax = plt.subplots(figsize=(5, 3)); ax.violinplot(groups, showmeans=True)
    ax.set_xticks(range(1,len(groups)+1)); ax.set_xticklabels(labels); ax.set_title("Violin by Group")
    png_vio = fig_to_png(fig_vio)
    means_table = pd.DataFrame({"group": labels, "n": ns, f"mean_{value}": means})
    summary = (f"One-way ANOVA\nGroups: {', '.join(labels)}\nF({df1}, {df2}) = {F:.4g}, p = {p:.4g}, η² = {eta2:.4g}\n" +
               "Group means (n): " + ", ".join([f"{lab}={m:.4g} (n={n})" for lab,m,n in zip(labels,means,ns)]))
    return ModelOutput("ANOVA", summary, means_table, [("Boxplot", png_box), ("Violin", png_vio)])

def run_ols(df: pd.DataFrame, formula: str) -> ModelOutput:
    model = smf.ols(formula, data=df).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index":"term"})
    # Scatter + fit if single numeric predictor
    lhs, rhs = [s.strip() for s in formula.split("~",1)]
    terms = [t.strip() for t in re.split(r"\+|:|\*", rhs) if t.strip()]
    fig_fit: Optional[bytes] = None
    if len(terms)==1 and terms[0] in df.columns and pd.api.types.is_numeric_dtype(df[terms[0]]):
        x = df[terms[0]]; y = df[lhs]; fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(x,y); order = np.argsort(x.values); ax.plot(x.values[order], model.fittedvalues.values[order])
        ax.set_xlabel(terms[0]); ax.set_ylabel(lhs); ax.set_title("Scatter + OLS fit"); fig_fit = fig_to_png(fig)
    # Residuals + QQ
    fig_r, ax = plt.subplots(figsize=(5,3))
    ax.scatter(model.fittedvalues, model.resid); ax.axhline(0, linestyle=":"); ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted"); png_r = fig_to_png(fig_r)
    sm.qqplot(model.resid, line="45", fit=True); png_qq = fig_to_png(plt.gcf())
    summary = (f"OLS Regression\nn = {int(model.nobs)}, df_model = {model.df_model:.0f}, df_resid = {model.df_resid:.0f}\n"
               f"R² = {model.rsquared:.4g}, adj. R² = {model.rsquared_adj:.4g}\nF = {model.fvalue:.4g}, p(F) = {model.f_pvalue:.4g}")
    return ModelOutput("OLS", summary + "\n\n" + str(model.summary()), coef, [("Scatter+Fit", fig_fit), ("Residuals", png_r), ("QQ", png_qq)])

def run_glm(df: pd.DataFrame, formula: str, family: str) -> ModelOutput:
    fam_map = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(), "poisson": sm.families.Poisson(), "gamma": sm.families.Gamma()}
    fam = fam_map.get(family, sm.families.Gaussian())
    model = smf.glm(formula, data=df, family=fam).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index":"term"})
    # Residuals + QQ (deviance)
    fig_r, ax = plt.subplots(figsize=(5,3))
    ax.scatter(model.fittedvalues, model.resid_deviance); ax.axhline(0, linestyle=":"); ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted (GLM)"); png_r = fig_to_png(fig_r)
    sm.qqplot(model.resid_deviance, line="45", fit=True); png_qq = fig_to_png(plt.gcf())
    summary = (f"GLM ({family})\n n = {int(model.nobs)}, df_model = {model.df_model:.0f}, df_resid = {model.df_resid:.0f}\nAIC = {model.aic:.4g}, BIC = {model.bic:.4g}")
    return ModelOutput(f"GLM ({family})", summary + "\n\n" + str(model.summary()), coef, [("Residuals", png_r), ("QQ", png_qq)])

# ---------- PLOTS & CHECKS ----------

def plot_hist(df: pd.DataFrame, col: str, bins: int = 30) -> Tuple[str, bytes]:
    series = df[col].dropna().astype(float); fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(series, bins=bins); ax.set_title(f"Histogram of {col}")
    return f"Histogram {col}", fig_to_png(fig)

def plot_box(df: pd.DataFrame, value: str, group: str) -> Tuple[str, bytes]:
    fig, ax = plt.subplots(figsize=(5,3)); df.boxplot(column=value, by=group, ax=ax)
    ax.set_title(f"Boxplot of {value} by {group}"); ax.figure.suptitle("")
    return f"Boxplot {value}~{group}", fig_to_png(fig)

def plot_violin(df: pd.DataFrame, value: str, group: str) -> Tuple[str, bytes]:
    fig, ax = plt.subplots(figsize=(5,3))
    data = [g[value].dropna().values for _, g in df[[group, value]].groupby(group)]
    ax.violinplot(data, showmeans=True); ax.set_xticks(range(1,len(data)+1))
    ax.set_xticklabels([str(k) for k in df[group].dropna().unique().tolist()]); ax.set_title(f"Violin of {value} by {group}")
    return f"Violin {value}~{group}", fig_to_png(fig)

def plot_bar_with_error(df: pd.DataFrame, value: str, group: str, error: str = "sem") -> Tuple[str, bytes]:
    dfg = df[[group, value]].dropna(); dfg[value] = dfg[value].astype(float)
    stats_tbl = dfg.groupby(group)[value].agg(['mean', 'std', 'count']).reset_index()
    means = stats_tbl['mean'].values; sds = stats_tbl['std'].values; ns = stats_tbl['count'].values.astype(float)
    if error.lower() == "sd":
        yerr = sds; err_label = "SD"
    elif error.lower() == "ci95":
        sem = sds / np.sqrt(np.where(ns>0, ns, np.nan)); tcrit = stats.t.ppf(0.975, np.maximum(ns-1, 1)); yerr = sem * tcrit; err_label = "95% CI"
    else:
        yerr = sds / np.sqrt(np.where(ns>0, ns, np.nan)); err_label = "SEM"
    yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)
    labels = stats_tbl[group].astype(str).tolist(); x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x, means, yerr=yerr, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(f"Mean {value}"); ax.set_title(f"{value} by {group} (error: {err_label})")
    return f"Bar {value}~{group} ({err_label})", fig_to_png(fig)

def check_normality(df: pd.DataFrame, col: str) -> Tuple[str, Optional[bytes]]:
    series = df[col].dropna().astype(float)
    stat, p = stats.shapiro(series) if len(series) <= 5000 else (np.nan, np.nan)
    msg = f"Shapiro–Wilk (n={len(series)}): W={stat:.4f}, p={p:.4g}" if not np.isnan(stat) else "Shapiro skipped (n>5000)."
    sm.qqplot(series, line="45", fit=True); png = fig_to_png(plt.gcf())
    return msg, png

def check_homogeneity(df: pd.DataFrame, value: str, group: str) -> str:
    arrays = [g[value].dropna().values for _, g in df[[group, value]].groupby(group)]
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
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(outputs_dir):
            for fn in files:
                if fn.endswith('.zip'): continue
                z.write(os.path.join(root, fn), arcname=fn)
    return archive

def clear_outputs():
    if os.path.exists(outputs_dir): shutil.rmtree(outputs_dir)
    os.makedirs(outputs_dir, exist_ok=True)

# ---------- LLM + CHAT ----------

def _get_llm_client():
    global _llm_client
    if _llm_client is not None: return _llm_client
    if Together is None or not TOGETHER_API_KEY: return None
    _llm_client = Together(api_key=TOGETHER_API_KEY)
    return _llm_client

def ask_llm(chat_history, user_input):
    client = _get_llm_client()
    if client is None:
        return None, ("TOGETHER_API_KEY not set. You can still run explicit commands like:\n"
                      " - ttest value=score group=sex\n - anova value=score group=group\n - plot bar value=score group=sex error=sem\n"
                      "Or set the key in Settings → Variables & secrets.")
    messages = [{"role":"system","content":SYSTEM_PROMPT}] + chat_history + [{"role":"user","content":user_input}]
    try:
        resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.0).choices[0].message.content
        try:
            call = json.loads(resp); return call, resp
        except Exception:
            conv = client.chat.completions.create(model=LLM_MODEL, messages=[{"role":"system","content":FALLBACK_PROMPT}]+messages, temperature=0.7).choices[0].message.content
            return None, conv
    except Exception as e:
        return None, f"(LLM unavailable: {getattr(e, 'message', str(e))}) You can run explicit commands, e.g. 'plot bar value=score group=sex error=sem'."

def handle_upload(file):
    global cached_df
    clear_outputs()
    try:
        df = pd.read_csv(file)
        cached_df = df
        cols = ", ".join(df.columns.astype(str))
        preview = df.head(200)
        # Reset plot navigator state on new upload
        return (
            [{"role": "assistant", "content": f"CSV uploaded. Columns detected: {cols}. Ask me for t-test, ANOVA, OLS/GLM, histograms, box/violin, bar w/ error bars, normality checks, or power analysis."}],
            gr.update(value=preview, visible=True),
            gr.update(visible=True),
            [], 0  # plots_state, plot_index
        )
    except Exception as e:
        return [{"role": "assistant", "content": f"Failed to read CSV: {e}"}], gr.update(visible=False), gr.update(visible=False), [], 0

def parse_explicit(user_message: str) -> Optional[Tuple[str, dict]]:
    s = user_message.lower()
    if re.search(r"\bbar\s*plot\b|\bbar\s*chart\b|\bbar\b", s):
        value = None; group = None; error = None
        m = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message);  value = m.group(1) if m else None
        m = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message);  group = m.group(1) if m else None
        m = re.search(r"error\s*=\s*(sem|sd|ci95)", s);               error = m.group(1) if m else None
        return ("plot", {"bar": {"value": value, "group": group, "error": error or "sem"}})
    if re.search(r"\bt[-\s]?test\b|\bttest\b", s):
        v = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message)
        g = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message)
        return ("ttest", {"value": v.group(1) if v else None, "group": g.group(1) if g else None})
    if re.search(r"\banova\b", s):
        v = re.search(r"value\s*=\s*([A-Za-z0-9_]+)", user_message)
        g = re.search(r"group\s*=\s*([A-Za-z0-9_]+)", user_message)
        return ("anova", {"value": v.group(1) if v else None, "group": g.group(1) if g else None})
    return None

def handle_chat(chat_history, user_message):
    global cached_df
    chat_history = list(chat_history)

    explicit = parse_explicit(user_message)
    tool, llm_output = (None, None)
    if not explicit:
        tool, llm_output = ask_llm(chat_history, user_message)

    if explicit:
        action = explicit[0]; args = explicit[1]; tool_ok = True
    elif tool and tool.get("tool") == "stats":
        action = tool.get("action"); args = tool.get("args", {}); tool_ok = True
    else:
        tool_ok = False

    if tool_ok:
        if cached_df is None:
            chat_history.append({"role":"assistant","content":"Please upload a CSV first."})
            return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0
        df = cached_df.copy()
        sections: List[Tuple[str, str, Optional[bytes]]] = []
        plot_filepaths: List[str] = []
        preview_path: Optional[str] = None

        try:
            if action == "ttest":
                val, grp = guess_ttest_args(df, args.get("value"), args.get("group"))
                out = run_ttest(df, value=val, group=grp,
                                paired=bool(args.get("paired", False)),
                                equal_var=bool(args.get("equal_var", False)))
                if out.table is not None:
                    out.table.to_csv(os.path.join(outputs_dir, "ttest_table.csv"), index=False)
                # Save and queue all plots (Bar±SEM first for convenience)
                for title, png in (out.plots or []):
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').replace('±','pm').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                sections.append((out.kind, out.summary_text, None))

            elif action == "anova":
                val, grp = guess_anova_args(df, args.get("value"), args.get("group"))
                out = run_anova(df, value=val, group=grp)
                for title, png in (out.plots or []):
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                sections.append((out.kind, out.summary_text, None))

            elif action == "ols":
                out = run_ols(df, formula=args.get("formula"))
                if out.table is not None:
                    out.table.to_csv(os.path.join(outputs_dir, "ols_coef.csv"), index=False)
                for title, png in (out.plots or []):
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                sections.append((out.kind, out.summary_text, None))

            elif action == "glm":
                out = run_glm(df, formula=args.get("formula"), family=args.get("family", "gaussian"))
                if out.table is not None:
                    out.table.to_csv(os.path.join(outputs_dir, "glm_coef.csv"), index=False)
                for title, png in (out.plots or []):
                    if png:
                        fp = save_png(png, f"plot_{title.replace(' ','_').lower()}.png")
                        plot_filepaths.append(fp); preview_path = fp
                sections.append((out.kind, out.summary_text, None))

            elif action == "plot":
                if "hist" in args:
                    p = args["hist"]; title, png = plot_hist(df, p.get("col"), int(p.get("bins", 30)))
                    if png: 
                        fp = save_png(png, f"plot_hist_{p.get('col')}.png")
                        plot_filepaths.append(fp); preview_path = fp
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
                    val, grp = guess_bar_args(df, p.get("value"), p.get("group"))
                    err = p.get("error", "sem")
                    title, png = plot_bar_with_error(df, val, grp, err)
                    if png:
                        fp = save_png(png, f"plot_bar_{val}_{grp}_{err}.png")
                        plot_filepaths.append(fp); preview_path = fp
                    sections.append((title, f"Bar chart of mean {val} by {grp} with {err.upper()} error bars.", png))

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

            elif action == "power":
                if "ttest_ind" in args:
                    p = args["ttest_ind"]
                    msg = power_ttest_ind(p.get("effect_size"), float(p.get("alpha", 0.05)),
                                          p.get("power"), float(p.get("ratio", 1.0)),
                                          p.get("solve_for", "n_total"))
                    sections.append(("Power – t-test (ind)", msg, None))
                if "anova_oneway" in args:
                    p = args["anova_oneway"]
                    msg = power_anova_oneway(p.get("effect_size"), int(p.get("k_groups", 3)),
                                             float(p.get("alpha", 0.05)), p.get("power"),
                                             p.get("solve_for", "n_per_group"))
                    sections.append(("Power – ANOVA (one-way)", msg, None))
            else:
                sections.append(("Note", f"Unknown action: {action}", None))

        except Exception as e:
            sections.append(("Error", str(e), None))

        chat_summary = "\n\n".join([f"## {title}\n{txt}" for (title, txt, _) in sections if txt])
        _ = write_report(sections)
        zip_fp = save_zip()
        chat_history.append({"role":"user","content":user_message})
        chat_history.append({"role":"assistant","content":chat_summary or "Done. See preview and use Download Results."})
        # Initialize plot navigator state to this command's plots
        if not plot_filepaths:
            return chat_history, gr.update(value=None), gr.update(value=zip_fp, visible=True), [], 0
        return chat_history, gr.update(value=preview_path), gr.update(value=zip_fp, visible=True), plot_filepaths, len(plot_filepaths)-1

    # No tool recognized → natural language
    if llm_output:
        chat_history.append({"role":"user","content":user_message})
        chat_history.append({"role":"assistant","content":llm_output})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0
    else:
        chat_history.append({"role":"assistant","content":"Try: 'ttest value=score group=sex' (auto plots Bar±SEM, Box, Violin), or 'plot bar value=score group=sex error=ci95'."})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False), [], 0

# ---------- PLOT NAVIGATION ----------
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
    gr.Image(value="logo_long1.png", show_label=False, show_download_button=False, show_share_button=False, type="filepath", elem_id="logo-img")
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

    plots_state = gr.State([])   # list of filepaths for latest command
    plot_index  = gr.State(0)    # current index into plots_state

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat", show_label=True, type="messages",
                value=[{"role":"assistant","content":"Welcome! Upload a CSV, then ask: t-test (auto Bar±SEM, Box, Violin), ANOVA, OLS/GLM, hist/box/violin/bar, normality, or power."}]
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder="e.g., ttest value=score group=sex  |  ols score ~ weight", lines=1)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        with gr.Column(scale=3):
            with gr.Row():
                preview_plot = gr.Image(label="Preview (navigate plots)", type="filepath")
            with gr.Row():
                prev_btn = gr.Button("◀ Prev plot", variant="secondary")
                next_btn = gr.Button("Next plot ▶", variant="secondary")
            data_preview = gr.Dataframe(label="Data Preview (first 200 rows)", interactive=False, wrap=True)
            download_btn = gr.DownloadButton("📥 Download Results", value=None, visible=False)

    file_input.change(handle_upload, inputs=file_input, outputs=[chatbot, data_preview, download_btn, plots_state, plot_index])
    user_input.submit(handle_chat, inputs=[chatbot, user_input], outputs=[chatbot, preview_plot, download_btn, plots_state, plot_index])
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

    prev_btn.click(show_prev, inputs=[plots_state, plot_index], outputs=[preview_plot, plot_index])
    next_btn.click(show_next, inputs=[plots_state, plot_index], outputs=[preview_plot, plot_index])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
