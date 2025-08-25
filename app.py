import os
import io
import re
import json
import zipfile
import shutil
import time
import random
import threading
import sys
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv

# LLM providers
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, ServiceUnavailableError

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

print("Starting SpatChat: Stats Room (chat-first)")

# ---------- GLOBALS ----------
cached_df: Optional[pd.DataFrame] = None
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Track if we asked a follow-up (e.g., “which outcome?”)
pending: Dict[str, Optional[str]] = {"action": None, "need": None, "args": None}

# ---------- LLM (HF primary, Together fallback) ----------
load_dotenv()

HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def _choice_content(choice):
    """
    Extract assistant text from HF/Together pydantic/dict choices.
    Handles str or list-of-parts content.
    """
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")

    content = None
    if msg is not None:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)

    return content or ""

def _delta_text(delta):
    if isinstance(delta, dict):
        return delta.get("content", "")
    return getattr(delta, "content", "")

class _SpacedCallLimiter:
    """Ensure at least `min_interval_seconds` between calls (process-wide)."""
    def __init__(self, min_interval_seconds: float):
        self.min_interval = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._last = 0.0
    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()

class UnifiedLLM:
    """
    Primary: Hugging Face (Serverless or Endpoint via HF_ENDPOINT_URL)
    Fallback: Together.ai (if TOGETHER_API_KEY set)
    Returns plain string content.
    """
    def __init__(self):
        hf_model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()

        self.hf_client = InferenceClient(
            model=hf_model_or_url,
            token=hf_token,
            timeout=300,
        )

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
            self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)  # ≈0.6 QPM

    @staticmethod
    def _messages_to_prompt(messages):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            else:
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _hf_chat(self, messages, max_tokens=256, temperature=0.0, stream=False):
        tries, delay = 3, 2.0
        last_err = None
        for _ in range(tries):
            try:
                if hasattr(self.hf_client, "chat_completion"):
                    resp = self.hf_client.chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                    )
                    if stream:
                        text = "".join(_delta_text(ch.choices[0].delta) for ch in resp)
                    else:
                        text = _choice_content(resp.choices[0])
                    return text
                else:
                    prompt = self._messages_to_prompt(messages)
                    text = self.hf_client.text_generation(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                        return_full_text=False,
                    )
                    return text
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay *= 1.8
        raise last_err

    def chat(self, messages, temperature=0.0, max_tokens=256, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as hf_err:
            print(f"[LLM] HF primary failed: {hf_err}", file=sys.stderr)
            if self.together is None:
                raise

            # pace Together BEFORE first attempt
            self._tg_limiter.wait()
            backoff = 12.0
            for attempt in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if attempt == 3:
                        raise
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8

llm = UnifiedLLM()

SYSTEM_PROMPT = """
You are SpatChat, an expert statistics assistant for basic analyses.
When the user asks for an analysis, respond ONLY in compact JSON using this schema:
{"tool":"stats","action":"ttest|anova|ols|glm|plot|check|power|summary|recommend","args":{...}}

Actions and args:
- ttest: {"value":"colY","group":"colG", "paired": false, "equal_var": false}
- anova: {"value":"colY","group":"colG"}
- ols: {"formula":"y ~ x1 + x2"}
- glm: {"formula":"y ~ x1 + x2", "family":"gaussian|binomial|poisson|gamma"}
- plot: one of {"hist":{"col":"c","bins":30}, "box":{"value":"y","group":"g"}, "violin":{"value":"y","group":"g"}, "bar":{"value":"y","group":"g","error":"sem|sd|ci95"}}
- check: {"normality":{"col":"y"}}
- power: one of
   {"ttest_ind": {"effect_size": 0.5, "alpha": 0.05, "power": 0.8, "ratio": 1.0, "solve_for":"n_total|power|effect_size"}}
   {"anova_oneway": {"effect_size": 0.25, "k_groups": 3, "alpha": 0.05, "power": 0.8, "solve_for":"n_per_group|power|effect_size"}}
- summary: {"col":"y", "by":"group_col_or_null"}  # quick stats (mean/sd/min/max), optional by-group
- recommend: {}  # dataset-aware suggestions

If unclear, pick sensible defaults from the dataset. NEVER include prose; JSON only.
For general questions, answer in <=3 sentences of plain text.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a concise statistics tutor. If you can't map to a tool call, answer naturally in <=3 sentences.
""".strip()

# ---------- PLOTTING HELPERS ----------
def fig_to_np(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import PIL.Image as Image
    img = Image.open(buf).convert("RGB")
    return np.array(img)

def save_image_np(arr: np.ndarray, fname: str) -> str:
    fp = os.path.join(outputs_dir, fname)
    import PIL.Image as Image
    Image.fromarray(arr).save(fp, format="PNG")
    return fp

# ---------- REPORT & ZIP ----------
def add_to_report(parts: List[str], title: str, html_fragment: str):
    parts.append(f"<h2>{title}</h2>{html_fragment}")

def write_report(sections: List[Tuple[str, str, Optional[np.ndarray]]]) -> str:
    parts = ["<h1>SpatChat – Stats Report</h1>"]
    for title, html_text, img_np in sections:
        add_to_report(parts, title, f"<pre style='white-space:pre-wrap'>{html_text}</pre>")
        if img_np is not None:
            import PIL.Image as Image, base64
            buf = io.BytesIO()
            Image.fromarray(img_np).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
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
                full = os.path.join(root, fn)
                z.write(full, arcname=fn)
    return archive

def clear_outputs():
    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
    os.makedirs(outputs_dir, exist_ok=True)

# ---------- SCHEMA & RECOMMENDATIONS ----------
ID_NAME_PAT = re.compile(r"(?:^|[_\W])(id|identifier|index|subject|animal|record|row)(?:[_\W]|$)", re.I)

def is_id_like(df: pd.DataFrame, col: str) -> bool:
    name = str(col).strip().lower()
    if ID_NAME_PAT.search(name) or name.endswith("_id"):
        return True
    s = df[col].dropna()
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return False
    as_float = s.astype(float)
    if not np.all(np.isfinite(as_float)):
        return False
    frac = np.abs(as_float - np.round(as_float))
    if (frac > 1e-9).mean() > 0.05:
        return False
    u = np.sort(pd.unique(np.round(as_float).astype(int)))
    if len(u) < 5:
        return False
    diffs = np.diff(u)
    if len(diffs) == 0:
        return False
    return (diffs == 1).mean() >= 0.9

def is_integer_like(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return False
    frac = np.abs(s.astype(float) - np.round(s.astype(float)))
    return (frac > 1e-9).mean() <= 0.05

def infer_schema(df: pd.DataFrame) -> Dict:
    n, p = df.shape
    numeric_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical: List[str] = []

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            uniq = df[c].nunique(dropna=True)
            if uniq <= max(20, int(0.2 * max(n, 1))):
                categorical.append(c)

    cat_thresh = min(5, max(2, int(0.1 * max(n, 1))))
    for c in numeric_all:
        uniq = df[c].nunique(dropna=True)
        if is_integer_like(df[c]) and uniq <= cat_thresh:
            categorical.append(c)

    for c in ["group", "sex", "treatment", "class", "category"]:
        if c in df.columns and c not in categorical:
            categorical.append(c)

    binary = []
    for c in df.columns:
        u = df[c].nunique(dropna=True)
        if u == 2 and (not pd.api.types.is_numeric_dtype(df[c]) or is_integer_like(df[c])):
            binary.append(c)

    return {
        "n_rows": n, "n_cols": p,
        "numeric_all": numeric_all,
        "categorical": list(dict.fromkeys(categorical)),
        "binary": binary
    }

def usable_numeric_cols(df: pd.DataFrame) -> List[str]:
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    safe = [c for c in nums if not is_id_like(df, c)]
    priority = ["score", "height", "weight", "age"]
    ordered = [c for c in priority if c in safe]
    ordered += [c for c in safe if c not in ordered]
    if not ordered and nums:
        ordered = nums
    return ordered

def recommend_text_and_examples(df: pd.DataFrame) -> Tuple[str, str]:
    info = infer_schema(df)
    n, p = info["n_rows"], info["n_cols"]
    nums_all = info["numeric_all"]
    cats = info["categorical"]
    bins = info["binary"]

    nums_safe = usable_numeric_cols(df)
    y_num = nums_safe[0] if nums_safe else (nums_all[0] if nums_all else None)

    def valid_group(col: str, outcome: Optional[str]) -> bool:
        if outcome and col == outcome:
            return False
        u = df[col].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[col]):
            return u >= 2 and is_integer_like(df[col])
        else:
            return u >= 2

    g_bin = next((c for c in bins if valid_group(c, y_num)), None)
    g_multi = next((c for c in cats if df[c].nunique(dropna=True) >= 3 and valid_group(c, y_num)), None)

    overview_lines = [f"Dataset overview: n={n} rows, p={p} columns."]
    if nums_all:
        overview_lines.append(f"- Numeric columns ({len(nums_all)}): {', '.join(map(str, nums_all[:8]))}{'…' if len(nums_all)>8 else ''}")
    if cats:
        overview_lines.append(f"- Categorical/low-cardinality ({len(cats)}): {', '.join(map(str, cats[:8]))}{'…' if len(cats)>8 else ''}")

    rec_lines = ["Here are some analysis ideas you can run next:"]
    if y_num:
        rec_lines.append(f"• See a quick summary of **{y_num}** (mean, sd, min/max).")
        rec_lines.append(f"• **Histogram** to see the shape of **{y_num}**.")
    if y_num and g_bin:
        rec_lines.append(f"• Compare average **{y_num}** between the two groups in **{g_bin}** (t-test).")
        rec_lines.append(f"• **Bar chart with error bars** for **{y_num}** by **{g_bin}**.")
    if y_num and g_multi:
        rec_lines.append(f"• Compare **{y_num}** across levels of **{g_multi}** (one-way ANOVA).")
        rec_lines.append(f"• **Box/violin plots** of **{y_num}** across **{g_multi}**.")
    if len(nums_safe) >= 2:
        rec_lines.append(f"• See how one value predicts another (linear regression), e.g., **{nums_safe[0]} ~ {nums_safe[1]}**.")
    rec_lines.append("• **Normality check** with a QQ plot (assumption check).")
    if g_bin:
        rec_lines.append("• **Power analysis** for a t-test to estimate sample size.")
    if g_multi:
        rec_lines.append("• **Power analysis** for one-way ANOVA to estimate sample size.")

    ex = ["Here are some things you can say:"]
    if y_num:
        ex.append(f'• "What is the average {y_num}?"')
        ex.append(f'• "Show a histogram of {y_num}."')
    if y_num and g_bin:
        ex.append(f'• "I want to do a t-test on {y_num} by {g_bin}."')
    if y_num and g_multi:
        ex.append(f'• "Run a one-way ANOVA of {y_num} by {g_multi}."')
        ex.append(f'• "Show box and violin plots for {y_num} by {g_multi}."')
    if len(nums_safe) >= 2:
        ex.append(f'• "Fit a linear regression: {nums_safe[0]} ~ {nums_safe[1]}."')
    ex.append('• "Check normality of the main outcome and show a QQ plot."')
    if g_bin:
        ex.append('• "Power analysis for a t-test with 80% power and effect size 0.5."')
    if g_multi:
        ex.append('• "Power analysis for a one-way ANOVA with 3 groups and 80% power."')

    return "\n".join(overview_lines + [""] + rec_lines), "\n".join(ex)

# ---------- GROUP ORDER ----------
def ordered_groups(df: pd.DataFrame, group_col: str) -> List[str]:
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

# ---------- NEW: NULL-LIKE SANITIZATION ----------
NULLY_STRINGS = {"", "null", "none", "na", "n/a", "nil"}

def clean_null_like(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in NULLY_STRINGS else v

def sanitize_args(action: Optional[str], args: Dict) -> Dict:
    """Normalize any 'group'/'by' keys so 'null'/'none'/'' → None."""
    if not isinstance(args, dict):
        return {}
    if action == "summary":
        args["by"] = clean_null_like(args.get("by"))
    elif action in {"ttest", "anova"}:
        args["group"] = clean_null_like(args.get("group"))
    elif action == "plot":
        for k in ("box", "violin", "bar"):
            if k in args and isinstance(args[k], dict):
                args[k]["group"] = clean_null_like(args[k].get("group"))
    return args

# ---------- ANALYSES ----------
def ttest_summary(a: np.ndarray, b: np.ndarray, g1: str, g2: str, equal_var=False, paired=False) -> str:
    if paired:
        stat, p = stats.ttest_rel(a, b, nan_policy="omit")
        test_name = "Paired t-test"
        df_est = len(a) - 1 if len(a) == len(b) else np.nan
    else:
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
        test_name = "Student t-test" if equal_var else "Welch t-test"
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        na, nb = len(a), len(b)
        df_est = (va/na + vb/nb)**2 / ((va**2)/((na**2)*(na-1)) + (vb**2)/((nb**2)*(nb-1)))

    mean_a, mean_b = a.mean(), b.mean()
    sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
    diff = mean_a - mean_b
    se = np.sqrt(sd_a**2/len(a) + sd_b**2/len(b))
    try:
        tcrit = stats.t.ppf(0.975, df=df_est)
        ci_low, ci_high = diff - tcrit*se, diff + tcrit*se
    except Exception:
        ci_low = ci_high = np.nan

    sp = np.sqrt(((len(a)-1)*sd_a**2 + (len(b)-1)*sd_b**2) / (len(a)+len(b)-2)) if len(a)+len(b)-2 > 0 else np.nan
    d = diff / sp if sp and sp > 0 else np.nan
    J = 1 - (3 / (4*(len(a)+len(b)-2) - 1)) if (len(a)+len(b)-2) > 1 else 1.0
    g = d * J if d is not None else np.nan

    lines = [
        "t-test",
        test_name,
        f"Groups: {g1} (n={len(a)}, mean={mean_a:.3g}, sd={sd_a:.3g}) vs {g2} (n={len(b)}, mean={mean_b:.3g}, sd={sd_b:.3g})",
        f"Mean difference = {diff:.3g}, 95% CI [{ci_low:.4g}, {ci_high:.4g}]",
        f"t({df_est:.2f}) = {stat:.3g}, p = {p:.5g}",
        f"Effect size: Cohen's d = {d:.3g}, Hedges' g = {g:.3g}"
    ]
    return "\n".join(lines)

def run_ttest(df: pd.DataFrame, value: str, group: str, paired=False, equal_var=False) -> Tuple[str, List[np.ndarray]]:
    gorder = ordered_groups(df, group)
    if len(gorder) != 2:
        raise gr.Error(f"t-test requires exactly 2 groups; {group} has {len(gorder)} levels.")
    a = df[df[group].astype(str) == gorder[0]][value].dropna().astype(float).values
    b = df[df[group].astype(str) == gorder[1]][value].dropna().astype(float).values

    fig_box, ax = plt.subplots(figsize=(5,3))
    ax.boxplot([a, b], tick_labels=[gorder[0], gorder[1]])
    ax.set_title(f"Boxplot of {value} by {group}")
    img_box = fig_to_np(fig_box)

    fig_vio, ax = plt.subplots(figsize=(5,3))
    ax.violinplot([a, b], showmeans=True)
    ax.set_xticks([1,2]); ax.set_xticklabels([gorder[0], gorder[1]])
    ax.set_title(f"Violin of {value} by {group}")
    img_vio = fig_to_np(fig_vio)

    img_bar = bar_with_error_plot(df, value, group, error="sem", gorder=gorder)

    txt = ttest_summary(a, b, gorder[0], gorder[1], equal_var=equal_var, paired=paired)
    return txt, [img_bar, img_box, img_vio]

def run_anova(df: pd.DataFrame, value: str, group: str) -> Tuple[str, List[np.ndarray]]:
    gorder = ordered_groups(df, group)
    data = [df[df[group].astype(str) == g][value].dropna().astype(float).values for g in gorder]
    if len(data) < 2:
        raise gr.Error("ANOVA requires at least two groups.")
    F, p = stats.f_oneway(*data)
    txt = f"One-way ANOVA on {value} by {group}\nGroups: {', '.join(gorder)}\nF = {F:.4g}, p = {p:.5g}"

    fig_box, ax = plt.subplots(figsize=(6,3))
    ax.boxplot(data, tick_labels=gorder)
    ax.set_title(f"Boxplot of {value} by {group}")
    img_box = fig_to_np(fig_box)

    fig_vio, ax = plt.subplots(figsize=(6,3))
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(gorder)+1)); ax.set_xticklabels(gorder)
    ax.set_title(f"Violin of {value} by {group}")
    img_vio = fig_to_np(fig_vio)

    return txt, [img_box, img_vio]

def run_ols(df: pd.DataFrame, formula: str) -> Tuple[str, List[np.ndarray]]:
    model = smf.ols(formula, data=df).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index":"term"})
    coef.to_csv(os.path.join(outputs_dir, "ols_coef.csv"), index=False)

    images: List[np.ndarray] = []

    lhs, rhs = [s.strip() for s in formula.split("~",1)]
    terms = [t.strip() for t in re.split(r"\+", rhs) if t.strip()]
    if len(terms) == 1 and terms[0] in df.columns and pd.api.types.is_numeric_dtype(df[terms[0]]):
        x = df[terms[0]].astype(float); y = df[lhs].astype(float)
        order = np.argsort(x.values)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.scatter(x, y)
        ax.plot(x.values[order], model.fittedvalues.values[order])
        ax.set_xlabel(terms[0]); ax.set_ylabel(lhs); ax.set_title("Scatter + OLS fit")
        images.append(fig_to_np(fig))

    fig_r, ax = plt.subplots(figsize=(6,3))
    ax.scatter(model.fittedvalues, model.resid)
    ax.axhline(0, linestyle=":")
    ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted")
    images.append(fig_to_np(fig_r))

    sm.qqplot(model.resid, line="45", fit=True)
    images.append(fig_to_np(plt.gcf()))

    return str(model.summary()), images

def run_glm(df: pd.DataFrame, formula: str, family: str) -> Tuple[str, List[np.ndarray]]:
    fam_map = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
    }
    fam = fam_map.get(family.lower(), sm.families.Gaussian())
    model = smf.glm(formula, data=df, family=fam).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index":"term"})
    coef.to_csv(os.path.join(outputs_dir, "glm_coef.csv"), index=False)

    images: List[np.ndarray] = []
    fig_r, ax = plt.subplots(figsize=(6,3))
    ax.scatter(model.fittedvalues, model.resid_deviance)
    ax.axhline(0, linestyle=":")
    ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title(f"Residuals vs Fitted (GLM: {family})")
    images.append(fig_to_np(fig_r))

    sm.qqplot(model.resid_deviance, line="45", fit=True)
    images.append(fig_to_np(plt.gcf()))

    return str(model.summary()), images

# ---------- PLOTS & CHECKS ----------
def plot_hist(df: pd.DataFrame, col: str, bins: int = 30) -> Tuple[str, np.ndarray]:
    s = df[col].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(s, bins=bins)
    ax.set_title(f"Histogram of {col}")
    return f"Histogram {col}", fig_to_np(fig)

def plot_box(df: pd.DataFrame, value: str, group: str) -> Tuple[str, np.ndarray]:
    gorder = ordered_groups(df, group)
    data = [df[df[group].astype(str) == g][value].dropna().astype(float).values for g in gorder]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.boxplot(data, tick_labels=gorder)
    ax.set_title(f"Boxplot of {value} by {group}")
    return f"Boxplot {value}~{group}", fig_to_np(fig)

def plot_violin(df: pd.DataFrame, value: str, group: str) -> Tuple[str, np.ndarray]:
    gorder = ordered_groups(df, group)
    data = [df[df[group].astype(str) == g][value].dropna().astype(float).values for g in gorder]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(gorder)+1))
    ax.set_xticklabels(gorder)
    ax.set_title(f"Violin of {value} by {group}")
    return f"Violin {value}~{group}", fig_to_np(fig)

def bar_with_error_plot(df: pd.DataFrame, value: str, group: str, error: str = "sem", gorder: Optional[List[str]] = None) -> np.ndarray:
    if gorder is None:
        gorder = ordered_groups(df, group)
    means, errs = [], []
    for g in gorder:
        vals = df[df[group].astype(str) == g][value].dropna().astype(float).values
        n = len(vals)
        m = vals.mean() if n > 0 else np.nan
        if error == "sd":
            e = vals.std(ddof=1) if n > 1 else np.nan
        elif error == "ci95":
            sd = vals.std(ddof=1) if n > 1 else np.nan
            se = sd/np.sqrt(n) if n > 0 else np.nan
            tcrit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
            e = se * tcrit if (se is not None and tcrit is not None) else np.nan
        else:  # sem
            sd = vals.std(ddof=1) if n > 1 else np.nan
            e = sd/np.sqrt(n) if n > 0 else np.nan
        means.append(m); errs.append(e)

    x = np.arange(len(gorder))
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(x, means, yerr=errs, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(gorder)
    ax.set_title(f"Bar ± {error.upper()} of {value} by {group}")
    return fig_to_np(fig)

def check_normality(df: pd.DataFrame, col: str) -> Tuple[str, np.ndarray]:
    series = df[col].dropna().astype(float)
    W, p = (stats.shapiro(series) if 3 <= len(series) <= 5000 else (np.nan, np.nan))
    msg = f"Shapiro–Wilk normality on {col} (n={len(series)}): W={W:.4g}, p={p:.5g}" if not np.isnan(W) else f"Shapiro–Wilk skipped for {col} (requires 3–5000 values)."
    sm.qqplot(series, line="45", fit=True)
    return msg, fig_to_np(plt.gcf())

# ---------- POWER ----------
def power_ttest_ind(effect_size: Optional[float], alpha: float, power: Optional[float], ratio: float, solve_for: str) -> str:
    tool = TTestIndPower()
    if solve_for == "n_total":
        n1 = tool.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=ratio, alternative="two-sided")
        n2 = n1*ratio
        return f"Required total sample size (two-sample t-test): group1 ≈ {int(np.ceil(n1))}, group2 ≈ {int(np.ceil(n2))} (total ≈ {int(np.ceil(n1+n2))})"
    elif solve_for == "power":
        pw = tool.solve_power(effect_size=effect_size, nobs1=power, alpha=alpha, ratio=ratio, alternative="two-sided")
        return f"Achieved power ≈ {pw:.3f}"
    elif solve_for == "effect_size":
        es = tool.solve_power(effect_size=None, nobs1=power, alpha=alpha, ratio=ratio, alternative="two-sided")
        return f"Implied effect size d ≈ {es:.3f}"
    else:
        return "Unknown solve_for for t-test power."

def power_anova_oneway(effect_size: Optional[float], k_groups: int, alpha: float, power: Optional[float], solve_for: str) -> str:
    tool = FTestAnovaPower()
    if solve_for == "n_per_group":
        n = tool.solve_power(effect_size=effect_size, k_groups=k_groups, alpha=alpha, power=power)
        return f"Required n per group (one-way ANOVA): ≈ {int(np.ceil(n))}"
    elif solve_for == "power":
        pw = tool.solve_power(effect_size=effect_size, k_groups=k_groups, alpha=alpha, nobs=power)
        return f"Achieved power ≈ {pw:.3f}"
    elif solve_for == "effect_size":
        es = tool.solve_power(effect_size=None, k_groups=k_groups, alpha=alpha, nobs=power)
        return f"Implied effect size f ≈ {es:.3f}"
    else:
        return "Unknown solve_for for ANOVA power."

# ---------- QUICK SUMMARY ----------
def quick_summary(df: pd.DataFrame, col: str, by: Optional[str] = None) -> str:
    if by and by in df.columns and by != col and df[by].nunique(dropna=True) > 1:
        lines = [f"Summary of {col} by {by}:"]
        gorder = ordered_groups(df, by)
        for g in gorder:
            s = df[df[by].astype(str) == g][col].dropna().astype(float)
            if len(s) == 0:
                lines.append(f"- {g}: n=0")
            else:
                lines.append(f"- {g}: n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, min={s.min():.4g}, max={s.max():.4g}")
        return "\n".join(lines)
    else:
        s = df[col].dropna().astype(float)
        return f"Summary of {col}: n={len(s)}, mean={s.mean():.4g}, sd={s.std(ddof=1):.4g}, min={s.min():.4g}, max={s.max():.4g}"

# ---------- LLM + LOCAL PARSER ----------
def ask_llm(chat_history, user_input):
    messages = [{"role":"system","content":SYSTEM_PROMPT}] + chat_history + [{"role":"user","content":user_input}]
    resp = llm.chat(messages=messages, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = llm.chat(
            messages=[{"role":"system","content":FALLBACK_PROMPT}] + messages,
            temperature=0.7, max_tokens=256, stream=False
        )
        return None, conv

def local_parse(user_input: str) -> Optional[Dict]:
    s = user_input.strip().lower()

    if re.search(r"\b(what can i do|how should i analyze|recommend|suggestion|what analyses)\b", s):
        return {"tool":"stats","action":"recommend","args":{}}

    m = re.search(r"(?:what(?:'s| is) the )?(?:mean|average|summary|summarize|describe)\s+([a-zA-Z0-9_]+)(?:\s+by\s+([a-zA-Z0-9_]+))?", s)
    if m:
        col = m.group(1)
        by = m.group(2) if m.group(2) else None
        return {"tool":"stats","action":"summary","args":{"col":col, "by":by}}

    m = re.search(r"(t[\-\s]?test).*?(?:on|of)?\s*([a-zA-Z0-9_]+).*(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool":"stats","action":"ttest","args":{"value":m.group(2), "group":m.group(3)}}

    m = re.search(r"(anova).*(?:on|of)?\s*([a-zA-Z0-9_]+).*(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool":"stats","action":"anova","args":{"value":m.group(2), "group":m.group(3)}}

    m = re.search(r"(?:ols|regression)\s+([a-zA-Z0-9_]+)\s*~\s*([a-zA-Z0-9_+\s]+)", s)
    if m:
        return {"tool":"stats","action":"ols","args":{"formula": f"{m.group(1)} ~ {m.group(2)}"}}

    m = re.search(r"(glm|poisson|binomial|gaussian|gamma)\s+([a-zA-Z0-9_]+)\s*~\s*([a-zA-Z0-9_+\s]+)", s)
    if m:
        fam = "gaussian" if m.group(1) == "glm" else m.group(1)
        return {"tool":"stats","action":"glm","args":{"formula": f"{m.group(2)} ~ {m.group(3)}", "family":fam}}

    m = re.search(r"(hist(?:ogram)?)\s+(?:of|on|for)?\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool":"stats","action":"plot","args":{"hist":{"col":m.group(2),"bins":30}}}

    m = re.search(r"(box|violin|bar)\s+(?:plot\s+)?(?:of|on|for)?\s*([a-zA-Z0-9_]+)\s+(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        kind, val, grp = m.group(1), m.group(2), m.group(3)
        if kind == "box":
            return {"tool":"stats","action":"plot","args":{"box":{"value":val,"group":grp}}}
        if kind == "violin":
            return {"tool":"stats","action":"plot","args":{"violin":{"value":val,"group":grp}}}
        if kind == "bar":
            return {"tool":"stats","action":"plot","args":{"bar":{"value":val,"group":grp,"error":"ci95"}}}

    m = re.search(r"(normality|qq)\s+(?:check|plot)?\s*(?:for|of)?\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool":"stats","action":"check","args":{"normality":{"col":"{}".format(m.group(2))}}}

    if "power" in s and "t-test" in s:
        return {"tool":"stats","action":"power","args":{"ttest_ind":{"effect_size":0.5,"alpha":0.05,"power":0.8,"ratio":1.0,"solve_for":"n_total"}}}
    if "power" in s and "anova" in s:
        return {"tool":"stats","action":"power","args":{"anova_oneway":{"effect_size":0.25,"k_groups":3,"alpha":0.05,"power":0.8,"solve_for":"n_per_group"}}}

    try:
        maybe = json.loads(user_input)
        if isinstance(maybe, dict) and "tool" in maybe:
            return maybe
    except Exception:
        pass
    return None

# ---------- CHAT HANDLERS ----------
def handle_upload(file):
    global cached_df
    clear_outputs()
    pending.update({"action": None, "need": None, "args": None})
    try:
        df = pd.read_csv(file)
        cached_df = df
        cols = ", ".join(map(str, df.columns))
        preview = df.head(200)
        return (
            [{"role":"assistant","content":f"CSV uploaded. Columns detected: {cols}. Ask me for t-test, ANOVA, OLS/GLM, hist/box/violin/bar, normality checks, power, quick summaries (e.g., 'What is the average height?'), or say 'what can I do with my data?'"}],
            gr.update(value=preview, visible=True),
            gr.update(visible=True)
        )
    except Exception as e:
        return (
            [{"role":"assistant","content":f"Failed to read CSV: {e}"}],
            gr.update(visible=False),
            gr.update(visible=False)
        )

def need_value_and_group(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    nums = usable_numeric_cols(df)
    groups = []
    for c in df.columns:
        if df[c].nunique(dropna=True) >= 2:
            if not pd.api.types.is_numeric_dtype(df[c]) or is_integer_like(df[c]):
                groups.append(c)
    prefer = ["sex","group","treatment","class","category"]
    ordered_groups = [c for c in prefer if c in groups] + [c for c in groups if c not in prefer]
    return nums, ordered_groups

# ---------- GALLERY (Prev/Next) ----------
def _safe_last(paths: List[str]) -> Tuple[Optional[str], List[str], int]:
    """Return (preview, paths, idx). If empty, preview=None and idx=-1."""
    if paths:
        return paths[-1], paths, len(paths) - 1
    return None, [], -1

def _step(paths: List[str], idx: int, delta: int) -> Tuple[Optional[str], int]:
    """Move idx by delta with wrap-around. Returns (preview, new_idx)."""
    if not paths:
        return None, -1
    n = len(paths)
    new_idx = (idx + delta) % n
    return paths[new_idx], new_idx

def handle_chat(chat_history, user_message, data_preview):
    global cached_df, pending

    chat_history = list(chat_history)
    text = user_message.strip()

    parsed = None
    llm_error_text = None
    try:
        tool, _ = ask_llm(chat_history, text)
        parsed = tool
    except Exception as e:
        llm_error_text = f"(LLM unavailable: {e})"
        parsed = local_parse(text)

    if pending["action"] and parsed and parsed.get("tool") == "stats":
        pending = {"action": None, "need": None, "args": None}

    if not parsed:
        reply = "I can help with t-tests, ANOVA, OLS/GLM, hist/box/violin/bar, normality, power, and quick summaries. Try: 'I want a t-test on score by sex' or 'Show a histogram of height'."
        if llm_error_text:
            reply = llm_error_text + " " + reply
        chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":reply}])
        # no images produced
        preview, paths, idx = _safe_last([])
        return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx

    if parsed.get("tool") != "stats":
        chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":"I'm not sure how to handle that yet."}])
        preview, paths, idx = _safe_last([])
        return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx

    action = parsed.get("action")
    args = sanitize_args(action, parsed.get("args", {}))  # sanitize 'null'-like args

    if cached_df is None:
        chat_history.append({"role":"assistant","content":"Please upload a CSV first."})
        preview, paths, idx = _safe_last([])
        return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx

    df = cached_df.copy()
    sections: List[Tuple[str, str, Optional[np.ndarray]]] = []
    images: List[np.ndarray] = []
    image_paths: List[str] = []

    try:
        # -------- Recommendations --------
        if action == "recommend":
            tech, examples = recommend_text_and_examples(df)
            msg = "Recommendations\n" + tech + "\n\nSay it like this\n" + examples
            sections.append(("Recommendations", msg, None))
            chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":msg}])
            report_fp = write_report(sections); zip_fp = save_zip()
            preview, paths, idx = _safe_last([])
            return chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx

        # -------- Summary --------
        if action == "summary":
            col = args.get("col")
            by = args.get("by")  # already sanitized to None if null-like
            if col not in df.columns:
                raise gr.Error(f"Column '{col}' not found.")
            if by is not None and by not in df.columns:
                raise gr.Error(f"Group column '{by}' not found.")
            msg = quick_summary(df, col, by)
            sections.append(("Summary", msg, None))
            chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":msg}])
            report_fp = write_report(sections); zip_fp = save_zip()
            preview, paths, idx = _safe_last([])
            return chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx

        # -------- t-test --------
        if action == "ttest":
            value = args.get("value")
            group = args.get("group")  # sanitized
            if not value or not group:
                nums, groups = need_value_and_group(df)
                pending.update({"action":"ttest","need":"value" if not value else "group","args":{"value":value,"group":group}})
                ask = "Which numeric outcome should I test? Candidates: " + ", ".join(nums) if not value else "Which group column? Candidates: " + ", ".join(groups)
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":ask}])
                preview, paths, idx = _safe_last([])
                return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
            if value == group:
                raise gr.Error("Outcome and group must be different.")
            txt, imgs = run_ttest(df, value=value, group=group, paired=bool(args.get("paired", False)), equal_var=bool(args.get("equal_var", False)))
            sections.append(("t-test", txt, None))
            for i, im in enumerate(imgs):
                if im is not None:
                    images.append(im)
                    pth = save_image_np(im, f"ttest_plot_{i+1}.png")
                    image_paths.append(pth)
            chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":txt}])

        # -------- ANOVA --------
        elif action == "anova":
            value = args.get("value")
            group = args.get("group")  # sanitized
            if not value or not group:
                nums, groups = need_value_and_group(df)
                pending.update({"action":"anova","need":"value" if not value else "group","args":{"value":value,"group":group}})
                ask = "Which numeric outcome for ANOVA? Candidates: " + ", ".join(nums) if not value else "Which group column for ANOVA? Candidates: " + ", ".join(groups)
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":ask}])
                preview, paths, idx = _safe_last([])
                return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
            if value == group:
                raise gr.Error("Outcome and group must be different.")
            if df[group].nunique(dropna=True) < 2:
                raise gr.Error(f"ANOVA needs ≥2 groups; '{group}' has {df[group].nunique(dropna=True)}.")
            txt, imgs = run_anova(df, value=value, group=group)
            sections.append(("ANOVA", txt, None))
            for i, im in enumerate(imgs):
                pth = save_image_np(im, f"anova_plot_{i+1}.png")
                image_paths.append(pth)
            chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":txt}])

        # -------- OLS --------
        elif action == "ols":
            formula = args.get("formula")
            if not formula or "~" not in formula:
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":"Please specify a formula like: ols score ~ age + weight"}])
                preview, paths, idx = _safe_last([])
                return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
            txt, imgs = run_ols(df, formula=formula)
            sections.append(("OLS", txt, None))
            for i, im in enumerate(imgs):
                pth = save_image_np(im, f"ols_plot_{i+1}.png")
                image_paths.append(pth)
            chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":txt}])

        # -------- GLM --------
        elif action == "glm":
            formula = args.get("formula"); family = args.get("family","gaussian")
            if not formula or "~" not in formula:
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":"Please specify a formula like: glm count ~ age family=poisson"}])
                preview, paths, idx = _safe_last([])
                return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
            txt, imgs = run_glm(df, formula=formula, family=family)
            sections.append((f"GLM ({family})", txt, None))
            for i, im in enumerate(imgs):
                pth = save_image_np(im, f"glm_plot_{i+1}.png")
                image_paths.append(pth)
            chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":txt}])

        # -------- PLOTS --------
        elif action == "plot":
            if "hist" in args:
                p = args["hist"]; title, im = plot_hist(df, p.get("col"), int(p.get("bins",30)))
                pth = save_image_np(im, f"plot_hist_{p.get('col')}.png"); image_paths.append(pth)
                sections.append((title, "", im))
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":f"{title} generated."}])
            if "box" in args:
                p = args["box"]
                grp = p.get("group")
                if grp is None:
                    nums, groups = need_value_and_group(df)
                    ask = "Which group column for the box plot? Candidates: " + ", ".join(groups)
                    chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":ask}])
                    preview, paths, idx = _safe_last([])
                    return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
                if p.get("value") == grp:
                    raise gr.Error("Outcome and group must be different.")
                title, im = plot_box(df, p.get("value"), grp)
                pth = save_image_np(im, f"plot_box_{p.get('value')}_{grp}.png"); image_paths.append(pth)
                sections.append((title, "", im))
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":f"{title} generated."}])
            if "violin" in args:
                p = args["violin"]
                grp = p.get("group")
                if grp is None:
                    nums, groups = need_value_and_group(df)
                    ask = "Which group column for the violin plot? Candidates: " + ", ".join(groups)
                    chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":ask}])
                    preview, paths, idx = _safe_last([])
                    return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
                if p.get("value") == grp:
                    raise gr.Error("Outcome and group must be different.")
                title, im = plot_violin(df, p.get("value"), grp)
                pth = save_image_np(im, f"plot_violin_{p.get('value')}_{grp}.png"); image_paths.append(pth)
                sections.append((title, "", im))
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":f"{title} generated."}])
            if "bar" in args:
                p = args["bar"]
                grp = p.get("group")
                if grp is None:
                    nums, groups = need_value_and_group(df)
                    ask = "Which group column for the bar chart? Candidates: " + ", ".join(groups)
                    chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":ask}])
                    preview, paths, idx = _safe_last([])
                    return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
                if p.get("value") == grp:
                    raise gr.Error("Outcome and group must be different.")
                err = p.get("error","sem")
                gorder = ordered_groups(df, grp)
                im = bar_with_error_plot(df, p.get("value"), grp, error=err, gorder=gorder)
                pth = save_image_np(im, f"plot_bar_{p.get('value')}_{grp}_{err}.png"); image_paths.append(pth)
                sections.append((f"Bar {p.get('value')}~{grp}", "", im))
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":f"Bar ± {err.upper()} generated."}])

        # -------- CHECKS --------
        elif action == "check":
            if "normality" in args:
                p = args["normality"]; msg, im = check_normality(df, p.get("col"))
                pth = save_image_np(im, f"plot_qq_{p.get('col')}.png"); image_paths.append(pth)
                sections.append(("Normality", msg, im))
                chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":msg}])

        else:
            chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":f"Unknown action: {action}"}])

    except gr.Error as e:
        chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":str(e)}])
        preview, paths, idx = _safe_last([])
        return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx
    except Exception as e:
        msg = f"Error: {e}"
        chat_history.extend([{"role":"user","content":text},{"role":"assistant","content":msg}])
        preview, paths, idx = _safe_last([])
        return chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx

    report_fp = write_report(sections)
    zip_fp = save_zip()
    preview, paths, idx = _safe_last(image_paths)
    return chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx

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
    .dataframe-wrap { max-height: 340px; overflow: auto; border: 1px solid #ddd; border-radius: 6px; padding: 6px; }
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
        If you use SpatChat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Stats Room.</i>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role":"assistant","content":"Welcome! Upload a CSV, then ask: t-test, ANOVA, OLS/GLM, histogram, box/violin/bar, normality, power, quick summaries (e.g., 'What is the average height?'), or 'what can I do with my data?'."}]
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder="e.g., I want a t-test on score by sex", lines=1)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        with gr.Column(scale=3):
            preview_plot = gr.Image(label="Preview (last figure)", value=None, type="filepath")
            data_preview = gr.Dataframe(label="Data Preview (first 200 rows)", interactive=False, visible=False)
            download_btn = gr.DownloadButton("📥 Download Results", value=None, visible=False)

            with gr.Row():
                prev_btn = gr.Button("◀️ Prev", variant="secondary")
                next_btn = gr.Button("Next ▶️", variant="secondary")

            # Gallery states
            gallery_paths = gr.State([])
            gallery_index = gr.State(-1)

    # Enable queue (older-Gradio-safe signature)
    demo.queue(max_size=16)

    file_input.change(handle_upload, inputs=file_input, outputs=[chatbot, data_preview, download_btn])

    user_input.submit(
        handle_chat,
        inputs=[chatbot, user_input, data_preview],
        outputs=[chatbot, preview_plot, download_btn, data_preview, gallery_paths, gallery_index]
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

    # Button callbacks
    def on_prev(paths, idx):
        preview, new_idx = _step(paths, idx, -1)
        return gr.update(value=preview), new_idx

    def on_next(paths, idx):
        preview, new_idx = _step(paths, idx, +1)
        return gr.update(value=preview), new_idx

    prev_btn.click(on_prev, inputs=[gallery_paths, gallery_index], outputs=[preview_plot, gallery_index])
    next_btn.click(on_next, inputs=[gallery_paths, gallery_index], outputs=[preview_plot, gallery_index])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
