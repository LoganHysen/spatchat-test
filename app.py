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
from together import Together
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

print("Starting SpatChat: Stats Room (chat-first)")

# ---------- GLOBAL STATE ----------
cached_df: Optional[pd.DataFrame] = None
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# ---------- LLM SETUP (Together API) ----------
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

SYSTEM_PROMPT = """
You are SpatChat, an expert statistics assistant for basic analyses.
When the user asks for an analysis, respond ONLY in compact JSON using this schema:
{"tool":"stats","action":"ttest|anova|ols|glm|plot|check|power","args":{...}}

Actions and args:
- ttest: {"value":"colY","group":"colG", "paired": false, "equal_var": false}
- anova: {"value":"colY","group":"colG"}
- ols: {"formula":"y ~ x1 + x2"}
- glm: {"formula":"y ~ x1 + x2", "family":"gaussian|binomial|poisson|gamma"}
- plot: one of {"hist":{"col":"c","bins":30}, "box":{"value":"y","group":"g"}, "violin":{"value":"y","group":"g"}}
- check: {"normality":{"col":"y"}} or {"homogeneity":{"value":"y","group":"g"}}
- power: one of
   {"ttest_ind": {"effect_size": 0.5, "alpha": 0.05, "power": 0.8, "ratio": 1.0, "solve_for":"n_total|power|effect_size"}}
   {"anova_oneway": {"effect_size": 0.25, "k_groups": 3, "alpha": 0.05, "power": 0.8, "solve_for":"n_per_group|power|effect_size"}}
If the user mentions a formula like y ~ x + z, emit action="ols" (or glm if they say logistic/poisson).
If something is unclear, make a best guess and include minimal defaults.
NEVER include explanatory prose; JSON ONLY for tool calls.
For general questions (not a tool call), answer in 1–3 sentences of plain text.
"""

FALLBACK_PROMPT = """
You are SpatChat, a concise statistics tutor. If you can't map to a tool call, answer naturally in <=3 sentences.
"""

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


def df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False)


# ---------- ANALYSES ----------

@dataclass
class ModelOutput:
    kind: str
    summary_text: str
    table: Optional[pd.DataFrame] = None
    plots: Optional[List[Tuple[str, Optional[bytes]]]] = None


def run_ttest(df: pd.DataFrame, value: str, group: str, paired=False, equal_var=False) -> ModelOutput:
    gvals = df[group].dropna().unique().tolist()
    if len(gvals) != 2:
        raise gr.Error("t-test requires exactly two groups in the group column.")
    a = df[df[group] == gvals[0]][value].dropna()
    b = df[df[group] == gvals[1]][value].dropna()
    if paired:
        if len(a) != len(b):
            raise gr.Error("Paired t-test requires equal-length paired samples.")
        stat, p = stats.ttest_rel(a, b, nan_policy="omit")
        test_name = "Paired t-test"
    else:
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
        test_name = "Welch t-test" if not equal_var else "Student t-test"

    # Boxplot
    fig_box, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot([a, b], labels=[str(gvals[0]), str(gvals[1])])
    ax.set_title("Boxplot by Group")
    png_box = fig_to_png(fig_box)

    # Violin
    fig_vio, ax = plt.subplots(figsize=(5, 3))
    ax.violinplot([a, b], showmeans=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([str(gvals[0]), str(gvals[1])])
    ax.set_title("Violin by Group")
    png_vio = fig_to_png(fig_vio)

    table = pd.DataFrame({
        "group": [str(gvals[0]), str(gvals[1])],
        "n": [len(a), len(b)],
        "mean": [a.mean(), b.mean()],
        "sd": [a.std(ddof=1), b.std(ddof=1)]
    })
    txt = f"{test_name}: t={stat:.4f}, p={p:.4g}"
    return ModelOutput("t-test", txt, table, [("Boxplot", png_box), ("Violin", png_vio)])


def run_anova(df: pd.DataFrame, value: str, group: str) -> ModelOutput:
    groups = [g[value].dropna().values for _, g in df[[group, value]].dropna().groupby(group)]
    if len(groups) < 2:
        raise gr.Error("ANOVA requires at least two groups.")
    F, p = stats.f_oneway(*groups)

    # Boxplot
    fig_box, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column=value, by=group, ax=ax)
    ax.set_title("Boxplot by Group"); ax.figure.suptitle("")
    png_box = fig_to_png(fig_box)

    # Violin
    fig_vio, ax = plt.subplots(figsize=(5, 3))
    data = [g[value].dropna().values for _, g in df[[group, value]].groupby(group)]
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([str(k) for k in df[group].dropna().unique().tolist()])
    ax.set_title("Violin by Group")
    png_vio = fig_to_png(fig_vio)

    txt = f"One-way ANOVA: F={F:.4f}, p={p:.4g}"
    return ModelOutput("ANOVA", txt, None, [("Boxplot", png_box), ("Violin", png_vio)])


def run_ols(df: pd.DataFrame, formula: str) -> ModelOutput:
    model = smf.ols(formula, data=df).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})

    # Scatter + fit if single numeric predictor
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    terms = [t.strip() for t in re.split(r"\+|:|\*", rhs) if t.strip()]
    fig_fit: Optional[bytes] = None
    if len(terms) == 1 and terms[0] in df.columns and pd.api.types.is_numeric_dtype(df[terms[0]]):
        x = df[terms[0]]; y = df[lhs]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(x, y)
        order = np.argsort(x.values)
        ax.plot(x.values[order], model.fittedvalues.values[order])
        ax.set_xlabel(terms[0]); ax.set_ylabel(lhs); ax.set_title("Scatter + OLS fit")
        fig_fit = fig_to_png(fig)

    # residuals + QQ
    fig_r, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(model.fittedvalues, model.resid)
    ax.axhline(0, linestyle=":")
    ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted")
    png_r = fig_to_png(fig_r)

    sm.qqplot(model.resid, line="45", fit=True)
    png_qq = fig_to_png(plt.gcf())

    return ModelOutput("OLS", str(model.summary()), coef, [("Scatter+Fit", fig_fit), ("Residuals", png_r), ("QQ", png_qq)])


def run_glm(df: pd.DataFrame, formula: str, family: str) -> ModelOutput:
    fam_map = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
    }
    fam = fam_map.get(family, sm.families.Gaussian())
    model = smf.glm(formula, data=df, family=fam).fit()
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})

    # residuals + QQ (deviance)
    fig_r, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(model.fittedvalues, model.resid_deviance)
    ax.axhline(0, linestyle=":")
    ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs Fitted (GLM)")
    png_r = fig_to_png(fig_r)

    sm.qqplot(model.resid_deviance, line="45", fit=True)
    png_qq = fig_to_png(plt.gcf())

    return ModelOutput(f"GLM ({family})", str(model.summary()), coef, [("Residuals", png_r), ("QQ", png_qq)])


# ---------- PLOTS & CHECKS ----------

def plot_hist(df: pd.DataFrame, col: str, bins: int = 30) -> Tuple[str, bytes]:
    series = df[col].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(series, bins=bins)
    ax.set_title(f"Histogram of {col}")
    png = fig_to_png(fig)
    return f"Histogram {col}", png


def plot_box(df: pd.DataFrame, value: str, group: str) -> Tuple[str, bytes]:
    fig, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column=value, by=group, ax=ax)
    ax.set_title(f"Boxplot of {value} by {group}"); ax.figure.suptitle("")
    png = fig_to_png(fig)
    return f"Boxplot {value}~{group}", png


def plot_violin(df: pd.DataFrame, value: str, group: str) -> Tuple[str, bytes]:
    fig, ax = plt.subplots(figsize=(5, 3))
    data = [g[value].dropna().values for _, g in df[[group, value]].groupby(group)]
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([str(k) for k in df[group].dropna().unique().tolist()])
    ax.set_title(f"Violin of {value} by {group}")
    png = fig_to_png(fig)
    return f"Violin {value}~{group}", png


def check_normality(df: pd.DataFrame, col: str) -> Tuple[str, Optional[bytes]]:
    series = df[col].dropna().astype(float)
    stat, p = stats.shapiro(series) if len(series) <= 5000 else (np.nan, np.nan)
    msg = f"Shapiro–Wilk (n={len(series)}): W={stat:.4f}, p={p:.4g}" if not np.isnan(stat) else "Shapiro skipped (n>5000)."
    sm.qqplot(series, line="45", fit=True)
    png = fig_to_png(plt.gcf())
    return msg, png


def check_homogeneity(df: pd.DataFrame, value: str, group: str) -> str:
    arrays = [g[value].dropna().values for _, g in df[[group, value]].groupby(group)]
    stat, p = stats.levene(*arrays)
    return f"Levene test for equal variances: W={stat:.4f}, p={p:.4g}"


# ---------- POWER ----------

def power_ttest_ind(effect_size: Optional[float], alpha: float, power: Optional[float], ratio: float, solve_for: str) -> str:
    tool = TTestIndPower()
    if solve_for == "n_total":
        n = tool.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=ratio, alternative="two-sided")
        return f"Required total sample size (two-sample t-test): n_total ≈ {np.ceil((1+ratio)*n).astype(int)} (group1 ≈ {np.ceil(n).astype(int)}, group2 ≈ {np.ceil(n*ratio).astype(int)})"
    elif solve_for == "power":
        pw = tool.solve_power(effect_size=effect_size, nobs1=None, alpha=alpha, ratio=ratio, alternative="two-sided")
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
        return f"Required n per group (one-way ANOVA): ≈ {np.ceil(n).astype(int)}"
    elif solve_for == "power":
        pw = tool.solve_power(effect_size=effect_size, k_groups=k_groups, alpha=alpha, nobs=None)
        return f"Achieved power ≈ {pw:.3f}"
    elif solve_for == "effect_size":
        es = tool.solve_power(effect_size=None, k_groups=k_groups, alpha=alpha, nobs=power)
        return f"Implied effect size f ≈ {es:.3f}"
    else:
        return "Unknown solve_for for ANOVA power."


# ---------- FILE/ZIP ----------

def write_report(sections: List[Tuple[str, str, Optional[bytes]]]) -> str:
    parts = ["<h1>SpatChat – Stats Report</h1>"]
    for title, html_text, png in sections:
        add_to_report(parts, title, f"<pre style='white-space:pre-wrap'>{html_text}</pre>")
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


# ---------- CHAT HANDLERS ----------

def ask_llm(chat_history, user_input):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history + [{"role": "user", "content": user_input}]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.0
    ).choices[0].message.content
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
            temperature=0.7
        ).choices[0].message.content
        return None, conv


def handle_upload(file):
    global cached_df
    clear_outputs()
    try:
        df = pd.read_csv(file)
        cached_df = df
        cols = ", ".join(df.columns.astype(str))
        return [{"role": "assistant", "content": f"CSV uploaded. Columns detected: {cols}. Ask me for t-test, ANOVA, OLS/GLM, histograms, box/violin, normality checks, or power analysis."}], gr.update(visible=True)
    except Exception as e:
        return [{"role": "assistant", "content": f"Failed to read CSV: {e}"}], gr.update(visible=False)


def handle_chat(chat_history, user_message):
    global cached_df
    chat_history = list(chat_history)
    tool, llm_output = ask_llm(chat_history, user_message)

    if tool and tool.get("tool") == "stats":
        action = tool.get("action"); args = tool.get("args", {})
        if cached_df is None:
            chat_history.append({"role": "assistant", "content": "Please upload a CSV first."})
            return chat_history, gr.update(value=None), gr.update(value=None, visible=False)
        df = cached_df.copy()
        sections: List[Tuple[str, str, Optional[bytes]]] = []
        images: List[bytes] = []
        # --- Dispatch ---
        try:
            if action == "ttest":
                out = run_ttest(df, value=args.get("value"), group=args.get("group"), paired=bool(args.get("paired", False)), equal_var=bool(args.get("equal_var", False)))
                if out.table is not None:
                    out.table.to_csv(os.path.join(outputs_dir, "ttest_table.csv"), index=False)
                for title, png in (out.plots or []):
                    if png:
                        images.append(png)
                        save_png(png, f"plot_{title.replace(' ', '_').lower()}.png")
                sections.append((out.kind, out.summary_text, None))

            elif action == "anova":
                out = run_anova(df, value=args.get("value"), group=args.get("group"))
                for title, png in (out.plots or []):
                    if png:
                        images.append(png)
                        save_png(png, f"plot_{title.replace(' ', '_').lower()}.png")
                sections.append((out.kind, out.summary_text, None))

            elif action == "ols":
                out = run_ols(df, formula=args.get("formula"))
                if out.table is not None:
                    out.table.to_csv(os.path.join(outputs_dir, "ols_coef.csv"), index=False)
                for title, png in (out.plots or []):
                    if png:
                        images.append(png)
                        save_png(png, f"plot_{title.replace(' ', '_').lower()}.png")
                sections.append((out.kind, out.summary_text, None))

            elif action == "glm":
                out = run_glm(df, formula=args.get("formula"), family=args.get("family", "gaussian"))
                if out.table is not None:
                    out.table.to_csv(os.path.join(outputs_dir, "glm_coef.csv"), index=False)
                for title, png in (out.plots or []):
                    if png:
                        images.append(png)
                        save_png(png, f"plot_{title.replace(' ', '_').lower()}.png")
                sections.append((out.kind, out.summary_text, None))

            elif action == "plot":
                if "hist" in args:
                    p = args["hist"]; title, png = plot_hist(df, p.get("col"), int(p.get("bins", 30)))
                    if png:
                        images.append(png)
                        save_png(png, f"plot_hist_{p.get('col')}.png")
                    sections.append((title, "", png))
                if "box" in args:
                    p = args["box"]; title, png = plot_box(df, p.get("value"), p.get("group"))
                    if png:
                        images.append(png)
                        save_png(png, f"plot_box_{p.get('value')}_{p.get('group')}.png")
                    sections.append((title, "", png))
                if "violin" in args:
                    p = args["violin"]; title, png = plot_violin(df, p.get("value"), p.get("group"))
                    if png:
                        images.append(png)
                        save_png(png, f"plot_violin_{p.get('value')}_{p.get('group')}.png")
                    sections.append((title, "", png))

            elif action == "check":
                if "normality" in args:
                    p = args["normality"]; msg, png = check_normality(df, p.get("col"))
                    if png:
                        images.append(png)
                        save_png(png, f"plot_qq_{p.get('col')}.png")
                    sections.append(("Normality", msg, png))
                if "homogeneity" in args:
                    p = args["homogeneity"]; msg = check_homogeneity(df, p.get("value"), p.get("group"))
                    sections.append(("Homogeneity of variances", msg, None))

            elif action == "power":
                if "ttest_ind" in args:
                    p = args["ttest_ind"]
                    msg = power_ttest_ind(p.get("effect_size"), float(p.get("alpha", 0.05)), p.get("power"), float(p.get("ratio", 1.0)), p.get("solve_for", "n_total"))
                    sections.append(("Power – t-test (ind)", msg, None))
                if "anova_oneway" in args:
                    p = args["anova_oneway"]
                    msg = power_anova_oneway(p.get("effect_size"), int(p.get("k_groups", 3)), float(p.get("alpha", 0.05)), p.get("power"), p.get("solve_for", "n_per_group"))
                    sections.append(("Power – ANOVA (one-way)", msg, None))

            else:
                sections.append(("Note", f"Unknown action: {action}", None))
        except Exception as e:
            sections.append(("Error", str(e), None))

        # Build HTML report and ZIP
        _ = write_report(sections)
        zip_fp = save_zip()
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "Done. See preview and use Download Results."})

        preview = images[-1] if images else None
        return chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True)

    # Not a tool call → natural language reply
    if llm_output:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": llm_output})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False)
    else:
        chat_history.append({"role": "assistant", "content": "How can I help? Upload a CSV and try: 'ttest value=mpg group=am', 'ols mpg ~ wt + hp', 'plot hist col=hp', or 'power ttest_ind effect_size=0.5 power=0.8'."})
        return chat_history, gr.update(value=None), gr.update(value=None, visible=False)


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

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role": "assistant", "content": "Welcome! Upload a CSV, then ask: t-test, ANOVA, OLS/GLM, histogram, box/violin, normality, or power analysis."}]
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder="e.g., ols mpg ~ wt + hp", lines=1)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        with gr.Column(scale=3):
            preview_plot = gr.Image(label="Preview (last figure)")
            download_btn = gr.DownloadButton("📥 Download Results", value=None, visible=False)

    file_input.change(handle_upload, inputs=file_input, outputs=[chatbot, download_btn])
    user_input.submit(handle_chat, inputs=[chatbot, user_input], outputs=[chatbot, preview_plot, download_btn])
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
