# app.py (full — core logic + existing UI layout preserved)
# Wires together LLM, parsing, stats submodules, and handlers used by the UI layer.

# =========================
# Imports
# =========================
import os
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr  # used for gr.Error and gr.update return values
from dotenv import load_dotenv

# LLM
from llm_utils import UnifiedLLM

# Parsing & prompts
from parsing_utils import (
    set_llm,
    ask_llm,
    sanitize_args,
    local_parse,
    clarify_message,
)

# Report / outputs
from report_utils import (
    write_report,
    save_zip,
    clear_outputs,
    save_image_np,
)

# Plot/theme
from plot_helpers import set_house_style

# Core utils for schema-ish decisions
from core_utils import usable_numeric_cols, is_integer_like

# Stats API (from stats/ package)
from stats import (
    # two-group & paired
    ttest_summary,
    run_ttest,
    mann_whitney,
    wilcoxon_signed,
    point_biserial,
    # k-group & variance + posthocs
    run_anova,
    kruskal_wallis,
    levene_test,
    tukey_hsd,
    dunn_posthoc,
    # correlations
    corr_pair,
    corr_matrix_plot,
    partial_corr,
    # categorical associations
    chisq_test,
    # modeling
    run_ols,
    run_glm,
    # descriptives & checks
    plot_hist,
    plot_box,
    plot_violin,
    check_normality,
    # power
    power_ttest_ind,
    power_anova_oneway,
    # recommendations & summaries
    quick_summary,
    recommend_text_and_examples,
)

print("Starting SpatChat: Stats Room (chat-first)")

# =========================
# Globals
# =========================
cached_df: Optional[pd.DataFrame] = None
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Track if we asked a follow-up
pending: Dict[str, Optional[str]] = {"action": None, "need": None, "args": None}

# =========================
# LLM init & styling
# =========================
load_dotenv()
llm = UnifiedLLM()
set_llm(llm)  # provide the LLM client to parsing_utils
set_house_style()  # apply house plotting theme once


# =========================
# Small helpers
# =========================
def _fence(text: str) -> str:
    """Wrap multi-line stats text to prevent Markdown table parsing in the chat UI."""
    from textwrap import dedent
    return "```text\n" + dedent(str(text)).strip() + "\n```"


def _safe_last(paths: List[str]) -> Tuple[Optional[str], List[str], int]:
    if paths:
        return paths[-1], paths, len(paths) - 1
    return None, [], -1


def _step(paths: List[str], idx: int, delta: int) -> Tuple[Optional[str], int]:
    if not paths:
        return None, -1
    n = len(paths)
    new_idx = (idx + delta) % n
    return paths[new_idx], new_idx


# --- Case-insensitive column resolution (supports *, data, dataset, everything) ---
SPECIAL_ALL = {"*", "data", "dataset", "everything"}

def resolve_col(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    if s.lower() in SPECIAL_ALL:
        return s  # understood by quick_summary
    cols = list(df.columns)
    if s in cols:
        return s
    lower_map = {c.lower(): c for c in cols}
    return lower_map.get(s.lower())

def resolve_group(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    cols = list(df.columns)
    if s in cols:
        return s
    lower_map = {c.lower(): c for c in cols}
    return lower_map.get(s.lower())


# =========================
# Data & chat handlers (used by UI)
# =========================
def handle_upload(file):
    """
    UI wires: file_input.change(handle_upload, ...)
    Returns: (chat_history, data_preview_component, download_btn_component)
    """
    global cached_df
    clear_outputs()
    pending.update({"action": None, "need": None, "args": None})
    try:
        df = pd.read_csv(file)
        cached_df = df
        cols = ", ".join(map(str, df.columns))
        preview = df.head(200)
        return (
            [
                {
                    "role": "assistant",
                    "content": (
                        "CSV uploaded. Columns detected: "
                        f"{cols}. Ask me for t-test, ANOVA (+ Tukey), Kruskal (+ Dunn), "
                        "correlations/partial/pbiserial, chi-square/Fisher, rank tests, Levene’s, "
                        "OLS/GLM, histogram, box/violin/bar, normality, power, quick summaries, "
                        "or 'what can I do with my data?'"
                    ),
                }
            ],
            gr.update(value=preview, visible=True),
            gr.update(visible=True),
        )
    except Exception as e:
        return (
            [{"role": "assistant", "content": f"Failed to read CSV: {e}"}],
            gr.update(visible=False),
            gr.update(visible=False),
        )


def need_value_and_group(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Helper used when the user omits value/group; prefers common group-like columns.
    """
    nums = usable_numeric_cols(df)
    groups = []
    for c in df.columns:
        if df[c].nunique(dropna=True) >= 2:
            if not pd.api.types.is_numeric_dtype(df[c]) or is_integer_like(df[c]):
                groups.append(c)
    prefer = ["sex", "group", "treatment", "class", "category"]
    ordered_g = [c for c in prefer if c in groups] + [c for c in groups if c not in prefer]
    return nums, ordered_g


def handle_chat(chat_history, user_message, data_preview):
    """
    Core router for user requests. The UI passes (chat_history, user_input, data_preview)
    and expects:
      (chat_history, preview_plot_component, download_btn_component,
       data_preview, gallery_paths_state, gallery_index_state)
    """
    global cached_df, pending
    chat_history = list(chat_history)
    text = str(user_message or "").strip()

    parsed = None
    llm_error_text = None
    try:
        tool, _ = ask_llm(chat_history, text)
        parsed = tool
    except Exception as e:
        llm_error_text = f"(LLM unavailable: {e})"

    if not parsed:
        lp = local_parse(text)
        if lp:
            parsed = lp

    if pending["action"] and parsed and parsed.get("tool") == "stats":
        pending = {"action": None, "need": None, "args": None}

    if not parsed or parsed.get("tool") != "stats":
        msg = clarify_message(cached_df, llm_error_text, text)
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (
            chat_history,
            gr.update(value=preview),
            gr.update(value=None, visible=False),
            data_preview,
            paths,
            idx,
        )

    if cached_df is None:
        msg = "Please upload a CSV first."
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (
            chat_history,
            gr.update(value=preview),
            gr.update(value=None, visible=False),
            data_preview,
            paths,
            idx,
        )

    action = parsed.get("action")
    args = sanitize_args(action, parsed.get("args", {}))
    df = cached_df.copy()

    sections: List[Tuple[str, str, Optional[np.ndarray]]] = []
    image_paths: List[str] = []

    try:
        # -------- Recommendations --------
        if action == "recommend":
            tech, examples = recommend_text_and_examples(df)
            msg = "Recommendations\n" + tech + "\n\nSay it like this\n" + examples
            sections.append(("Recommendations", msg, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
            write_report(sections)
            zip_fp = save_zip()
            preview, paths, idx = _safe_last([])
            return (chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx)

        # -------- Summary (robust to case mismatches and stringy groups) --------
        if action == "summary":
            req_col = args.get("col")
            req_by  = args.get("by")

            # Resolve names safely (case-insensitive + supports *,data,… for col)
            col = resolve_col(df, req_col)
            by  = resolve_group(df, req_by) if req_by is not None else None

            if col is None:
                raise gr.Error(
                    f"Column '{req_col}' not found. Available: " + ", ".join(map(str, df.columns))
                )
            if by is not None and by not in df.columns:
                raise gr.Error(
                    f"Group column '{req_by}' not found. Available: " + ", ".join(map(str, df.columns))
                )

            # All numeric coercions happen inside quick_summary (errors='coerce'); no float-casting here.
            msg = quick_summary(df, col, by)
            sections.append(("Summary", msg, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])
            write_report(sections)
            zip_fp = save_zip()
            preview, paths, idx = _safe_last([])
            return (chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx)

        # -------- t-test --------
        if action == "ttest":
            value = args.get("value")
            group = args.get("group")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                pending.update({"action": "ttest", "need": "value" if not value else "group", "args": {"value": value, "group": group}})
                ask = ("Which numeric outcome should I test? Candidates: " + ", ".join(nums)
                       if not value else "Which group column? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = run_ttest(
                df, value=value, group=group,
                paired=bool(args.get("paired", False)),
                equal_var=bool(args.get("equal_var", False)),
            )
            sections.append(("t-test", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"ttest_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- ANOVA --------
        elif action == "anova":
            value = args.get("value"); group = args.get("group")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                pending.update({"action": "anova", "need": "value" if not value else "group", "args": {"value": value, "group": group}})
                ask = ("Which numeric outcome for ANOVA? Candidates: " + ", ".join(nums)
                       if not value else "Which group column for ANOVA? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = run_anova(df, value=value, group=group)
            sections.append(("ANOVA", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"anova_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- POSTHOC: TUKEY --------
        elif action == "posthoc_tukey":
            value = args.get("value"); group = args.get("group")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Tukey? Candidates: " + ", ".join(nums)
                       if not value else "Which group column (≥2 levels)? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = tukey_hsd(df, value=value, group=group)
            sections.append(("Post-hoc: Tukey HSD", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"tukey_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- KRUSKAL --------
        elif action == "kruskal":
            value = args.get("value"); group = args.get("group")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Kruskal–Wallis? Candidates: " + ", ".join(nums)
                       if not value else "Which group column (≥2 levels)? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = kruskal_wallis(df, value=value, group=group)
            sections.append(("Kruskal–Wallis", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"kruskal_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- POSTHOC: DUNN --------
        elif action == "posthoc_dunn":
            value = args.get("value"); group = args.get("group"); padj = args.get("p_adjust", "holm")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Dunn’s test? Candidates: " + ", ".join(nums)
                       if not value else "Which group column (≥2 levels)? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, _ = dunn_posthoc(df, value=value, group=group, p_adjust=padj)
            sections.append(("Post-hoc: Dunn’s", txt, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- CHI-SQUARE / FISHER --------
        elif action == "chisq":
            row = args.get("row"); col = args.get("col"); exact = bool(args.get("exact", False))
            if not row or not col:
                cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                ask = "Which two categorical columns for chi-square? Candidates: " + ", ".join(cats)
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = chisq_test(df, row=row, col=col, exact=exact)
            sections.append(("Chi-square / Fisher", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"chisq_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- CORRELATION(S) --------
        elif action == "corr":
            if bool(args.get("matrix", False)):
                method = args.get("method", "pearson")
                txt, imgs = corr_matrix_plot(df, cols=args.get("cols"), method=method)
                sections.append(("Correlation matrix", txt, None))
                for i, im in enumerate(imgs):
                    image_paths.append(save_image_np(im, f"corr_matrix_{method}_{i+1}.png"))
                chat_history.extend(
                    [{"role": "user", "content": text}, {"role": "assistant", "content": "Correlation matrix generated."}]
                )
            else:
                x = args.get("x"); y = args.get("y"); method = args.get("method", "pearson")
                if not x or not y:
                    nums = usable_numeric_cols(df)
                    ask = "Which two numeric columns for correlation? Candidates: " + ", ".join(nums)
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

                txt, imgs = corr_pair(df, x=x, y=y, method=method)
                sections.append(("Correlation", txt, None))
                for i, im in enumerate(imgs):
                    image_paths.append(save_image_np(im, f"corr_pair_{method}_{i+1}.png"))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- PARTIAL CORRELATION --------
        elif action == "pcorr":
            x = args.get("x"); y = args.get("y"); ctrls = args.get("controls", []); method = args.get("method", "pearson")
            if not x or not y or not ctrls:
                nums = usable_numeric_cols(df)
                ask = "Specify: partial correlation <x> and <y> controlling for <a, b>. Numeric candidates: " + ", ".join(nums)
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = partial_corr(df, x=x, y=y, controls=ctrls, method=method)
            sections.append(("Partial correlation", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"pcorr_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- POINT-BISERIAL --------
        elif action == "pbiserial":
            value = args.get("value"); group = args.get("group")
            if not value or not group:
                nums = usable_numeric_cols(df)
                groups = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
                ask = ("Which numeric outcome for point-biserial? Candidates: " + ", ".join(nums)
                       if not value else "Which binary group? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = point_biserial(df, value=value, group=group)
            sections.append(("Point-biserial correlation", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"pbiserial_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- MANN–WHITNEY --------
        elif action == "mwutest":
            value = args.get("value"); group = args.get("group")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Mann–Whitney? Candidates: " + ", ".join(nums)
                       if not value else "Which group column (2 levels needed)? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = mann_whitney(df, value=value, group=group)
            sections.append(("Mann–Whitney U", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"mwutest_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- WILCOXON SIGNED-RANK --------
        elif action == "wilcoxon":
            a = args.get("a"); b = args.get("b")
            if not a or not b:
                nums = usable_numeric_cols(df)
                ask = "Which two columns are paired? Example: 'wilcoxon pre vs post'. Candidates: " + ", ".join(nums)
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = wilcoxon_signed(df, a=a, b=b)
            sections.append(("Wilcoxon signed-rank", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"wilcoxon_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- LEVENE --------
        elif action == "levene":
            value = args.get("value"); group = args.get("group"); center = args.get("center", "median")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Levene? Candidates: " + ", ".join(nums)
                       if not value else "Which group column? Candidates: " + ", ".join(groups))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, _ = levene_test(df, value=value, group=group, center=center)
            sections.append(("Levene’s test", txt, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- OLS --------
        elif action == "ols":
            formula = args.get("formula")
            if not formula or "~" not in formula:
                ask = "Please specify a formula like: ols score ~ age + weight."
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = run_ols(df, formula=formula)
            sections.append(("OLS", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"ols_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- GLM --------
        elif action == "glm":
            formula = args.get("formula"); family = args.get("family", "gaussian")
            if not formula or "~" not in formula:
                ask = "Please specify a formula like: glm count ~ age with family=poisson."
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = run_glm(df, formula=formula, family=family)
            sections.append((f"GLM ({family})", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"glm_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- PLOTS --------
        elif action == "plot":
            # Histogram
            if "hist" in args:
                p = args["hist"]
                title, im = plot_hist(df, p.get("col"), int(p.get("bins", 30)))
                image_paths.append(save_image_np(im, f"plot_hist_{p.get('col')}.png"))
                sections.append((title, "", im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": f"{title} generated."}])

            # Box
            if "box" in args:
                p = args["box"]; grp = p.get("group")
                if grp is None:
                    _, groups = need_value_and_group(df)
                    ask = "Which group column for the box plot? Candidates: " + ", ".join(groups)
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)
                if p.get("value") == grp:
                    raise gr.Error("Outcome and group must be different.")
                title, im = plot_box(df, p.get("value"), grp)
                image_paths.append(save_image_np(im, f"plot_box_{p.get('value')}_{grp}.png"))
                sections.append((title, "", im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": f"{title} generated."}])

            # Violin
            if "violin" in args:
                p = args["violin"]; grp = p.get("group")
                if grp is None:
                    _, groups = need_value_and_group(df)
                    ask = "Which group column for the violin plot? Candidates: " + ", ".join(groups)
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)
                if p.get("value") == grp:
                    raise gr.Error("Outcome and group must be different.")
                title, im = plot_violin(df, p.get("value"), grp)
                image_paths.append(save_image_np(im, f"plot_violin_{p.get('value')}_{grp}.png"))
                sections.append((title, "", im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": f"{title} generated."}])

            # Bar
            if "bar" in args:
                p = args["bar"]; grp = p.get("group")
                if grp is None:
                    _, groups = need_value_and_group(df)
                    ask = "Which group column for the bar chart? Candidates: " + ", ".join(groups)
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)
                if p.get("value") == grp:
                    raise gr.Error("Outcome and group must be different.")
                err = p.get("error", "sem")
                from plot_helpers import bar_with_error_plot  # local import to avoid circulars
                gorder = df[grp].dropna().astype(str).unique().tolist()
                im = bar_with_error_plot(df, p.get("value"), grp, error=err, gorder=gorder)
                image_paths.append(save_image_np(im, f"plot_bar_{p.get('value')}_{grp}_{err}.png"))
                sections.append((f"Bar {p.get('value')}~{grp}", "", im))
                chat_history.extend(
                    [{"role": "user", "content": text}, {"role": "assistant", "content": f"Bar ± {str(err).upper()} generated."}]
                )

        # -------- CHECKS --------
        elif action == "check":
            if "normality" in args:
                p = args["normality"]
                msg, im = check_normality(df, p.get("col"))
                image_paths.append(save_image_np(im, f"plot_qq_{p.get('col')}.png"))
                sections.append(("Normality", msg, im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])

        # -------- POWER --------
        elif action == "power":
            if "ttest_ind" in args:
                p = args["ttest_ind"]
                msg = power_ttest_ind(
                    p.get("effect_size", 0.5),
                    float(p.get("alpha", 0.05)),
                    p.get("power", 0.8),
                    float(p.get("ratio", 1.0)),
                    p.get("solve_for", "n_total"),
                )
                sections.append(("Power – t-test (ind)", msg, None))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])
            if "anova_oneway" in args:
                p = args["anova_oneway"]
                msg = power_anova_oneway(
                    p.get("effect_size", 0.25),
                    int(p.get("k_groups", 3)),
                    float(p.get("alpha", 0.05)),
                    p.get("power", 0.8),
                    p.get("solve_for", "n_per_group"),
                )
                sections.append(("Power – ANOVA (one-way)", msg, None))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])

        else:
            msg = clarify_message(df, llm_error_text, text)
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])

    except gr.Error as e:
        msg = f"Sorry — {e}"
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

    except Exception as e:
        msg = f"Sorry — something went wrong: {e}"
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

    # Finalize report & zip, prep gallery preview
    write_report(sections)
    zip_fp = save_zip()
    preview, paths, idx = _safe_last(image_paths)
    return (chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx)


# =========================
# Optional: gallery nav helpers the UI may bind to buttons
# =========================
def on_prev(paths, idx):
    preview, new_idx = _step(paths, idx, -1)
    return gr.update(value=preview), new_idx


def on_next(paths, idx):
    preview, new_idx = _step(paths, idx, +1)
    return gr.update(value=preview), new_idx

# --------------------------
# UI (layout preserved)
# --------------------------
with gr.Blocks(title="SpatChat: Stats Room") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img",
    )
    gr.HTML(
        """
<style>
#logo-img img { height: 90px; margin: 10px 50px 10px 10px; border-radius: 6px; }
.dataframe-wrap { max-height: 340px; overflow: auto; border: 1px solid #ddd; border-radius: 6px; padding: 6px; }
#preview-plot { margin-bottom: 6px; }
#plot-nav { margin: 0 0 10px 0; display: flex; gap: 8px; justify-content: flex-end; }
</style>
"""
    )
    gr.Markdown("## 📊 SpatChat: Stats Room {stats}")
    gr.HTML(
        """
<div style="margin-top: -10px; margin-bottom: 15px;">
  <input type="text" value="https://spatchat.org/browse/?room=stats" id="shareLink" readonly
         style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
  <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)"
          style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
    📋 Copy Share Link
  </button>
  <div style="margin-top: 10px; font-size: 14px;">
    <b>Share:</b>
    <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=stats" target="_blank">🐦 Twitter</a> |
    <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=stats" target="_blank">📘 Facebook</a>
  </div>
</div>
"""
    )
    gr.Markdown(
        """
<div style="font-size: 14px;">
© 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
If you use SpatChat in research, please cite:<br>
<b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Stats Room.</i>
</div>
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[
                    {
                        "role": "assistant",
                        "content": "Welcome! Upload a CSV, then ask: t-test, ANOVA (+Tukey), Kruskal (+Dunn), correlations (Pearson/Spearman), partial/pbiserial, chi-square/Fisher, rank tests (Mann-Whitney, Wilcoxon), Levene’s, OLS/GLM, histogram, box/violin/bar, normality, power, quick summaries, or 'what can I do with my data?'.",
                    }
                ],
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder="e.g., Tukey HSD score by group", lines=1)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])

        with gr.Column(scale=3):
            preview_plot = gr.Image(label="Preview (last figure)", value=None, type="filepath", elem_id="preview-plot")
            with gr.Row(elem_id="plot-nav"):
                prev_btn = gr.Button("◀️ Prev", variant="secondary")
                next_btn = gr.Button("Next ▶️", variant="secondary")
            data_preview = gr.Dataframe(label="Data Preview (first 200 rows)", interactive=False, visible=False)
            download_btn = gr.DownloadButton("📥 Download Results", value=None, visible=False)
            gallery_paths = gr.State([])
            gallery_index = gr.State(-1)

    demo.queue(max_size=16)

    file_input.change(
        handle_upload,
        inputs=file_input,
        outputs=[chatbot, data_preview, download_btn],
    )

    user_input.submit(
        handle_chat,
        inputs=[chatbot, user_input, data_preview],
        outputs=[chatbot, preview_plot, download_btn, data_preview, gallery_paths, gallery_index],
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

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

