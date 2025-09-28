# parsing_utils.py

from typing import Dict, List, Optional
import json
import re
import pandas as pd

# pull dataset-aware guidance for clarify_message()
from stats.recommendations import recommend_text_and_examples

# --------------------------
# PROMPTS (unchanged)
# --------------------------
SYSTEM_PROMPT = """
You are SpatChat, an expert statistics assistant for basic analyses.
When the user asks for an analysis, respond ONLY in compact JSON using this schema:
{"tool":"stats","action":"ttest|anova|ols|glm|plot|check|power|summary|recommend|chisq|corr|mwutest|wilcoxon|kruskal|levene|posthoc_tukey|posthoc_dunn|pcorr|pbiserial","args":{...}}

Actions and args:
- ttest: {"value":"colY","group":"colG", "paired": false, "equal_var": false}
- anova: {"value":"colY","group":"colG"}
- posthoc_tukey: {"value":"colY","group":"colG"}
- kruskal: {"value":"colY","group":"colG"}
- posthoc_dunn: {"value":"colY","group":"colG","p_adjust":"holm"}
- ols: {"formula":"y ~ x1 + x2"}
- glm: {"formula":"y ~ x1 + x2", "family":"gaussian|binomial|poisson|gamma"}
- chisq: {"row":"catA","col":"catB","exact": false}
- corr: {"x":"col1","y":"col2","method":"pearson|spearman","matrix": false, "cols": ["colA","colB", "..."]}
- pcorr: {"x":"col1","y":"col2","controls":["c1","c2"],"method":"pearson|spearman"}
- pbiserial: {"value":"numericCol","group":"binaryCol"}
- mwutest: {"value":"colY","group":"colG"}
- wilcoxon: {"a":"colA","b":"colB"}
- levene: {"value":"colY","group":"colG","center":"median|mean|trimmed"}
- plot: one of {"hist":{"col":"c","bins":30}, "box":{"value":"y","group":"g"}, "violin":{"value":"y","group":"g"}, "bar":{"value":"y","group":"g","error":"sem|sd|ci95"}}
- check: {"normality":{"col":"y"}}
- power: one of
  {"ttest_ind": {"effect_size": 0.5, "alpha": 0.05, "power": 0.8, "ratio": 1.0, "solve_for":"n_total|power|effect_size"}}
  {"anova_oneway": {"effect_size": 0.25, "k_groups": 3, "alpha": 0.05, "power": 0.8, "solve_for":"n_per_group|power|effect_size"}}
- summary: {"col":"y", "by":"group_col_or_null"}
- recommend: {}

If unclear, pick sensible defaults from the dataset.
NEVER include prose; JSON only.
For general questions, answer in <=3 sentences of plain text.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a concise statistics tutor. If you can't map to a tool call, answer naturally in <=3 sentences.
""".strip()

# --------------------------
# LLM bridge (module-level setter to avoid UI signature changes)
# --------------------------
_LLM = None  # set once by app.py via set_llm()

def set_llm(llm_obj) -> None:
    """Call this once in app.py after creating the UnifiedLLM: parsing_utils.set_llm(llm)."""
    global _LLM
    _LLM = llm_obj

def ask_llm(chat_history, user_input):
    """
    Keep original signature used by app.py.
    Relies on a module-level LLM set via set_llm().
    Returns (parsed_tool_dict_or_None, raw_text_response)
    """
    if _LLM is None:
        raise RuntimeError("LLM client is not set. Call parsing_utils.set_llm(llm) in app.py after creating the client.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(chat_history) + [{"role": "user", "content": user_input}]
    resp = _LLM.chat(messages=messages, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = _LLM.chat(
            messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
            temperature=0.7,
            max_tokens=256,
            stream=False,
        )
        return None, conv

# --------------------------
# Argument sanitization
# --------------------------
_NULLY_STRINGS = {"", "null", "none", "na", "n/a", "nil"}

def clean_null_like(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in _NULLY_STRINGS else v

def sanitize_args(action: Optional[str], args: Dict) -> Dict:
    if not isinstance(args, dict):
        return {}
    if action == "summary":
        args["by"] = clean_null_like(args.get("by"))
    elif action in {"ttest", "anova", "mwutest", "kruskal", "levene", "posthoc_tukey", "posthoc_dunn", "pbiserial"}:
        args["group"] = clean_null_like(args.get("group"))
    elif action == "plot":
        for k in ("box", "violin", "bar"):
            if k in args and isinstance(args[k], dict):
                args[k]["group"] = clean_null_like(args[k].get("group"))
    elif action == "chisq":
        args["row"] = clean_null_like(args.get("row"))
        args["col"] = clean_null_like(args.get("col"))
    elif action == "corr":
        args["x"] = clean_null_like(args.get("x"))
        args["y"] = clean_null_like(args.get("y"))
    elif action == "pcorr":
        args["x"] = clean_null_like(args.get("x"))
        args["y"] = clean_null_like(args.get("y"))
        if isinstance(args.get("controls"), list):
            args["controls"] = [c for c in args["controls"] if clean_null_like(c)]
    elif action == "wilcoxon":
        args["a"] = clean_null_like(args.get("a"))
        args["b"] = clean_null_like(args.get("b"))
    return args

# --------------------------
# Local (regex) parser for plain-English commands
# --------------------------
def _split_controls(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,+]", s)
    return [p.strip() for p in parts if p.strip()]

def local_parse(user_input: str) -> Optional[Dict]:
    s = user_input.strip().lower()

    if re.search(r"\b(what can i do|how should i analyze|recommend|suggestion|what analyses)\b", s):
        return {"tool": "stats", "action": "recommend", "args": {}}

    m = re.search(
        r"(?:what(?:'s| is) the )?(?:mean|average|summary|summarize|describe)\s+([a-zA-Z0-9_]+)(?:\s+by\s+([a-zA-Z0-9_]+))?",
        s,
    )
    if m:
        col = m.group(1)
        by = m.group(2) if m.group(2) else None
        return {"tool": "stats", "action": "summary", "args": {"col": col, "by": by}}

    m = re.search(r"(t[\-\s]?test).*?(?:on|of)?\s*([a-zA-Z0-9_]+).*(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "ttest", "args": {"value": m.group(2), "group": m.group(3)}}

    m = re.search(r"(anova).*(?:on|of)?\s*([a-zA-Z0-9_]+).*(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "anova", "args": {"value": m.group(2), "group": m.group(3)}}

    # Post-hoc Tukey
    m = re.search(
        r"(?:tukey|hsd|post[\-\s]*hoc)(?:\s+test)?\s+(?:on|of|for)\s+([a-zA-Z0-9_]+)\s+(?:by|across|vs)\s+([a-zA-Z0-9_]+)",
        s,
    )
    if m:
        return {"tool": "stats", "action": "posthoc_tukey", "args": {"value": m.group(1), "group": m.group(2)}}

    # Kruskal–Wallis
    m = re.search(r"(kruskal|kw)(?:\s+test)?\s+(?:on|of|for)\s+([a-zA-Z0-9_]+)\s+(?:by|across)\s+([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "kruskal", "args": {"value": m.group(2), "group": m.group(3)}}

    # Dunn
    m = re.search(r"(dunn)(?:\s+test)?\s+(?:on|of|for)\s+([a-zA-Z0-9_]+)\s+(?:by|across)\s+([a-zA-Z0-9_]+)", s)
    if m:
        return {
            "tool": "stats",
            "action": "posthoc_dunn",
            "args": {"value": m.group(2), "group": m.group(3), "p_adjust": "holm"},
        }

    # Chi-square / Fisher
    m = re.search(
        r"(chi(?:-?square)?|chi2|chisq|fisher).*?(?:of|between|for)?\s*([a-zA-Z0-9_]+)\s*(?:by|x|vs|versus|and)\s*([a-zA-Z0-9_]+)",
        s,
    )
    if m:
        exact = "fisher" in m.group(1)
        return {"tool": "stats", "action": "chisq", "args": {"row": m.group(2), "col": m.group(3), "exact": exact}}

    # Correlation (pair)
    m = re.search(
        r"(pearson|spearman|correlation|correlate).*?(?:of|between)?\s*([a-zA-Z0-9_]+)\s*(?:and|&|,|vs|x)\s*([a-zA-Z0-9_]+)",
        s,
    )
    if m:
        method = "pearson" if "pearson" in m.group(1) else ("spearman" if "spearman" in m.group(1) else "pearson")
        return {"tool": "stats", "action": "corr", "args": {"x": m.group(2), "y": m.group(3), "method": method, "matrix": False}}

    # Correlation heatmap/matrix
    if re.search(r"(corr(?:elation)?\s*(matrix|heat\s*map|heatmap))", s):
        return {"tool": "stats", "action": "corr", "args": {"matrix": True, "method": "pearson"}}

    # Partial correlation
    m = re.search(
        r"(pcorr|partial\s+correlation).*?([a-zA-Z0-9_]+)\s*(?:and|&|,|vs|~)\s*([a-zA-Z0-9_]+)"
        r"(?:.*?(?:\||controlling|adjusting|for)\s*([a-zA-Z0-9_\s\+,\.;:]+))?",
        s,
    )
    if m:
        x, y = m.group(2), m.group(3)
        ctrls = _split_controls(m.group(4) or "")
        return {"tool": "stats", "action": "pcorr", "args": {"x": x, "y": y, "controls": ctrls, "method": "pearson"}}

    # Point-biserial
    m = re.search(r"(point[\-\s]*biserial|pbiserial).*?(?:of|between|on)?\s*([a-zA-Z0-9_]+)\s*(?:by|with|and)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "pbiserial", "args": {"value": m.group(2), "group": m.group(3)}}

    # Mann–Whitney
    m = re.search(r"(mann[\-\s]?whitney|wilcoxon\s*rank\s*sum|rank\s*test).*?(?:on|of)?\s*([a-zA-Z0-9_]+).*(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "mwutest", "args": {"value": m.group(2), "group": m.group(3)}}

    # Wilcoxon signed-rank (paired)
    m = re.search(r"(wilcoxon).*?([a-zA-Z0-9_]+)\s*(?:vs|and|,)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "wilcoxon", "args": {"a": m.group(2), "b": m.group(3)}}

    # Levene
    m = re.search(
        r"(levene).*?(?:on|of)?\s*([a-zA-Z0-9_]+).*(?:by|across)\s*([a-zA-Z0-9_]+)(?:.*center\s*=\s*(median|mean|trimmed))?",
        s,
    )
    if m:
        center = m.group(4) if m.group(4) else "median"
        return {"tool": "stats", "action": "levene", "args": {"value": m.group(2), "group": m.group(3), "center": center}}

    # OLS / GLM
    m = re.search(r"(?:ols|regression)\s+([a-zA-Z0-9_]+)\s*~\s*([a-zA-Z0-9_+\s]+)", s)
    if m:
        return {"tool": "stats", "action": "ols", "args": {"formula": f"{m.group(1)} ~ {m.group(2)}"}}

    m = re.search(r"(glm|poisson|binomial|gaussian|gamma)\s+([a-zA-Z0-9_]+)\s*~\s*([a-zA-Z0-9_+\s]+)", s)
    if m:
        fam = "gaussian" if m.group(1) == "glm" else m.group(1)
        return {"tool": "stats", "action": "glm", "args": {"formula": f"{m.group(2)} ~ {m.group(3)}", "family": fam}}

    # Plots
    m = re.search(r"(hist(?:ogram)?)\s+(?:of|on|for)?\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "plot", "args": {"hist": {"col": m.group(2), "bins": 30}}}

    m = re.search(r"(box|violin|bar)\s+(?:plot\s+)?(?:of|on|for)?\s*([a-zA-Z0-9_]+)\s+(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        kind, val, grp = m.group(1), m.group(2), m.group(3)
        if kind == "box":
            return {"tool": "stats", "action": "plot", "args": {"box": {"value": val, "group": grp}}}
        if kind == "violin":
            return {"tool": "stats", "action": "plot", "args": {"violin": {"value": val, "group": grp}}}
        if kind == "bar":
            return {"tool": "stats", "action": "plot", "args": {"bar": {"value": val, "group": grp, "error": "ci95"}}}

    # Normality / QQ
    m = re.search(r"(normality|qq)\s+(?:check|plot)?\s*(?:for|of)?\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "check", "args": {"normality": {"col": "{}".format(m.group(2))}}}

    # Power
    if "power" in s and "t-test" in s:
        return {
            "tool": "stats",
            "action": "power",
            "args": {"ttest_ind": {"effect_size": 0.5, "alpha": 0.05, "power": 0.8, "ratio": 1.0, "solve_for": "n_total"}},
        }

    if "power" in s and "anova" in s:
        return {
            "tool": "stats",
            "action": "power",
            "args": {"anova_oneway": {"effect_size": 0.25, "k_groups": 3, "alpha": 0.05, "power": 0.8, "solve_for": "n_per_group"}},
        }

    # Raw JSON fallback if user typed it
    try:
        maybe = json.loads(user_input)
        if isinstance(maybe, dict) and "tool" in maybe:
            return maybe
    except Exception:
        pass
    return None

# --------------------------
# Clarification helper
# --------------------------
def clarify_message(df: Optional[pd.DataFrame], llm_error_text: Optional[str], user_text: str) -> str:
    prefix = "Sorry — I didn’t understand that request."
    if llm_error_text:
        prefix = f"{prefix} {llm_error_text}"

    examples = [
        "What is the average score?",
        "Show a histogram of height.",
        "I want to do a t-test on score by sex.",
        "Kruskal-Wallis score by group.",
        "Tukey HSD score by group.",
        "Dunn's test score by group.",
        "Pearson correlation height and weight.",
        "Partial correlation height and score controlling for age, weight.",
        "Point-biserial score by sex.",
        "Chi-square sex by group.",
        "Check normality of score (QQ plot).",
    ]

    if df is not None:
        try:
            tech, ex = recommend_text_and_examples(df)
            return f"{prefix}\n\nCould you clarify what you want to do?\n\n{tech}\n\nSay it like this\n{ex}"
        except Exception:
            pass

    cols = []
    if df is not None:
        cols = list(map(str, df.columns))
    col_text = f"\nDetected columns: {', '.join(cols)}" if cols else ""
    return f"{prefix}\n\nTry one of these examples:\n• " + "\n• ".join(examples) + col_text
