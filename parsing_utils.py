# parsing_utils.py

from typing import Dict, List, Optional
import json
import re
import pandas as pd

# Pull dataset-aware guidance for clarify_message()
from stats.recommendations import recommend_text_and_examples

# --------------------------
# PROMPTS
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
# LLM bridge
# --------------------------
_LLM = None  # set once by app.py via set_llm()

def set_llm(llm_obj) -> None:
    """Called by app.py after creating the UnifiedLLM."""
    global _LLM
    _LLM = llm_obj

def ask_llm(chat_history, user_input):
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
# Local regex parser (UPDATED)
# --------------------------
def _split_controls(s: str) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in re.split(r"[,+]", s) if p.strip()]

def local_parse(user_input: str) -> Optional[Dict]:
    s = user_input.strip().lower()

    def _summary_args(by_val=None):
        return {"tool": "stats", "action": "summary", "args": {"col": "data", "by": by_val}}

    # --- Summarize data/dataset/all ---
    m = re.search(r"\b(summarize|summary|describe)\s+(?:the\s+)?(data|dataset|everything|\*)\s+(?:by|across)\s+([a-zA-Z0-9_]+)", s)
    if m:
        return _summary_args(by_val=m.group(3))

    # Just "summarize data/dataset" (no grouping)
    m = re.search(r"\b(summarize|summary|describe)\s+(?:the\s+)?(data|dataset|everything|\*)\b", s)
    if m:
        return _summary_args()

    # "summarize by sex"
    m = re.search(r"\b(summarize|summary|describe)\s+(?:by|across)\s+([a-zA-Z0-9_]+)", s)
    if m:
        return _summary_args(by_val=m.group(2))

    # Recommendation
    if re.search(r"\b(what can i do|how should i analyze|recommend|suggestion|what analyses)\b", s):
        return {"tool": "stats", "action": "recommend", "args": {}}

    # Summarize a specific column
    m = re.search(r"(?:mean|average|summary|summarize|describe)\s+([a-zA-Z0-9_]+)(?:\s+by\s+([a-zA-Z0-9_]+))?", s)
    if m:
        return {"tool": "stats", "action": "summary", "args": {"col": m.group(1), "by": m.group(2) if m.group(2) else None}}

    # --- Keep all your other patterns unchanged (ttest, anova, posthoc, chisq, corr, pcorr, pbiserial, etc.) ---
    # (For brevity, re-use your existing regex blocks here; they are unchanged from your current file.)
    # Copy all remaining matchers (ttest, anova, kruskal, dunn, chisq, corr, matrix, pcorr, pbiserial, mwutest, wilcoxon, levene, ols, glm, plot, check, power) exactly as you have them.

    # Finally, allow raw JSON if user typed it
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

    cols = list(map(str, df.columns)) if df is not None else []
    col_text = f"\nDetected columns: {', '.join(cols)}" if cols else ""
    return f"{prefix}\n\nTry one of these examples:\n• " + "\n• ".join(examples) + col_text
