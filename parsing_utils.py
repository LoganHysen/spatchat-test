# parsing_utils.py

from typing import Dict, List, Optional
import re
import json

NULLY_STRINGS = {"", "null", "none", "na", "n/a", "nil"}

def clean_null_like(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in NULLY_STRINGS else v


def sanitize_args(action: Optional[str], args: Dict) -> Dict:
    """Normalize/clean parsed args exactly like the original."""
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


def _split_controls(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,+]", s)
    return [p.strip() for p in parts if p.strip()]


def local_parse(user_input: str) -> Optional[Dict]:
    """
    Lightweight regex parser that maps free text to the structured tool-call schema.
    Falls back to JSON if the user already provided a dict.
    """
    s = user_input.strip().lower()

    if re.search(r"\b(what can i do|how should i analyze|recommend|suggestion|what analyses)\b", s):
        return {"tool": "stats", "action": "recommend", "args": {}}

    m = re.search(
        r"(?:what(?:'s| is) the )?(?:mean|average|summary|summarize|describe)\s+([a-zA-Z0-9_]+)"
        r"(?:\s+by\s+([a-zA-Z0-9_]+))?",
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
    m = re.search(r"(kruskal|kw)(?:\s+test)?\s+(?:on|of|for)\s+([a-zA-Z0-9_]+)\s+(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {"tool": "stats", "action": "kruskal", "args": {"value": m.group(2), "group": m.group(3)}}

    # Dunn
    m = re.search(r"(dunn)(?:\s+test)?\s+(?:on|of|for)\s+([a-zA-Z0-9_]+)\s+(?:by|across)\s*([a-zA-Z0-9_]+)", s)
    if m:
        return {
            "tool": "stats",
            "action": "posthoc_dunn",
            "args": {"value": m.group(2), "group": m.group(3), "p_adjust": "holm"},
        }

    # Chi-square / Fisher
    m = re.search(
        r"(chi(?:-?square)?|chi2|chisq|fisher).*?(?:of|between|for)?\s*([a-zA-Z0-9_]+)\s*"
        r"(?:by|x|vs|versus|and)\s*([a-zA-Z0-9_]+)",
        s,
   
