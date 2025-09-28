# llm_utils.py

import os
import time
import random
import threading
import sys
import json
from typing import Optional, Tuple, Dict, List

from dotenv import load_dotenv
load_dotenv()

# Providers
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, ServiceUnavailableError

# --------------------------
# Prompts (unchanged semantics)
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
# Small helpers
# --------------------------
def _choice_content(choice):
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
    if i
