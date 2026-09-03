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
from together.error import RateLimitError

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
    if isinstance(delta, dict):
        return delta.get("content", "")
    return getattr(delta, "content", "")

class _SpacedCallLimiter:
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

# --------------------------
# Unified LLM Client
# --------------------------
HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

class UnifiedLLM:
    def __init__(self):
        hf_model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()
        self.hf_client = InferenceClient(model=hf_model_or_url, token=hf_token, timeout=300)

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
        self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)

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
                        messages=messages, max_tokens=max_tokens, temperature=temperature, stream=stream
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
                except RateLimitError:
                    if attempt == 3:
                        raise
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8

# Singleton (matches original usage pattern)
llm = UnifiedLLM()

# --------------------------
# Public helper: ask_llm
# --------------------------
def ask_llm(
    chat_history: List[Dict[str, str]],
    user_input: str,
    system_prompt: str = SYSTEM_PROMPT,
    fallback_prompt: str = FALLBACK_PROMPT,
) -> Tuple[Optional[Dict], str]:
    """
    Try to produce a structured tool call JSON from the primary prompt.
    On failure, return (None, fallback_plain_text).
    """
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    resp = llm.chat(messages=messages, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = llm.chat(
            messages=[{"role": "system", "content": fallback_prompt}] + messages,
            temperature=0.7,
            max_tokens=256,
            stream=False,
        )
        return None, conv
