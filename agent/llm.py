"""Answering-model plumbing, with provider fallback.

Preference order is Anthropic → OpenAI → Perplexity, and the order is a quality
judgement, not alphabetical: claude-opus-5 is the model this agent's prompts and
citation discipline were written against. The fallbacks exist so the system ships
and runs on a machine that has no Anthropic account — which is the situation
this deployment is actually in — not because the providers are interchangeable.

Two things hold regardless of provider:
  * Every stage still has a deterministic fallback. If all three providers are
    unavailable, retrieval and the safety screen keep working.
  * The structured-output contract is the same shape everywhere, so classify.py
    and core.py never branch on provider.
"""

import json
import os
import re
import subprocess
from pathlib import Path

ANTHROPIC_MODEL = "claude-opus-5"
OPENAI_MODEL = "gpt-5.1"
PERPLEXITY_MODEL = "sonar-pro"
OLLAMA_MODEL = os.environ.get("MINGTU_LLM_MODEL", "qwen3:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# 默认全本地。用户的健康问题不出本机，规避《个人信息保护法》下的数据出境问题。
# 需要调用云端模型时显式设置 MINGTU_LOCAL_ONLY=0。
def local_only():
    return os.environ.get("MINGTU_LOCAL_ONLY", "1").strip() not in ("0", "false", "no")

KEY_ENVS = [
    Path.home() / "Desktop/dev/curio/.env",
    Path.home() / "Desktop/dev/Curio-Automation/.env",
    Path.home() / "Desktop/dev/curio-international/.env",
]

# A local .env in the project root wins, so a client machine can configure the
# agent without touching anything outside its own directory.
PROJECT_ENV = Path(__file__).resolve().parent.parent / ".env"


def _read_key(name):
    if os.environ.get(name):
        return os.environ[name]
    for env in [PROJECT_ENV, *KEY_ENVS]:
        if not env.exists():
            continue
        m = re.search(rf"^{name}=(.*)$", env.read_text(), re.M)
        if m and m.group(1).strip():
            return m.group(1).strip().strip('"').strip("'")
    return None


_clients = {}


def _anthropic():
    if "anthropic" not in _clients:
        key = _read_key("ANTHROPIC_API_KEY")
        if not key:
            _clients["anthropic"] = None
        else:
            import anthropic
            _clients["anthropic"] = anthropic.Anthropic(api_key=key)
    return _clients["anthropic"]


def _openai():
    if "openai" not in _clients:
        key = _read_key("OPENAI_API_KEY")
        if not key:
            _clients["openai"] = None
        else:
            from openai import OpenAI
            _clients["openai"] = OpenAI(api_key=key)
    return _clients["openai"]


def ollama_up():
    """True when the local runtime is serving the configured chat model."""
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "3", f"{OLLAMA_HOST}/api/tags"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return False
        names = [m.get("name", "") for m in json.loads(proc.stdout).get("models", [])]
        base = OLLAMA_MODEL.split(":")[0]
        return any(n.split(":")[0] == base for n in names)
    except Exception:
        return False


def active_provider():
    """Which provider will answer, or None when nothing is configured.

    Local first, always. In local-only mode (the default) there is no fallback
    to a cloud provider — a missing local model is an error the operator must
    see, not something to silently paper over by shipping user health questions
    to a US API.
    """
    if ollama_up():
        return "ollama"
    if local_only():
        return None
    if _anthropic() is not None:
        return "anthropic"
    if _openai() is not None:
        return "openai"
    if _read_key("PERPLEXITY_API_KEY"):
        return "perplexity"
    return None


def provider_label():
    return {
        "ollama": f"本地 {OLLAMA_MODEL}",
        "anthropic": f"Anthropic {ANTHROPIC_MODEL}",
        "openai": f"OpenAI {OPENAI_MODEL}",
        "perplexity": f"Perplexity {PERPLEXITY_MODEL}",
    }.get(active_provider(), "未配置")


def _ollama_chat(system, messages, max_tokens, fmt=None):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": max_tokens},
        "messages": [{"role": "system", "content": system}] + _fold_system_into_user(messages),
    }
    if fmt:
        payload["format"] = fmt
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "600", "-X", "POST",
             f"{OLLAMA_HOST}/api/chat", "-H", "content-type: application/json",
             "--data-binary", "@-"],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True, text=True,
        )
        data = json.loads(proc.stdout)
    except Exception:
        return None
    content = (data.get("message") or {}).get("content")
    if not content:
        return None
    # qwen3 emits reasoning in <think> blocks; strip before the user sees it.
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S).strip()


def available():
    return active_provider() is not None


# --- message normalisation ----------------------------------------------------
# core.py speaks one dialect: a system string plus messages with roles
# user/assistant/system. Each adapter converts. Mid-conversation system messages
# are the repair channel (citation and compliance corrections), so every adapter
# must carry them through without losing the instruction.

def _fold_system_into_user(messages):
    """For providers that only accept a leading system message."""
    folded = []
    for msg in messages:
        if msg["role"] == "system":
            if folded and folded[-1]["role"] == "user":
                folded[-1]["content"] += f"\n\n{msg['content']}"
            else:
                folded.append({"role": "user", "content": msg["content"]})
        elif folded and folded[-1]["role"] == msg["role"]:
            folded[-1]["content"] += f"\n\n{msg['content']}"
        else:
            folded.append(dict(msg))
    return folded


def _perplexity_chat(system, messages, max_tokens, json_mode=False):
    key = _read_key("PERPLEXITY_API_KEY")
    if not key:
        return None
    payload = {
        "model": PERPLEXITY_MODEL,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + _fold_system_into_user(messages),
    }
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "120", "-X", "POST",
             "https://api.perplexity.ai/chat/completions",
             "-H", f"Authorization: Bearer {key}",
             "-H", "content-type: application/json", "--data-binary", "@-"],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True, text=True,
        )
        data = json.loads(proc.stdout)
    except Exception:
        return None
    if "choices" not in data:
        return None
    return data["choices"][0]["message"]["content"]


# --- structured output (classify, rerank) -------------------------------------

def call_json(system, user, schema, effort="low", max_tokens=4096):
    """One structured-output call. Returns the parsed object, or None."""
    provider = active_provider()

    if provider == "anthropic":
        try:
            resp = _anthropic().messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system,
                output_config={"effort": effort,
                               "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": user}],
            )
            if resp.stop_reason == "refusal":
                return None
            return json.loads(next((b.text for b in resp.content if b.type == "text"), ""))
        except Exception:
            return None

    if provider == "openai":
        try:
            resp = _openai().chat.completions.create(
                model=OPENAI_MODEL,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "result", "schema": schema, "strict": True}},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return None

    if provider == "perplexity":
        instruction = (
            f"{system}\n\n只输出一个 JSON 对象，不要任何解释、不要代码块标记。"
            f"必须严格符合这个 JSON Schema：\n{json.dumps(schema, ensure_ascii=False)}"
        )
        text = _perplexity_chat(instruction, [{"role": "user", "content": user}], max_tokens)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(match.group(0)) if match else None
        except Exception:
            return None

    return None


# --- free-form answering ------------------------------------------------------

def generate(system, messages, max_tokens=8000, effort="high"):
    """Produce the user-facing answer. Returns text, or None on failure."""
    provider = active_provider()

    if provider == "anthropic":
        try:
            with _anthropic().messages.stream(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                # Cached: the policy layer is stable across every request, and
                # the per-turn evidence deliberately rides in the user message.
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": effort},
                messages=messages,
            ) as stream:
                resp = stream.get_final_message()
            if resp.stop_reason == "refusal":
                return None
            return next((b.text for b in resp.content if b.type == "text"), "")
        except Exception:
            return None

    if provider == "openai":
        try:
            resp = _openai().chat.completions.create(
                model=OPENAI_MODEL,
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}] + messages,
            )
            return resp.choices[0].message.content
        except Exception:
            return None

    if provider == "perplexity":
        return _perplexity_chat(system, messages, max_tokens)

    return None


# Kept for callers that only need to know whether answering is possible.
def client():
    return active_provider()


MODEL = ANTHROPIC_MODEL
