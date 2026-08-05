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
# 分流与重排是机械任务，用小模型即可，速度差一个量级。作答仍用大模型。
OLLAMA_FAST_MODEL = os.environ.get("MINGTU_FAST_MODEL", "qwen3:4b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
GEMINI_MODEL = os.environ.get("MINGTU_GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_FALLBACK = "gemini-2.5-flash"
# 分流与重排是机械任务，用 lite 档即可，两个阶段合计能省约 7 秒。
GEMINI_FAST_MODEL = os.environ.get("MINGTU_GEMINI_FAST", "gemini-3.1-flash-lite")

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


def _norm_tag(name):
    return name if ":" in name else f"{name}:latest"


def _ollama_has(name):
    """Exact tag match, not base-name match.

    Comparing only the part before the colon made qwen3:14b satisfy a request
    for qwen3:4b. Ollama then 404s, call_json swallows the error and returns
    None, and classify/rerank quietly drop to their deterministic fallbacks —
    the pipeline looks fast and healthy while two model stages never run.
    """
    try:
        proc = subprocess.run(["curl", "-sS", "--max-time", "3", f"{OLLAMA_HOST}/api/tags"],
                              capture_output=True, text=True)
        names = {_norm_tag(m.get("name", "")) for m in json.loads(proc.stdout).get("models", [])}
        return _norm_tag(name) in names
    except Exception:
        return False


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
    # OpenAI 优先于 Gemini：Gemini 免费额度已耗尽（429 prepayment credits
    # depleted），线上一度整片失败。付费且可用的通道必须排在免费额度前面，
    # 否则「免费」换来的是客户看到的服务不可用。
    if _openai() is not None:
        return "openai"
    if _read_key("GEMINI_API_KEY"):
        return "gemini"
    if _read_key("PERPLEXITY_API_KEY"):
        return "perplexity"
    return None


def provider_label():
    return {
        "ollama": f"本地 {OLLAMA_MODEL}",
        "anthropic": f"Anthropic {ANTHROPIC_MODEL}",
        "gemini": f"Gemini {GEMINI_MODEL}",
        "openai": f"OpenAI {OPENAI_MODEL}",
        "perplexity": f"Perplexity {PERPLEXITY_MODEL}",
    }.get(active_provider(), "未配置")


def _strip_unsupported(node):
    """Gemini accepts a JSON-Schema subset; drop the keys it rejects."""
    if isinstance(node, dict):
        return {k: _strip_unsupported(v) for k, v in node.items()
                if k not in ("additionalProperties", "strict", "$schema")}
    if isinstance(node, list):
        return [_strip_unsupported(v) for v in node]
    return node


def _gemini_chat(system, messages, max_tokens, schema=None, fast=False):
    """Gemini uses systemInstruction plus role user/model turns."""
    key = _read_key("GEMINI_API_KEY")
    if not key:
        return None
    contents = []
    for msg in _fold_system_into_user(messages):
        contents.append({
            "role": "model" if msg["role"] == "assistant" else "user",
            "parts": [{"text": msg["content"]}],
        })
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
    }
    if schema:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        # additionalProperties is not part of Gemini's schema dialect; sending it
        # makes the request 400, which then looks like a model failure.
        payload["generationConfig"]["responseSchema"] = _strip_unsupported(schema)

    # Fall back down the model ladder on rate limits — the free tier throttles
    # the strongest model first, and a throttled demo looks like a broken one.
    ladder = (GEMINI_FAST_MODEL, GEMINI_MODEL) if fast else (GEMINI_MODEL, GEMINI_FALLBACK)
    for model in ladder:
        try:
            proc = subprocess.run(
                ["curl", "-sS", "--max-time", "180", "-X", "POST",
                 f"https://generativelanguage.googleapis.com/v1beta/models/"
                 f"{model}:generateContent?key={key}",
                 "-H", "content-type: application/json", "--data-binary", "@-"],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True, text=True,
            )
            data = json.loads(proc.stdout)
        except Exception:
            continue
        if "candidates" not in data:
            continue
        parts = data["candidates"][0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if text:
            return text
    return None


def _ollama_chat(system, messages, max_tokens, fmt=None, think=False, fast=False):
    """Local chat. Thinking is OFF by default, and that is a measured decision.

    qwen3 emits long <think> blocks before answering. On an M4 that took ~400s
    per turn across the three LLM stages — unusable. With think=False the same
    turn lands in ~5-20s. The quality cost is small here because the retrieved
    passages already carry the reasoning: the model is summarising graded
    evidence, not deriving it. Set MINGTU_THINK=1 to turn it back on.
    """
    model = OLLAMA_FAST_MODEL if (fast and _ollama_has(OLLAMA_FAST_MODEL)) else OLLAMA_MODEL
    payload = {
        "model": model,
        "stream": False,
        "think": think or os.environ.get("MINGTU_THINK", "").strip() in ("1", "true", "yes"),
        # 必须显式给 num_ctx：ollama 会按内存自行决定上下文长度（实测只给 2048），
        # 超出部分从**前面**丢弃，于是整段系统提示词——人设、证据分级、引用规则、
        # 合规红线——根本没进模型，而 HTTP 仍然 200、done_reason 仍然是 stop。
        "options": {"temperature": 0.2, "num_predict": max_tokens,
                    "num_ctx": int(os.environ.get("MINGTU_NUM_CTX", "16384"))},
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


def _loose_json(text):
    """Parse a JSON object OR array out of a model reply.

    Models wrap JSON in prose or code fences, and return a bare array where the
    schema asked for an object. Insisting on one shape made whole stages fail.
    """
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


# --- structured output (classify, rerank) -------------------------------------

def call_json(system, user, schema, effort="low", max_tokens=4096, fast=False):
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

    if provider == "ollama":
        text = _ollama_chat(system + "\n\n只输出一个 JSON 对象，不要解释、不要代码块标记。",
                            [{"role": "user", "content": user}], max_tokens,
                            fmt=schema, fast=fast)
        if not text:
            return None
        return _loose_json(text)

    if provider == "gemini":
        text = _gemini_chat(
            system + "\n\n只输出一个 JSON 对象，不要解释、不要代码块标记。必须符合此 Schema："
            + json.dumps(schema, ensure_ascii=False),
            [{"role": "user", "content": user}], max_tokens, schema=schema, fast=fast)
        if not text:
            return None
        return _loose_json(text)

    if provider == "perplexity":
        instruction = (
            f"{system}\n\n只输出一个 JSON 对象，不要任何解释、不要代码块标记。"
            f"必须严格符合这个 JSON Schema：\n{json.dumps(schema, ensure_ascii=False)}"
        )
        text = _perplexity_chat(instruction, [{"role": "user", "content": user}], max_tokens)
        if not text:
            return None
        return _loose_json(text)

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

    if provider == "ollama":
        return _ollama_chat(system, messages, max_tokens)

    if provider == "gemini":
        return _gemini_chat(system, messages, max_tokens)

    if provider == "perplexity":
        return _perplexity_chat(system, messages, max_tokens)

    return None


def stream_generate(system, messages, max_tokens=2000):
    """逐块产出模型输出。目前只有 Gemini 走真流式，其余回退成一次性产出。

    回退不是降级：调用方拿到的仍是完整正确的答案，只是没有逐字效果。
    宁可退回一次性，也不要为了"看起来在流"而绕开合规闸门。
    """
    provider = active_provider()

    if provider == "gemini":
        key = _read_key("GEMINI_API_KEY")
        contents = []
        for msg in _fold_system_into_user(messages):
            contents.append({"role": "model" if msg["role"] == "assistant" else "user",
                             "parts": [{"text": msg["content"]}]})
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
        }
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{GEMINI_MODEL}:streamGenerateContent?alt=sse&key={key}")
        proc = subprocess.Popen(
            ["curl", "-sS", "--no-buffer", "--max-time", "180", "-X", "POST", url,
             "-H", "content-type: application/json", "--data-binary", "@-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            proc.stdin.write(json.dumps(payload, ensure_ascii=False))
            proc.stdin.close()
            for line in proc.stdout:
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                for cand in d.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text"):
                            yield part["text"]
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
        return

    text = generate(system, messages, max_tokens=max_tokens)
    if text:
        yield text
