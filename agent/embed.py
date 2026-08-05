"""Pluggable embeddings, so the whole system can run with no outbound calls.

Why local matters here beyond cost: this agent answers health questions from
Chinese users. Sending that text to a US API is a 数据出境 question under
《个人信息保护法》and the 数据出境安全评估 rules. A fully local stack removes the
question instead of answering it, which is usually what a Chinese enterprise
customer actually needs.

Backends, in preference order:
  ollama   bge-m3, 1024-dim, strong multilingual/Chinese retrieval, fully local
  openai   text-embedding-3-large, 3072-dim, needs an outbound call

The index records which backend built it. Mixing backends is not a degraded
result, it is a meaningless one — cosine similarity across two different vector
spaces is noise — so a mismatch is a hard error, never a silent fallback.
"""

import json
import os
import subprocess

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.environ.get("MINGTU_EMBED_MODEL", "bge-m3")
OPENAI_EMBED_MODEL = "text-embedding-3-large"

def local_only():
    """Single source of truth, imported from llm.py.

    These two modules previously carried their own copies with opposite
    defaults — llm.py defaulted to local-only, embed.py defaulted to allowing
    cloud. That is the worst possible split: a deployment believing it was
    airtight would still ship every user question to a cloud embedding API,
    and nothing in the logs would say so.
    """
    from agent.llm import local_only as _local_only
    return _local_only()


def ollama_up():
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "3", f"{OLLAMA_HOST}/api/tags"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return False
        models = [m.get("name", "") for m in json.loads(proc.stdout).get("models", [])]
        return any(m.startswith(OLLAMA_EMBED_MODEL) for m in models)
    except Exception:
        return False


def active_backend():
    # 显式指定后端。没有这个开关时，只要本机 ollama 在跑就一定选本地，
    # 想单独重建云端索引都做不到——而索引与向量后端一旦错配，
    # 行数还对得上、维度也对得上，只有答案是错的。
    forced = os.environ.get("MINGTU_EMBED_BACKEND", "").strip().lower()
    if forced in ("ollama", "gemini", "openai"):
        return forced
    if ollama_up():
        return "ollama"
    if local_only():
        return None
    from agent.llm import _read_key
    # 与作答同样的理由：可用性优先于免费。
    if _read_key("OPENAI_API_KEY"):
        return "openai"
    return "gemini" if _read_key("GEMINI_API_KEY") else None


def backend_id():
    """Identifier stored alongside the index so a mismatch is detectable."""
    backend = active_backend()
    if backend == "ollama":
        return f"ollama:{OLLAMA_EMBED_MODEL}"
    if backend == "gemini":
        return f"gemini:{GEMINI_EMBED_MODEL}"
    if backend == "openai":
        return f"openai:{OPENAI_EMBED_MODEL}"
    return "none"


def _embed_ollama(texts):
    vectors = []
    for text in texts:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "120", "-X", "POST",
             f"{OLLAMA_HOST}/api/embed", "-H", "content-type: application/json",
             "--data-binary", "@-"],
            input=json.dumps({"model": OLLAMA_EMBED_MODEL, "input": text}),
            capture_output=True, text=True,
        )
        data = json.loads(proc.stdout)
        embeddings = data.get("embeddings") or ([data["embedding"]] if "embedding" in data else None)
        if not embeddings:
            raise RuntimeError(f"ollama embed failed: {proc.stdout[:200]}")
        vectors.append(embeddings[0])
    return vectors


def _embed_openai(texts):
    from openai import OpenAI
    from agent.llm import _read_key
    client = OpenAI(api_key=_read_key("OPENAI_API_KEY"))
    vectors = []
    for i in range(0, len(texts), 64):
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts[i:i + 64])
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def embed(texts, progress=False):
    """Embed a list of strings with the active backend. Returns a float32 array."""
    import numpy as np

    backend = active_backend()
    if backend is None:
        raise SystemExit(
            "没有可用的向量模型。\n"
            "本地方案：安装 ollama 并执行 ollama pull bge-m3\n"
            "云端方案：配置 OPENAI_API_KEY（注意会有数据出境问题）"
        )

    fn = {"ollama": _embed_ollama, "gemini": _embed_gemini,
          "openai": _embed_openai}[backend]
    out = []
    step = 32 if backend == "ollama" else 64
    for i in range(0, len(texts), step):
        out.extend(fn(texts[i:i + step]))
        if progress:
            print(f"  已向量化 {min(i + step, len(texts))}/{len(texts)}", flush=True)

    arr = np.asarray(out, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


# --- Gemini backend (free tier) ----------------------------------------------
# Used for cloud deployment where ollama cannot run. Free-tier limits are
# generous enough for a demo; the vectors land in their own index file so the
# local bge-m3 index is never mixed with them.
GEMINI_EMBED_MODEL = os.environ.get("MINGTU_GEMINI_EMBED", "gemini-embedding-2")


def _gemini_key():
    from agent.llm import _read_key
    return _read_key("GEMINI_API_KEY")


def _embed_gemini(texts):
    key = _gemini_key()
    vectors = []
    for text in texts:
        payload = {"content": {"parts": [{"text": text[:8000]}]}}
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "60", "-X", "POST",
             f"https://generativelanguage.googleapis.com/v1beta/models/"
             f"{GEMINI_EMBED_MODEL}:embedContent?key={key}",
             "-H", "content-type: application/json", "--data-binary", "@-"],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True, text=True,
        )
        data = json.loads(proc.stdout)
        if "embedding" not in data:
            raise RuntimeError(f"gemini embed failed: {proc.stdout[:200]}")
        vectors.append(data["embedding"]["values"])
    return vectors
