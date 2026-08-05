"""Anthropic client plumbing shared by the classifier, reranker and answerer.

Every LLM stage in this system has a deterministic fallback. That is not a
convenience — it is the safety property. If the model call fails, retrieval
must still filter sensibly and the emergency screen must still fire, because
the failure mode we cannot accept is a red-flag symptom slipping through
because an API call timed out.
"""

import json
import os
import re
from pathlib import Path

MODEL = "claude-opus-5"

KEY_ENVS = [
    Path.home() / "Desktop/dev/curio/.env",
    Path.home() / "Desktop/dev/Curio-Automation/.env",
]


def anthropic_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for env in KEY_ENVS:
        if not env.exists():
            continue
        m = re.search(r"^ANTHROPIC_API_KEY=(.*)$", env.read_text(), re.M)
        if m and m.group(1).strip():
            return m.group(1).strip().strip('"').strip("'")
    return None


_client = None


def client():
    """Return a client, or None when no credential is configured."""
    global _client
    if _client is not None:
        return _client
    key = anthropic_key()
    if not key:
        return None
    import anthropic
    _client = anthropic.Anthropic(api_key=key)
    return _client


def available():
    return client() is not None


def call_json(system, user, schema, effort="low", max_tokens=4096):
    """One structured-output call. Returns the parsed object, or None on failure.

    Used for the mechanical stages (classify, rerank) where the output is a
    fixed shape and low effort is the right tier.
    """
    c = client()
    if c is None:
        return None
    try:
        resp = c.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            return None
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return json.loads(text)
    except Exception:
        return None
