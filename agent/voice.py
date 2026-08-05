"""语音朗读（ElevenLabs）。

为什么加它：这个顾问的用户里有相当一部分是给父母查资料的人，以及本身年纪较大、
读长段文字吃力的人。把回答念出来是实际有用的，不是装饰。

三个必要处理：
  1 朗读前要清掉引用标记 [3] 和 markdown 星号，否则会被念成「三」和「星号」。
  2 音频按文本哈希缓存到本地。同一段回答不该重复计费，
    而缓存命中的回答本来就会被反复朗读。
  3 本地模式（MINGTU_LOCAL_ONLY=1）下直接不提供朗读——
    朗读要把用户看到的健康内容发到外部服务，这与「数据不出本机」直接冲突。
    宁可没有这个功能，也不能让一个宣称全本地的部署偷偷出网。
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

AUDIO_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "audio"
MODEL = "eleven_multilingual_v2"
# Rachel。多语种模型下中文发音自然，语速适合健康科普这种需要听清楚的内容。
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"
MAX_CHARS = 2500          # 超长回答只念前面部分，避免一次烧掉大量额度


def available():
    """本地模式下不提供朗读，见模块开头第 3 点。"""
    from agent.llm import local_only, _read_key
    if local_only():
        return False
    return bool(_read_key("ELEVENLABS_API_KEY"))


def speakable(text):
    """把回答转成适合朗读的纯文本。"""
    t = re.sub(r"\[(\d{1,3})\]", "", text)          # 引用标记不念
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)        # 去掉加粗标记
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)    # 去掉标题井号
    t = re.sub(r"^\s*[-*•]\s*", "", t, flags=re.M)  # 去掉项目符号
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()[:MAX_CHARS]


def _cache_path(text, voice):
    key = hashlib.sha1(f"{voice}:{MODEL}:{text}".encode("utf-8")).hexdigest()[:20]
    return AUDIO_DIR / f"{key}.mp3"


def synthesize(text, voice=DEFAULT_VOICE):
    """返回 mp3 字节；失败返回 None。命中缓存则不计费。"""
    from agent.llm import _read_key

    clean = speakable(text)
    if not clean:
        return None

    path = _cache_path(clean, voice)
    if path.exists() and path.stat().st_size > 1024:
        return path.read_bytes()

    key = _read_key("ELEVENLABS_API_KEY")
    if not key:
        return None

    payload = {
        "text": clean,
        "model_id": MODEL,
        # stability 略高、similarity 适中：健康内容要稳，不要有戏剧化起伏。
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.75, "speed": 0.95},
    }
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "120", "-X", "POST",
             f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
             "-H", f"xi-api-key: {key}",
             "-H", "content-type: application/json",
             "--data-binary", "@-"],
            # 输出是二进制 mp3，所以整个调用必须走 bytes 模式，
            # 输入也要跟着编码——传 str 会抛 TypeError 并被下面的 except 吞掉。
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
        )
    except Exception:
        return None

    audio = proc.stdout
    if not audio or not (audio[:3] == b"ID3" or audio[:2] == b"\xff\xfb"):
        return None

    try:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
    except Exception:
        pass
    return audio
