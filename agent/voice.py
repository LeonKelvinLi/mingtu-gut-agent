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
import os
import re
import subprocess
from pathlib import Path

# 音频同样放持久盘：临时文件系统上，每次部署都要把所有回答重新合成一遍，
# 既慢又重复计费。
_ROOT = Path(os.environ.get("MINGTU_DATA_DIR", "")) if os.environ.get("MINGTU_DATA_DIR") \
    else Path(__file__).resolve().parent.parent / "knowledge"
AUDIO_DIR = _ROOT / "audio"
MODEL = "eleven_multilingual_v2"
# Rachel。多语种模型下中文发音自然，语速适合健康科普这种需要听清楚的内容。
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"
MAX_CHARS = 1200          # 超长回答只念前面部分，避免一次烧掉大量额度


def available():
    """朗读是否可用。

    默认在本地模式下关闭，因为朗读必然要把回答文本发到 ElevenLabs——
    而本地版对客户的承诺是「数据不出本机」。

    但这是个可以由部署方自己权衡的取舍，不是安全边界：朗读发出去的是
    智能体生成的回答，不是用户原话。所以给一个显式开关 MINGTU_ALLOW_TTS=1，
    打开后本地模式也能朗读。默认仍然关闭——要放弃一条承诺，必须是有人
    主动按下的，不能是默认发生的。
    """
    from agent.llm import local_only, _read_key
    if not _read_key("ELEVENLABS_API_KEY"):
        return False
    if local_only() and os.environ.get("MINGTU_ALLOW_TTS", "").strip() not in ("1", "true", "yes"):
        return False
    return True


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


def cached_path(text, voice=DEFAULT_VOICE):
    """这段文本是否已经有落盘的音频。有就返回路径，没有返回 None。

    调用方靠它决定走哪条路：已经有完整文件就整段发并支持 Range（可以拖进度条），
    还没有就边合成边发（第一声快，但这一遍不可定位）。
    """
    clean = speakable(text)
    if not clean:
        return None
    path = _cache_path(clean, voice)
    if path.exists() and path.stat().st_size > 1024:
        return path
    return None

def synthesize_stream(text, voice=DEFAULT_VOICE):
    """逐块产出 mp3。用 ElevenLabs 的 /stream 端点。

    一次性合成 1350 字要 51 秒，这段时间里按钮只能一直显示「生成语音…」——
    用户看到的就是「朗读坏了」。流式下第一块音频一秒左右就到，
    浏览器可以边下边播。命中本地缓存时直接整块吐出，不再请求。
    """
    from agent.llm import _read_key

    clean = speakable(text)
    if not clean:
        return

    path = _cache_path(clean, voice)
    if path.exists() and path.stat().st_size > 1024:
        yield path.read_bytes()
        return

    key = _read_key("ELEVENLABS_API_KEY")
    if not key:
        return

    payload = {
        "text": clean,
        "model_id": MODEL,
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.75, "speed": 0.95},
    }
    proc = subprocess.Popen(
        ["curl", "-sS", "--no-buffer", "--max-time", "300", "-X", "POST",
         f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=mp3_44100_64",
         "-H", f"xi-api-key: {key}",
         "-H", "content-type: application/json", "--data-binary", "@-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    collected = bytearray()
    try:
        proc.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        proc.stdin.close()
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            collected += chunk
            yield chunk
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

    # 播完再落盘，下次同一段回答直接命中缓存，不重复计费。
    if collected[:3] == b"ID3" or collected[:2] == b"\xff\xfb":
        try:
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(collected))
        except Exception:
            pass


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
