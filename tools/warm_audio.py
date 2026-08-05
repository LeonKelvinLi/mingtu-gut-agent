#!/usr/bin/env python3
"""给已缓存的回答预先生成音频，让演示时点「朗读」是秒开。

合成一段一千多字的回答要二三十秒。预热之后这些常见问题的音频已经在盘上，
点下去立刻出声。运行： MINGTU_ALLOW_TTS=1 .venv/bin/python tools/warm_audio.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MINGTU_ALLOW_TTS", "1")

from agent.voice import synthesize, speakable, _cache_path, DEFAULT_VOICE  # noqa: E402

rows = [json.loads(l) for l in (Path(__file__).resolve().parent.parent
                                / "knowledge/cache.jsonl").read_text().splitlines() if l.strip()]
print(f"待预热 {len(rows)} 条")
done = skipped = failed = 0
for i, r in enumerate(rows, 1):
    clean = speakable(r["answer"])
    if _cache_path(clean, DEFAULT_VOICE).exists():
        skipped += 1
        continue
    t = time.time()
    audio = synthesize(r["answer"])
    if audio:
        done += 1
        print(f"[{i}/{len(rows)}] {r['query'][:22]} — {len(audio)//1024}KB, {time.time()-t:.0f}s")
    else:
        failed += 1
        print(f"[{i}/{len(rows)}] {r['query'][:22]} — 失败")
print(f"\n新增 {done}，已有 {skipped}，失败 {failed}")
