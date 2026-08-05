"""语义答案缓存。让常见问题即时返回。

为什么需要它：本地版跑 qwen3 一轮要几十秒到几分钟，云端也要 30 秒。而真实用户
问的问题高度集中——「益生菌怎么选」「酸奶和益生菌一样吗」「宝宝能不能吃」这类
占了绝大多数。同一个问题重算一遍没有任何意义。

三级命中，从快到慢：
  1 归一化字面命中   完全不需要模型和向量，微秒级
  2 语义相似命中     只需给问题算一次向量（本地 bge-m3 约 0.3 秒）
  3 未命中           走完整流程，答完后写回缓存

安全边界（不可协商）：
  * 缓存查询发生在确定性报警症状筛查**之后**。命中报警症状的输入永不查缓存、
    也永不写入缓存——急症必须每次都走固定话术，不能被任何缓存短路。
  * 语料变了，缓存全部失效。答案是语料的函数，语料换了旧答案就是错的。
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

import os

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "knowledge"          # 语料索引始终在仓库里

# 缓存要写在持久盘上。云主机的文件系统是临时的：每次部署都会重置，
# 运行期间新缓存的回答会全部消失，只剩随仓库带上去的那份预热。
# 挂了盘之后，用户问过的问题下次部署仍然是秒回。
DATA_DIR = Path(os.environ.get("MINGTU_DATA_DIR", "")) if os.environ.get("MINGTU_DATA_DIR") else CACHE_DIR
CACHE_PATH = DATA_DIR / "cache.jsonl"
CACHE_VECS = DATA_DIR / "cache-vectors.npy"
SEED_PATH = CACHE_DIR / "cache.jsonl"          # 仓库里那份预热，用来给空盘播种
SEED_VECS = CACHE_DIR / "cache-vectors.npy"

# 0.93 是刻意定高的。定低了会把「益生菌能治肠炎吗」和「益生菌能预防肠炎吗」
# 当成同一个问题——这两个问题的合规答案并不相同。宁可少命中，不可答错。
SIMILARITY_THRESHOLD = 0.93
MAX_ENTRIES = 2000


def normalize(query):
    """归一化：去标点、空白、语气词，让同义写法落到同一个键上。"""
    q = query.strip().lower()
    q = re.sub(r"[\s，。！？、；：""''（）()\[\]【】~…·]+", "", q)
    q = re.sub(r"^(请问|想问一下|我想问|请教一下|问一下)", "", q)
    q = re.sub(r"(呀|啊|呢|吗|吧|哦|嘛|哈)+$", "", q)
    return q


_FINGERPRINT = None


def corpus_fingerprint():
    """语料指纹：按**内容**算，不能用 mtime。

    最初用的是 size+mtime，结果是缓存在生产环境永远不命中——
    git checkout 会给每个文件一个新的 mtime，所以每次部署指纹都变，
    整份预热缓存被当作过期直接丢掉。本地测着好好的，线上白跑。
    内容哈希在任何机器上都一致，这才是「语料是否变了」的真实判据。
    """
    global _FINGERPRINT
    if _FINGERPRINT:
        return _FINGERPRINT
    chunks = CACHE_DIR / "chunks.jsonl"
    if not chunks.exists():
        return "none"
    h = hashlib.sha1()
    with chunks.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    _FINGERPRINT = h.hexdigest()[:12]
    return _FINGERPRINT


class AnswerCache:
    def __init__(self):
        self.entries = []
        self.vectors = None
        self._by_key = {}
        self._load()

    # -- 持久化 --------------------------------------------------------------
    def _load(self):
        # 持久盘第一次挂上来是空的，用仓库里那份预热给它播种，
        # 否则新部署的第一位访客要等完整生成。
        if DATA_DIR != CACHE_DIR and not CACHE_PATH.exists() and SEED_PATH.exists():
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_bytes(SEED_PATH.read_bytes())
                if SEED_VECS.exists():
                    CACHE_VECS.write_bytes(SEED_VECS.read_bytes())
            except Exception:
                pass
        if not CACHE_PATH.exists():
            return
        fingerprint = corpus_fingerprint()
        rows = []
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            # 语料换了就整体丢弃，不做部分保留：旧答案引用的段落编号已经对不上了。
            if row.get("fingerprint") != fingerprint:
                continue
            rows.append(row)

        self.entries = rows[-MAX_ENTRIES:]
        self._by_key = {r["key"]: i for i, r in enumerate(self.entries)}
        # 行数对不上就整体放弃：随仓库携带的 jsonl 与 npy 可能来自不同时刻，
        # 截断对齐（原来的 vecs[-n:]）会让每一条都指向错误的向量。
        if CACHE_VECS.exists() and self.entries:
            try:
                vecs = np.load(CACHE_VECS)
                if vecs.shape[0] == len(self.entries):
                    self.vectors = vecs
                else:
                    self.entries, self._by_key, self.vectors = [], {}, None
            except Exception:
                self.entries, self._by_key, self.vectors = [], {}, None
        elif self.entries:
            self.entries, self._by_key = [], {}

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            for row in self.entries:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.vectors is not None:
            np.save(CACHE_VECS, self.vectors)

    # -- 查询 ----------------------------------------------------------------
    def lookup_literal(self, query):
        """字面命中。不需要向量，微秒级。"""
        idx = self._by_key.get(normalize(query))
        return self.entries[idx] if idx is not None else None

    def lookup_semantic(self, query_vector):
        """语义命中。需要调用方已经算好问题向量。"""
        if self.vectors is None or not len(self.entries):
            return None
        if self.vectors.shape[1] != len(query_vector):
            return None          # 向量后端换了，维度对不上，直接不命中
        scores = self.vectors @ query_vector
        best = int(np.argmax(scores))
        if float(scores[best]) >= SIMILARITY_THRESHOLD:
            entry = dict(self.entries[best])
            entry["_similarity"] = round(float(scores[best]), 4)
            return entry
        return None

    # -- 写入 ----------------------------------------------------------------
    def put(self, query, answer, sources, query_vector=None):
        key = normalize(query)
        if not key or not answer:
            return
        row = {
            "key": key,
            "query": query,
            "answer": answer,
            "sources": sources or [],
            "fingerprint": corpus_fingerprint(),
            "created_at": int(time.time()),
        }
        # entries 与 vectors 靠下标隐式对齐，一旦错位，语义命中会以 sim=1.0
        # 返回**另一个问题**的答案——对健康产品来说这是最坏的一类缺陷。
        # 所以没有向量就不写入：宁可这条不进缓存，也不能让整个缓存错位。
        if query_vector is None:
            return
        vec = np.asarray(query_vector, dtype="float32")[None, :]
        # 维度对不上说明换过向量后端，硬塞会让 np.vstack 直接抛异常，
        # 而调用方的 try/except 会把它吞掉——表现成「写了但没写进去」。
        if self.vectors is not None and self.vectors.shape[1] != vec.shape[1]:
            return

        if key in self._by_key:
            idx = self._by_key[key]
            self.entries[idx] = row
            if self.vectors is not None and idx < len(self.vectors):
                self.vectors[idx] = vec[0]
        else:
            self.entries.append(row)
            self._by_key[key] = len(self.entries) - 1
            self.vectors = vec if self.vectors is None else np.vstack([self.vectors, vec])

        # 不变式：写完必须等长，否则整体作废重来，绝不带着错位继续服务。
        if self.vectors is None or len(self.vectors) != len(self.entries):
            self.entries, self._by_key, self.vectors = [], {}, None
            return
        self._save()

    def __len__(self):
        return len(self.entries)
