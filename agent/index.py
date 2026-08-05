"""Build the retrieval index: chunks + BM25 + dense embeddings.

Hybrid, not vector-only, and the reason is specific to this corpus: the highest
value facts here are strain identifiers (LGG ATCC 53103, DSM 17938, CNCM I-745),
dosages (10^10 CFU), and regulation numbers. Embeddings blur exactly those —
"DSM 17938" and "DSM 9843" sit almost on top of each other in vector space and
are different organisms with different evidence. BM25 keeps them apart. Dense
retrieval earns its place on the other side: paraphrased consumer questions
("吃了益生菌还是胀气") that share no keywords with the clinical passage.

Run: python -m agent.index build
"""

import json
import os
import re
import sys
from pathlib import Path

import jieba
import numpy as np

from agent.chunker import chunk_document
from agent.embed import backend_id, embed as embed_texts

REPO = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO / "research"
INDEX_DIR = REPO / "knowledge"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"


def _slug(backend):
    return backend.replace(":", "_").replace("/", "_")


def vectors_path(backend=None):
    """Vectors are stored per embedding backend.

    Local (bge-m3, 1024-dim) and cloud (OpenAI, 3072-dim) live in different
    vector spaces, so they get different files and can coexist. This is what
    lets the same repo run fully offline on a laptop and on a cloud host.
    """
    from agent.embed import backend_id
    return INDEX_DIR / f"vectors-{_slug(backend or backend_id())}.npy"

EMBED_MODEL = "text-embedding-3-large"
EMBED_BATCH = 64

KEY_ENVS = [
    Path.home() / "Desktop/dev/curio/.env",
    Path.home() / "Desktop/dev/Curio-Automation/.env",
]


def openai_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for env in KEY_ENVS:
        if not env.exists():
            continue
        m = re.search(r"^OPENAI_API_KEY=(.*)$", env.read_text(), re.M)
        if m and m.group(1).strip():
            return m.group(1).strip().strip('"').strip("'")
    raise SystemExit("no OPENAI_API_KEY found")


def tokenize(text):
    """Chinese needs segmentation before BM25; English/IDs pass through.

    search mode over the default: it emits overlapping sub-terms, so a query for
    「双歧杆菌」 still matches a chunk that wrote 「长双歧杆菌」.
    """
    tokens = [t.strip().lower() for t in jieba.cut_for_search(text) if t.strip()]
    # Keep alphanumeric strain identifiers intact as their own tokens.
    tokens += [m.lower() for m in re.findall(r"[A-Za-z]{2,}[\s\-]?\d{3,}|[A-Za-z]{3,}", text)]
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def load_chunks():
    chunks = []
    for path in sorted(RESEARCH_DIR.glob("*.md")):
        # README.md 是这个目录的说明文件，不是研究资料。
        # 把它切进语料等于让智能体拿目录说明当医学证据回答用户。
        if path.stem.lower() in ("readme", "index"):
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_document(text, path.stem))
    return chunks




def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"no research documents in {RESEARCH_DIR}")
    print(f"{len(chunks)} chunks from {len(list(RESEARCH_DIR.glob('*.md')))} documents")

    graded = sum(1 for c in chunks if c["evidence_grade"])
    regulatory = sum(1 for c in chunks if c["is_regulatory"])
    print(f"  {graded} carry an evidence grade, {regulatory} are regulatory")

    print(f"  向量后端：{backend_id()}")
    vectors = embed_texts([c["text"] for c in chunks], progress=True)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    np.save(vectors_path(), vectors)
    print(f"已写入 {CHUNKS_PATH} 和 {vectors_path()} {vectors.shape}")


def load_index():
    vpath = vectors_path()
    if not CHUNKS_PATH.exists() or not vpath.exists():
        raise SystemExit(
            f"当前向量后端 {backend_id()} 还没有建过索引。\n"
            f"请运行： python -m agent.index build")
    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = np.load(vpath)
    return chunks, vectors


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print(__doc__)
