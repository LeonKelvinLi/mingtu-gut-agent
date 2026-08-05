"""Split researched source documents into retrievable chunks with metadata.

Metadata is the point of this file. Vector similarity alone cannot tell that a
claim is D-grade animal evidence from 2015 about mice, versus an A-grade 2024
meta-analysis in adults — but the retriever has to, because those two passages
answer the same question with completely different authority.

Every chunk carries what the metadata filter needs:
  evidence_grade  A/B/C/D/E  — parsed from the source's own grading
  population      成人 / 儿童 / 婴幼儿 / 孕妇 / 老年 / 免疫抑制
  year            most recent 4-digit year mentioned
  authority       meta-analysis > rct > cohort > animal > opinion
  is_regulatory   True for 法规/标准 passages, which override evidence grading
"""

import hashlib
import re

# Target chunk size in Chinese characters. Chinese runs ~1-1.5 tokens/char for
# Claude, so ~450 chars lands near 500-650 tokens — small enough that a reranked
# top-8 stays cheap, large enough to keep a claim with its own citation.
TARGET_CHARS = 450
OVERLAP_CHARS = 90
MIN_CHARS = 120

_GRADE = re.compile(r"证据等级[：:\s]*([A-E])|[（(]([A-E])\s*级[）)]|\b([A-E])级证据")
_YEAR = re.compile(r"(19[89]\d|20[0-3]\d)\s*年?")
_URL = re.compile(r"https?://[^\s）)、，,；;】\]]+")

_POPULATIONS = [
    ("婴幼儿", r"婴幼儿|婴儿|新生儿|早产儿|幼儿"),
    ("儿童", r"儿童|小儿|学龄|青少年"),
    ("孕产妇", r"孕妇|妊娠|孕期|哺乳期"),
    ("老年", r"老年|老人|高龄"),
    ("免疫抑制", r"免疫抑制|免疫缺陷|化疗|器官移植|危重症"),
    ("成人", r"成人|成年"),
]

# Ordered strongest-first; the first match wins.
_AUTHORITY = [
    ("meta-analysis", r"荟萃分析|Meta分析|meta分析|系统综述|系统评价|Cochrane"),
    ("guideline", r"指南|共识|专家共识|标准|规范"),
    ("rct", r"随机对照|RCT|随机双盲|安慰剂对照"),
    ("cohort", r"队列研究|观察性研究|横断面|病例对照"),
    ("animal", r"动物实验|小鼠|大鼠|体外实验|细胞实验"),
    ("opinion", r"专家意见|机制推测|理论上"),
]

_REGULATORY = re.compile(
    r"广告法|食品安全法|卫健委|国家卫生健康委|市场监管总局|"
    r"保健食品|可用于食品的菌种名单|团体标准|国标|GB\s?\d|法规|文号"
)


def _first(pattern_list, text, default=None):
    for label, pattern in pattern_list:
        if re.search(pattern, text):
            return label
    return default


def _all(pattern_list, text):
    return [label for label, pattern in pattern_list if re.search(pattern, text)]


def extract_metadata(text, source, heading):
    grade_match = _GRADE.search(text)
    grade = None
    if grade_match:
        grade = next(g for g in grade_match.groups() if g)

    years = [int(y) for y in _YEAR.findall(text)]

    return {
        "source": source,
        "heading": heading,
        "evidence_grade": grade,
        "population": _all(_POPULATIONS, text) or ["通用"],
        "year": max(years) if years else None,
        "authority": _first(_AUTHORITY, text, default="unspecified"),
        "is_regulatory": bool(_REGULATORY.search(text)),
        "citations": _URL.findall(text),
    }


def _split_long(body):
    """Split an over-long section on paragraph boundaries, with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        # A single paragraph over target gets hard-split on sentence ends.
        if len(para) > TARGET_CHARS * 2:
            sentences = re.split(r"(?<=[。！？；])", para)
            for sentence in sentences:
                if len(current) + len(sentence) > TARGET_CHARS and len(current) >= MIN_CHARS:
                    chunks.append(current)
                    current = current[-OVERLAP_CHARS:] + sentence
                else:
                    current += sentence
            continue

        if len(current) + len(para) > TARGET_CHARS and len(current) >= MIN_CHARS:
            chunks.append(current)
            current = current[-OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(current)
    return chunks


def chunk_document(text, source):
    """Split one research markdown file into metadata-carrying chunks."""
    # Section on markdown headings so a chunk keeps its topical context.
    parts = re.split(r"\n(?=#{1,4}\s)", text)
    chunks = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        heading_match = re.match(r"#{1,4}\s*(.+)", part)
        heading = heading_match.group(1).strip() if heading_match else "正文"
        body = part[heading_match.end():].strip() if heading_match else part
        if len(body) < MIN_CHARS:
            continue

        for piece in _split_long(body):
            piece = piece.strip()
            if len(piece) < MIN_CHARS:
                continue
            # The heading rides with the text so retrieval sees the topic.
            full = f"【{heading}】\n{piece}"
            meta = extract_metadata(piece, source, heading)
            meta["text"] = full
            meta["id"] = hashlib.sha1(
                f"{source}:{full}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(meta)

    return chunks
