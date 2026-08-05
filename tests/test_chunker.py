"""Chunk metadata extraction.

Metadata is what lets the agent say "this is only mouse data" instead of stating
a D-grade claim in the same voice as an A-grade one. If extraction silently
fails, nothing downstream errors — the chunk just loses its grade, the retriever
loses its boost, and the answer loses the hedge. So these tests check extraction
against the exact way the research corpus writes things, not against tidy
synthetic strings.
"""

import re

import pytest

from agent.chunker import MIN_CHARS, TARGET_CHARS, chunk_document, extract_metadata
from conftest import RESEARCH_DIR, RESEARCH_READY, known_gap


def meta(text, source="01-strains", heading="测试标题"):
    return extract_metadata(text, source, heading)


# --- evidence grade -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("证据等级：A。一项荟萃分析显示……", "A"),
    ("证据等级: B，单个人体RCT", "B"),
    ("证据等级 C 观察性研究", "C"),
    ("这条结论（D级）来自小鼠实验", "D"),
    ("该说法（E 级）属于机制推测", "E"),
    ("没有任何等级标注的一段话", None),
])
def test_evidence_grade_forms(text, expected):
    assert meta(text)["evidence_grade"] == expected


@pytest.mark.parametrize("text,expected", [
    ("可归入 A 级证据。", "A"),          # corpus form: space between letter and 级
    ("属于典型 E 级证据，不适合作为结论", "E"),
    ("该结论为C级证据", "C"),            # no space, but preceded by a Chinese char
])
def test_evidence_grade_spaced_and_inline_forms(text, expected):
    assert meta(text)["evidence_grade"] == expected


def test_first_grade_marker_wins():
    assert meta("证据等级：A。后文提到（D级）动物数据。")["evidence_grade"] == "A"


# --- population ---------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("婴幼儿配方食品中的菌种", {"婴幼儿"}),
    ("儿童急性腹泻的疗程", {"儿童"}),
    ("孕妇与哺乳期女性的补充", {"孕产妇"}),
    ("老年人便秘的干预", {"老年"}),
    ("化疗期间与器官移植后的风险", {"免疫抑制"}),
    ("成人受试者每日剂量", {"成人"}),
])
def test_population_detection(text, expected):
    assert expected <= set(meta(text)["population"])


def test_population_is_multi_valued():
    populations = set(meta("纳入婴幼儿、儿童与孕妇三组人群")["population"])
    assert {"婴幼儿", "儿童", "孕产妇"} <= populations


def test_population_defaults_to_generic():
    assert meta("一段没有指明人群的通用描述")["population"] == ["通用"]


# --- year ---------------------------------------------------------------------

def test_year_takes_the_most_recent():
    assert meta("2015年的小鼠实验，2023年有人体RCT验证")["year"] == 2023


def test_year_absent_is_none():
    assert meta("没有写年份的一段话")["year"] is None


def test_year_ignores_non_year_numbers():
    assert meta("每日剂量100亿CFU，疗程14天")["year"] is None


# --- authority ----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("一项荟萃分析的结果", "meta-analysis"),
    ("Cochrane 系统综述", "meta-analysis"),
    ("《中国专家共识》建议", "guideline"),
    ("随机双盲安慰剂对照试验", "rct"),
    ("一项前瞻性队列研究", "cohort"),
    ("小鼠实验与体外实验数据", "animal"),
    ("这更多是机制推测", "opinion"),
    ("一段没有研究类型描述的话", "unspecified"),
])
def test_authority_detection(text, expected):
    assert meta(text)["authority"] == expected


def test_authority_takes_the_strongest_when_several_appear():
    """A meta-analysis of RCTs is a meta-analysis, not an RCT."""
    assert meta("对多项随机对照试验的荟萃分析")["authority"] == "meta-analysis"
    assert meta("指南引用了队列研究")["authority"] == "guideline"


# --- regulatory ---------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "《广告法》第十七条规定",
    "《食品安全法》第七十三条",
    "国家卫生健康委发布的公告",
    "国家卫健委可用于食品的菌种名单",
    "市场监管总局的处罚案例",
    "保健食品批准文号",
    "团体标准 T/CNFIA 131",
    "GB 7101 相关规定",
])
def test_regulatory_detection(text):
    assert meta(text)["is_regulatory"] is True


@pytest.mark.parametrize("text", [
    "可用于婴幼儿食品的菌种名单",
    "婴幼儿配方乳粉中允许添加的菌株见可用于婴幼儿食品的菌种名单",
])
def test_infant_strain_list_is_regulatory(text):
    assert meta(text)["is_regulatory"] is True


@pytest.mark.parametrize("text", [
    "每日25至30克膳食纤维的人体证据",
    "LGG ATCC 53103 缩短腹泻病程",
    "发酵食品与菌群多样性的关系",
])
def test_non_regulatory_text_is_not_flagged(text):
    assert meta(text)["is_regulatory"] is False


# --- citations ----------------------------------------------------------------

def test_citation_urls_are_extracted_without_trailing_punctuation():
    text = ("见 https://pubmed.ncbi.nlm.nih.gov/12345678/ 与 http://example.cn/a?b=1，"
            "以及（https://who.int/x）")
    citations = meta(text)["citations"]
    assert "https://pubmed.ncbi.nlm.nih.gov/12345678/" in citations
    assert "http://example.cn/a?b=1" in citations
    assert "https://who.int/x" in citations
    for url in citations:
        assert not url.endswith(("，", "）", "、", "；"))


# --- document chunking --------------------------------------------------------

DOC = (
    "# 一级标题\n\n" + "第一节的内容。" * 40 + "\n\n"
    "## 1.1 子标题\n\n证据等级：A。" + "第二节的内容，随机对照试验的荟萃分析，2024年。" * 20 + "\n\n"
    "## 1.2 太短\n\n短。\n"
)


def test_chunks_carry_their_heading_in_the_text():
    chunks = chunk_document(DOC, "test-doc")
    assert chunks
    for chunk in chunks:
        assert chunk["text"].startswith("【")
        assert chunk["heading"] in chunk["text"][:40]


def test_chunk_metadata_is_attached_per_chunk():
    chunks = chunk_document(DOC, "test-doc")
    graded = [c for c in chunks if c["heading"] == "1.1 子标题"]
    assert graded
    assert graded[0]["evidence_grade"] == "A"
    assert graded[0]["authority"] == "meta-analysis"
    assert graded[0]["year"] == 2024
    assert graded[0]["source"] == "test-doc"


def test_sections_below_minimum_length_are_dropped():
    headings = {c["heading"] for c in chunk_document(DOC, "test-doc")}
    assert "1.2 太短" not in headings


def test_chunk_ids_are_stable_and_source_scoped():
    first = chunk_document(DOC, "test-doc")
    again = chunk_document(DOC, "test-doc")
    assert [c["id"] for c in first] == [c["id"] for c in again], "重建索引不该改变块ID"
    other = chunk_document(DOC, "other-doc")
    assert first[0]["id"] != other[0]["id"], "不同来源的同一段文字必须是不同的块"


def test_chunk_ids_are_unique_within_a_document():
    ids = [c["id"] for c in chunk_document(DOC, "test-doc")]
    assert len(ids) == len(set(ids))


def test_long_section_is_split_with_overlap():
    long_doc = "# 大节\n\n" + "这是一个很长的段落。" * 300
    chunks = chunk_document(long_doc, "long")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= TARGET_CHARS * 2
        assert chunk["heading"] == "大节"


def test_document_without_headings_still_chunks():
    chunks = chunk_document("没有任何标题的正文。" * 30, "plain")
    assert chunks
    assert chunks[0]["heading"] == "正文"


def test_every_chunk_meets_the_minimum_length():
    for chunk in chunk_document(DOC, "test-doc"):
        assert len(chunk["text"]) >= MIN_CHARS


# --- against the real corpus --------------------------------------------------

@pytest.mark.skipif(not RESEARCH_READY, reason="research/*.md not present")
def test_regulatory_corpus_is_actually_marked():
    """The regulatory quota in retrieve.py is worthless if nothing is flagged."""
    text = (RESEARCH_DIR / "03-china-reg.md").read_text(encoding="utf-8")
    chunks = chunk_document(text, "03-china-reg")
    assert len(chunks) > 30
    regulatory = sum(1 for c in chunks if c["is_regulatory"])
    assert regulatory / len(chunks) > 0.4, (
        f"法规文档只有 {regulatory}/{len(chunks)} 块被标为法规，配额检索会落空")


@pytest.mark.skipif(not RESEARCH_READY, reason="research/*.md not present")
def test_no_chunk_with_a_visible_grade_marker_is_left_ungraded():
    marker = re.compile(r"([A-E])\s*级证据")
    missed = []
    for path in sorted(RESEARCH_DIR.glob("*.md")):
        for chunk in chunk_document(path.read_text(encoding="utf-8"), path.stem):
            if chunk["evidence_grade"] is None and marker.search(chunk["text"]):
                missed.append((path.stem, chunk["heading"]))
    assert not missed, f"{len(missed)} 个块写了证据等级却没被解析出来"
