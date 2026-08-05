#!/usr/bin/env python3
"""把九篇研究报告合成一份 .docx 综述。

真正的难点是引用编号。每篇报告各自维护一套 [1]–[20]，指向自己文末的来源列表。
直接首尾相接会让所有引用错位——第三章的 [5] 会指到第一章的第五条来源。
所以这里做的是：抽出每篇的来源表 → 建全局去重表 → 把每篇正文里的 [n]
重映射到全局编号。合并后引用仍然可追溯，这是这份文档能不能当研究材料用的前提。
"""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

REPO = Path(__file__).resolve().parent.parent
RESEARCH = REPO / "research"
OUT = REPO / "肠道菌群健康循证综述.docx"

CHAPTERS = [
    ("01-strains", "菌株级人体证据：按适应症梳理"),
    ("06-symptoms", "常见消化道症状的循证处理"),
    ("02-diet", "膳食纤维、益生元与发酵食品"),
    ("08-china-diet", "中国居民饮食与生活方式对肠道菌群的影响"),
    ("07-drugs", "药物与肠道菌群的相互作用"),
    ("05-systemic", "肠道菌群与全身健康：关联强度分级"),
    ("04-limits", "临床边界、安全性与常见误区"),
    ("03-china-reg", "中国监管框架与合规边界"),
    ("09-product", "产品标签解读与选购"),
]

CITE = re.compile(r"\[(\d{1,3})\]")


def set_cjk_font(run, name="等线", size=None, bold=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def parse(path):
    """返回 (正文, 来源URL列表)。丢掉生成脚手架（模型行、问题段等）。"""
    text = path.read_text(encoding="utf-8")

    sources = []
    m = re.search(r"\n## 来源\n(.*)$", text, re.S)
    if m:
        sources = re.findall(r"^-\s*(https?://\S+)", m.group(1), re.M)
        text = text[:m.start()]

    # 去掉 "# 01-strains" / "> 模型 ..." / "## 问题 ... ## 回答"
    text = re.sub(r"^#\s*\d+-[\w-]+\s*$", "", text, flags=re.M)
    text = re.sub(r"^>\s*模型.*$", "", text, flags=re.M)
    text = re.sub(r"^##\s*问题\s*\n.*?(?=^##\s*回答)", "", text, flags=re.S | re.M)
    text = re.sub(r"^##\s*回答\s*$", "", text, flags=re.M)
    return text.strip(), sources


def remap(text, local_sources, global_index):
    """把本篇的 [n] 换成全局编号。找不到来源的编号原样保留但不再指向错的条目。"""
    def sub(match):
        n = int(match.group(1))
        if 1 <= n <= len(local_sources):
            return f"[{global_index[local_sources[n - 1]]}]"
        return ""      # 没有对应来源的编号直接删掉，胜过指向错误条目
    return CITE.sub(sub, text)


def add_md_paragraph(doc, line, style=None):
    """渲染一行 markdown（支持 **加粗**）。"""
    p = doc.add_paragraph(style=style)
    for part in re.split(r"(\*\*[^*]+\*\*)", line):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            set_cjk_font(run, size=10.5, bold=True)
        else:
            run = p.add_run(part)
            set_cjk_font(run, size=10.5)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.4
    return p


def render_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
    if not cells:
        return
    width = max(len(r) for r in cells)
    table = doc.add_table(rows=0, cols=width)
    table.style = "Light Grid Accent 1"
    for i, row in enumerate(cells):
        tr = table.add_row().cells
        for j in range(width):
            txt = row[j] if j < len(row) else ""
            tr[j].text = ""
            para = tr[j].paragraphs[0]
            run = para.add_run(re.sub(r"\*\*", "", txt))
            set_cjk_font(run, size=9, bold=(i == 0))
    doc.add_paragraph()


def render_body(doc, text):
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        # 表格
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            render_table(doc, block)
            continue

        heading = re.match(r"^(#{1,6})\s*(.+)$", line)
        if heading:
            level = min(len(heading.group(1)) + 1, 4)
            text_ = re.sub(r"\*\*", "", heading.group(2)).strip()
            h = doc.add_heading(level=level)
            run = h.add_run(text_)
            set_cjk_font(run, size={2: 15, 3: 12.5, 4: 11}.get(level, 11), bold=True)
            run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
            i += 1
            continue

        bullet = re.match(r"^\s*[-*•]\s+(.*)$", line)
        if bullet:
            add_md_paragraph(doc, bullet.group(1), style="List Bullet")
            i += 1
            continue

        numbered = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if numbered:
            add_md_paragraph(doc, numbered.group(1), style="List Number")
            i += 1
            continue

        add_md_paragraph(doc, line)
        i += 1


def build():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(10.5)
    set_cjk_font(style.element.rPr and style.paragraphs[0].runs[0] if False else doc.add_paragraph().add_run(""))
    doc.paragraphs[-1]._element.getparent().remove(doc.paragraphs[-1]._element)

    for section in doc.sections:
        section.left_margin = Cm(2.6)
        section.right_margin = Cm(2.6)
        section.top_margin = Cm(2.4)
        section.bottom_margin = Cm(2.4)

    # ---- 封面 ----
    for _ in range(6):
        doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_cjk_font(title.add_run("肠道菌群健康循证综述"), size=26, bold=True)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_cjk_font(sub.add_run("面向中国用户的证据分级知识库"), size=13)

    for _ in range(2):
        doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_cjk_font(meta.add_run("九个专题 · 证据分级 A–E · 全文引用可追溯"), size=10.5)

    doc.add_page_break()

    # ---- 摘要与方法 ----
    h = doc.add_heading(level=1)
    set_cjk_font(h.add_run("摘要与方法说明"), size=18, bold=True)

    for para in [
        "本文汇总九个专题的检索结果，覆盖菌株级人体证据、症状处理、膳食干预、"
        "药物相互作用、全身健康关联、临床边界与安全性、中国监管框架、以及产品选购。"
        "撰写目的是作为面向中国普通用户的健康问答知识库，因此每一条结论都标注证据等级，"
        "并明确区分「有人体试验支持」与「仅有动物或机制推测」。",
        "检索方法：使用 Perplexity sonar-deep-research 进行多轮代理式检索。"
        "统一的撰写约束为：涉及菌株必须写到菌株号（如 LGG ATCC 53103、DSM 17938），"
        "涉及剂量必须写明单位、疗程与研究人群，涉及法规必须给出文号与发布年份，"
        "优先采用 2020 年后的证据并注明是否存在相反结论，学界无定论时直接说明无定论。",
        "重要限制：本文是科普与循证梳理，不构成医疗建议，不能用于疾病诊断或治疗决策。"
        "文中所有关于食品与益生菌的表述均限定在「有助于」「可能改善」的范围内，"
        "不涉及疾病的预防、治疗或治愈功能。出现消化道报警症状者应直接就医。",
    ]:
        add_md_paragraph(doc, para)

    h = doc.add_heading(level=2)
    set_cjk_font(h.add_run("证据分级体系"), size=15, bold=True)
    render_table(doc, [
        "| 等级 | 含义 | 可否对用户断言 |",
        "| --- | --- | --- |",
        "| A | 人体随机对照试验的荟萃分析或系统综述 | 可以，但需说明异质性 |",
        "| B | 单个或少数高质量人体随机对照试验 | 可以，但限定在研究条件内 |",
        "| C | 队列或其他观察性研究 | 只能说相关，不能说因果 |",
        "| D | 动物实验或体外实验 | 不可断言，必须写明证据不足 |",
        "| E | 专家意见或机制推测 | 不可断言，必须写明证据不足 |",
    ])

    doc.add_page_break()

    # ---- 正文 ----
    global_index, ordered_urls = {}, []
    for key, title_text in CHAPTERS:
        path = RESEARCH / f"{key}.md"
        if not path.exists():
            print(f"  跳过（缺失）：{key}")
            continue
        body, sources = parse(path)
        for url in sources:
            if url not in global_index:
                ordered_urls.append(url)
                global_index[url] = len(ordered_urls)
        body = remap(body, sources, global_index)

        h = doc.add_heading(level=1)
        set_cjk_font(h.add_run(title_text), size=18, bold=True)
        render_body(doc, body)
        doc.add_page_break()
        print(f"  已合入 {key}：{len(body)} 字，{len(sources)} 条来源")

    # ---- 参考文献 ----
    h = doc.add_heading(level=1)
    set_cjk_font(h.add_run("参考文献"), size=18, bold=True)
    add_md_paragraph(doc, f"共 {len(ordered_urls)} 条，编号与正文中的方括号引用一一对应。")
    for i, url in enumerate(ordered_urls, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        set_cjk_font(p.add_run(f"[{i}] "), size=9, bold=True)
        set_cjk_font(p.add_run(url), size=9)

    doc.save(OUT)
    print(f"\n已生成：{OUT}")
    print(f"参考文献 {len(ordered_urls)} 条（去重后）")
    return OUT


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
