#!/usr/bin/env python3
"""Perplexity deep-research runner for the mingtu gut-microbiome agent.

Standing rules this file encodes:
  - deepest available model always (sonar-deep-research)
  - prompts are composed on ONE line
  - quota exhaustion is a FULL STOP, never a silent downgrade to a weaker model
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Leon's standing location for the Perplexity key. All three carry the same key
# today; curio/.env is the one to read, the others are fallback only.
KEY_ENVS = [
    Path.home() / "Desktop/dev/curio/.env",
    Path.home() / "Desktop/dev/Curio-Automation/.env",
    Path.home() / "Desktop/dev/curio-international/.env",
]

DEFAULT_MODEL = "sonar-deep-research"

# Appended to every question. Chinese output, evidence-tiered, citation-carrying.
RIGOUR = """
用中文回答。这份材料要用来做一个面向中国普通用户的肠道健康智能体的知识库，因此要求：
1 每一条结论必须标注证据等级：A=人体随机对照试验的荟萃分析/系统综述，B=单个人体RCT，C=队列/观察性研究，D=动物或体外研究，E=专家意见或机制推测。
2 凡是只有D或E级证据的说法，必须明确写出"证据不足，不可对用户断言"。
3 涉及具体菌株必须写到菌株号（例如 LGG ATCC 53103、BB-12、DSM 17938），不能只写属种。
4 涉及剂量必须写出单位（CFU/日、克/日）、疗程和研究人群。
5 给出发表年份，优先2020年之后的证据，并注明是否有相反结论的研究。
6 最后附参考文献列表，带可访问的URL。
7 如果某个问题目前学界没有定论，直接说没有定论，不要编造共识。
"""

QUESTIONS = {
    "01-strains": "哪些具体益生菌菌株在人体随机对照试验或荟萃分析中被证明对特定适应症有效？请分适应症整理：抗生素相关性腹泻、儿童急性感染性腹泻、肠易激综合征IBS、功能性便秘、幽门螺杆菌根除辅助治疗、旅行者腹泻、婴幼儿绞痛、特应性皮炎预防。每个适应症给出有效菌株的菌株号、每日剂量CFU、疗程天数、效应量（RR或NNT）、证据等级和主要文献。同时指出哪些常见商业菌株其实缺乏该适应症的人体证据。",
    "02-diet": "膳食纤维与益生元改善肠道菌群的人体证据是什么？请覆盖：菊粉、低聚果糖FOS、低聚半乳糖GOS、抗性淀粉、聚葡萄糖、β-葡聚糖的有效剂量区间与产气腹胀的耐受阈值；发酵食品对菌群多样性和炎症标志物影响的RCT证据（含Stanford 2021 Wastyk等研究）；每日25至30克纤维和每周30种植物这两个流行建议各自的证据强度与来源；中国居民实际膳食纤维摄入量与中式饮食中可行的高纤维来源。",
    "03-china-reg": "中国关于益生菌与肠道健康产品的监管现状是什么？请覆盖：国家卫健委发布的可用于食品的菌种名单最新版本及其收录菌种、可用于婴幼儿食品的菌种名单、保健食品与普通食品在功能宣称上的法规边界、食品安全法与广告法禁止的健康宣称类型、益生菌产品必须标注到菌株号的相关规定或团体标准、中国益生菌市场规模与主要品牌及其所用菌株。请给出法规文号和发布年份。",
    "04-limits": "肠道健康咨询中的临床边界与常见误区有哪些？请覆盖：需要立即转诊就医的消化道报警症状清单及其依据；粪菌移植FMT在中国的适应症、监管状态与风险；面向消费者的商业肠道菌群检测在科学界受到的批评与其可重复性问题；益生菌的安全性禁忌人群（免疫抑制、危重症、中心静脉置管、短肠综合征、早产儿）及已报告的不良事件；对以下流行说法的循证反驳或支持：酸奶等于益生菌、益生菌必须冷藏、菌种越多越好、益生菌需要长期不间断服用、益生菌需要空腹服用。",
    "06-symptoms": "针对以下具体消化道症状，目前有哪些循证的日常处理方案？请分症状整理：慢性便秘、腹胀与排气过多、功能性腹泻、餐后不适与消化不良、反复口臭。每个症状给出：常见的非疾病性成因、有人体证据支持的饮食与行为干预及其效应量、有证据的益生菌菌株与剂量、多久应该看到改善、什么情况下说明不是功能性问题需要就医。注明每条建议的证据等级。",
    "07-drugs": "常见药物与肠道菌群的相互作用有哪些人体证据？请覆盖：不同种类抗生素对菌群的破坏程度与恢复时间、质子泵抑制剂PPI长期使用对菌群和感染风险的影响、二甲双胍与菌群的双向关系、非甾体抗炎药NSAIDs对肠黏膜的影响、益生菌与抗生素的服用间隔建议及其依据、益生菌与免疫抑制剂的相互作用风险。给出证据等级、研究人群和主要文献。",
    "08-china-diet": "中国居民的实际饮食与生活方式对肠道菌群的影响有哪些证据？请覆盖：中式高油高盐饮食、外卖与预制菜、饮酒（白酒啤酒）、辛辣食物、熬夜与作息紊乱、久坐、以及中国人乳糖不耐受高发对肠道健康的影响；中国居民膳食指南对膳食纤维和发酵食品的建议及其依据；中式饮食中可行的高纤维与发酵食品来源及其实际含量。给出证据等级并区分中国人群研究与西方人群研究。",
    "09-product": "中国消费者应该怎样看懂和挑选益生菌产品？请覆盖：产品标签上必须标注和应该标注的信息（菌株号、活菌数、保质期末活菌数、菌种名单合规性）、活菌数标注的常见误导方式、常温与冷藏剂型的差异与证据、益生菌与益生元与合生元的区别、婴幼儿配方与成人产品的差异、酸奶与发酵乳与益生菌补充剂的实际差别、以及在中国市场上如何核验一个产品宣称是否合规。给出法规依据与证据等级。",
    "05-systemic": "肠道菌群与全身健康关联的人体证据强度分别是多少？请按证据等级分级评估：肠脑轴与情绪或焦虑抑郁、肠肝轴与脂肪肝、肠皮轴与痤疮及特应性皮炎、菌群与肥胖及2型糖尿病、菌群与免疫功能和呼吸道感染、菌群与口臭。特别说明哪些关联目前只有动物实验或横断面关联研究支持而没有人体干预证据，哪些已有RCT支持。并指出人体肠道免疫细胞占全身免疫细胞70%这一常见说法的原始出处与准确性。",
}


def key():
    for env in KEY_ENVS:
        if not env.exists():
            continue
        m = re.search(r"^PERPLEXITY_API_KEY=(.*)$", env.read_text(), re.M)
        if m and m.group(1).strip():
            return m.group(1).strip().strip('"').strip("'")
    return ""


def one_line(text):
    return re.sub(r"\s+", " ", text).strip()


def send(prompt, model=DEFAULT_MODEL, timeout=2400):
    k = key()
    if not k:
        return None, "no PERPLEXITY_API_KEY"
    payload = {"model": model, "temperature": 0.2,
               "messages": [{"role": "user", "content": one_line(prompt)}]}
    tmp = REPO / f".payload-{abs(hash(prompt)) % 10**8}.json"
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    try:
        p = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout), "-X", "POST",
             "https://api.perplexity.ai/chat/completions",
             "-H", f"Authorization: Bearer {k}",
             "-H", "content-type: application/json",
             "--data-binary", f"@{tmp}"],
            capture_output=True, text=True)
    finally:
        tmp.unlink(missing_ok=True)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None, f"non-JSON: {p.stdout[:200]}"
    if "choices" not in d:
        return None, json.dumps(d.get("error", d), ensure_ascii=False)[:300]
    body = d["choices"][0]["message"]["content"]
    cites = d.get("citations") or d.get("search_results") or []
    if cites:
        body += "\n\n## 来源\n"
        for c in cites[:60]:
            body += f"- {c if isinstance(c, str) else c.get('url', '')}\n"
    usage = d.get("usage", {})
    return body, f"ok tokens={usage.get('total_tokens', '?')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="cheap key check on the small model")
    ap.add_argument("--ask", help="run one named question key")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    if a.probe:
        body, err = send("用一句话说明什么是肠道菌群。", model="sonar", timeout=90)
        print(err)
        print((body or "")[:300])
        return 0 if body else 1

    if a.ask not in QUESTIONS:
        print(f"unknown question: {a.ask}; have {list(QUESTIONS)}", file=sys.stderr)
        return 2

    out = REPO / "research" / f"{a.ask}.md"
    body, err = send(QUESTIONS[a.ask] + RIGOUR, model=a.model)
    if not body:
        # Quota dry is a full stop. Park the prompt so it can be pasted by hand.
        park = REPO / "research" / f"{a.ask}.PARKED.txt"
        park.write_text(one_line(QUESTIONS[a.ask] + RIGOUR))
        print(f"FAILED {a.ask}: {err}\nprompt parked at {park}", file=sys.stderr)
        return 1
    out.write_text(f"# {a.ask}\n\n> 模型 {a.model}\n\n## 问题\n\n{one_line(QUESTIONS[a.ask])}\n\n## 回答\n\n{body}\n")
    print(f"OK {a.ask} -> {out} ({err}, {len(body)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
