#!/usr/bin/env python3
"""预热缓存：把最常问的问题先跑一遍，让真实用户第一次问就是秒回。

为什么值得做：真实用户的问题分布极度集中，「益生菌怎么选」「酸奶和益生菌一样吗」
这类占了绝大多数。预热之后，绝大部分访客根本不会等模型。

题目怎么选的：不是拍脑袋列的，是按九篇语料实际覆盖的范围倒推——
每个问题都必须能在语料里找到有证据等级的段落，否则预热出来的是弃答，
把「我不知道」固化成标准答案比不缓存更糟。

运行： MINGTU_LOCAL_ONLY=0 .venv/bin/python tools/warm_cache.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent          # noqa: E402
from agent.llm import provider_label  # noqa: E402

QUESTIONS = [
    # 最基础的认知
    "益生菌到底有没有用",
    "益生菌怎么选",
    "酸奶和益生菌是一回事吗",
    "益生菌和益生元有什么区别",
    "益生菌菌株号是什么意思",
    "活菌数越高越好吗",
    # 服用方式
    "益生菌什么时候吃最好",
    "益生菌需要冷藏吗",
    "益生菌可以天天吃吗",
    "益生菌要吃多久才有效果",
    "吃益生菌之后胀气怎么办",
    # 具体场景
    "吃完抗生素要不要补益生菌",
    "便秘可以吃益生菌吗",
    "拉肚子能吃益生菌吗",
    "肠易激综合征怎么调理",
    "总是腹胀放屁多怎么回事",
    "口臭和肠道有关系吗",
    # 人群
    "宝宝可以吃益生菌吗",
    "孕妇能不能吃益生菌",
    "老年人适合吃益生菌吗",
    "什么人不能吃益生菌",
    # 饮食与生活方式
    "每天要吃多少膳食纤维",
    "哪些食物对肠道菌群好",
    "发酵食品对肠道有好处吗",
    "熬夜会影响肠道菌群吗",
    "经常喝酒对肠道有什么影响",
    "外卖吃多了肠道会怎么样",
    # 常见误解
    "肠道菌群检测有用吗",
    "益生菌能减肥吗",
    "益生菌能提高免疫力吗",
    "益生菌能改善皮肤吗",
    "怎么知道自己肠道菌群失衡了",
    # 合规边界
    "益生菌能不能治疗肠炎",
    "保健食品和普通食品有什么区别",
]


def main():
    agent = Agent()
    print(f"作答模型：{provider_label()}")
    print(f"预热前缓存条目：{len(agent.cache)}")
    print(f"待预热问题：{len(QUESTIONS)} 条\n")

    warmed = skipped = failed = 0
    for i, q in enumerate(QUESTIONS, 1):
        if agent.cache.lookup_literal(q):
            print(f"[{i:2d}/{len(QUESTIONS)}] 已有缓存  {q}")
            skipped += 1
            continue
        t = time.time()
        try:
            r = agent.answer(q)
        except Exception as exc:
            print(f"[{i:2d}/{len(QUESTIONS)}] 异常      {q} — {exc!r:.80}")
            failed += 1
            continue

        elapsed = time.time() - t
        # 弃答、兜底、无来源的结果一律不算预热成功——写进缓存会把失败固化。
        if r.get("error") or not r.get("sources"):
            print(f"[{i:2d}/{len(QUESTIONS)}] 未缓存    {q} "
                  f"（{r.get('error') or '无有效引用'}，{elapsed:.0f}s）")
            failed += 1
        else:
            print(f"[{i:2d}/{len(QUESTIONS)}] 已缓存    {q} "
                  f"（{len(r['sources'])} 条来源，{elapsed:.0f}s）")
            warmed += 1

    print(f"\n新增 {warmed} 条，跳过 {skipped} 条，未缓存 {failed} 条")
    print(f"缓存总条目：{len(agent.cache)}")
    if failed:
        print("提示：未缓存的问题说明语料覆盖不足，考虑补一篇针对性研究再重跑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
