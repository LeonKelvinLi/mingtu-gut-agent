"""逐句流式输出，且每句都先过合规闸门再放出去。

为什么不能直接流式：合规校验原本在整段生成完之后跑。如果照搬常见做法把 token
直接推给前端，那么一句违规的疾病治疗宣称会**先被用户看到**，几秒后才被撤回。
对一个中国的健康产品来说，这不是体验问题，是《广告法》问题。

这里的做法是：按句缓冲，每凑满一个完整句子就跑一次 screen_output，
干净的句子才发出去；一旦有句子违规，立刻停止流式、丢弃整段，
退回非流式的重写流程。违规内容一个字都不会到达前端。

引用编号越界同样在收尾时校验——编号错的答案宁可重写，不能将就。
"""

import re

from agent import safety

# 句子边界。中文标点为主，同时兼容英文句号和换行。
_BOUNDARY = re.compile(r"(?<=[。！？；\n])")
# 最小放行长度：太碎的片段逐句发反而闪烁。
MIN_EMIT = 12


class SentenceGate:
    """把 token 流整理成「已过合规检查的句子」流。"""

    def __init__(self):
        self.buffer = ""
        self.emitted = []
        self.blocked = None

    def feed(self, chunk):
        """喂入一段新文本，返回本次可以安全放出的字符串（可能为空）。"""
        if self.blocked:
            return ""
        self.buffer += chunk

        parts = _BOUNDARY.split(self.buffer)
        # 最后一段可能是不完整的句子，留在缓冲里等下一次。
        complete, self.buffer = parts[:-1], parts[-1]

        out = []
        for sentence in complete:
            if not sentence.strip():
                continue
            violations = safety.screen_output(sentence)
            if violations:
                # 违规句子绝不放行，整段作废走重写。
                self.blocked = violations
                return ""
            out.append(sentence)

        text = "".join(out)
        if text:
            self.emitted.append(text)
        return text

    def finish(self):
        """收尾：把缓冲里剩下的半句也检查后放出。"""
        if self.blocked:
            return ""
        tail = self.buffer
        self.buffer = ""
        if not tail.strip():
            return ""
        violations = safety.screen_output(tail)
        if violations:
            self.blocked = violations
            return ""
        self.emitted.append(tail)
        return tail

    @property
    def text(self):
        return "".join(self.emitted)


def stream_answer(system, messages, passages, max_tokens=2000):
    """生成器：产出 (kind, payload)。

    kind 取值：
      "delta"  —— 可以直接追加显示的干净文本
      "done"   —— 完整答案（已通过逐句合规检查与引用校验）
      "reject" —— 流式路径不可用或校验未过，调用方应退回非流式重写流程
    """
    from agent.core import Agent
    from agent.llm import stream_generate

    gate = SentenceGate()
    produced = False

    for chunk in stream_generate(system, messages, max_tokens=max_tokens):
        produced = True
        safe = gate.feed(chunk)
        if gate.blocked:
            yield ("reject", gate.blocked)
            return
        if len(safe) >= MIN_EMIT or (safe and "\n" in safe):
            yield ("delta", safe)
        elif safe:
            yield ("delta", safe)

    tail = gate.finish()
    if gate.blocked:
        yield ("reject", gate.blocked)
        return
    if tail:
        yield ("delta", tail)

    if not produced or not gate.text.strip():
        yield ("reject", "生成为空")
        return

    problems = Agent.validate_citations(gate.text, passages)
    if problems:
        # 引用有问题的答案已经流出去了，前端会用重写结果整体替换。
        yield ("reject", problems)
        return

    yield ("done", gate.text)
