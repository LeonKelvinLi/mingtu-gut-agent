#!/usr/bin/env python3
"""肠道菌群健康顾问 —— 中文交互式终端入口。

运行：.venv/bin/python cli.py

这一层只做三件事：把问题交给 agent.core.Agent、把这一轮的过程指标显示出来、
把回答真正引用的来源列清楚。它不做任何医学判断，也不改写回答内容——
急症话术、引用校验、合规校验都在 agent/ 里，CLI 只负责如实呈现。
"""

import os
import re
import sys
import threading
import time
import unicodedata

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# jieba 第一次加载会往 stderr 打英文日志，压掉它，界面里不该出现英文噪音。
try:
    import jieba

    jieba.setLogLevel(60)
except Exception:
    pass


# --- 终端能力 ---------------------------------------------------------------

_TTY = sys.stdout.isatty()
_COLOR = _TTY and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"

BOLD, DIM, RED, YELLOW, CYAN, GREEN, REVERSE = "1", "2", "31", "33", "36", "32", "7"


def paint(text, *codes):
    if not _COLOR or not codes:
        return text
    return "\033[" + ";".join(codes) + "m" + text + "\033[0m"


def term_width():
    try:
        return max(40, min(os.get_terminal_size().columns, 78))
    except OSError:
        return 72


def _char_width(ch):
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def clip(text, limit):
    """按显示宽度截断，中文按两格算，避免来源行折行。"""
    text = " ".join(text.split())
    if sum(_char_width(c) for c in text) <= limit:
        return text
    out, used = [], 0
    for ch in text:
        w = _char_width(ch)
        if used + w > limit - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


class Progress:
    """作答期间的等待提示。一轮要跑分流、检索、实时补充、重排、作答，
    可能几十秒，不给反馈会让人以为卡死了。"""

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if _TTY:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        else:
            print(self.label, flush=True)
        return self

    def _tick(self):
        start = time.time()
        while not self._stop.wait(0.4):
            sys.stdout.write("\r" + paint(f"{self.label} {int(time.time() - start)} 秒", DIM))
            sys.stdout.flush()

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        return False


# --- 展示用的中文标签 -------------------------------------------------------

RISK_LABEL = {"normal": "一般", "caution": "谨慎", "emergency": "急症"}
RISK_COLOR = {"normal": (), "caution": (YELLOW,), "emergency": (RED, BOLD)}

SOURCE_LABEL = {
    "01-strains": "菌株证据",
    "02-diet": "饮食与益生元",
    "03-china-reg": "中国法规",
    "04-limits": "安全边界与局限",
    "05-systemic": "机制与全身健康",
    "live:perplexity": "实时检索",
}

GRADE_LABEL = {
    "A": "A级 人体RCT荟萃分析",
    "B": "B级 单个人体试验",
    "C": "C级 观察性研究",
    "D": "D级 动物或体外",
    "E": "E级 专家意见",
}

ERROR_NOTE = {
    "no_model_key": "没有配置任何模型 API Key，本轮只跑了检索与安全层，没有生成回答。",
    "generation_failed": "本轮生成失败，可能是网络或接口异常。可以直接再问一次。",
    "validation_failed": "本轮回答没通过引用校验或合规校验，已按规则弃答，不会把不合格的内容发给你。",
}

_CITATION = re.compile(r"\[\d{1,2}\]")
_URL_BODY = re.compile(r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")

MAX_HISTORY_TURNS = 6


def source_label(source):
    return SOURCE_LABEL.get(source, source or "本地资料")


def grade_label(source_item):
    if source_item.get("is_live"):
        return "实时检索 未分级"
    grade = source_item.get("grade")
    return GRADE_LABEL.get(grade, "未标注等级")


def clean_url(url):
    """来源里混进过中文标点和残缺的脚注编号，展示前先切干净。"""
    url = (url or "").strip()
    match = _URL_BODY.match(url)
    if match:
        url = match.group(0)
    return url.rstrip(".,;:)]}")


def strip_citations(answer):
    """存进历史前去掉引用编号：编号只对生成它的那一轮有效，
    留在上文里会诱导模型在下一轮复用不存在的编号。"""
    text = _CITATION.sub("", answer)
    text = re.sub(r"[ \t]+([。，、；：！？）】」』])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# --- 输出 -------------------------------------------------------------------

def rule(char="─"):
    print(paint(char * term_width(), DIM))


def print_banner():
    width = term_width()
    print()
    print(paint("肠道菌群健康顾问", BOLD))
    print(paint("按证据等级说话的肠道健康科普助手。做科普和就医导航，不做诊断、不开处方、不解读化验单。", DIM))
    print(paint("─" * width, DIM))
    print(paint("命令： /退出 结束    /清空 清空对话    /调试 显示检索细节    /帮助 查看全部", DIM))
    print()


def print_help():
    print()
    print(paint("可用命令", BOLD))
    for cmd, desc in (
        ("/退出", "结束对话（也可以用 /quit、/exit，或按 Ctrl-D）"),
        ("/清空", "清空上下文，重新开始一段对话（也可以用 /clear）"),
        ("/调试", "开关调试模式，显示分流细节和全部候选资料（也可以用 /debug）"),
        ("/帮助", "显示这份说明（也可以用 /help）"),
    ):
        print(f"  {paint(cmd, CYAN)}  {desc}")
    print()


def print_status(result):
    routing = result.get("routing") or {}
    risk = routing.get("risk", "normal")
    parts = [
        "风险 " + paint(RISK_LABEL.get(risk, risk), *RISK_COLOR.get(risk, ())),
        f"资料 {len(result.get('passages') or [])} 条",
        "实时检索 " + ("已使用" if result.get("used_live") else "未使用"),
        f"重写 {result.get('repairs', 0)} 次",
    ]
    if routing.get("_fallback"):
        parts.append("分流 降级兜底")
    # 换了作答模型要看得见：提示词是按 claude-opus-5 写的，别让人误以为一直是它在答。
    if result.get("provider"):
        parts.append("模型 " + result["provider"])
    print(paint(" · ".join(parts), DIM))


def print_emergency_banner():
    width = term_width()
    print()
    print(paint("━" * width, RED, BOLD))
    print(paint("  需要就医 · 请优先处理这件事", RED, BOLD))
    print(paint("━" * width, RED, BOLD))
    print()


def print_sources(sources):
    if not sources:
        return
    print()
    print(paint("来源", BOLD))
    body = term_width() - 6
    for item in sources:
        head = " · ".join([
            grade_label(item),
            source_label(item.get("source")),
            clip(item.get("heading") or "未标注小节", max(20, body - 34)),
        ])
        print(f"  {paint('[%d]' % item['n'], CYAN)} {head}")
        urls = [clean_url(u) for u in (item.get("citations") or [])]
        urls = [u for u in urls if u.startswith("http")]
        if urls:
            for url in urls[:3]:
                print(paint(f"      {url}", DIM))
        elif item.get("is_live"):
            print(paint("      实时检索结果，未附来源链接，请自行核实", DIM))
        else:
            print(paint(f"      research/{item.get('source')}.md", DIM))


def print_debug(result):
    routing = result.get("routing") or {}
    print()
    print(paint("调试信息", BOLD))
    print(paint(
        "  分流    风险={risk}  类别={cat}  人群={pop}  法规={reg}  需要时效={cur}  分类器={mode}".format(
            risk=routing.get("risk"),
            cat=routing.get("category") or "未标注",
            pop="、".join(routing.get("populations") or []) or "通用",
            reg="是" if routing.get("regulatory") else "否",
            cur="是" if routing.get("needs_current_info") else "否",
            mode="降级兜底" if routing.get("_fallback") else "模型",
        ), DIM))
    if routing.get("red_flags"):
        print(paint("  报警症状  " + "、".join(routing["red_flags"]), YELLOW))
    for i, sub in enumerate(routing.get("sub_queries") or [], start=1):
        head = "  子查询  " if i == 1 else "          "
        print(paint(f"{head}{i} {clip(sub, term_width() - 14)}", DIM))

    passages = result.get("passages") or []
    print(paint(f"  候选资料  {len(passages)} 条（编号即回答里的引用号）", DIM))
    for i, chunk in enumerate(passages, start=1):
        meta = [
            f"综合 {chunk.get('_score', 0):.2f}",
            f"BM25 {chunk.get('_bm25', 0):.2f}",
            f"向量 {chunk.get('_dense', 0):.3f}",
        ]
        if chunk.get("_rerank_score") is not None:
            meta.append(f"重排 {chunk['_rerank_score']}")
        meta.append(chunk.get("evidence_grade") or "未分级")
        meta.append(source_label(chunk.get("source")))
        print(paint(f"    [{i}] " + " | ".join(meta), DIM))
        print(paint("        " + clip(chunk.get("text", ""), term_width() - 10), DIM))

    if result.get("problems"):
        print(paint("  未通过的校验  " + "；".join(result["problems"]), YELLOW))


def print_turn(result, debug):
    print()
    if result.get("escalated"):
        print_emergency_banner()
    print_status(result)
    print()
    print(result.get("answer", "").strip())
    if result.get("escalated"):
        print()
        print(paint("这段话来自固定的安全话术，不经过模型生成，也不会被任何提问绕过。", DIM))

    error = result.get("error")
    if error:
        print()
        print(paint("说明：" + ERROR_NOTE.get(error, error), YELLOW))

    print_sources(result.get("sources") or [])
    if debug:
        print_debug(result)
    print()


# --- 启动检查 ---------------------------------------------------------------

def model_provider():
    """返回 (provider_id, 展示名)。取不到就降级，不要因为一个提示行崩掉整个入口。"""
    try:
        from agent.llm import active_provider, provider_label
        return active_provider(), provider_label()
    except Exception:
        return None, "未知"


def check_credentials():
    """把缺 key 的后果讲清楚，而不是等到中途报错。"""
    provider, label = model_provider()

    if provider is None:
        print(paint("提醒：没有配置任何模型 API Key，这一轮不会生成回答。", YELLOW, BOLD))
        print(paint("      安全层和检索仍然工作：报警症状照常触发就医提示，也能看到检索到哪些资料。", DIM))
        print(paint("      配置方式：export ANTHROPIC_API_KEY=sk-ant-...", DIM))
        print(paint("      或写进项目根目录的 .env，键名 ANTHROPIC_API_KEY。", DIM))
        print(paint("      也接受 OPENAI_API_KEY 或 PERPLEXITY_API_KEY 作为备选。", DIM))
        print()
    elif provider != "anthropic":
        print(paint(f"提醒：没有找到 ANTHROPIC_API_KEY，本次由 {label} 作答。", YELLOW, BOLD))
        print(paint("      提示词、证据分级和引用规则都是按 claude-opus-5 写的，", DIM))
        print(paint("      换模型后措辞、引用密度和弃答判断都可能有出入，正式上线前请核对。", DIM))
        print(paint("      配置方式：export ANTHROPIC_API_KEY=sk-ant-...，或写进项目根目录的 .env。", DIM))
        print()
    else:
        print(paint(f"作答模型：{label}", DIM))

    try:
        from agent.index import openai_key
        openai_key()
    except BaseException:
        print(paint("提醒：没有找到 OPENAI_API_KEY，查询向量化无法工作，提问会直接失败。", YELLOW, BOLD))
        print(paint("      配置方式：export OPENAI_API_KEY=sk-...", DIM))
        print()


def load_agent():
    with Progress("正在加载知识库索引……"):
        from agent.core import Agent
        return Agent()


# --- 主循环 -----------------------------------------------------------------

QUIT = {"/退出", "/quit", "/exit", "/q"}
CLEAR = {"/清空", "/clear", "/reset"}
DEBUG = {"/调试", "/debug"}
HELP = {"/帮助", "/help", "/?", "/？"}


def main():
    print_banner()
    check_credentials()

    try:
        agent = load_agent()
    except BaseException as exc:
        print(paint("知识库索引加载失败，没法启动。", RED, BOLD))
        print(paint(f"原因：{exc}", DIM))
        print(paint("如果提示索引缺失，先运行：.venv/bin/python -m agent.index build", DIM))
        return 1

    print(paint("准备好了，直接输入你的问题。", DIM))

    history, debug = [], False

    while True:
        try:
            question = input(paint("\n你：", BOLD)).strip()
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print("\n再见。")
            return 0

        if not question:
            continue

        command = question.lower()
        if command in QUIT:
            print("再见。")
            return 0
        if command in HELP:
            print_help()
            continue
        if command in CLEAR:
            history = []
            print(paint("已清空上下文，下一句会当作新话题。", DIM))
            continue
        if command in DEBUG:
            debug = not debug
            print(paint("调试模式：" + ("开" if debug else "关"), DIM))
            continue
        if question.startswith("/"):
            print(paint("没有这个命令。输入 /帮助 看可用命令。", DIM))
            continue

        try:
            with Progress("正在检索并作答……"):
                result = agent.answer(question, history=history)
        except KeyboardInterrupt:
            print(paint("已中断这一轮。", DIM))
            continue
        except BaseException as exc:
            print()
            print(paint("这一轮出错了，没有生成回答。", RED, BOLD))
            print(paint(f"原因：{type(exc).__name__}: {exc}", DIM))
            print(paint("常见原因：网络不通、API key 无效、额度用尽。修好后再问一次即可。", DIM))
            continue

        print_turn(result, debug)
        rule()

        # 出错的一轮不写进上下文：一句道歉或一段占位文本当不了后文的依据。
        if not result.get("error"):
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": strip_citations(result.get("answer", ""))})
            history[:] = history[-MAX_HISTORY_TURNS * 2:]


if __name__ == "__main__":
    sys.exit(main() or 0)
