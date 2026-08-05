"""明途 · 肠道菌群健康顾问 —— 网页服务。

只用标准库 http.server。这个服务要做的事只有两件：把一个单文件页面发出去，
把一次问答转给 agent.core。为这两件事引入 flask/fastapi，等于给部署环境多加
一层需要维护的依赖，而它换不来任何这里用得上的能力。

路由
    GET  /            index.html
    GET  /health      运行自检：作答模型、索引规模、当前并发
    POST /api/chat    {message, history} -> JSON

几个不显眼但重要的决定：

  * Agent 在启动时构建一次。构建过程要加载全部索引块、向量矩阵和 BM25，
    放进请求里做等于每个用户都付一次这个代价。
  * 并发用信号量限流，不是无限开线程。后端串着嵌入接口和作答模型，
    云端的有速率限制，本地的抢同一块显存，放任并发只会把超时换成另一种超时。
  * 浏览器传来的 history 在服务端重新规整（角色、长度、严格交替）。
    agent.core 会在末尾追加本轮提问，所以历史必须以 assistant 结尾，
    否则拼出来的对话结构是坏的。
  * 任何异常都不把堆栈发给浏览器：页面只拿到一句中文说明，堆栈进服务端日志。

启动
    .venv/bin/python web/server.py [--host 127.0.0.1] [--port 8000]
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

WEB_DIR = Path(__file__).resolve().parent
REPO = WEB_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.core import Agent                       # noqa: E402  路径要先就位
from agent.llm import available, provider_label    # noqa: E402

INDEX_HTML = WEB_DIR / "index.html"

MAX_BODY_BYTES = 64 * 1024      # 请求体上限，超过直接拒绝，不读进内存
MAX_MESSAGE_CHARS = 2000        # 单条提问上限
MAX_HISTORY_MESSAGES = 12       # 6 轮对话
MAX_HISTORY_CHARS = 4000        # 单条历史消息上限
MAX_CONCURRENT = 3              # 同时进行的问答数
QUEUE_TIMEOUT = 150             # 排队等待上限（秒）

# research/*.md 的文件名不适合直接给用户看，这里给出对应的中文标题。
# 语料还在扩，没登记的 source 会原样显示文件名——难看但不会出错，
# 加了新的研究文件就在这里补一行。
SOURCE_TITLES = {
    "01-strains": "菌株证据",
    "02-diet": "膳食纤维与益生元",
    "03-china-reg": "中国监管与市场",
    "04-limits": "临床边界与常见误区",
    "05-systemic": "菌群与全身健康",
    "06-symptoms": "症状的日常处理",
    "07-drugs": "药物与菌群",
    "08-china-diet": "中国人的饮食与作息",
    "09-product": "看懂与挑选产品",
    "live:perplexity": "实时检索",
}

GRADE_LABELS = {
    "A": "A 级 · 人体RCT荟萃分析",
    "B": "B 级 · 单个人体RCT",
    "C": "C 级 · 观察性研究",
    "D": "D 级 · 动物或体外",
    "E": "E 级 · 专家意见",
}
GRADE_UNKNOWN = "未标注等级"

RISK_LABELS = {
    "emergency": "急症转诊",
    "caution": "需谨慎",
    "normal": "一般咨询",
}

# 面向用户的失败说明。core 返回的是 error 码，把它翻译成人话是这一层的事。
# 遇到没见过的码走 DEFAULT_NOTICE：管线换了错误码，页面也不会露出英文码。
NOTICES = {
    "no_model_key": (
        "服务器没有配置任何模型 API Key，系统处于降级模式："
        "报警症状筛查与资料检索正常工作，但回答无法生成。"
    ),
    "generation_failed": "这一轮模型调用没有成功返回，请再问一次。",
    "validation_failed": (
        "这一轮生成的内容没有通过引用校验或合规校验，系统按规则拒绝输出，"
        "不会把没有来源支撑的说法发给你。"
    ),
}
DEFAULT_NOTICE = "这一轮没有产出可用的回答，请再问一次。"


def live_available():
    """实时检索是否可用。问的是 live.py 自己用的那个密钥函数，不另起一套判断。"""
    try:
        from agent.live import perplexity_key
        return bool(perplexity_key())
    except BaseException:
        return False


# --- 载荷整形 -----------------------------------------------------------------

def clean_message(raw):
    """返回 (message, 错误说明)。"""
    if not isinstance(raw, str):
        return None, "请求缺少 message 字段。"
    message = raw.strip()
    if not message:
        return None, "请输入内容。"
    if len(message) > MAX_MESSAGE_CHARS:
        return None, f"消息太长，请控制在 {MAX_MESSAGE_CHARS} 字以内。"
    return message, None


def clean_history(raw):
    """把浏览器传来的历史整理成作答模型能接受的结构。

    规则：只留 user/assistant，角色严格交替，以 user 开头、assistant 结尾。
    最后一条必须是 assistant，因为 agent.core 会在其后追加本轮提问。
    浏览器传什么都不算数：role 是自由字符串，混进 system 就等于把注入通道
    交给了前端，所以这里只认白名单。
    """
    if not isinstance(raw, list):
        return []

    cleaned = []
    for item in raw[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[:MAX_HISTORY_CHARS]
        if not content:
            continue
        if not cleaned and role != "user":
            continue
        if cleaned and cleaned[-1]["role"] == role:
            merged = (cleaned[-1]["content"] + "\n\n" + content)[:MAX_HISTORY_CHARS]
            cleaned[-1]["content"] = merged
            continue
        cleaned.append({"role": role, "content": content})

    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()
    return cleaned


def link_label(url):
    host = urlsplit(url).netloc
    return host[4:] if host.startswith("www.") else (host or url)


def source_cards(sources):
    """把 core 返回的 sources 补成页面直接可渲染的卡片。"""
    cards = []
    for s in sources:
        source = s.get("source") or ""
        grade = s.get("grade")
        links = []
        for url in (s.get("citations") or [])[:4]:
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                links.append({"url": url, "label": link_label(url)})
        cards.append({
            "n": s.get("n"),
            "doc": SOURCE_TITLES.get(source, source or "未标注来源"),
            "heading": (s.get("heading") or "").strip(),
            "grade": grade,
            "grade_label": GRADE_LABELS.get(grade, GRADE_UNKNOWN),
            "is_live": bool(s.get("is_live")),
            "links": links,
        })
    return cards


def build_payload(result, elapsed):
    routing = result.get("routing") or {}
    passages = result.get("passages") or []
    error = result.get("error")

    return {
        "ok": not error,
        "error": error,
        "notice": NOTICES.get(error, DEFAULT_NOTICE) if error else None,
        "answer": result.get("answer") or "",
        "escalated": bool(result.get("escalated")),
        "sources": source_cards(result.get("sources") or []),
        "stats": {
            "passages": len(passages),
            "live_passages": sum(1 for p in passages if p.get("is_live")),
            "cited": len(result.get("sources") or []),
            "used_live": bool(result.get("used_live")),
            "repairs": int(result.get("repairs") or 0),
            "risk": routing.get("risk", "normal"),
            "risk_label": RISK_LABELS.get(routing.get("risk"), "一般咨询"),
            "category": routing.get("category") or "",
            "populations": routing.get("populations") or [],
            "sub_queries": routing.get("sub_queries") or [],
            "red_flags": routing.get("red_flags") or [],
            "classifier_fallback": bool(routing.get("_fallback")),
            "gate_problems": result.get("problems") or [],
            "provider": result.get("provider") or provider_label(),
            "elapsed": round(elapsed, 1),
        },
    }


# --- 服务状态 -----------------------------------------------------------------

class Service:
    """进程内共享的一份 Agent，加上并发闸门。"""

    def __init__(self):
        self.agent = None
        self.chunks = 0
        self.warm = False
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)
        self._busy = 0
        self._lock = threading.Lock()

    def start(self, warmup=True):
        t0 = time.time()
        self.agent = Agent()
        self.chunks = len(self.agent.retriever.chunks)
        if warmup:
            # 走一次真实检索来预热：把嵌入客户端的惰性初始化从并发路径里拿掉，
            # 顺带在第一个用户提问之前就验证嵌入链路是通的。
            try:
                self.agent.retriever.search("肠道菌群", None, top_n=1)
                self.warm = True
            except Exception as exc:
                print(f"  预热失败（嵌入链路不可用）：{exc}", file=sys.stderr)
        return time.time() - t0

    def busy(self):
        with self._lock:
            return self._busy

    def answer(self, message, history):
        if not self.slots.acquire(timeout=QUEUE_TIMEOUT):
            raise TimeoutError
        with self._lock:
            self._busy += 1
        try:
            return self.agent.answer(message, history=history)
        finally:
            with self._lock:
                self._busy -= 1
            self.slots.release()

    def health(self):
        return {
            "ok": True,
            "chunks": self.chunks,
            "warm": self.warm,
            "provider": provider_label(),
            "degraded": not available(),
            "live": live_available(),
            "busy": self.busy(),
            "capacity": MAX_CONCURRENT,
        }


SERVICE = Service()


# --- HTTP ---------------------------------------------------------------------

class ChatHandler(BaseHTTPRequestHandler):
    server_version = "mingtu"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # -- 响应工具 ----------------------------------------------------------
    def _headers(self, status, content_type, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _fail(self, status, message, close=False):
        body = json.dumps({"ok": False, "message": message},
                          ensure_ascii=False).encode("utf-8")
        extra = {"Connection": "close"} if close else None
        self._headers(status, "application/json; charset=utf-8", len(body), extra)
        self.wfile.write(body)
        if close:
            self.close_connection = True

    # -- 路由 --------------------------------------------------------------
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_page()
        if path == "/health":
            return self._json(SERVICE.health())
        if path == "/favicon.ico":
            self._headers(HTTPStatus.NO_CONTENT, "image/x-icon", 0)
            return
        self._fail(HTTPStatus.NOT_FOUND, "页面不存在。")

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/api/chat":
            return self._chat()
        if path == "/api/chat/stream":
            return self._chat_stream()
        if path == "/api/speak":
            return self._speak()
        self._fail(HTTPStatus.NOT_FOUND, "接口不存在。")

    def _chat_stream(self):
        """SSE 流式作答。每句先过合规闸门再发出，见 agent/stream.py。"""
        data, problem = self._read_json()
        if problem:
            return self._fail(HTTPStatus.BAD_REQUEST, problem)
        message = (data.get("message") or "").strip()
        if not message:
            return self._fail(HTTPStatus.BAD_REQUEST, "请先输入内容。")
        history = data.get("history") or []

        # SSE 的长度事先不可知，所以必须显式关闭连接来终止流。
        # 少了这一步，事件其实都发出去了，但连接一直挂着，
        # 浏览器的 reader 永远等不到结束——表现成「一直转圈」。
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(kind, payload):
            blob = json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False)
            self.wfile.write(f"data: {blob}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            for kind, payload in SERVICE.agent.answer_stream(message, history):
                emit(kind, payload)
        except Exception:
            emit("done", {"answer": "这一轮没能生成回答，请再问一次。", "sources": []})
        emit("end", {})
        try:
            self.wfile.flush()
        except Exception:
            pass

    def _speak(self):
        """朗读。本地模式下不提供——朗读必然要把内容发到外部服务。"""
        from agent import voice

        if not voice.available():
            return self._fail(HTTPStatus.SERVICE_UNAVAILABLE, "当前部署未开启朗读功能。")

        data, problem = self._read_json()
        if problem:
            return self._fail(HTTPStatus.BAD_REQUEST, problem)

        text = (data.get("text") or "").strip()
        if not text:
            return self._fail(HTTPStatus.BAD_REQUEST, "没有可朗读的内容。")
        if len(text) > 6000:
            text = text[:6000]

        audio = voice.synthesize(text)
        if not audio:
            return self._fail(HTTPStatus.BAD_GATEWAY, "语音合成失败，请稍后再试。")

        self._headers(HTTPStatus.OK, "audio/mpeg", len(audio),
                      extra={"Cache-Control": "public, max-age=86400"})
        self.wfile.write(audio)

    # -- 处理 --------------------------------------------------------------
    def _serve_page(self):
        try:
            body = INDEX_HTML.read_bytes()
        except OSError:
            return self._fail(HTTPStatus.INTERNAL_SERVER_ERROR,
                              "页面文件缺失：web/index.html。")
        self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None, "请求头不合法。"
        if length <= 0:
            return None, "请求内容为空。"
        if length > MAX_BODY_BYTES:
            return None, "请求内容过大。"
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8")), None
        except Exception:
            return None, "请求格式不是合法的 JSON。"

    def _chat(self):
        data, problem = self._read_json()
        if problem:
            return self._fail(HTTPStatus.BAD_REQUEST, problem, close=True)
        if not isinstance(data, dict):
            return self._fail(HTTPStatus.BAD_REQUEST, "请求格式不正确。")

        message, problem = clean_message(data.get("message"))
        if problem:
            return self._fail(HTTPStatus.BAD_REQUEST, problem)
        history = clean_history(data.get("history"))

        started = time.time()
        try:
            result = SERVICE.answer(message, history)
        except TimeoutError:
            return self._fail(HTTPStatus.SERVICE_UNAVAILABLE,
                              "当前咨询量较大，请稍后重试。")
        except Exception:
            # 检索或模型链路里的任何意外都在这里落地：用户看到一句中文，
            # 完整堆栈留在服务端日志里。
            traceback.print_exc()
            return self._fail(HTTPStatus.INTERNAL_SERVER_ERROR,
                              "服务处理这条消息时出错了，请再试一次。")

        elapsed = time.time() - started
        print(f"  问答完成 {elapsed:.1f}s  资料 {len(result.get('passages') or [])} 条"
              f"  实时检索 {'是' if result.get('used_live') else '否'}"
              f"  风险 {(result.get('routing') or {}).get('risk', '-')}",
              file=sys.stderr)
        self._json(build_payload(result, elapsed))

    # -- 日志 --------------------------------------------------------------
    def log_message(self, fmt, *args):
        sys.stderr.write("%s  %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            # 用户关掉页面或中途刷新。一轮问答要跑好几分钟，这会经常发生，
            # 它是正常情况，不该在日志里留一段堆栈。
            print("  连接被客户端提前断开", file=sys.stderr)
            return
        super().handle_error(request, client_address)


# --- 入口 ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="明途 · 肠道菌群健康顾问 网页服务")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                        help="监听地址，局域网演示用 0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动预热")
    args = parser.parse_args()

    # 日志常被重定向到文件，块缓冲会让启动信息迟迟不出现。
    sys.stdout.reconfigure(line_buffering=True)

    print("明途 · 肠道菌群健康顾问")
    print("  正在加载索引与检索模型…")
    try:
        seconds = SERVICE.start(warmup=not args.no_warmup)
    except BaseException as exc:
        print(f"  启动失败：{exc}", file=sys.stderr)
        return 1

    health = SERVICE.health()
    print(f"  索引 {health['chunks']} 段资料，用时 {seconds:.1f} 秒")
    print(f"  作答模型 {health['provider']}"
          f" · 实时检索 {'可用' if health['live'] else '不可用'}")
    if health["degraded"]:
        print("  注意：没有配置任何模型 API Key，服务以降级模式运行——"
              "报警症状筛查与资料检索可用，回答无法生成。")

    server = ChatServer((args.host, args.port), ChatHandler)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"  已启动：http://{shown}:{args.port}  （Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
