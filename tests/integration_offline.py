"""离线端到端集成测试。不 mock 任何模型调用。

单元测试证明不了这套东西真的能离线跑：它们可以把模型调用 mock 掉，而真正会坏的
恰恰是 ollama 连接、索引维度、引用编号、以及「本地模式下会不会偷偷走云端」。
这个脚本做的是真调用 + 真抓包。

运行： MINGTU_LOCAL_ONLY=1 .venv/bin/python tests/integration_offline.py
"""

import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = "通过", "失败"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return ok


# --- 1. 失败关闭：本地模式下绝不允许回落到云端 --------------------------------
def test_fail_closed():
    print("\n1. 失败关闭（本地模式禁止云端回落）")
    from agent import llm

    os.environ["MINGTU_LOCAL_ONLY"] = "1"
    check("local_only() 默认开启", llm.local_only())

    # 云端 key 全部存在的情况下，本地模式仍然只能选本地
    have_cloud = any(llm._read_key(k) for k in
                     ("GEMINI_API_KEY", "OPENAI_API_KEY", "PERPLEXITY_API_KEY"))
    provider = llm.active_provider()
    check("云端 key 存在时仍不回落", provider in ("ollama", None),
          f"云端key={'有' if have_cloud else '无'} provider={provider}")

    # 把 ollama 打挂，本地模式必须返回 None 而不是任何云端 provider
    real_host = llm.OLLAMA_HOST
    llm.OLLAMA_HOST = "http://127.0.0.1:9"
    llm._clients.clear()
    check("ollama 不可用时返回 None 而非云端", llm.active_provider() is None,
          f"实际={llm.active_provider()}")
    llm.OLLAMA_HOST = real_host

    from agent import embed
    check("embed 与 llm 的 local_only 一致", embed.local_only() == llm.local_only())


# --- 2. 向量后端与索引维度一致 ------------------------------------------------
def test_index_consistency():
    print("\n2. 索引与向量后端一致性")
    from agent.embed import backend_id
    from agent.index import load_index, vectors_path

    bid = backend_id()
    check("向量后端为本地", bid.startswith("ollama:"), bid)
    chunks, vectors = load_index()
    check("索引已加载", len(chunks) > 0 and vectors.shape[0] == len(chunks),
          f"{len(chunks)} 块 / {vectors.shape}")

    q = __import__("agent.embed", fromlist=["embed"]).embed(["测试查询"])
    check("查询向量与索引维度一致", q.shape[1] == vectors.shape[1],
          f"查询{q.shape[1]}维 vs 索引{vectors.shape[1]}维")


# --- 3. 急症拦截：不经过模型 --------------------------------------------------
def test_emergency():
    print("\n3. 急症拦截（口语化表述，且不经过模型）")
    from agent import safety
    cases = ["我最近大便有血，还瘦了好几斤", "两个月掉了8斤，也没节食",
             "我爸得过肠癌，我要注意什么", "夜里总是腹泻", "拉出来是黑的"]
    for text in cases:
        flags, _ = safety.screen_input(text)
        check(f"命中：{text[:16]}", bool(flags), "、".join(flags))
    for benign in ["益生菌怎么选", "酸奶和益生菌一样吗"]:
        flags, _ = safety.screen_input(benign)
        check(f"不误报：{benign}", not flags)


# --- 4. 真实离线问答 + 引用校验 ----------------------------------------------
def test_end_to_end():
    print("\n4. 真实离线问答（qwen3 本地作答）")
    from agent.core import Agent
    from agent.llm import provider_label

    agent = Agent()
    print(f"  作答模型：{provider_label()}")

    r = agent.answer("我最近大便有血，还瘦了好几斤")
    check("急症走固定话术", r["escalated"] and "就医" in r["answer"])
    check("急症不消耗检索", len(r["passages"]) == 0)

    q = "抗生素吃完之后怎么恢复肠道菌群"
    t0 = time.time()
    r = agent.answer(q)
    elapsed = time.time() - t0
    check("正常问题有检索", len(r["passages"]) > 0, f"{len(r['passages'])} 条")
    check("生成了回答", bool(r["answer"]) and not r.get("error"),
          f"{len(r['answer'])}字 / {elapsed:.0f}秒 / 重写{r['repairs']}次")

    import re
    cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", r["answer"])}
    valid = set(range(1, len(r["passages"]) + 1))
    check("引用编号全部有效（无编造）", cited <= valid,
          f"引用={sorted(cited)} 有效范围=1-{len(r['passages'])}")
    check("有实质引用或明确弃答", bool(cited) or "资料" in r["answer"])

    violations = __import__("agent.safety", fromlist=["screen_output"]).screen_output(r["answer"])
    check("回答无违规宣称", not violations, str(violations[:1]))
    print(f"\n  ── 回答节选 ──\n  {r['answer'][:260]}")


# --- 5. 出网监控：本地模式下不得有对外连接 -----------------------------------
def test_no_egress():
    print("\n5. 出网监控（本地模式不得有对外连接）")
    pid = os.getpid()
    seen = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["lsof", "-nP", "-i", "-a", "-p", str(pid)],
                    capture_output=True, text=True, timeout=8).stdout
                for line in out.splitlines():
                    if "ESTABLISHED" not in line:
                        continue
                    if "127.0.0.1" in line or "[::1]" in line or "localhost" in line:
                        continue
                    seen.append(line.split()[-2] if len(line.split()) > 1 else line)
            except Exception:
                pass
            time.sleep(0.4)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    from agent.core import Agent
    Agent().answer("益生菌对便秘有用吗")
    stop.set()
    t.join(timeout=3)
    check("问答过程中无对外连接", not seen, f"检测到 {len(set(seen))} 个外部连接: {list(set(seen))[:3]}")


if __name__ == "__main__":
    os.environ["MINGTU_LOCAL_ONLY"] = "1"
    print("离线端到端集成测试（MINGTU_LOCAL_ONLY=1）")
    print("=" * 68)
    for fn in (test_fail_closed, test_index_consistency, test_emergency,
               test_end_to_end, test_no_egress):
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} 执行异常", False, repr(exc)[:160])

    print("\n" + "=" * 68)
    failed = [n for n, ok, _ in results if not ok]
    print(f"共 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for n in failed:
        print(f"  失败：{n}")
    sys.exit(1 if failed else 0)
