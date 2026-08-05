# 部署与运维

面向接手这套代码的工程团队。架构和设计理由看 [README.md](README.md)，这份只讲怎么跑起来、怎么维护、出问题怎么查。

> **快照说明**：对应 2026-08-04 23:31 的工作区状态。作答 provider 层（`agent/llm.py`）当时仍在改造中，第 3 节和第 8 节记录的是那一刻复核过的状态。密钥的权威说明在 `.env.example` 里（那份文件跟着代码走，比这份更新更频繁）——本文档只讲它没讲的部分：模块间的不一致、启动顺序、排障。

---

## 1. 环境要求

| 项 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.13 | 虚拟环境在 `.venv/`，uv 创建（解释器来自 `~/.local/share/uv/python/cpython-3.12-macos-aarch64-none`） |
| uv | 0.11.30 | `/opt/homebrew/bin/uv` |
| curl | 系统自带 | Perplexity 和 Ollama 都走 `subprocess` 调 curl，不用 Python HTTP 库。**系统必须有 curl。** |
| Ollama | 可选但推荐 | 本地向量后端（`bge-m3`）和本地作答模型（默认 `qwen3:14b`）。全本地栈的前提 |
| 磁盘 | 约 10 MB + 模型 | `research/` 约 700 KB，索引产物几 MB；Ollama 模型另算（`bge-m3` 约 1.2 GB，`qwen3:14b` 约 9 GB） |

**不要用系统 `python3`**（macOS 自带 3.9.6，太老）。所有命令走 `.venv/bin/python`。

**必须在仓库根目录运行。** `agent/` 和 `tools/` 没有 `__init__.py`，靠命名空间包工作。`cli.py` 和 `web/server.py` 会自己把仓库根插进 `sys.path`，直接 `python -m agent.xxx` 则依赖 cwd。

---

## 2. 依赖安装

```bash
cd /Users/leon/Desktop/dev/mingtu
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`requirements.txt` 精确锁定了 5 个直接依赖和全部传递依赖。直接依赖是 `anthropic`、`openai`、`numpy`、`jieba`、`rank-bm25`；`pyproject.toml` 里有同一份清单（改一处必须同步另一处）。

**`pytest` 不在 `requirements.txt` 里**，但 `tests/` 需要它。跑测试前：

```bash
uv pip install --python .venv/bin/python pytest
```

验证：

```bash
.venv/bin/python -c "import anthropic, openai, numpy, jieba, rank_bm25; print('ok')"
```

`pyproject.toml` 刻意不声明 `[build-system]`——这是应用不是库，声明一个永远不会被调用的构建后端只会让人以为可以 `pip install` 这个目录。`[tool.uv] package = false` 是为了让 `uv sync` 不去找构建后端。

---

## 3. 配置与密钥

**权威说明在 `.env.example`**，每个变量的用途、缺失时的行为、必需程度都写在那里。用法：

```bash
cp .env.example .env
# 填值
./run.sh
```

**必须用 `./run.sh` 启动，不要直接 `python web/server.py`。** 各模块的 key 查找规则不一致，只有 `agent/llm.py` 读项目根目录的 `.env`；`run.sh` 会先把 `.env` 的值 export 到进程环境，让所有模块看到同一份配置。绕过它容易出现「作答模型取到了 key、检索没取到」这种一半能跑的状态。

### 3.1 两条路线

**路线 A：全本地（推荐，也是当前代码的默认取向）**

```bash
ollama pull bge-m3        # 向量后端，1024 维
ollama pull qwen3:14b     # 作答模型
```

不需要任何 API key。用户的健康提问不出本机，规避《个人信息保护法》下的数据出境问题。

**路线 B：云端**

在 `.env` 里配 `ANTHROPIC_API_KEY`（作答，质量最好，提示词就是照着 `claude-opus-5` 写的）和 `OPENAI_API_KEY`（向量，`text-embedding-3-large`，3072 维）。

### 3.2 必须知道的三个不一致

这三条不在 `.env.example` 里，但会直接决定服务能不能跑起来。

**（1）`MINGTU_LOCAL_ONLY` 在两个模块里默认极性相反。**

| 位置 | 未设置时 | 行为 |
|---|---|---|
| `agent/llm.py:local_only()` | **True** | 禁止云端作答，本地模型缺失就不作答 |
| `agent/embed.py:local_only()` | **False** | 允许回落到 OpenAI 做向量化 |

`.env.example` 里写的「默认关闭」只对向量层成立。**当前默认状态下，作答层拒绝出网、向量层却会出网**，这和「全本地」的合规理由自相矛盾。上线前必须把两处统一。

排查用：

```bash
.venv/bin/python -c "
from agent import llm, embed
print('llm.local_only  :', llm.local_only())
print('embed.local_only:', embed.local_only())
"
```

**（2）作答用的本地模型和向量用的本地模型是两个，要分别拉。** `llm.ollama_up()` 查的是 chat 模型（`MINGTU_LLM_MODEL`，默认 `qwen3:14b`），`embed.ollama_up()` 查的是 embedding 模型（`MINGTU_EMBED_MODEL`，默认 `bge-m3`）。只 pull 了 `bge-m3` 的机器上，向量是本地的、作答是不可用的（因为 `llm.local_only()` 默认 True，不回退云端）。

**（3）`tools/research.py` 的 `key()` 不读环境变量也不读项目 `.env`**，只读 `~/Desktop/dev/{curio,Curio-Automation,curio-international}/.env`。所以 `PERPLEXITY_API_KEY` 填在项目 `.env` 里，实时检索这一路拿不到（作答回退那一路能拿到，走的是 `llm._read_key`）。`.env.example` 已记录。

### 3.3 一条龙状态检查

```bash
.venv/bin/python - <<'PY'
from agent import llm, embed
print("作答 provider :", llm.active_provider() or "无（不会生成回答）", "|", llm.provider_label())
print("  local_only  :", llm.local_only(), "| ollama chat 就绪:", llm.ollama_up())
print("向量 backend  :", embed.active_backend() or "无（检索起不来）", "|", embed.backend_id())
print("  local_only  :", embed.local_only(), "| ollama embed 就绪:", embed.ollama_up())
for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PERPLEXITY_API_KEY"):
    print(f"{name:<20}", "已配置" if llm._read_key(name) else "缺失")
PY
```

`./run.sh --help` 也会跑一遍启动前检查（解释器 → 依赖 → 密钥 → 索引 → 安全层 → 入口），输出中文诊断。

---

## 4. 索引

### 4.1 向量按后端分文件存

`agent/embed.py` 挑后端的顺序是：本地 Ollama `bge-m3`（1024 维）→ OpenAI `text-embedding-3-large`（3072 维）。两个是两个向量空间，跨空间算余弦相似度得到的是噪声，所以：

- 向量文件名带后端标识：`knowledge/vectors-ollama_bge-m3.npy`、`knowledge/vectors-openai_text-embedding-3-large.npy`
- `load_index()` 找的是「当前后端对应的那个文件」，找不到就**直接报错**，不做静默回落
- 换后端不会覆盖旧索引，两份可以共存，切回去不用重建

**换后端（或换 embedding 模型）必须重建索引。**

### 4.2 重建

```bash
cd /Users/leon/Desktop/dev/mingtu
.venv/bin/python -m agent.index build
```

或让 `run.sh` 代劳：

```bash
MINGTU_REBUILD_INDEX=1 ./run.sh
```

**什么时候需要重建：** `research/*.md` 有增删改；`chunker.py` 的切块参数或元数据抽取规则改了；`index.py` 的分词逻辑改了；换向量后端或 embedding 模型。

**只改 `retrieve.py` 的权重、`prompt.py` 的提示词、`safety.py` 的正则，不需要重建。**

**规模：** 当前 `research/` 下 9 篇文档 → 762 块。走本地 Ollama 只花时间不花钱（逐条请求，比 OpenAI 的批量慢）；走 OpenAI 是每批 64 条、约 46 万 token 的 embedding 调用。

**构建产物不进版本库**（见 `.gitignore` 里的详细理由）。新机器部署必须先跑一次 build。

### 4.3 重建后校验

`chunks.jsonl` 与向量文件必须逐行对齐——`retrieve.py` 用同一个下标同时索引两者，错位的表现是回答里的 `[3]` 指向另一段资料。对一个要求「逐条标注来源、编造编号视为严重缺陷」的系统，这是安全缺陷。

```bash
.venv/bin/python - <<'PY'
import json, numpy as np
from agent.index import CHUNKS_PATH, vectors_path, load_index
from agent.embed import backend_id

chunks = [json.loads(l) for l in CHUNKS_PATH.read_text().splitlines() if l.strip()]
vectors = np.load(vectors_path())
assert len(chunks) == vectors.shape[0], f"错位: {len(chunks)} 块 vs {vectors.shape[0]} 向量"
norms = np.linalg.norm(vectors, axis=1)
assert abs(norms.mean() - 1.0) < 0.01, f"向量未归一化: 均值 {norms.mean()}"
load_index()   # 后端不符会在这里 SystemExit
print(f"ok: {len(chunks)} 块 / {vectors.shape} / 后端 {backend_id()} / "
      f"带证据等级 {sum(1 for c in chunks if c['evidence_grade'])} / "
      f"法规 {sum(1 for c in chunks if c['is_regulatory'])}")
PY
```

当前预期输出：`ok: 762 块 / (762, 1024) / 后端 ollama:bge-m3 / 带证据等级 369 / 法规 102`

`run.sh` 在索引缺失或行数对不上时会拦住启动并给出命令。

---

## 5. 新增研究资料

`research/*.md` 是索引的唯一数据源。

### 5.1 用现成的研究流程

```bash
.venv/bin/python tools/research.py --probe          # 探活，走便宜的 sonar 模型
.venv/bin/python tools/research.py --ask 01-strains # 跑一个预置问题
```

**新增主题**：在 `tools/research.py` 的 `QUESTIONS` 字典里加一条，键名做文件名。`RIGOUR` 段会自动追加到每个问题后面，它规定了证据分级、菌株号、剂量单位、发表年份、参考文献 URL 这些要求——**不要绕过它**，否则新文档的元数据抽取率会明显低于现有文档。问题必须写成一行（`one_line()` 会把空白压成单空格）。

失败时不重试也不换模型：提示词落到 `research/<name>.PARKED.txt`，进程返回 1，需要人工去 Perplexity 网页端粘贴执行。

### 5.2 手动添加文档

要让元数据抽取生效，文档需满足：

| 元数据 | 抽取规则（`chunker.py`） | 写法建议 |
|---|---|---|
| `evidence_grade` | `证据等级：A` / `（A级）` / `A级证据` 三选一 | 每条结论后紧跟证据等级，用这三种写法之一 |
| `population` | 关键词：婴幼儿/儿童/孕产妇/老年/免疫抑制/成人 | 明确写出人群词，不要只写「患者」 |
| `year` | `19[89]\d` 或 `20[0-3]\d` 取最大值 | 年份写在结论所在段落内 |
| `authority` | 荟萃分析/指南/随机对照/队列/动物实验/专家意见，最先命中者胜 | 写明研究类型 |
| `is_regulatory` | 广告法/食品安全法/卫健委/市场监管总局/保健食品/国标/GB 等 | 法规内容写全名和文号 |
| `citations` | 段落内的 `http(s)://` URL | **把 URL 写在结论所在段落里，不要只堆在文末** |

切块参数：目标 450 字，重叠 90 字，小于 120 字的段落丢弃。按 1–4 级 Markdown 标题分节，标题会拼进块正文参与检索。

**已知坑：** Perplexity 输出自带 `[1][2]` 形式的引用标记，会原样进索引并与本系统的资料编号冲突（README 第 8.2 节）。手写文档时不要用方括号数字标注引用。

加完文档后重建索引并按 4.3 校验。

---

## 6. 本地运行

### 6.1 命令行

```bash
cd /Users/leon/Desktop/dev/mingtu
.venv/bin/python cli.py
```

启动时会先做密钥检查并把后果讲清楚（没有 provider / 用的不是 Anthropic / 缺 embedding key 各有不同提示），然后加载索引。

命令：

| 命令 | 作用 |
|---|---|
| `/退出`（`/quit`、`/exit`、`/q`、Ctrl-D） | 结束 |
| `/清空`（`/clear`、`/reset`） | 清空上下文 |
| `/调试`（`/debug`） | 开关调试模式：显示分流细节、子查询、全部候选资料及其 BM25/向量/重排分 |
| `/帮助`（`/help`） | 说明 |

每轮会显示一行状态：风险等级、资料条数、是否用了实时检索、重写次数、分类器是否降级兜底、**当前作答模型**。最后一项很重要——提示词是按 `claude-opus-5` 写的，换了模型措辞和引用密度都会变，界面上必须看得见。

上下文保留最近 6 轮，写进历史前会用 `strip_citations()` 去掉引用编号：编号只对生成它的那一轮有效，留在上文会诱导模型在下一轮复用不存在的编号。出错的一轮不写进上下文。

### 6.2 网页

```bash
./run.sh                                    # 推荐：先做启动前检查
# 或
.venv/bin/python web/server.py --host 127.0.0.1 --port 8000
```

只用标准库 `http.server`，没有引入 flask/fastapi。

| 路由 | 说明 |
|---|---|
| `GET /` | 单文件页面 `web/index.html` |
| `GET /health` | 自检：索引规模、当前 provider、是否降级、实时检索可用性、当前并发/容量 |
| `POST /api/chat` | `{message, history}` → JSON |

关键行为：

- **Agent 进程内单例**，启动时构建一次并做一次真实检索预热（把嵌入客户端的惰性初始化从并发路径里拿掉，顺带在第一个用户提问前验证嵌入链路通不通）
- **并发上限 3**（`MAX_CONCURRENT`），信号量限流，排队超过 150 秒返回 503。后端串着有速率限制的接口，放任并发只会把超时换成另一种超时
- **请求体上限 64 KB，单条提问 2000 字，历史 12 条**，服务端重新规整历史（只留 user/assistant、严格交替、以 assistant 结尾——因为 `agent.core` 会在末尾追加本轮提问）
- **异常不外泄堆栈**：页面只拿到一句中文说明，完整堆栈进服务端日志
- 响应带 CSP、`X-Content-Type-Options`、`Referrer-Policy`，默认只监听 `127.0.0.1`

**不要在没有反向代理和访问控制的情况下把 `--host 0.0.0.0` 暴露到公网。** 这是健康咨询服务。

### 6.3 直接调用

```bash
.venv/bin/python - <<'PY'
from agent.core import Agent
a = Agent()          # 复用这一个实例，不要每个请求都新建
r = a.answer("LGG 对儿童急性腹泻有用吗")
print(r["routing"]["risk"], r["provider"], len(r["passages"]), "条资料")
print(r["answer"])
for s in r["sources"]:
    print(s["n"], s["source"], s["heading"], s["grade"], s["is_live"])
PY
```

`Agent.answer()` 返回字段：

| 字段 | 含义 |
|---|---|
| `answer` | 面向用户的中文回答 |
| `sources` | 回答真正引用过的来源：编号、标题、文件、证据等级、是否实时、URL |
| `routing` | 分流结果：`risk`、`red_flags`、`sub_queries`、`populations`、`_fallback`（是否走了确定性兜底） |
| `passages` | 送进模型的资料块，带 `_score` / `_bm25` / `_dense` / `_rerank_score` |
| `escalated` | 是否走了急症短路 |
| `used_live` | 本轮是否用了实时检索 |
| `repairs` | 触发了几轮重写 |
| `provider` | 本轮实际作答的模型 |
| `error` | `no_model_key` / `generation_failed` / `validation_failed` |
| `problems` | 仅 `validation_failed` 时存在，列出未通过的校验项 |

---

## 7. 上线检查清单

按顺序执行。任何一项失败都不要上线。

### 环境

- [ ] `.venv/bin/python --version` 返回 3.12.x
- [ ] `uv pip install -r requirements.txt` 后五个直接依赖导入无报错
- [ ] 系统有 `curl`
- [ ] `./run.sh --help` 的六项启动前检查全绿

### 配置

- [ ] `.env` 已从 `.env.example` 复制并填值
- [ ] **`llm.local_only()` 与 `embed.local_only()` 语义已统一**（第 3.2 节，当前默认相反）
- [ ] 走本地路线：`ollama pull bge-m3` 和 `ollama pull qwen3:14b` 都完成，两个 `ollama_up()` 都为 True
- [ ] 走云端路线：`ANTHROPIC_API_KEY` 和 `OPENAI_API_KEY` 都可读，且已显式 `MINGTU_LOCAL_ONLY=0`
- [ ] `llm.active_provider()` 不是 `None`
- [ ] `embed.active_backend()` 不是 `None`

### 索引

- [ ] 已跑过 `python -m agent.index build`
- [ ] 4.3 的校验脚本通过（行数对齐、已归一化、后端相符）
- [ ] `research/*.md` 与索引同步（改过文档就要重建）
- [ ] 索引块数与维度符合预期（当前 762 块 / 1024 维 / `ollama:bge-m3`）

### 安全层（不需要任何模型，必须逐条过）

- [ ] `screen_input("我最近大便有血")` → `['便血']`
- [ ] `screen_input("我最近大便有血，还瘦了五斤")` → `['便血', '体重下降']`，`mentions_self` 为 True
- [ ] 在没有任何可用 provider 的情况下，`answer("我最近大便有血，还瘦了五斤")` 仍返回 `escalated=True` 和完整就医话术
- [ ] `screen_output("益生菌可以治疗肠易激综合征。")` 命中违规
- [ ] `screen_output("益生菌不能治疗肠易激综合征。")` 不命中（否定豁免生效）
- [ ] 模型返回 `normal` 而正则判 emergency 时，最终 risk 仍是 emergency
- [ ] `pytest tests/ -q` 全绿（当前 228 passed / 3 skipped / 24 xfailed）

### 端到端

- [ ] 普通科普提问能出答案，`sources` 非空，每条实质结论带编号
- [ ] 法规提问的资料里出现 `is_regulatory` 段落
- [ ] 特殊人群提问（「宝宝能吃益生菌吗」）的资料里出现婴幼儿段落，且回答里有「先咨询医生」的提示
- [ ] 冷门提问走弃答路径，不硬凑答案
- [ ] **重写循环实测有效**：构造一个会触发合规校验的场景，确认 `repairs > 0` 且最终输出合规，而不是返回「抱歉，这一轮没能生成回答」（README 第 8.3 节记录了 Anthropic 路径上的已知缺陷，必须先修掉并验证）
- [ ] 引用编号全部落在 `1..len(passages)` 区间，随机抽 3 条人工核对编号指向的段落确实支撑该结论
- [ ] 回答里没有 AI 腔（「希望对您有帮助」「作为一个 AI」），没有表情符号，全中文
- [ ] 网页 `GET /health` 返回 `degraded: false`

### 容量与降级演练

- [ ] 单次提问的耗时和 token 用量实测过
- [ ] 网页并发上限 3 对预期流量够用（不够就改 `MAX_CONCURRENT`，同时确认下游速率限制撑得住）
- [ ] 临时停掉作答 provider，确认安全层照常工作、服务不崩、页面显示降级说明
- [ ] 临时清掉 `PERPLEXITY_API_KEY`，确认主链路不受影响
- [ ] 断网跑一次，确认异常被捕获而不是把堆栈打给用户

---

## 8. 常见故障排查

### `ModuleNotFoundError: No module named 'agent'`

不在仓库根目录。`cd /Users/leon/Desktop/dev/mingtu` 后重试，或改用 `cli.py` / `web/server.py`（它们会自己修 `sys.path`）。

### `当前向量后端 ollama:bge-m3 还没有建过索引。`

向量后端和索引对不上（换了后端、换了 embedding 模型，或新机器还没建过）。跑 `python -m agent.index build`。**不要试图绕过这个检查**——跨向量空间检索返回的是噪声，不是降级结果。

### 索引明明在，`run.sh` 说索引不可用

大概率是向量文件名的后端标识和当前后端不符（比如索引是 OpenAI 建的、现在 Ollama 起来了）。`ls knowledge/` 看有哪些 `vectors-*.npy`，再跟 `embed.backend_id()` 对一下。

### 回答是「【未配置任何模型 API Key，无法生成回答】」

`result["error"] == "no_model_key"`，`llm.active_provider()` 返回 `None`。检索和安全层正常。三种原因：

1. 本地 chat 模型没拉（`ollama pull qwen3:14b`）
2. `llm.local_only()` 为 True（默认）挡住了云端回退——要用云端就显式 `export MINGTU_LOCAL_ONLY=0`
3. 云端 key 都没配

按第 3.3 节的脚本逐项确认。

### 回答是「抱歉，这一轮没能生成回答，请再问一次。」

`generate()` 返回了 `None`，`result["error"] == "generation_failed"`。可能是：

1. **模型拒答**（Anthropic 路径的 `stop_reason == "refusal"`）。健康医疗类提问有一定概率触发安全分类器
2. **API 异常被吞掉**——认证失败、限流、超时、参数错误在外部完全无法区分
3. **命中了重写循环的已知缺陷**（README 第 8.3 节）：如果这句出现在「本来应该重写」的场景下，几乎可以确定是 Anthropic 路径上 system 消息位置的问题
4. 本地模型推理超时（`_ollama_chat` 的 curl 超时是 600 秒）

排障时把 `agent/llm.py` 里对应 provider 分支的 `except Exception:` 临时改成打印异常：

```python
except Exception as e:
    print(f"[{provider} generate failed] {type(e).__name__}: {e}")
    return None
```

`call_json` 同理。**这几处静默吞异常是当前最大的可观测性缺口，建议尽快改成有日志的形式。**

### 回答是「这个问题我没能给出符合规范的回答……」

两轮重写后仍未通过校验，`result["error"] == "validation_failed"`。看 `result["problems"]`：

- 「引用了不存在的资料编号 …」→ 模型编号越界，或抄了段落里 Perplexity 自带的 `[N]` 标记（README 第 8.2 节）
- 「宣称疾病治疗/治愈功能」等 → 合规校验命中

**不要为了让它通过而放宽 `BANNED_CLAIMS`。** 问题在生成端就在生成端修。

### 检索结果不对题 / 张冠李戴

1. 索引与代码是否同步（改过 `chunker.py` 或 `tokenize` 但没重建）
2. 行数是否对齐（4.3 的校验脚本；错位是**静默**的）
3. 用 `cli.py` 的 `/调试` 看每条候选的 BM25、向量、重排分。`_bm25` 高而 `_dense` 低说明是关键词命中（菌株号、法规号），反之是语义命中；两个都低说明它是靠元数据加权或配额挤进来的

### 向量维度报错（`matmul` shape 不匹配）

查询向量和索引向量不在同一个空间。`retrieve.py:_embed_query` 现在走 `agent.embed`，正常情况下不会出现；如果出现，说明有代码绕过了 `embed.py` 直接调某个 provider。查 `grep -rn "embeddings.create" agent/` 看有没有第二条向量化路径。

### CLI 提示「没有找到 OPENAI_API_KEY，查询向量化无法工作」但其实能跑

`cli.py` 的启动检查还在问 `agent.index.openai_key()`，那是切换到可插拔向量后端之前的判断。走本地 `bge-m3` 时这句提示是错的——本地后端不需要这个 key。以 `embed.active_backend()` 为准。

### 实时检索不起作用

先看 `result["used_live"]`。为 `False` 说明 `needs_live()` 判定不需要（问题没有时效词，且本地最高分 ≥ 2.0）。

为 `True` 但答案里没有实时内容——这是**已知缺陷**（README 第 8.3 节）：live 段落 `_score` 是 0.0，排序后被 `pool[:24]` 切掉。

另外注意 `tools/research.py` 的 `key()` 不读环境变量也不读项目 `.env`（第 3.2 节第 3 条），key 填错地方会让这条链路静默失效。验证：

```bash
.venv/bin/python tools/research.py --probe
```

### 改了 `chunker.py` 的元数据正则之后

任何影响 `evidence_grade` / `population` / `is_regulatory` / `authority` 抽取的改动，都会改变检索加权和配额的输入，**必须重建索引**，否则 `chunks.jsonl` 里还是旧标记。测试能过不代表索引是新的。

### 网页返回 503「当前咨询量较大」

并发打满（`MAX_CONCURRENT = 3`）且排队超过 `QUEUE_TIMEOUT = 150` 秒。要么调大这两个常量，要么确认下游模型接口的速率限制撑得住。

### 首次运行 stderr 有 jieba 的英文输出

正常，jieba 首次加载词典，约 0.5 秒，之后走缓存。`cli.py` 已用 `jieba.setLogLevel(60)` 压掉。

### 分类器好像没生效

`result["routing"]["_fallback"] == True` 说明走了确定性兜底——要么没有可用 provider，要么 `call_json` 失败了。

Anthropic 路径上一个容易忽略的原因：`claude-opus-5` 默认开启自适应思考，而 `max_tokens` 是思考加正文的总上限。分流用的是 `max_tokens=1024`，思考一旦吃掉预算，JSON 会被截断，`json.loads` 抛异常，静默返回 `None` 退回兜底。功能不崩，但分类器实际生效率可能远低于预期。怀疑时先加日志确认，再考虑调高 `classify.py` 里的 `max_tokens`。

---

## 9. 运维注意事项

- **`Agent` 实例要复用。** 构造时要加载索引并建 BM25。`cli.py` 和 `web/server.py` 都只建一次，不要在请求路径上新建。
- **Anthropic 路径的系统提示词走了 prompt caching**（`cache_control: ephemeral`）。策略层稳定、每轮资料放在 user 消息里，就是为了不破坏缓存前缀。**往 `build_system_prompt()` 里塞任何每轮变化的内容（时间戳、用户 ID、本轮资料）都会让缓存全线失效。**
- **改 `safety.py` 前后各跑一次 `pytest tests/test_safety.py`。** 这个文件是产品的法律和医疗底线。
- **`research/*.log` 是每次研究调用的结果行**，含 token 用量，是判断某份文档什么时候生成、花了多少的唯一记录，保留它们。
- **构建产物不进版本库**，但 `research/*.md` 和 `agent/chunker.py`、`agent/embed.py` 进——产物可以完全重建，没有信息丢失。
- **换作答模型要让用户看得见。** CLI 的状态行和网页的 `/health` 都会显示当前 provider，非 Anthropic 时 CLI 会额外提示「提示词是按 claude-opus-5 写的，换模型后措辞、引用密度和弃答判断都可能有出入」。不要把这个提示去掉。
