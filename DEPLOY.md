# 部署与运维

面向接手这套代码的工程团队。架构和设计理由看 [README.md](README.md)，这份只讲怎么跑起来、怎么维护、出问题怎么查。

---

## 1. 环境要求

| 项 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.13 | 虚拟环境在 `.venv/`，由 uv 管理（解释器来自 `~/.local/share/uv/python/cpython-3.12-macos-aarch64-none`） |
| uv | 0.11.30 | `/opt/homebrew/bin/uv` |
| curl | 系统自带 | `live.py` 和 `tools/research.py` 用 `subprocess` 调 curl 发 Perplexity 请求，不走 Python HTTP 库 |
| 磁盘 | 约 6 MB | `knowledge/` 5.7 MB（主要是 `vectors.npy`）+ `research/` 364 KB |

**不要用系统 `python3`。** macOS 自带的是 3.9.6，代码用了 3.10+ 的语法和依赖。所有命令一律走 `.venv/bin/python`。

**必须在仓库根目录运行。** `agent/` 和 `tools/` 没有 `__init__.py`，靠命名空间包和 cwd 在 `sys.path` 上工作。在别处运行会 `ModuleNotFoundError: No module named 'agent'`。

---

## 2. 依赖安装

仓库里没有 `pyproject.toml` 或 `requirements.txt`，依赖只存在于现有 `.venv` 中。从零重建：

```bash
cd /Users/leon/Desktop/dev/mingtu
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "anthropic==0.120.2" \
  "openai==2.53.0" \
  "numpy==2.5.1" \
  "jieba==0.42.1" \
  "rank-bm25==0.2.2"
```

这五个是直接依赖，其余（httpx、pydantic、tqdm 等）由它们带入。

验证：

```bash
.venv/bin/python -c "import anthropic, openai, numpy, jieba, rank_bm25; print('ok')"
```

**建议尽早补一份 `pyproject.toml` 和 lock 文件。** 现在这五个版本号只写在这份文档里，环境一丢就没有权威来源了。

---

## 3. API key

三个 key，用途和读取顺序各不相同。**没有 `.env.example`，也没有集中的配置文件**，读取逻辑硬编码在三个模块里。

### 3.1 `OPENAI_API_KEY`（必需）

**用途：** 向量化。构建索引时批量向量化 443 段文本；**运行时每一次检索都要调一次**，把用户提问变成查询向量。所以这个 key 不是只在构建期用，线上每个请求都依赖它。

**模型：** `text-embedding-3-large`，3072 维。

**读取顺序**（`agent/index.py:41-50`）：

1. 环境变量 `OPENAI_API_KEY`
2. `~/Desktop/dev/curio/.env`
3. `~/Desktop/dev/Curio-Automation/.env`

都找不到 → `SystemExit: no OPENAI_API_KEY found`。

**当前状态：** 后两个文件里都有，可用。

**注意：** 这几个 key 共享同一个 OpenAI 账户额度，额度用尽会影响所有依赖它的项目。`index.build()` 在批量向量化之前会先发一个单条的探针请求，让额度问题在第 1 次调用就暴露，而不是跑到第 400 次才失败。

### 3.2 `ANTHROPIC_API_KEY`（必需，当前缺失）

**用途：** 三个 LLM 阶段——风险分流、候选重排、作答与重写。模型是 `claude-opus-5`。

**读取顺序**（`agent/llm.py:23-32`）：

1. 环境变量 `ANTHROPIC_API_KEY`
2. `~/Desktop/dev/curio/.env`
3. `~/Desktop/dev/Curio-Automation/.env`

找不到时 `client()` 返回 `None`，**不抛异常**，整条链路降级：分流走确定性兜底、重排走原序、作答返回一句「【未配置 ANTHROPIC_API_KEY，无法生成回答】」。

**当前状态：缺失。** 上述两个 `.env` 里都没有这一项，环境变量也没有。**这是当前的第一号上线阻塞项**——检索和安全层正常，但产品最核心的作答能力完全不工作。

配置方式（二选一）：

```bash
# 方式一：环境变量（推荐，作用域清晰）
export ANTHROPIC_API_KEY=sk-ant-...

# 方式二：写进 llm.py 已经会读的 .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> ~/Desktop/dev/curio/.env
```

配完立刻验证：

```bash
.venv/bin/python -c "from agent import llm; print('anthropic key:', 'ok' if llm.available() else 'MISSING')"
```

### 3.3 `PERPLEXITY_API_KEY`（可选）

**用途：** 两处。`tools/research.py` 用 `sonar-deep-research` 生成知识库原料（离线，产出 `research/*.md`）；`agent/live.py` 用 `sonar-pro` 做运行时实时补充检索（在线，60 秒超时）。

分两个模型是刻意的：深度研究可以跑几分钟，实时检索在一次对话轮次内，跑深度研究会让产品不可用。

**读取顺序**（`tools/research.py:50-57`，`live.py` 复用同一个函数）：

1. `~/Desktop/dev/curio/.env`
2. `~/Desktop/dev/Curio-Automation/.env`
3. `~/Desktop/dev/curio-international/.env`

**注意这条路径不读环境变量**，只读文件。找不到时 `live.search_live()` 返回 `[]`，主链路继续；`tools/research.py` 则会把提示词落盘成 `research/<name>.PARKED.txt` 并以非零码退出——配额耗尽是全停，不会自动降级到弱模型。

**当前状态：** 三个文件里都有，可用。

### 3.4 key 检查一条龙

```bash
.venv/bin/python - <<'PY'
from agent import llm
from agent.index import openai_key
from tools.research import key as pplx_key
try:
    openai_key(); print("OPENAI_API_KEY      ok")
except SystemExit as e:
    print("OPENAI_API_KEY      MISSING —", e)
print("ANTHROPIC_API_KEY  ", "ok" if llm.available() else "MISSING（作答层将完全降级）")
print("PERPLEXITY_API_KEY ", "ok" if pplx_key() else "MISSING（实时检索将静默返回空）")
PY
```

---

## 4. 重建索引

### 什么时候需要重建

- `research/*.md` 有增删改
- `chunker.py` 的切块参数或元数据抽取规则改了
- `index.py` 的分词逻辑（`tokenize`）改了
- 换 embedding 模型

**只改 `retrieve.py` 的权重、`prompt.py` 的提示词、`safety.py` 的正则，不需要重建。**

### 怎么重建

```bash
cd /Users/leon/Desktop/dev/mingtu
.venv/bin/python -m agent.index build
```

输出形如：

```
443 chunks from 5 documents
  171 carry an evidence grade, 68 are regulatory
  embedding probe OK
  embedded 64/443
  ...
  embedded 443/443
wrote .../knowledge/chunks.jsonl and .../knowledge/vectors.npy (443, 3072)
```

**规模参考：** 443 段 × 约 600 token ≈ 27 万 token，一次全量重建。每批 64 条，约 7 个请求。

**没有增量更新。** 加一份研究文档也是全量重跑。

### 重建后必须校验

`load_index` 不检查 `chunks.jsonl` 行数与 `vectors.npy` 行数是否一致，不一致时不会报错，而是静默返回**错位的检索结果**——这是最难发现的一类故障。每次重建后跑一遍：

```bash
.venv/bin/python - <<'PY'
import json, numpy as np
from pathlib import Path
chunks = [json.loads(l) for l in Path("knowledge/chunks.jsonl").read_text().splitlines() if l.strip()]
vectors = np.load("knowledge/vectors.npy")
assert len(chunks) == vectors.shape[0], f"错位: {len(chunks)} 块 vs {vectors.shape[0]} 向量"
assert vectors.shape[1] == 3072, f"维度异常: {vectors.shape[1]}"
norms = np.linalg.norm(vectors, axis=1)
assert abs(norms.mean() - 1.0) < 0.01, f"向量未归一化: 均值 {norms.mean()}"
graded = sum(1 for c in chunks if c["evidence_grade"])
print(f"ok: {len(chunks)} 块 / {vectors.shape} / 带证据等级 {graded} 块 / 法规 {sum(1 for c in chunks if c['is_regulatory'])} 块")
PY
```

**重建前先备份。** 旧索引一旦被覆盖，重新生成要再花一次向量化费用：

```bash
cp knowledge/chunks.jsonl knowledge/chunks.jsonl.bak
cp knowledge/vectors.npy  knowledge/vectors.npy.bak
```

---

## 5. 新增研究资料

`research/*.md` 是索引的唯一数据源。有两条路径。

### 5.1 用现成的研究流程（推荐）

先探活（走便宜的 `sonar` 模型）：

```bash
.venv/bin/python tools/research.py --probe
```

跑一个预置问题：

```bash
.venv/bin/python tools/research.py --ask 01-strains
```

五个预置问题：`01-strains`（菌株证据）、`02-diet`（膳食纤维与益生元）、`03-china-reg`（中国监管）、`04-limits`（临床边界与误区）、`05-systemic`（菌群与全身健康）。

**新增一个主题**：在 `tools/research.py` 的 `QUESTIONS` 字典里加一条，键名做文件名，值是问题。`RIGOUR` 段会自动追加到每个问题后面，它规定了证据分级、菌株号、剂量单位、发表年份、参考文献 URL 这些要求——**不要绕过它**，否则新文档的元数据抽取率会明显低于现有文档。

问题必须写成一行（`one_line()` 会把所有空白压成单空格再发出去）。

失败时不会重试也不会换模型：提示词落到 `research/<name>.PARKED.txt`，进程返回 1，需要人工去 Perplexity 网页端粘贴执行。

### 5.2 手动添加文档

放一个 `.md` 到 `research/`。要让元数据抽取生效，文档需要满足：

| 元数据 | 抽取规则（`chunker.py`） | 写法建议 |
|---|---|---|
| `evidence_grade` | `证据等级：A` / `（A级）` / `A级证据` 三选一 | 每条结论后紧跟证据等级，用这三种写法之一 |
| `population` | 关键词匹配：婴幼儿/儿童/孕产妇/老年/免疫抑制/成人 | 明确写出人群词，不要只写「患者」 |
| `year` | `19[89]\d` 或 `20[0-3]\d` 取最大值 | 年份写在结论所在段落内 |
| `authority` | 荟萃分析/指南/随机对照/队列/动物实验/专家意见，最先命中者胜 | 写明研究类型 |
| `is_regulatory` | 广告法/食品安全法/卫健委/市场监管总局/保健食品/国标/GB 等 | 法规内容写全名和文号 |
| `citations` | 段落内的 `http(s)://` URL | **把 URL 写在结论所在段落里，不要只堆在文末** |

切块参数：目标 450 字，重叠 90 字，小于 120 字的段落丢弃。文档按 1–4 级 Markdown 标题分节，标题会拼进块正文（`【标题】\n正文`）参与检索。

**已知坑：** Perplexity 输出自带 `[1][2]` 形式的引用标记，会原样进索引并与本系统的资料编号冲突（详见 README 第 9.1 节第 4 条）。手写文档时不要用方括号数字标注引用。

加完文档后重建索引，并按第 4 节校验。

---

## 6. 本地运行

### 6.1 CLI

**目前没有 CLI 入口。** 没有 `cli.py`，`core.py` 也没有 `__main__` 块。可用的调用方式是脚本或 REPL：

```bash
cd /Users/leon/Desktop/dev/mingtu
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...

.venv/bin/python - <<'PY'
from agent.core import Agent
a = Agent()          # 加载索引约 1.2 秒（含 jieba 初始化）
while True:
    q = input("\n问题（空行退出）> ").strip()
    if not q:
        break
    r = a.answer(q)
    print(f"\n[risk={r['routing']['risk']} 资料={len(r['passages'])} 实时={r['used_live']} 重写={r['repairs']}]")
    print(r["answer"])
    for s in r["sources"]:
        print(f"  [{s['n']}] {s['source']} · {s['heading']} · {s['grade'] or '未标注'}" + (" · 实时" if s["is_live"] else ""))
PY
```

单轮的写法：

```bash
.venv/bin/python -c "
from agent.core import Agent
print(Agent().answer('益生菌要空腹吃吗')['answer'])
"
```

`Agent()` 每次构造都要重新加载索引和建 BM25，**要复用同一个实例**，不要每个请求都新建。

**建议补一个 `cli.py`**（`argparse` + 交互循环 + `--verbose` 打印检索中间态），这是最低成本的一块补全。

### 6.2 网页

**未实现。** `web/` 是空目录，没有任何 HTTP 服务代码、模板或前端资源。

要做的话最小可用形态是：

- `Agent` 单例在进程启动时构造一次（加载索引约 1.2 秒，不能放在请求路径上）
- 一个 `POST /ask`，请求体 `{"query": str, "history": [...]}`，返回 `answer` / `sources` / `escalated`
- **`escalated == True` 时前端必须显著区分展示**——这是就医提示，不能和普通科普混在一起
- `sources` 至少展示来源文件、标题、证据等级；`is_live` 的条目要标出「实时检索，可信度低于已分级资料」
- `error` 字段非空时不要显示 `answer` 之外的技术细节给用户

依赖里没有任何 Web 框架，选型时另装。

---

## 7. 上线检查清单

按顺序执行。任何一项失败都不要上线。

### 环境

- [ ] `.venv/bin/python --version` 返回 3.12.x
- [ ] 五个直接依赖导入无报错
- [ ] 在仓库根目录执行，`import agent.core` 成功

### 密钥

- [ ] `OPENAI_API_KEY` 可读，且额度充足（跑一次探针）
- [ ] **`ANTHROPIC_API_KEY` 可读，`llm.available()` 返回 True**（当前缺失，必须先解决）
- [ ] `PERPLEXITY_API_KEY` 可读（不可用则接受实时检索静默失效，并确认这个降级可接受）

### 索引

- [ ] `knowledge/chunks.jsonl` 与 `knowledge/vectors.npy` 行数一致（第 4 节的校验脚本）
- [ ] 向量维度 3072，已归一化
- [ ] `research/*.md` 与索引同步（改过文档就要重建）

### 安全层（不需要任何 key，必须逐条过）

- [ ] `screen_input("我最近大便有血")` → `['便血']`
- [ ] `screen_input("我最近大便有血，还瘦了五斤")` → `['便血', '体重下降']`，且 `mentions_self` 为 True
- [ ] 无 `ANTHROPIC_API_KEY` 时 `answer("我最近大便有血，还瘦了五斤")` 仍返回 `escalated=True` 和完整就医话术
- [ ] `screen_output("益生菌可以治疗肠易激综合征。")` 命中违规
- [ ] `screen_output("益生菌不能治疗肠易激综合征。")` 不命中（否定豁免生效）
- [ ] `classify()` 在模型返回 `normal` 时，若正则判 emergency，最终 risk 仍是 emergency

### 端到端（需要真实 key）

- [ ] 普通科普提问（「LGG 对儿童急性腹泻有用吗」）能出答案，`sources` 非空，每条实质结论带编号
- [ ] 法规提问（「保健食品能宣称调节肠道菌群吗」）的资料里出现 `is_regulatory` 段落
- [ ] 特殊人群提问（「宝宝能吃益生菌吗」）的资料里出现婴幼儿相关段落，且回答里有「先咨询医生」的提示
- [ ] 冷门提问（知识库覆盖不到的）走弃答路径，不硬凑答案
- [ ] **重写循环实测有效**：构造一个会触发合规校验的场景，确认 `repairs > 0` 且最终输出合规，而不是返回「抱歉，这一轮没能生成回答」（README 第 9.1 节第 2 条是已知缺陷，上线前必须修掉并验证）
- [ ] 引用编号全部落在 `1..len(passages)` 区间，随机抽 3 条人工核对编号指向的段落确实支撑该结论
- [ ] 回答里没有 AI 腔（「希望对您有帮助」「作为一个 AI」这类），没有表情符号，全中文

### 成本与限流

- [ ] 单次提问的 token 用量实测过（分流 + 重排 + 作答，注意 thinking 也计入 `max_tokens`）
- [ ] 确认 OpenAI embedding 的 QPS 能撑住预期并发
- [ ] Perplexity 实时检索的 60 秒超时对产品体验可接受（或先关掉这条链路）

### 兜底演练

- [ ] 临时清掉 `ANTHROPIC_API_KEY`，确认安全层照常工作、系统不崩
- [ ] 临时清掉 `PERPLEXITY_API_KEY`，确认主链路不受影响
- [ ] 断网跑一次，确认异常被捕获而不是把栈打给用户

---

## 8. 常见故障排查

### `ModuleNotFoundError: No module named 'agent'`

不在仓库根目录。`agent/` 没有 `__init__.py`，靠 cwd 在 `sys.path` 上。`cd /Users/leon/Desktop/dev/mingtu` 后重试。

### `SystemExit: no OPENAI_API_KEY found`

三处都没读到。检查环境变量，以及 `~/Desktop/dev/curio/.env`、`~/Desktop/dev/Curio-Automation/.env` 里有没有 `OPENAI_API_KEY=` 开头的行（`agent/index.py` 用 `^OPENAI_API_KEY=(.*)$` 多行匹配，**行首不能有空格或 `export`**）。

### `SystemExit: index missing — run: python -m agent.index build`

`knowledge/` 下缺 `chunks.jsonl` 或 `vectors.npy`。按第 4 节重建。

### 回答是「【未配置 ANTHROPIC_API_KEY，无法生成回答】」

作答层降级了。`result["error"] == "no_anthropic_key"`。检索和安全层正常，只是没配 Anthropic key。见第 3.2 节。

### 回答是「抱歉，这一轮没能生成回答，请再问一次。」

`_generate()` 返回了 `None`，`result["error"] == "generation_failed"`。三种可能：

1. **模型拒答**（`stop_reason == "refusal"`）。健康医疗类提问有一定概率触发安全分类器。
2. **API 异常**被 `except Exception` 吞掉了——认证失败、限流、超时、参数错误在外部完全无法区分。
3. **命中了 README 第 9.1 节第 2 条的缺陷**：如果这一句出现在「本来应该重写」的场景下（比如上一轮回答里明显有违规词），几乎可以确定是重写循环的 system 消息位置问题。

排障时把 `core.py:192` 的 `except Exception:` 临时改成打印异常：

```python
except Exception as e:
    print(f"[generate failed] {type(e).__name__}: {e}")
    return None
```

`llm.py:79` 的 `call_json` 同理。**这两处的静默吞异常是当前最大的可观测性缺口，建议尽快改成有日志的形式。**

### 回答是「这个问题我没能给出符合规范的回答……」

两轮重写后仍未通过校验，`result["error"] == "validation_failed"`。看 `result["problems"]`，两类：

- 「引用了不存在的资料编号 …」→ 模型编号越界，或抄了段落里 Perplexity 自带的 `[N]` 标记（README 第 9.1 节第 4 条）
- 「宣称疾病治疗/治愈功能」等 → 合规校验命中

**不要为了让它通过而放宽 `BANNED_CLAIMS`。** 问题在生成端就在生成端修。

### 检索结果明显不对题 / 张冠李戴

按顺序查：

1. 索引与代码是否同步（改过 `chunker.py` 或 `tokenize` 但没重建）
2. `chunks.jsonl` 与 `vectors.npy` 行数是否一致（不一致会**静默错位**，第 4 节的校验脚本）
3. 打印检索中间态看是哪一路的问题：

```python
r = agent.retriever.search("你的问题", filters={}, top_n=8)
for c in r[:8]:
    print(f"{c['_score']:.3f}  bm25={c['_bm25']:.2f}  dense={c['_dense']:.3f}  {c['source']} · {c['heading'][:30]}")
```

`_bm25` 高而 `_dense` 低 → 关键词匹配主导，可能是菌株号/法规号命中；反过来是语义匹配主导。两个都低说明这个块是靠元数据加权或配额挤进来的。

### 实时检索完全不起作用

先确认触发了没有：`result["used_live"]`。若为 `False`，说明 `needs_live()` 判定不需要（问题没有时效词，且本地最高分 ≥ 2.0）。

若为 `True` 但答案里没有实时内容——这是**已知缺陷**（README 第 9.1 节第 3 条）：live 段落的 `_score` 是 0.0，排序后被 `pool[:24]` 切掉了。实测本地检索单个子查询就返回满 24 条，所以实时结果基本进不了重排。

### Perplexity 调用失败

`live.search_live()` 任何异常都返回 `[]`，不打日志。手工验证：

```bash
.venv/bin/python tools/research.py --probe
```

返回 `ok tokens=...` 说明 key 和网络正常。返回错误信息说明配额或 key 有问题——**配额耗尽是全停，不要换成弱模型凑合**。

### 首次运行 stderr 里有 jieba 的输出

```
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/.../jieba.cache
```

正常，jieba 首次加载词典。约 0.5 秒，之后走缓存。要屏蔽的话在导入前 `jieba.setLogLevel(logging.INFO)`。

### 分类器好像没生效

`result["routing"]["_fallback"] == True` 说明走了确定性兜底——要么没有 Anthropic key，要么 `call_json` 失败了。

一个容易忽略的原因：claude-opus-5 默认开启自适应思考，而 `max_tokens` 是思考加正文的总上限。分流用的是 `max_tokens=1024`，思考一旦吃掉预算，JSON 会被截断，`json.loads` 抛异常，静默返回 `None` 退回兜底。功能不崩，但分类器实际生效率可能远低于预期。怀疑这个问题时先加日志确认，再考虑调高 `classify.py` 里的 `max_tokens`。

---

## 9. 运维注意事项

- **`Agent` 实例要复用。** 构造时加载 443 段索引并建 BM25，约 1.2 秒。不要放在请求路径上。
- **`Retriever._client` 是懒加载的 OpenAI 客户端**，第一次检索时才建立，之后复用。
- **系统提示词走了 prompt caching**（`cache_control: ephemeral`）。策略层是稳定的、每轮资料放在 user 消息里，就是为了不破坏缓存前缀。**往 `build_system_prompt()` 里塞任何每轮变化的内容（时间戳、用户 ID、本轮资料），都会让缓存全线失效。**
- **改 `safety.py` 的正则前后各跑一次全量回归**（目前没有测试，只能手工构造样例）。这个文件是产品的法律和医疗底线。
- **`research/*.log` 是每次研究调用的结果行**，包含 token 用量，保留它们，是判断某份文档什么时候生成、花了多少的唯一记录。
- **索引重建前备份**（第 4 节）。
- **仓库不是 git 仓库**，改动没有历史也没法回滚。**建议第一件事就是 `git init` 并把 `.venv/` 加进 `.gitignore`**。`knowledge/vectors.npy` 有 5.4 MB，考虑是否入库（能从 `research/` 重建，但重建要花向量化费用）。
