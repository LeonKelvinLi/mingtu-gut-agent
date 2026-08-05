#!/usr/bin/env bash
#
# mingtu 一键启动
#
#   ./run.sh          检查环境和索引，然后启动网页服务
#   ./run.sh check    只做检查，不启动（部署前自检、CI 用）
#
# 检查顺序是有意安排的：先解释器和依赖，再密钥，然后索引，最后安全层。
# 顺序反过来的话，前面的失败会让后面的报错指向错误的原因。
#
# 索引缺失时默认拒绝启动，而不是自动重建 —— 重建要调 OpenAI embedding，
# 是花钱的动作，不该由一次误触发的启动替人决定。
# 确认要重建：MINGTU_REBUILD_INDEX=1 ./run.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
MODE="${1:-run}"

err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
say()  { printf '%s\n' "$*"; }

say "── mingtu 肠道菌群健康顾问 ──────────────────────────────"

# ---- 1. 解释器 --------------------------------------------------------------
if [ ! -x "$PY" ]; then
  err "找不到虚拟环境解释器：$PY"
  say "创建："
  say "  uv venv --python 3.12 .venv"
  say "  uv pip install --python .venv/bin/python -r requirements.txt"
  exit 1
fi
say "解释器   $("$PY" -V 2>&1)"

# ---- 2. 读取 .env -----------------------------------------------------------
# 只接受 KEY=VALUE，跳过注释和空行，不用 source —— 免得 .env 里的内容被当作
# shell 代码执行。已经在环境里的变量优先，不被文件覆盖。
if [ -f "$REPO/.env" ]; then
  loaded=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|'#'*) continue ;;
    esac
    line="${line#export }"
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      name="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value%\"}"; value="${value#\"}"
      value="${value%\'}"; value="${value#\'}"
      if [ -z "${!name:-}" ]; then
        export "$name=$value"
        loaded=$((loaded + 1))
      fi
    fi
  done < "$REPO/.env"
  say "配置     已从 .env 载入 $loaded 项"
else
  warn "配置     没有 .env（可以从 .env.example 复制一份）"
fi

# ---- 3. 依赖 ----------------------------------------------------------------
missing="$("$PY" - <<'PY'
import importlib.util
need = {"anthropic": "anthropic", "openai": "openai", "numpy": "numpy",
        "jieba": "jieba", "rank_bm25": "rank-bm25"}
print(" ".join(pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None))
PY
)"
if [ -n "$missing" ]; then
  err "缺少依赖：$missing"
  say "安装："
  say "  uv pip install --python .venv/bin/python -r requirements.txt"
  exit 1
fi
say "依赖     5 个直接依赖齐全"

if ! command -v curl >/dev/null 2>&1; then
  warn "警告     找不到 curl，Perplexity 实时补充检索会静默失效（本地作答不受影响）"
fi

# ---- 4. 密钥 ----------------------------------------------------------------
# 用项目自己的取值函数来查，而不是只看环境变量 —— 那几个函数还会回落到
# ~/Desktop/dev 下的共享 .env。检查逻辑必须和运行时取值逻辑是同一套，
# 否则会出现这里说「有」、用户提问时说「没有」。
keys="$("$PY" - <<'PY'
import sys
sys.path.insert(0, ".")

def probe(fn):
    try:
        return "yes" if fn() else "no"
    except SystemExit:
        return "no"
    except Exception:
        return "no"

from agent.llm import anthropic_key
from agent.index import openai_key
from tools.research import key as perplexity_key
print("%s %s %s" % (probe(anthropic_key), probe(openai_key), probe(perplexity_key)))
PY
)"
read -r has_anthropic has_openai has_perplexity <<< "$keys"

if [ "$has_openai" != "yes" ]; then
  err "OPENAI_API_KEY 没配"
  say "这是运行时硬依赖，不只是建索引时用：每一轮提问都要先把问题向量化，"
  say "取不到 key 检索会直接抛错，没有降级路径。请在 .env 里填好再启动。"
  exit 1
fi
say "密钥     OPENAI 已配置"

if [ "$has_anthropic" != "yes" ]; then
  warn "警告     ANTHROPIC_API_KEY 没配：安全层和检索照常工作，但不会生成回答，"
  warn "         只会返回本轮检索到的资料条数。生产环境必须配。"
else
  say "密钥     ANTHROPIC 已配置"
fi

if [ "$has_perplexity" != "yes" ]; then
  warn "提示     PERPLEXITY_API_KEY 没配：只用本地分级语料作答，涉及最新法规、"
  warn "         菌种名单、品牌价格的问题会按设计弃答，而不是猜。"
else
  say "密钥     PERPLEXITY 已配置"
fi

# ---- 5. 索引 ----------------------------------------------------------------
# chunks.jsonl 的行必须和 vectors.npy 的行一一对应：检索用同一个下标同时取
# 向量和原文，对不上的结果是「引用编号指向另一段资料」，那是安全缺陷不是小毛病。
check_index() {
  "$PY" - <<'PY'
import sys
from pathlib import Path
import numpy as np

chunks = Path("knowledge/chunks.jsonl")
vectors = Path("knowledge/vectors.npy")
if not chunks.exists() or not vectors.exists():
    print("MISSING")
    sys.exit(0)
rows = sum(1 for line in chunks.open(encoding="utf-8") if line.strip())
shape = np.load(vectors, mmap_mode="r").shape
if rows != shape[0]:
    print("MISMATCH %d %d" % (rows, shape[0]))
elif shape[1] != 3072:
    print("DIM %d" % shape[1])
else:
    print("OK %d %d" % (rows, shape[1]))
PY
}

index_state="$(check_index)"
if [[ "$index_state" == OK* ]]; then
  read -r _ rows dim <<< "$index_state"
  say "索引     $rows 个知识块，$dim 维向量"
else
  case "$index_state" in
    MISSING)
      reason="knowledge/ 下没有索引文件（构建产物不进版本库，新机器要自己建一次）" ;;
    MISMATCH*)
      read -r _ a b <<< "$index_state"
      reason="chunks.jsonl 有 $a 行，vectors.npy 有 $b 行，对不上（两者不是同一次构建的产物）" ;;
    DIM*)
      read -r _ d <<< "$index_state"
      reason="向量是 $d 维，期望 3072 维（embedding 模型换过？）" ;;
    *)
      reason="索引自检失败：$index_state" ;;
  esac

  if [ "${MINGTU_REBUILD_INDEX:-0}" = "1" ]; then
    warn "索引     $reason"
    warn "         MINGTU_REBUILD_INDEX=1，开始重建（会调用 OpenAI embedding，产生费用）"
    "$PY" -m agent.index build
    index_state="$(check_index)"
    if [[ "$index_state" == OK* ]]; then
      read -r _ rows dim <<< "$index_state"
      say "索引     重建完成，$rows 个知识块，$dim 维向量"
    else
      err "启动中止：重建之后索引仍然不一致（$index_state）"
      exit 1
    fi
  else
    err "索引不可用：$reason"
    say "重建（会调用 OpenAI embedding，产生费用）："
    say "  .venv/bin/python -m agent.index build"
    say "或者让本脚本代劳："
    say "  MINGTU_REBUILD_INDEX=1 ./run.sh"
    exit 1
  fi
fi

# ---- 6. 安全层自检 ----------------------------------------------------------
# 这一条不能省。急症判定是确定性的，不依赖模型也不依赖网络，所以它必须在
# 任何密钥缺失、任何 API 故障的情况下都能过。它过不了就绝不该对外提供服务。
# 正反两个用例都要测：只测正例的话，一个「把什么都判成急症」的坏正则也能通过，
# 那样顾问就废了。
if ! "$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
from agent import safety

hits, instruction = safety.screen_input("大便带血还一直拉肚子，人都瘦了")
if not hits or not instruction:
    print("确定性报警症状筛查漏掉了已知的红旗症状", file=sys.stderr)
    sys.exit(1)
if not safety.escalation_message(hits).strip():
    print("急症固定话术为空", file=sys.stderr)
    sys.exit(1)

benign, _ = safety.screen_input("益生菌一般什么时候吃比较好")
if benign:
    print("普通科普问题被误判成报警症状：%s" % benign, file=sys.stderr)
    sys.exit(1)
PY
then
  err "启动中止：安全层自检没过。在修好之前不要启动服务。"
  exit 1
fi
say "安全层   报警症状筛查正反用例均通过"

# ---- 7. 启动 ----------------------------------------------------------------
if [ "$MODE" = "check" ]; then
  say "───────────────────────────────────────────────────────"
  say "检查全部通过，未启动服务（check 模式）。"
  exit 0
fi

entry=""
for candidate in web/app.py web/server.py web/main.py; do
  if [ -f "$REPO/$candidate" ]; then
    entry="$candidate"
    break
  fi
done

if [ -z "$entry" ]; then
  err "找不到网页服务入口"
  say "已检查：web/app.py、web/server.py、web/main.py，都不存在。"
  say "环境、密钥、索引、安全层都是好的，缺的只有 web/ 这一层。"
  say "补上之后再跑 ./run.sh；只想复查环境用 ./run.sh check。"
  exit 1
fi

module="${entry%.py}"
module="${module//\//.}"

export MINGTU_HOST="${MINGTU_HOST:-127.0.0.1}"
export MINGTU_PORT="${MINGTU_PORT:-8000}"

say "───────────────────────────────────────────────────────"
say "启动     $entry"
say "地址     http://$MINGTU_HOST:$MINGTU_PORT"
say "停止     Ctrl+C"
say ""

# 用 -m 而不是直接跑文件路径：模块方式会把仓库根目录放进 sys.path，
# web 里的 from agent... / from tools... 才导得到。
# exec 让服务进程顶替这个 shell，Ctrl+C 和容器的停止信号直接送到它身上。
exec "$PY" -m "$module"
