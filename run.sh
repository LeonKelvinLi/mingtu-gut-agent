#!/usr/bin/env bash
#
# mingtu 一键启动
#
#   ./run.sh          检查环境和索引，然后启动网页服务
#   ./run.sh check    只做检查，不启动（部署前自检、CI 用）
#
# 检查顺序是有意安排的：解释器 → 依赖 → 后端与密钥 → 索引 → 安全层 → 入口。
# 顺序反过来的话，前面的失败会让后面的报错指向错误的原因。
#
# 有一件事这个脚本必须替 web/server.py 做：把 .env 里的值导出到进程环境。
# 各模块的 key 查找路径不完全一致，只有 agent/llm.py 会读项目根目录的 .env。
# 直接 python web/server.py 启动、配置又只写在项目 .env 里的话，容易出现
# 「作答模型取到了 key、别的模块没取到」这种一半能跑的状态。
# 所以生产环境请一律用 ./run.sh 启动。
#
# 索引缺失时默认拒绝启动，而不是自动重建。走 OpenAI 后端时重建是要花钱的动作，
# 不该由一次误触发的启动替人决定；走本地 ollama 后端也要几分钟，同样不该
# 在用户等着的时候悄悄发生。确认要重建：MINGTU_REBUILD_INDEX=1 ./run.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
MODE="${1:-run}"

err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
say()  { printf '%s\n' "$*"; }

say "── mingtu 肠道菌群健康顾问 · 启动前检查 ──────────────────"

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
  say "配置     已从 .env 载入并导出 $loaded 项"
else
  warn "配置     没有 .env（可以从 .env.example 复制一份）"
fi

# ---- 3. 依赖 ----------------------------------------------------------------
# 探针一律打印一个哨兵值（OK / MISSING ...），不靠「输出为空就是没问题」。
# 探针本身挂掉时输出也是空的，那种写法会把故障读成健康。
deps="$("$PY" - <<'PY' 2>&1 || true
import importlib.util
need = {"anthropic": "anthropic", "openai": "openai", "numpy": "numpy",
        "jieba": "jieba", "rank_bm25": "rank-bm25"}
miss = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
print("MISSING " + " ".join(miss) if miss else "OK")
PY
)"
case "$deps" in
  OK)
    say "依赖     5 个直接依赖齐全" ;;
  MISSING*)
    err "缺少依赖：${deps#MISSING }"
    say "安装："
    say "  uv pip install --python .venv/bin/python -r requirements.txt"
    exit 1 ;;
  *)
    err "依赖自检没能得出结论，解释器可能坏了："
    printf '%s\n' "$deps" >&2
    exit 1 ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  warn "警告     找不到 curl，Perplexity 实时补充检索会静默失效（本地作答不受影响）"
fi

# ---- 4. 向量后端与密钥 ------------------------------------------------------
# 这里一律调项目自己的判定函数，不自己复刻规则：
#   向量后端  agent/embed.py active_backend()（ollama 优先，其次 OpenAI）
#   作答模型  agent/llm.py provider_label()（Anthropic → OpenAI → Perplexity）
#   实时检索  tools/research.py key()
# 各模块的 key 查找路径本来就不完全一致（llm.py 认项目 .env，index.py 不认，
# research.py 连环境变量都不看），自己判断一定会出现这里说「有」、
# 用户提问时说「没有」。
# 输出写成 名字=值 的形式再按前缀取，不按行号取。Python 那边的报错会混进
# 同一个流里，按行号读会把 Traceback 的第一行当成某个字段的值 —— 那种错误
# 比没有检查更糟，因为它看起来像是检查通过了。
status="$("$PY" - <<'PY' 2>&1 || true
import sys
sys.path.insert(0, ".")

def field(name, fn):
    try:
        print("%s=%s" % (name, fn()))
    except BaseException as exc:
        print("%s=ERR:%s: %s" % (name, type(exc).__name__, exc))

try:
    from agent import embed, llm
    from tools.research import key as perplexity_key
except BaseException as exc:
    print("import=ERR:%s: %s" % (type(exc).__name__, exc))
    sys.exit(0)

field("backend", embed.backend_id)                             # 没有时是 none
field("provider", llm.provider_label)                          # 没有时是「未配置」
field("perplexity", lambda: "yes" if perplexity_key() else "no")
field("localonly", lambda: "yes" if embed.local_only() else "no")
PY
)"
pick() { printf '%s\n' "$status" | grep -m1 "^$1=" | cut -d= -f2- || true; }
backend="$(pick backend)"
provider="$(pick provider)"
perplexity_state="$(pick perplexity)"
local_only="$(pick localonly)"

for pair in "向量后端:$backend" "作答模型:$provider" "实时检索:$perplexity_state" "本地限定:$local_only"; do
  label="${pair%%:*}"; got="${pair#*:}"
  if [ -z "$got" ] || [[ "$got" == ERR:* ]]; then
    err "读不出「${label}」的状态，agent/embed.py 或 agent/llm.py 的接口可能变了："
    printf '%s\n' "$status" >&2
    exit 1
  fi
done

if [ "$backend" = "none" ]; then
  err "没有可用的向量后端，检索起不来"
  if [ "$local_only" = "yes" ]; then
    say "当前 MINGTU_LOCAL_ONLY 已开启，禁止出网，所以不会回落到 OpenAI。"
    say "本地方案：安装 ollama 后执行  ollama pull bge-m3"
  else
    say "本地方案：安装 ollama 后执行  ollama pull bge-m3"
    say "云端方案：在 .env 里配置 OPENAI_API_KEY（注意用户健康文本会出境）"
  fi
  exit 1
fi
if [ "$local_only" = "yes" ]; then
  say "向量     ${backend}（MINGTU_LOCAL_ONLY 已开，不会出网）"
else
  say "向量     $backend"
fi

if [ "$provider" = "未配置" ]; then
  warn "作答     没有任何可用的模型：安全层和检索照常工作，但不生成自然语言回答。"
  warn "         生产环境至少要配 ANTHROPIC_API_KEY。"
else
  say "作答     $provider"
fi

if [ "$perplexity_state" != "yes" ]; then
  warn "实时     PERPLEXITY_API_KEY 取不到：只用本地分级语料作答，涉及最新法规、"
  warn "         菌种名单、品牌价格的问题会按设计弃答，而不是猜。"
  warn "         注意 tools/research.py 的 key() 不读环境变量，只读 ~/Desktop/dev"
  warn "         下的共享 .env —— 填在项目 .env 里对这一路是不生效的。"
else
  say "实时     PERPLEXITY_API_KEY 已就位"
fi

# ---- 5. 索引 ----------------------------------------------------------------
# 走 agent.index.load_index()，也就是服务真正用的那条路径，而不是自己去拼
# 文件名 —— 向量文件是按后端分文件存的（vectors-ollama_bge-m3.npy 等），
# 路径规则写死在这里迟早会和代码脱节。
#
# 光「文件在」不够，还要比对行数：chunks.jsonl 的行和向量的行必须一一对应，
# 检索用同一个下标同时取向量和原文，对不上的结果是「引用编号指向另一段资料」。
# 对一个要求逐条标注来源的系统，这是安全缺陷，不是小毛病。
check_index() {
  "$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
try:
    from agent.index import load_index
    chunks, vectors = load_index()
except SystemExit as exc:
    print("MISSING %s" % str(exc).replace("\n", " "))
    sys.exit(0)
except Exception as exc:
    print("ERROR %s: %s" % (type(exc).__name__, exc))
    sys.exit(0)
if len(chunks) != vectors.shape[0]:
    print("MISMATCH %d %d" % (len(chunks), vectors.shape[0]))
else:
    print("OK %d %d" % (len(chunks), vectors.shape[1]))
PY
}

index_state="$(check_index)"
if [[ "$index_state" == OK* ]]; then
  read -r _ rows dim <<< "$index_state"
  say "索引     $rows 段资料，$dim 维向量，行数对齐"
else
  case "$index_state" in
    MISSING*)
      reason="${index_state#MISSING }" ;;
    MISMATCH*)
      read -r _ a b <<< "$index_state"
      reason="chunks.jsonl 有 $a 行，向量有 $b 行，对不上（两者不是同一次构建的产物）" ;;
    *)
      reason="索引自检失败：${index_state#ERROR }" ;;
  esac

  # 重建的代价取决于后端：本地 ollama 只花时间，OpenAI 要花钱。
  if [ "$backend" = "none" ] || [[ "$backend" == ollama:* ]]; then
    cost="用本地后端 $backend 重建，只花时间不花钱"
  else
    cost="用 $backend 重建，会产生 API 费用"
  fi

  if [ "${MINGTU_REBUILD_INDEX:-0}" = "1" ]; then
    warn "索引     $reason"
    warn "         MINGTU_REBUILD_INDEX=1，开始重建（${cost}）"
    "$PY" -m agent.index build
    index_state="$(check_index)"
    if [[ "$index_state" == OK* ]]; then
      read -r _ rows dim <<< "$index_state"
      say "索引     重建完成，$rows 段资料，$dim 维向量"
    else
      err "启动中止：重建之后索引仍然不可用（${index_state}）"
      exit 1
    fi
  else
    err "索引不可用：$reason"
    say "重建（${cost}）："
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
  err "启动中止：安全层自检没过。在修好之前不要对外提供服务。"
  exit 1
fi
say "安全层   报警症状筛查正反用例均通过"

# ---- 7. 网页入口 ------------------------------------------------------------
entry=""
for candidate in web/server.py web/app.py web/main.py; do
  if [ -f "$REPO/$candidate" ]; then
    entry="$candidate"
    break
  fi
done

if [ -z "$entry" ]; then
  err "找不到网页服务入口"
  say "已检查：web/server.py、web/app.py、web/main.py，都不存在。"
  say "环境、密钥、索引、安全层都是好的，缺的只有 web/ 这一层。"
  exit 1
fi

# 先确认入口导得进来再启动。这一步能挡住的是最难看的一种故障：模块之间接口
# 改了没对齐，服务在用户点开页面之后才崩。导入是安全的 —— 服务只在
# __name__ == "__main__" 时才真正监听端口。
set +e
import_error="$("$PY" - "$entry" <<'PY' 2>&1
import runpy, sys, traceback
sys.path.insert(0, ".")
try:
    # 以非 __main__ 的名字执行：模块级定义会跑，main() 不会，端口不会被占。
    runpy.run_path(sys.argv[1], run_name="mingtu_import_check")
except BaseException:
    traceback.print_exc(limit=3)
    sys.exit(1)
PY
)"
import_status=$?
set -e
if [ "$import_status" -ne 0 ]; then
  err "网页入口 $entry 导入失败，没有启动服务："
  printf '%s\n' "$import_error" >&2
  exit 1
fi
say "入口     $entry 可正常导入"

# ---- 8. 启动 ----------------------------------------------------------------
if [ "$MODE" = "check" ]; then
  say "───────────────────────────────────────────────────────"
  say "检查全部通过，未启动服务（check 模式）。"
  exit 0
fi

# web/server.py 的参数就是 --host / --port，环境变量回落到 HOST / PORT。
# 这里显式传参，免得依赖环境变量的隐式约定。
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

say "───────────────────────────────────────────────────────"
say ""

# exec 让服务进程顶替这个 shell，Ctrl+C 和容器的停止信号直接送到它身上，
# 中间不隔一层壳。启动横幅由 server.py 自己打印。
exec "$PY" "$REPO/$entry" --host "$HOST" --port "$PORT"
