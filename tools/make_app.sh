#!/bin/bash
# 生成一个可以双击打开的 macOS 应用：肠道菌群健康顾问.app
#
# 双击后它做四件事：确认 ollama 在跑 → 启动本地服务 → 打开浏览器 → 常驻。
# 全程 MINGTU_LOCAL_ONLY=1，不出网。
#
# 重新生成：bash tools/make_app.sh

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$REPO/肠道菌群健康顾问.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>肠道菌群健康顾问</string>
  <key>CFBundleDisplayName</key><string>肠道菌群健康顾问</string>
  <key>CFBundleIdentifier</key><string>cn.mingtu.gutagent</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <!-- 后台运行，不在 Dock 里占位，也不抢焦点 -->
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/launcher" <<LAUNCH
#!/bin/bash
# 双击入口。日志在 ~/Library/Logs/mingtu-agent.log
REPO="$REPO"
PORT=8830
LOG="\$HOME/Library/Logs/mingtu-agent.log"
mkdir -p "\$(dirname "\$LOG")"

note() { echo "[\$(date '+%H:%M:%S')] \$1" >> "\$LOG"; }
alert() { osascript -e "display notification \"\$1\" with title \"肠道菌群健康顾问\"" >/dev/null 2>&1 || true; }

note "启动"

# 已经在跑就直接开浏览器，不重复启动
if curl -sS --max-time 2 "http://127.0.0.1:\$PORT/health" >/dev/null 2>&1; then
  note "服务已在运行，直接打开浏览器"
  open "http://127.0.0.1:\$PORT/"
  exit 0
fi

# ollama：本地作答与向量都靠它
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
if ! curl -sS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  note "启动 ollama"
  alert "正在启动本地模型…"
  brew services start ollama >>"\$LOG" 2>&1 || (ollama serve >>"\$LOG" 2>&1 &)
  for i in \$(seq 1 30); do
    curl -sS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

if ! curl -sS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  note "ollama 起不来"
  osascript -e 'display alert "本地模型没能启动" message "请在终端执行：brew services start ollama" as critical' >/dev/null 2>&1
  exit 1
fi

# 全本地模式：健康问题不出本机
note "启动本地服务 :\$PORT"
alert "正在加载知识库…"
cd "\$REPO"
MINGTU_LOCAL_ONLY=1 "\$REPO/.venv/bin/python" web/server.py --host 127.0.0.1 --port \$PORT >>"\$LOG" 2>&1 &
SERVER_PID=\$!
echo \$SERVER_PID > /tmp/mingtu-agent.pid

# 等索引和 BM25 加载完再开浏览器，否则第一眼看到的是连接失败
for i in \$(seq 1 90); do
  if curl -sS --max-time 2 "http://127.0.0.1:\$PORT/health" >/dev/null 2>&1; then
    note "就绪"
    alert "已就绪，正在打开"
    open "http://127.0.0.1:\$PORT/"
    wait \$SERVER_PID
    exit 0
  fi
  kill -0 \$SERVER_PID 2>/dev/null || { note "服务进程退出"; break; }
  sleep 1
done

note "启动超时"
osascript -e 'display alert "服务没能启动" message "详情见 ~/Library/Logs/mingtu-agent.log" as critical' >/dev/null 2>&1
exit 1
LAUNCH

chmod +x "$APP/Contents/MacOS/launcher"

# 图标：用系统自带的方式生成一个简单的
ICON_SRC="$REPO/.appicon.png"
if command -v sips >/dev/null 2>&1 && [ -f "$ICON_SRC" ]; then
  ICONSET="$REPO/.icon.iconset"; rm -rf "$ICONSET"; mkdir -p "$ICONSET"
  for sz in 16 32 128 256 512; do
    sips -z $sz $sz "$ICON_SRC" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null 2>&1
    sips -z $((sz*2)) $((sz*2)) "$ICON_SRC" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns" 2>/dev/null || true
  rm -rf "$ICONSET"
fi

echo "已生成 $APP"
echo "双击即可运行。首次运行 macOS 可能提示未验证开发者：右键 → 打开。"
