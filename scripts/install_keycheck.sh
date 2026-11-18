#!/bin/sh
# Installs an hourly macOS LaunchAgent that runs src/key_status.py (records when the Jev key stops working).
# Run once yourself:  sh scripts/install_keycheck.sh      Remove:  sh scripts/install_keycheck.sh remove
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
P="$HOME/Library/LaunchAgents/com.wset-pipeline.keycheck.plist"
if [ "$1" = "remove" ]; then launchctl bootout "gui/$(id -u)/com.wset-pipeline.keycheck" 2>/dev/null || true; rm -f "$P"; echo removed; exit 0; fi
cat > "$P" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.wset-pipeline.keycheck</string>
  <key>ProgramArguments</key><array><string>$R/.venv/bin/python</string><string>-W</string><string>ignore</string><string>$R/src/key_status.py</string></array>
  <key>WorkingDirectory</key><string>$R</string>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$R/outputs/key_check.log</string>
  <key>StandardErrorPath</key><string>$R/outputs/key_check.log</string>
</dict></plist>
PLIST
launchctl bootout "gui/$(id -u)/com.wset-pipeline.keycheck" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$P"
echo "installed: hourly key check (log: $R/outputs/key_check.log)"
