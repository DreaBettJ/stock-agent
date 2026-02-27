#!/usr/bin/env bash
set -euo pipefail

# Batch strategy lab:
# - create sessions for all strategy templates
# - run all sessions in background via tmux
# - trigger daily_review for all online sessions
#
# Usage:
#   bash scripts/strategy_lab_runner.sh up
#   bash scripts/strategy_lab_runner.sh create
#   bash scripts/strategy_lab_runner.sh start
#   bash scripts/strategy_lab_runner.sh trigger
#   bash scripts/strategy_lab_runner.sh status
#   bash scripts/strategy_lab_runner.sh stop
#
# Env overrides:
#   WORKSPACE_DIR=/home/lijiang/workspace/Mini-Agent
#   TMUX_SESSION=strategy-lab
#   SESSION_NAME_PREFIX=lab
#   STRATEGY_DOC=docs/strategy_templates.md

ACTION="${1:-up}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
TMUX_SESSION="${TMUX_SESSION:-strategy-lab}"
SESSION_NAME_PREFIX="${SESSION_NAME_PREFIX:-lab}"
STRATEGY_DOC="${STRATEGY_DOC:-$WORKSPACE_DIR/docs/strategy_templates.md}"
STATE_FILE="$WORKSPACE_DIR/.strategy_lab_sessions"

CLI=(uv run python -m mini_agent.cli --workspace "$WORKSPACE_DIR")

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "❌ Missing command: $1" >&2
    exit 1
  fi
}

extract_template_ids() {
  awk '/^## [0-9]+\./ { gsub(/^## /, "", $0); split($0, a, "."); print a[1] }' "$STRATEGY_DOC"
}

create_sessions() {
  : > "$STATE_FILE"
  local ts
  ts="$(date +%Y%m%d-%H%M%S)"
  mapfile -t template_ids < <(extract_template_ids)
  if [ "${#template_ids[@]}" -eq 0 ]; then
    echo "❌ No template ids found in $STRATEGY_DOC" >&2
    exit 1
  fi

  echo "Creating sessions from strategy templates: ${template_ids[*]}"
  for tid in "${template_ids[@]}"; do
    local name output sid
    name="${SESSION_NAME_PREFIX}-s${tid}-${ts}"
    output="$("${CLI[@]}" session create \
      --name "$name" \
      --template "$tid" \
      --mode simulation \
      --initial-capital 1000000 \
      --risk-preference low \
      --max-single-loss-pct 1.5 \
      --single-position-cap-pct 15 \
      --stop-loss-pct 6 \
      --take-profit-pct 12 \
      --investment-horizon "中线" \
      --trade-notice-enabled \
      --event-filter daily_review)"
    sid="$(printf "%s\n" "$output" | awk '/^session_id:/ {print $2}')"
    if [ -z "${sid:-}" ]; then
      echo "❌ Failed to parse session_id for template $tid" >&2
      printf "%s\n" "$output" >&2
      exit 1
    fi
    printf "%s\t%s\t%s\n" "$sid" "$tid" "$name" >> "$STATE_FILE"
    echo "✅ created: template=$tid session_id=$sid name=$name"
  done
  echo "Saved session map to $STATE_FILE"
}

start_background() {
  require_cmd tmux
  if [ ! -s "$STATE_FILE" ]; then
    echo "❌ State file missing/empty: $STATE_FILE. Run create first." >&2
    exit 1
  fi

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "⚠️ tmux session already exists: $TMUX_SESSION"
    echo "   skip start; use status/stop as needed."
    return
  fi

  local first=1
  while IFS=$'\t' read -r sid tid name; do
    [ -z "${sid:-}" ] && continue
    local cmd
    cmd="cd \"$WORKSPACE_DIR\" && uv run python -m mini_agent.cli --workspace \"$WORKSPACE_DIR\" --session-id $sid"
    if [ "$first" -eq 1 ]; then
      tmux new-session -d -s "$TMUX_SESSION" -n "s${tid}_${sid}" "$cmd"
      first=0
    else
      tmux new-window -t "$TMUX_SESSION" -n "s${tid}_${sid}" "$cmd"
    fi
    echo "▶ started session_id=$sid (template=$tid)"
  done < "$STATE_FILE"

  echo "✅ tmux background started: $TMUX_SESSION"
  echo "   attach: tmux attach -t $TMUX_SESSION"
}

trigger_all() {
  "${CLI[@]}" event trigger daily_review --all
}

status_all() {
  echo "== Session List =="
  "${CLI[@]}" session list || true
  echo
  echo "== tmux =="
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux list-windows -t "$TMUX_SESSION"
  else
    echo "tmux session not running: $TMUX_SESSION"
  fi
  echo
  echo "== State File =="
  if [ -f "$STATE_FILE" ]; then
    cat "$STATE_FILE"
  else
    echo "state file not found"
  fi
}

stop_background() {
  require_cmd tmux
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux kill-session -t "$TMUX_SESSION"
    echo "✅ stopped tmux session: $TMUX_SESSION"
  else
    echo "tmux session not found: $TMUX_SESSION"
  fi
}

case "$ACTION" in
  up)
    create_sessions
    start_background
    ;;
  create)
    create_sessions
    ;;
  start)
    start_background
    ;;
  trigger)
    trigger_all
    ;;
  status)
    status_all
    ;;
  stop)
    stop_background
    ;;
  *)
    echo "Usage: $0 {up|create|start|trigger|status|stop}" >&2
    exit 1
    ;;
esac
