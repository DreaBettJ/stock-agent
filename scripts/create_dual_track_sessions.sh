#!/usr/bin/env bash
set -euo pipefail

# Create two comparable sessions for boundary exploration:
# 1) Strict: strategy template + strong process discipline.
# 2) Loose : only hard risk constraints, model chooses signal logic.
#
# Usage:
#   bash scripts/create_dual_track_sessions.sh
#   bash scripts/create_dual_track_sessions.sh /path/to/workspace

WORKSPACE_DIR="${1:-$(pwd)}"
CMD=(uv run python -m mini_agent.cli --workspace "$WORKSPACE_DIR" session create)

STRICT_PROMPT=$(cat <<'EOF'
你是A股交易研究助手（Strict轨道）。

目标：
在“Agent提醒 + 手动执行”模式下，输出可执行、可复盘、可审计的建议。

硬约束（必须遵守）：
1) 不自动下单；仅给建议与风险提示。
2) 必须给出数据时间（as_of_date）和数据口径（realtime/daily）。
3) 单票仓位建议不超过15%，组合仓位建议不超过70%。
4) 止损建议默认-6%，若调整必须说明理由。
5) 若关键信息不足，先声明“不足以判断”，再给最小化动作（观望/减仓）。

流程约束（必须按顺序）：
1) 先判断市场状态（bull/sideways/bear）。
2) 再给候选方向与触发条件。
3) 最后给动作建议（买入/减仓/观望）与失效条件。

输出模板（必须）：
- 市场状态：
- 关键依据：
- 建议动作：
- 风险位（止损/仓位）：
- 失效条件：
- 数据时间：
EOF
)

LOOSE_PROMPT=$(cat <<'EOF'
你是A股交易研究助手（Loose轨道）。

目标：
在“Agent提醒 + 手动执行”模式下，最大化你的自主判断能力，但必须守住风险底线。

仅保留硬约束（不可违反）：
1) 不自动下单；只输出建议。
2) 单票仓位建议 <= 15%，组合仓位建议 <= 70%。
3) 止损建议默认-6%（可调整，但要解释）。
4) 必须给出数据时间（as_of_date）与失效条件。

其余内容你可自主决定：
- 市场划分方法
- 信号组合方法
- 候选排序方法
- 表达方式

输出尽量简洁，但必须可执行、可复盘。
EOF
)

echo "Creating STRICT session..."
"${CMD[@]}" \
  --name "ab-strict-$(date +%m%d-%H%M%S)" \
  --prompt "$STRICT_PROMPT" \
  --mode simulation \
  --initial-capital 1000000 \
  --risk-preference low \
  --max-single-loss-pct 1.5 \
  --single-position-cap-pct 15 \
  --stop-loss-pct 6 \
  --take-profit-pct 12 \
  --investment-horizon "中线" \
  --trade-notice-enabled \
  --event-filter daily_review

echo "Creating LOOSE session..."
"${CMD[@]}" \
  --name "ab-loose-$(date +%m%d-%H%M%S)" \
  --prompt "$LOOSE_PROMPT" \
  --mode simulation \
  --initial-capital 1000000 \
  --risk-preference low \
  --max-single-loss-pct 1.5 \
  --single-position-cap-pct 15 \
  --stop-loss-pct 6 \
  --take-profit-pct 12 \
  --investment-horizon "中线" \
  --trade-notice-enabled \
  --event-filter daily_review

echo
echo "Done. List sessions:"
echo "  uv run python -m mini_agent.cli --workspace \"$WORKSPACE_DIR\" session list"
echo
echo "Suggested next step:"
echo "  Trigger the same event on both sessions and compare outputs:"
echo "  uv run python -m mini_agent.cli --workspace \"$WORKSPACE_DIR\" event trigger daily_review --session <strict_id>"
echo "  uv run python -m mini_agent.cli --workspace \"$WORKSPACE_DIR\" event trigger daily_review --session <loose_id>"
