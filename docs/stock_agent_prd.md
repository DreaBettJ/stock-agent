# Mini-Agent 股票助手 PRD（极简统一版）

## 1. 产品目标

本项目的最小可用目标（MVP）是：

1. 用一个统一的 SQLite 数据库支撑 `session + memory + kline + sim_trade`。
2. CLI 对话模式本身就是一个 session，和 `session/trade/backtest/event` 走同一套数据模型。
3. 通过事件驱动触发 LLM 决策，并可自动执行模拟交易。
4. 支持历史回测和日常同步 K 线数据。

---

## 2. 分层设计（当前实现）

### 2.1 CLI 适配层

- 文件：`mini_agent/cli.py`
- 职责：解析命令行、打印结果、参数校验、调用应用服务。
- 不再内嵌复杂业务逻辑。

### 2.2 应用服务层

- 文件：`mini_agent/app/decision_service.py`
  - 负责组装 LLM 决策运行时依赖（session manager / kline db / workflow / trade tool）。
  - 提供统一 `run_llm_decision(...)` 接口。
- 文件：`mini_agent/app/sync_service.py`
  - 负责股票列表解析、K 线同步、cron 推荐输出。

### 2.3 领域层

- `mini_agent/session.py`：session 生命周期管理（create/start/stop/finish/list/delete）。
- `mini_agent/backtest.py`：按交易日回放 + 绩效统计。
- `mini_agent/event_broadcaster.py`：向 listening session 广播事件。
- `mini_agent/auto_trading.py`：LLM 复盘提示构建、信号解析、交易执行编排。

### 2.4 基础设施层

- SQLite 数据存储：`/home/lijiang/workspace/Mini-Agent/.agent_memory.db`
- 数据工具：
  - `mini_agent/tools/kline_db_tool.py`
  - `mini_agent/tools/sim_trade_tool.py`
  - `mini_agent/tools/note_tool.py`

---

## 3. 核心实体与约束

### 3.1 Session

```python
session_id: int  # 主键，自增，CLI 统一使用 int
name: str
system_prompt: str
mode: Literal["simulation", "backtest"]
status: Literal["running", "stopped", "finished"]
is_listening: bool
initial_capital: float
current_cash: float
current_date: date | None
event_filter: list[str]
```

### 3.2 统一数据库

- 单一数据库文件：`/home/lijiang/workspace/Mini-Agent/.agent_memory.db`
- 典型表：
  - `sessions`
  - `memories`
  - `daily_kline`
  - `sim_trades`
  - `sim_positions`

说明：旧的多库设计（如 `stock_kline.db`）已收敛为单库。

---

## 4. 核心流程

### 4.1 对话模式（start）

1. 启动 `mini-agent`。
2. 自动创建并 `start` 一个 runtime session（命名 `cli-YYYYmmdd-HHMMSS`）。
3. 对话期间工具调用和记忆写入都绑定该 session。
4. 退出时自动 `stop` 该 session。

### 4.2 事件触发（event）

- `mini-agent event trigger <event_type> --all`
  - 广播给所有 listening session。
- `mini-agent event trigger <event_type> --session <id>`
  - 只触发单 session。
- 事件触发默认走 LLM 决策并尝试执行模拟交易。

### 4.3 模拟交易（trade）

- `buy/sell`：通过 `SimulateTradeTool` 下单并落库。
- `positions`：查看当前仓位。
- `profit`：统计现金、持仓市值、已实现与未实现盈亏。

### 4.4 回测（backtest）

1. 校验 session 和交易日历。
2. 清理该 session 历史模拟持仓与成交（保证可复现）。
3. 按交易日生成 `daily_review` 事件。
4. 每日触发 LLM 决策并可自动执行交易。
5. 汇总收益率、回撤、夏普、胜率等指标。

### 4.5 数据同步（sync）

1. `sync` 先解决 K 线数据可用性问题。
2. 支持指定股票或 `--all` 全市场。
3. 支持 `--cron` 输出定时任务建议（收盘后同步 + 事件触发）。

---

## 5. CLI 规范（统一）

```bash
# Session
mini-agent session create --name "demo" --prompt "..." [--mode simulation|backtest] [--initial-capital 100000]
mini-agent session list
mini-agent session start <id>
mini-agent session stop <id>
mini-agent session delete <id>

# Trade
mini-agent trade buy 600519 100 --session <id> [--date YYYY-MM-DD]
mini-agent trade sell 600519 100 --session <id> [--date YYYY-MM-DD]
mini-agent trade positions --session <id>
mini-agent trade profit --session <id>

# Backtest
mini-agent backtest run --session <id> --start 2024-01-01 --end 2024-12-31
mini-agent backtest result --session <id>

# Event
mini-agent event trigger daily_review --session <id>
mini-agent event trigger daily_review --all
mini-agent event trigger daily_review --session <id>

# Sync
mini-agent sync 600519,000001 --start 2020-01-01
mini-agent sync --all --start 1991-01-01
mini-agent sync --cron
```

---

## 6. 非目标（当前版本不做）

1. 独立 experiment 子系统（多版本提示词自动对比平台）。
2. 双账本或多数据库并存。
3. 与核心流程无关的冗余命令分叉。

---

## 7. 近期迭代清单

1. 统一 memory/session 类型：会话主键保持 `int`，memory 写入路径保持单库。
2. 完善 `event_filter` 与事件类型治理。
3. 增加 `sync` 定时任务落地脚本（systemd/cron 模板）。
4. 回测结果持久化与对比查询（在不引入新子系统前提下）。
