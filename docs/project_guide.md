# Mini-Agent 项目指南

## 1. 一句话理解项目

Mini-Agent 是一个“事件驱动 + LLM 决策 + 模拟交易 + 历史回测”的股票策略实验 CLI。

---

## 2. 项目目录（你最常用的）

- `mini_agent/cli.py`：命令入口与交互模式。
- `mini_agent/app/decision_service.py`：LLM 决策应用服务。
- `mini_agent/app/sync_service.py`：K 线同步应用服务。
- `mini_agent/session.py`：session 管理（核心状态机）。
- `mini_agent/backtest.py`：回测引擎与绩效统计。
- `mini_agent/event_broadcaster.py`：事件广播。
- `mini_agent/tools/sim_trade_tool.py`：模拟交易执行。
- `mini_agent/tools/kline_db_tool.py`：K 线读写查询。
- `mini_agent/tools/note_tool.py`：记忆读写。
- `docs/stock_agent_prd.md`：设计说明（已与当前实现对齐）。

---

## 3. 分层与职责

1. CLI 层：只管参数、输出、调用。
2. App 层：用例编排（决策、同步）。
3. Domain 层：session/backtest/event/trade 规则。
4. Infra 层：SQLite + 第三方数据源（AkShare）。

这个分层的目标是：避免 `cli.py` 再次变成“全能巨石文件”。

---

## 4. 数据存储（统一）

唯一数据库：

- `/home/lijiang/workspace/Mini-Agent/.agent_memory.db`

主要表：

- `sessions`
- `memories`
- `daily_kline`
- `sim_trades`
- `sim_positions`

`.gitignore` 已忽略 `.agent_memory.db`。

---

## 5. 日常命令手册

### 5.1 初始化

```bash
uv sync
```

### 5.2 启动交互会话（CLI 本身就是一个 session）

```bash
mini-agent
```

### 5.3 创建实验 session

```bash
mini-agent session create --name test --prompt "你是交易助手" --mode backtest --initial-capital 100000
mini-agent session list
mini-agent session start 1
```

### 5.4 同步 K 线

```bash
mini-agent sync 600519,000001 --start 2020-01-01
mini-agent sync --all --start 1991-01-01
mini-agent sync --cron
```

### 5.5 事件触发

```bash
mini-agent event trigger daily_review --session 1
mini-agent event trigger daily_review --all
mini-agent event trigger daily_review --session 1 --auto
```

### 5.6 模拟交易

```bash
mini-agent trade buy 600519 100 --session 1 --date 2024-06-03
mini-agent trade sell 600519 100 --session 1 --date 2024-06-05
mini-agent trade positions --session 1
mini-agent trade profit --session 1
```

### 5.7 回测

```bash
mini-agent backtest run --session 1 --start 2024-01-01 --end 2024-12-31
mini-agent backtest result --session 1
```

---

## 6. 关键运行逻辑速记

- `start`（交互模式）
  - 自动创建 session 并 start。
- `backtest`
  - 按交易日循环 -> 触发事件 -> LLM 决策 -> 执行模拟交易 -> 统计绩效。
- `trade`
  - 直接对 session 资金与仓位做买卖落库。
- `event`
  - 广播或单点触发；`--auto` 会启用 LLM 决策与下单。
- `sync`
  - 将 A 股日线写入 `daily_kline`，供 `trade/backtest/event` 共用。

---

## 7. 最小可用运行顺序（推荐）

1. `mini-agent sync --all --start 2018-01-01`
2. `mini-agent session create ... --mode backtest`
3. `mini-agent backtest run --session <id> --start ... --end ...`
4. 看结果后调整 prompt，再创建新 session 对比。

---

## 8. 开发者检查清单

```bash
uv run python -m py_compile mini_agent/cli.py mini_agent/app/decision_service.py mini_agent/app/sync_service.py
uv run pytest tests/test_stock_core_modules.py tests/test_backtest.py tests/test_note_tool.py -q
```

如果要扩展功能，优先在 `mini_agent/app/` 新增服务，不要先改 `cli.py`。
