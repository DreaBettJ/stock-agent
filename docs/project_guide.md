# Big-A-Helper 项目指南

## 1. 一句话理解项目

Big-A-Helper 是一个“事件驱动 + LLM 决策 + 模拟交易 + 历史回测”的股票策略实验 CLI。

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
big-a-helper
```

### 5.3 创建实验 session

```bash
big-a-helper session create --name test --prompt "你是交易助手" --mode backtest --initial-capital 100000
big-a-helper session list
big-a-helper session start 1
```

### 5.4 同步 K 线

```bash
big-a-helper sync 600519,000001 --start 2020-01-01
big-a-helper sync --all   # 默认当日增量同步（需 config.yaml 配置 tools.tushare_token）
big-a-helper sync --cron
```

### 5.4.1 配置 cron（每日自动运行）

推荐两种方式：

1. 自动安装（推荐）

```bash
big-a-helper sync --install-cron
crontab -l
```

会写入两条工作日任务：

- `16:00`：同步当日行情（`big-a-helper sync --all --start ...`）
- `16:05`：触发每日复盘事件（`big-a-helper event trigger daily_review --all`）

2. 手工安装（自定义时间/命令时使用）

```bash
crontab -e
```

加入（示例，按你的项目绝对路径修改）：

```cron
0 16 * * 1-5 cd /home/lijiang/workspace/Mini-Agent && big-a-helper sync --all
5 16 * * 1-5 cd /home/lijiang/workspace/Mini-Agent && big-a-helper event trigger daily_review --all
```

验收与排查：

```bash
crontab -l
big-a-helper session list   # 确认有在线 session，否则 --all 可能没有目标
```

注意：

- `event trigger --all` 只会投递给“在线”session（有运行中的 CLI runtime）。
- 生产建议用绝对路径命令（如 `uv run python -m mini_agent.cli ...`）避免 PATH 差异导致 cron 找不到 `big-a-helper`。

### 5.5 事件触发

```bash
big-a-helper event trigger daily_review --session 1
big-a-helper event trigger daily_review --all
big-a-helper event trigger daily_review --session 1 --debug  # 调试模式：强制唯一 event_id，跳过重复拦截
```

### 5.6 模拟交易

```bash
big-a-helper trade buy 600519 100 --session 1 --date 2024-06-03
big-a-helper trade sell 600519 100 --session 1 --date 2024-06-05
big-a-helper trade positions --session 1
big-a-helper trade profit --session 1
```

### 5.7 回测

```bash
big-a-helper backtest run --session 1 --start 2024-01-01 --end 2024-12-31
big-a-helper backtest run --session 1 --start 2024-01-01 --end 2024-12-31 --gate
big-a-helper backtest result --session 1
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
  - 广播或单点触发；默认会启用 LLM 决策与模拟下单。
- `sync`
  - 将 A 股日线写入 `daily_kline`，供 `trade/backtest/event` 共用。

---

## 6.1 决策协议（JSON 优先）

- `daily_review` 要求 LLM 优先输出 JSON：
  - `decision`: `trade | hold`
  - `actions[]`: `action/ticker/quantity/trade_date/reason`
- 系统兼容旧协议（`买入:代码,数量`、`不操作`），但建议只用 JSON。
- 自动执行时会按 `actions[]` 顺序逐条落单，结果写入执行结果列表。

---

## 6.2 数据自愈

- 决策前先检查目标交易日在 `daily_kline` 是否有数据。
- 若缺失且配置了 `tools.tushare_token`，系统会自动尝试单日批量同步。
- 若仍无数据，`daily_review` 回退到最近可用交易日。

---

## 6.3 关键记忆落库规则

- 会记录：`buy/sell/failed/planned` 相关操作。
- 默认不记录：`hold/no_trade`，避免噪音污染。
- 聊天模式下 agent 直接调用 `simulate_trade` 的结果也会写入 `critical_memories`。

---

## 6.4 回测门禁

- `--gate` 开启后会检查：
  - `--gate-min-trades`
  - `--gate-min-win-rate`
  - `--gate-max-drawdown`
  - `--gate-min-profit-factor`
- 门禁失败退出码为 `2`，可接 CI。

---

## 7. 最小可用运行顺序（推荐）

1. `big-a-helper sync --all --start 2018-01-01`
2. `big-a-helper session create ... --mode backtest`
3. `big-a-helper backtest run --session <id> --start ... --end ...`
4. 看结果后调整 prompt，再创建新 session 对比。

---

## 8. 开发者检查清单

```bash
uv run python -m py_compile mini_agent/cli.py mini_agent/app/decision_service.py mini_agent/app/sync_service.py
uv run pytest tests/test_stock_core_modules.py tests/test_backtest.py tests/test_note_tool.py -q
```

如果要扩展功能，优先在 `mini_agent/app/` 新增服务，不要先改 `cli.py`。

---

## 9. AI能力边界实验（Strict vs Loose）

如果你在做“约束强度 vs AI自主性”的边界测试，直接参考：

- `docs/dual_track_experiment.md`
- `scripts/create_dual_track_sessions.sh`

---

## 10. 批量策略后台实验（2~4周）

如果你想“遍历策略模板 + 后台常驻运行 + 周期观察结果”，使用：

- `scripts/strategy_lab_runner.sh`

### 10.1 一键创建并后台启动（tmux）

```bash
cd /home/lijiang/workspace/Mini-Agent
bash scripts/strategy_lab_runner.sh up
```

该命令会：

1. 从 `docs/strategy_templates.md` 自动提取所有模板编号
2. 为每个模板创建一个 simulation session（统一风险参数）
3. 将所有 session 以 `tmux` 多窗口方式后台运行
4. 记录映射到 `.strategy_lab_sessions`（session_id / template_id / session_name）

### 10.2 日常触发（建议工作日）

```bash
bash scripts/strategy_lab_runner.sh trigger
```

该命令等价于：

```bash
uv run python -m mini_agent.cli event trigger daily_review --all
```

### 10.3 查看状态

```bash
bash scripts/strategy_lab_runner.sh status
```

### 10.4 停止后台

```bash
bash scripts/strategy_lab_runner.sh stop
```

### 10.5 2~4周后对比建议

按 `.strategy_lab_sessions` 中的 session_id，逐个查看：

```bash
uv run python -m mini_agent.cli trade profit --session <session_id>
uv run python -m mini_agent.cli trade action-logs --session <session_id> --limit 50
```

重点比较：

- 最大回撤是否可接受
- 胜率和盈亏比是否稳定
- 信号是否可执行（是否明确到动作/仓位/失效条件）
