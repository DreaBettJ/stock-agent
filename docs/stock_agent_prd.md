# Big-A-Helper 股票助手 PRD（极简统一版）

## 1. 产品目标

本项目的最小可用目标（MVP）是：

1. 用一个统一的 SQLite 数据库支撑 `session + memory + kline + sim_trade`。
2. CLI 对话模式本身就是一个 session，和 `session/trade/backtest/event` 走同一套数据模型。
3. 通过事件驱动触发 LLM 决策，并可自动执行模拟交易。
4. 支持历史回测和日常同步 K 线数据。
5. 工具链遵循“本地优先（local-first）”，先保证系统可运行，再逐步把关键数据能力内化到本地。

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

### 2.6 数据与工具原则（Local-First）

1. Tool 执行优先使用本地数据（SQLite / 本地缓存 / 本地计算），降低外部网络依赖。
2. 数据按重要性分层，不要求“一次性全部本地化”：
   - P0（关键）：交易执行、持仓、资金、事件、关键记忆、K线基础行情（必须本地可用）。
   - P1（增强）：技术信号、主线/龙头分类、策略评分（优先本地计算，外部仅补充）。
   - P2（可选）：新闻、舆情、题材扩展信息（可外部降级，不阻塞主流程）。
3. 运行策略：先跑通主链路（可观测、可回放、可重试），再逐步替换为本地实现。

### 2.5 自进化治理层（EVO-HIL）

- 模块名：`EVO-HIL`（Evolution with Human-in-the-Loop）
- 目标：对自动运行后的行为做“事后发现”，沉淀为可注入系统提示词的执行 `use case`。
- 核心原则：
  1. 低耦合：独立服务模块，不侵入交易执行核心链路。
  2. 只产出提示词资产（use case prompt），不直接改交易规则代码。
  3. LLM 智能总结优先，规则模板兜底。

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
  - `event_logs`
  - `critical_memories`
  - `evolution_use_cases`
  - `reasoning_trace_events`（由 `agent_intercept_s*.jsonl` 同步得到的结构化推理链）

说明：旧的多库设计（如 `stock_kline.db`）已收敛为单库。

---

## 4. 核心流程

### 4.1 对话模式（start）

1. 启动 `big-a-helper`。
2. 自动创建并 `start` 一个 runtime session（命名 `cli-YYYYmmdd-HHMMSS`）。
3. 对话期间工具调用和记忆写入都绑定该 session。
4. 退出时自动 `stop` 该 session。

### 4.2 事件触发（event）

- `big-a-helper event trigger <event_type> --all`
  - 广播给所有 listening session。
- `big-a-helper event trigger <event_type> --session <id>`
  - 只触发单 session。
- 事件触发默认走 LLM 决策并尝试执行模拟交易。
- `big-a-helper event trigger <event_type> --session <id> --debug`
  - 调试模式：自动为事件生成唯一 `event_id`，跳过重复事件幂等拦截（用于重放/排查）。

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
4. `sync --all` 在未显式传 `--start/--end` 时默认执行“当日增量同步”。
5. `sync --all` 依赖 `config.yaml -> tools.tushare_token`；未配置时会直接报错并提示配置路径。

### 4.6 自进化数据流（EVO-HIL）

1. `scan` 阶段（发现问题/规律）
   - 输入：`event_logs`（当前）及后续可扩展的 `sim_trades/sim_positions/critical_memories`。
   - 处理：默认 LLM 生成 use case 提示词；失败时规则模板兜底。
   - 输出：`evolution_use_cases`（`enabled=1`）。

2. `inject` 阶段（运行时注入）
   - 启动 `big-a-helper` 时加载启用的 use case，拼接到系统提示词。
   - 目标是让 Agent 直接吸收经验规律，减少重复犯错。

3. `manage` 阶段（轻量治理）
   - 通过 `enable/disable` 开关 use case，快速控制注入内容。
   - 不直接修改交易引擎代码。

---

## 5. CLI 规范（统一）

```bash
# Session
big-a-helper session create --name "demo" --prompt "..." [--mode simulation|backtest] [--initial-capital 100000]
big-a-helper session list
big-a-helper session start <id>
big-a-helper session stop <id>
big-a-helper session delete <id>

# Trade
big-a-helper trade buy 600519 100 --session <id> [--date YYYY-MM-DD]
big-a-helper trade sell 600519 100 --session <id> [--date YYYY-MM-DD]
big-a-helper trade positions --session <id>
big-a-helper trade profit --session <id>

# Backtest
big-a-helper backtest run --session <id> --start 2024-01-01 --end 2024-12-31
big-a-helper backtest result --session <id>

# Event
big-a-helper event trigger daily_review --session <id>
big-a-helper event trigger daily_review --all
big-a-helper event trigger daily_review --session <id> --debug   # 调试重放，跳过重复事件拦截

# Sync
big-a-helper sync 600519,000001 --start 2020-01-01
big-a-helper sync --all                      # 默认当日增量
big-a-helper sync --all --start 2026-02-24 --end 2026-02-24
big-a-helper sync --cron

# Evolve (EVO-HIL)
big-a-helper evolve scan --limit 200              # 默认 LLM 生成提案
big-a-helper evolve scan --no-llm --limit 200     # 规则模板兜底
big-a-helper evolve list --enabled on
big-a-helper evolve enable <use_case_id>
big-a-helper evolve disable <use_case_id>
big-a-helper evolve prompt --limit 12
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
5. EVO-HIL 接入 `alpha` 模式（基于收益归因生成策略型 use case）。
6. 为 `evolution_use_cases` 增加更细粒度 source 映射（use_case -> event_log/trade ids）。
7. Local-First 收敛计划：
   - 第一步：把每日复盘关键信号全部切到本地 K 线计算（不依赖外网）。
   - 第二步：为外部数据增加本地缓存与回退策略（失败不阻断交易主流程）。
   - 第三步：逐步减少非关键外部 tool 在实时链路中的占比。
