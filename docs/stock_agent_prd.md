# Mini-Agent 股票投资助手 PRD（实验平台版）

## 1. 产品定位

**核心功能**：支持多策略并行实验的 AI 投资助手平台

**核心场景**：
1. **模拟交易** - 虚拟资金实时运行
2. **回测** - 历史数据验证策略
3. **提示词实验** - 批量对比不同提示词效果

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      实验平台核心                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │  Session 管理   │  │  事件广播       │  │  提示词管理   │ │
│  │  - 创建/启动    │  │  - 广播到多个   │  │  - 系统提示词 │ │
│  │  - 监听状态     │  │  - 监听中的session  │  - 版本管理   │ │
│  │  - 批量操作     │  │  - 事件过滤     │  - 效果对比   │ │
│  └─────────────────┘  └─────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      实验类型                                   │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │  实时模拟交易    │  │  历史回测        │                  │
│  │  - 虚拟资金 10万  │  │  - 2020-2024    │                  │
│  │  - 每日复盘      │  │  - 事件模拟      │                  │
│  │  - 事件驱动      │  │  - 绩效报告      │                  │
│  └──────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心概念

### 3.1 Session（实验会话）

```python
class ExperimentSession:
    session_id: str           # 唯一标识
    name: str                # 实验名称，如 "均线策略v1"
    system_prompt: str       # 系统提示词（实验变量）
    mode: str                # "simulation" | "backtest"
    
    # 模拟交易专用
    initial_capital: float  # 初始资金
    current_cash: float     # 当前现金
    positions: dict         # 持仓
    
    # 状态
    status: str             # "running" | "stopped" | "finished"
    is_listening: bool      # 是否监听事件
    
    # 时间（回测专用）
    backtest_start: date    # 回测开始日期
    backtest_end: date      # 回测结束日期
    current_date: date      # 当前回测日期
```

### 3.2 提示词版本管理

```python
class PromptVersion:
    version_id: str          # v1, v2, v3...
    content: str             # 提示词内容
    created_at: datetime
    description: str         # "增加止损规则"
    
# 实验对比
class Experiment:
    experiment_id: str
    prompt_versions: list[PromptVersion]  # 对比的提示词版本
    results: dict           # 每个版本的结果
```

---

## 4. Session 管理

### 4.1 CLI 命令

```bash
# 创建实验 session
mini-agent create --name "均线v1" --prompt "你的提示词"
mini-agent create --name "趋势v1" --prompt-file prompts/trend.txt

# 启动/停止 session
mini-agent start session_id
mini-agent stop session_id

# 查看所有 session
mini-agent list
# 输出：
# session_id    name      mode      status    listening    profit
# abc123        均线v1    simulation running   true         +5.2%
# def456        趋势v1    backtest  finished  false        +12.3%

# 批量操作
mini-agent start --all          # 启动所有
mini-agent stop --all          # 停止所有

# 对比实验
mini-agent compare exp_id1 exp_id2
```

### 4.2 Session 状态

```python
class SessionManager:
    """Session 管理器"""
    
    def create_session(self, name: str, system_prompt: str, mode: str, **kwargs) -> str:
        """创建新 session"""
    
    def start_session(self, session_id: str):
        """启动 session（开始监听事件）"""
    
    def stop_session(self, session_id: str):
        """停止 session"""
    
    def list_sessions(self) -> list[Session]:
        """列出所有 session"""
    
    def get_session(self, session_id: str) -> Session:
        """获取 session 详情"""
    
    def delete_session(self, session_id: str):
        """删除 session"""
    
    def broadcast_event(self, event: dict):
        """广播事件到所有 listening 的 session"""
```

---

## 5. 事件广播

### 5.1 事件类型

| 事件 | 触发时间 | 说明 |
|------|----------|------|
| 每日复盘 | 15:00 | 收盘后复盘 |
| 上涨监控 | 15:05 | 主线发现 |
| 历史复盘 | 回测中 | 回测专用 |

### 5.2 广播机制

```python
class EventBroadcaster:
    """事件广播器"""
    
    def broadcast(self, event: dict):
        """广播事件到所有 listening=True 的 session"""
        for session in self.get_listening_sessions():
            # 异步触发每个 session 的 Agent
            asyncio.create_task(self.trigger_session(session, event))
    
    def trigger_session(self, session: Session, event: dict):
        """触发单个 session"""
        # 构建 prompt
        prompt = self.build_event_prompt(session, event)
        # 调用 Agent
        result = await session.agent.run(prompt)
        # 记录结果
        self.record_result(session, event, result)
```

### 5.3 事件过滤

```python
# 某些 session 可能只关注特定事件
session.event_filter = ["daily_review"]  # 只接收复盘事件
```

---

## 6. 模拟交易

### 6.1 规则

- 初始资金：10 万元（可配置）
- 成交价格：次日开盘价
- 手续费：万三
- 印花税：千一（卖出）

### 6.2 工具

```python
class SimulateTradeTool(Tool):
    """模拟交易"""
    
    async def execute(self, action: str, ticker: str, quantity: int) -> ToolResult:
        # 查询次日开盘价
        price = await self.get_next_open_price(ticker)
        
        # 计算成本
        if action == "buy":
            cost = price * quantity * 1.001  # 万三佣金
            # 检查资金
            if self.session.current_cash < cost:
                return ToolResult(success=False, error="资金不足")
        else:
            revenue = price * quantity * 0.998  # 扣佣金+印花税
            # 检查持仓
            # ...
```

---

## 7. 回测系统

### 7.1 回测流程

```
1. 创建回测 session
2. 设置回测时间范围（2020-01-01 ~ 2024-12-31）
3. 加载历史数据
4. 按日期遍历：
   a. 触发复盘事件
   b. Agent 决策
   c. 记录交易
   d. 更新持仓
5. 生成绩效报告
```

### 7.2 历史事件模拟

```python
class HistoricalEvent模拟:
    """用历史数据模拟事件"""
    
    def generate_daily_review_event(self, date: str) -> dict:
        """生成某日的复盘事件"""
        return {
            "type": "daily_review",
            "date": date,
            "positions": self.get_historical_positions(date),
            "market": self.get_market_summary(date),
            "top_gainers": self.get_top_gainers(date)
        }
    
    def run_backtest(self, session: Session, start: str, end: str):
        """运行回测"""
        dates = self.get_trading_days(start, end)
        for date in dates:
            event = self.generate_daily_review_event(date)
            session.current_date = date
            # 触发 Agent
            result = session.agent.run(event)
            # 执行模拟交易
            self.execute_pending_trades(session, date)
```

### 7.3 绩效指标

| 指标 | 说明 |
|------|------|
| 总收益率 | (最终资产 - 初始资金) / 初始资金 |
| 年化收益率 | 年化后的收益 |
| 夏普比率 | 风险调整收益 |
| 最大回撤 | 最大亏损幅度 |
| 胜率 | 盈利交易占比 |
| 盈利因子 | 总盈利 / 总亏损 |

---

## 8. 提示词实验

### 8.1 对比实验

```bash
# 创建对比实验
mini-agent experiment create \
    --name "提示词对比实验" \
    --prompt-v1 "你是均线策略..." \
    --prompt-v2 "你是趋势策略..." \
    --mode backtest \
    --start 2023-01-01 \
    --end 2024-12-31

# 查看结果
mini-agent experiment result exp_001
# 输出：
# Version    TotalReturn    Sharpe    MaxDD    WinRate
# v1         +15%          1.2       -8%      55%
# v2         +22%          1.8       -6%      62%
```

### 8.2 提示词模板变量

```python
# 提示词中可用的变量
prompt_template = """
【时间】{{date}}
【持仓】{{positions}}
【资金】{{cash}}
【市场】{{market_summary}}
{{custom_rules}}
"""

# 变量由系统自动填充
variables = {
    "date": "2024-01-01",
    "positions": {...},
    "cash": 100000,
    "market_summary": {...},
    "custom_rules": "..."  # 用户自定义规则
}
```

---

## 9. 数据存储

### 9.1 Session 表

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    mode TEXT NOT NULL,          -- simulation / backtest
    initial_capital REAL DEFAULT 100000,
    current_cash REAL,
    status TEXT DEFAULT 'stopped',  -- running / stopped / finished
    is_listening INTEGER DEFAULT 0,
    backtest_start DATE,
    backtest_end DATE,
    current_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);
```

### 9.2 提示词版本表

```sql
CREATE TABLE prompt_versions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    version TEXT NOT NULL,
    content TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 9.3 实验结果表

```sql
CREATE TABLE experiment_results (
    id INTEGER PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    total_trades INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 9.4 模拟交易表

```sql
CREATE TABLE sim_trades (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    price REAL,
    quantity INT,
    amount REAL,
    fee REAL,
    profit REAL,
    trade_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sim_positions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    quantity INT,
    avg_cost REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);
```

### 9.5 记忆表

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INT DEFAULT 0,
    is_deleted INT DEFAULT 0,
    created_at DATETIME
);

CREATE INDEX idx_mem_session ON memories(session_id, is_deleted);
```

---

## 10. CLI 命令汇总

```bash
# Session 管理
mini-agent session create --name "实验名" --prompt "提示词" [--mode simulation|backtest]
mini-agent session start <session_id>
mini-agent session stop <session_id>
mini-agent session list
mini-agent session delete <session_id>
mini-agent session status <session_id>

# 模拟交易
mini-agent trade buy <ticker> <quantity> --session <id>
mini-agent trade sell <ticker> <quantity> --session <id>
mini-agent trade positions --session <id>
mini-agent trade profit --session <id>

# 回测
mini-agent backtest run --session <id> --start 2023-01-01 --end 2024-12-31
mini-agent backtest result <session_id>

# 实验
mini-agent experiment create --name "对比实验" --prompt-v1 "..." --prompt-v2 "..."
mini-agent experiment list
mini-agent experiment result <experiment_id>

# 事件
mini-agent event trigger daily_review --all
mini-agent event trigger daily_review --session <id>
```

---

## 11. 实施计划

### Phase 1: Session 管理
- [ ] Session 数据模型
- [ ] CLI 命令
- [ ] 状态管理

### Phase 2: 事件广播
- [ ] EventBroadcaster
- [ ] 监听机制
- [ ] 事件过滤

### Phase 3: 模拟交易
- [ ] SimulateTradeTool
- [ ] 持仓管理
- [ ] 盈利计算

### Phase 4: 回测
- [ ] HistoricalEvent模拟
- [ ] 回测引擎
- [ ] 绩效报告

### Phase 5: 实验对比
- [ ] 提示词版本管理
- [ ] 对比实验
- [ ] 结果可视化

---

**版本**：v3.0（实验平台版）
