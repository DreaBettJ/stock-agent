# Mini-Agent 股票投资助手 PRD（极简版）

## 1. 产品定位

**核心功能**：T+1 辅助投资智能体，每天收盘后分析市场、发现主线、给出调仓建议

**目标用户**：个人投资者，不懂选股，希望 AI 辅助发现主线并调仓

---

## 2. 系统架构

```
┌─────────────────────────────────────────┐
│           事件触发 (Cron)               │
│  每日复盘 15:00                         │
│  上涨监控 15:05                         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           Agent 核心                    │
│  session_id 管理记忆和操作记录           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           工具层                        │
│  - 行情查询 (AShareQuoteTool)          │
│  - 选股筛选 (AShareScreenTool)         │
│  - 交易记录 (TradeRecordTool)          │
│  - 持仓查询 (PositionTool)             │
└─────────────────────────────────────────┘
```

---

## 3. 核心流程

### 3.1 每日复盘（15:00）

```
1. 查询持仓股票行情
2. 分析今日盈亏
3. 分析市场整体表现
4. 判断是否需要调仓
5. 生成复盘报告
```

### 3.2 上涨监控（15:05）

```
1. 获取今日涨幅前 50
2. Agent 分析主线逻辑
3. 存入长期记忆
4. 给出是否调仓建议
```

---

## 4. 工具清单

| Tool | 功能 | 来源 |
|------|------|------|
| AShareQuoteTool | 股票行情 | 已有 |
| AShareScreenTool | 选股筛选 | 已有 |
| TradeRecordTool | 记录买卖 | 已有 |
| PositionTool | 查询持仓 | 已有 |
| CalculateProfitTool | 盈利计算 | 需新增 |

---

## 5. 模拟交易（可选）

### 5.1 模式说明

| 模式 | 说明 |
|------|------|
| **模拟模式** | 用虚拟资金交易，不涉及真实账户 |
| **实盘模式** | 仅给出建议，人确认后执行 |

### 5.2 模拟交易规则

- 初始虚拟资金：10 万元
- 买入规则：次日开盘价成交
- 卖出规则：次日开盘价成交
- 手续费：万三（模拟）
- 印花税：千一（卖出模拟）

### 5.3 模拟交易工具

```python
class SimulateTradeTool(Tool):
    """模拟交易工具"""
    
    name = "simulate_trade"
    
    async def execute(self, action: str, ticker: str, quantity: int, price: float = None) -> ToolResult:
        """
        模拟买入/卖出
        action: "buy" 或 "sell"
        ticker: 股票代码
        quantity: 数量
        price: 价格（可选，不填则用次日开盘价）
        """
        # 检查资金是否充足
        # 记录模拟交易
        # 更新模拟持仓
```

### 5.4 模拟持仓

```sql
CREATE TABLE sim_positions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    quantity INT,
    avg_cost REAL,        -- 平均成本
    created_at DATETIME,
    updated_at DATETIME
);
```

### 5.5 模拟账户

```sql
CREATE TABLE sim_account (
    id INTEGER PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    initial_capital REAL DEFAULT 100000,  -- 初始资金
    current_cash REAL,                    -- 当前现金
    created_at DATETIME,
    updated_at DATETIME
);
```

### 5.6 模拟交易记录

```sql
CREATE TABLE sim_trades (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,     -- buy/sell
    price REAL,
    quantity INT,
    amount REAL,              -- 成交金额
    fee REAL,                 -- 手续费
    profit REAL,              -- 盈亏（仅卖出）
    timestamp DATETIME,
    created_at DATETIME
);
```

### 5.7 模拟盈利计算

```python
async def calculate_sim_profit(session_id: str) -> dict:
    """计算模拟盈利"""
    # 当前持仓市值
    position_value = sum(quantity * current_price for each position)
    
    # 总资产 = 现金 + 持仓市值
    total = current_cash + position_value
    
    # 盈利 = 总资产 - 初始资金
    profit = total - initial_capital
    profit_rate = profit / initial_capital
    
    return {
        "initial_capital": 100000,
        "current_cash": current_cash,
        "position_value": position_value,
        "total_assets": total,
        "profit": profit,
        "profit_rate": profit_rate
    }
```

---

## 6. 数据存储

### 6.1 记忆表（逻辑删除）

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INT DEFAULT 0,  -- 0普通 1重要 2主线
    is_deleted INT DEFAULT 0,
    created_at DATETIME
);

CREATE INDEX idx_mem_session ON memories(session_id, is_deleted);
```

### 5.2 交易记录表

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,  -- buy/sell
    price REAL,
    quantity INT,
    reason TEXT,
    timestamp DATETIME,
    created_at DATETIME
);
```

### 5.3 日线数据

```sql
CREATE TABLE daily_kline (
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,
    PRIMARY KEY (ticker, date)
);
```

---

## 7. Session 管理

- 外部传入 session_id
- 启动时加载未删除的记忆（is_deleted=0）
- 超过 80k token 时裁剪（逻辑删除 importance=0 的记忆）

---

## 8. Cron 配置

| 任务 | 时间 | 说明 |
|------|------|------|
| 每日复盘 | 0 15 * * 1-5 | 收盘后复盘 |
| 上涨监控 | 5 15 * * 1-5 | 发现主线 |

---

## 9. 实施计划

### Phase 1: 基础能力
- [ ] session_id 传入
- [ ] 记忆模块（存储+裁剪）
- [ ] 盈利计算

### Phase 2: 事件驱动
- [ ] 每日复盘
- [ ] 上涨监控

### Phase 3: 数据
- [ ] 日线同步

---

## 10. 附录

### 提示词示例

**复盘**：
```
【时间】2025-01-01 15:00
【持仓】{positions}
【盈亏】{daily_pnl}
今天已收盘，是否需要调仓？
```

**主线分析**：
```
【时间】2025-01-01 15:05
【涨幅前50】{top_gainers}
请分析主线，判断是否需要调整持仓。
```

---

**版本**：v2.0（极简版）
