# Mini-Agent 股票投资助手 PRD

## 1. 产品定位

**产品名称**：Mini-Agent 股票投资助手

**核心功能**：T+1 辅助投资智能体，每天收盘后分析市场，给出买入/卖出建议，人工确认后执行

**目标用户**：个人投资者，希望通过 AI 辅助分析市场、做出更理性的投资决策

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      事件触发层                              │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │  每日复盘   │  │  上涨监控   │                          │
│  │  (15:00)   │  │  (15:05)   │                          │
│  └──────┬──────┘  └──────┬──────┘                          │
└─────────┼────────────────┼──────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent 核心引擎                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Session 管理 (session_id)                           │   │
│  │  - 对话历史                                          │   │
│  │  - 记忆存储                                          │   │
│  │  - 操作记录                                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      工具层                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ 股票查询工具 │ │ 交易记录工具 │ │  数据同步    │        │
│  │ - 行情      │ │ - 买入/卖出  │ │  - 日线数据 │        │
│  │ - 基本面    │ │ - 持仓查询   │ │  - 基本面   │        │
│  │ - 选股      │ │ - 历史记录   │ │  - 北向资金 │        │
│  │              │ │ - 盈利计算   │ │              │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心功能

### 3.1 Session 管理

**功能**：通过 session_id 复用历史记忆和操作记录

**设计**：

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | TEXT PRIMARY KEY | 会话唯一标识 |
| create_time | DATETIME | 创建时间 |
| last_active | DATETIME | 最后活跃时间 |
| memory_summary | TEXT | 记忆摘要（裁剪后） |

**实现**：
- Agent 启动时接受外部传入 session_id
- session_id 存入上下文，每次请求携带
- 支持跨session查询历史操作

### 3.2 事件驱动机制

> 注：暂不包含新闻监听（散户获取新闻存在信息差，且T+1模式不需要实时新闻）

#### 3.2.1 每日复盘事件

**触发条件**：每天 15:00（收盘后）自动触发

**处理流程**：
```
1. 获取今日行情数据
2. 分析持仓股票表现
3. 分析市场整体情况
4. 判断是否需要调仓
5. 生成复盘报告
```

**提示词模板**：
```
【时间】2025-01-01 15:30:00
【事件】每日复盘
【今日盈亏】{daily_pnl}
【持仓表现】{positions_performance}
【市场表现】{market_summary}
今天已经收盘，是否需要操作？
```

#### 3.2.2 上涨股票监控

**触发条件**：收盘后，每天 15:05 执行

**处理流程**：
```
1. 获取今日涨幅前 N 的股票
2. 按行业/概念分组
3. 分析主线逻辑
4. 如有主线，存入长期记忆 (NoteTool)
5. 询问是否需要调整自选/关注新方向
```

**提示词模板**：
```
【时间】2025-01-01 15:05:00
【事件】上涨股票监控
【涨幅前50】{top_gainers}
【分组结果】{sector_analysis}
请分析市场主线，并将主线判断存入长期记忆。
是否需要调整持仓或关注方向？
```

### 3.3 记忆模块

#### 3.3.1 记忆分层

| 等级 | 说明 | 存储 | 裁剪策略 |
|------|------|------|----------|
| **0 普通** | 一般对话内容 | memories表 | 超过80k token时逻辑删除 |
| **1 重要** | 买入/卖出操作 | trades表 | 永不删除 |
| **2 主线** | 主线判断、市场分析 | memories表(importance=2) | 永不删除 |

#### 3.3.2 记忆存储机制

**核心原则**：逻辑删除，裁剪后或重启只需加载未删除的记忆

```sql
-- 对话记忆表（逻辑删除）
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,          -- user/assistant/system
    content TEXT NOT NULL,       -- 记忆内容
    importance INT DEFAULT 0,    -- 0普通 1重要 2主线
    is_deleted INT DEFAULT 0,    -- 0未删除 1已删除（逻辑删除）
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted_at DATETIME         -- 删除时间
);

-- 索引
CREATE INDEX idx_memories_session ON memories(session_id, is_deleted);
CREATE INDEX idx_memories_importance ON memories(session_id, importance, is_deleted);
```

**加载逻辑**：
```python
class MemoryManager:
    """记忆管理器"""
    
    def load_session_memory(self, session_id: str) -> list[Message]:
        """加载会话记忆（仅加载未删除的记忆）"""
        sql = """
            SELECT role, content 
            FROM memories 
            WHERE session_id = ? AND is_deleted = 0 
            ORDER BY created_at ASC
        """
        return self.db.query(sql, [session_id])
    
    def save_memory(self, session_id: str, role: str, content: str, importance: int = 0):
        """保存记忆"""
        sql = """
            INSERT INTO memories (session_id, role, content, importance)
            VALUES (?, ?, ?, ?)
        """
        self.db.execute(sql, [session_id, role, content, importance])
    
    def soft_delete(self, memory_id: int):
        """逻辑删除（裁剪时使用）"""
        sql = "UPDATE memories SET is_deleted = 1, deleted_at = NOW() WHERE id = ?"
        self.db.execute(sql, [memory_id])
    
    def prune_memory(self, session_id: str, keep_importance: int = 1):
        """裁剪记忆：删除 importance < keep_importance 的普通记忆"""
        sql = """
            UPDATE memories 
            SET is_deleted = 1, deleted_at = NOW() 
            WHERE session_id = ? 
            AND importance < ? 
            AND is_deleted = 0
        """
        self.db.execute(sql, [session_id, keep_importance])
```

**启动加载流程**：
```
1. 接收 session_id 参数
2. 查询 memories 表，条件：session_id = ? AND is_deleted = 0
3. 按时间顺序加载到上下文
4. 裁剪时：UPDATE is_deleted = 1
5. 重启后：自动加载未删除的记忆
```

#### 3.3.3 裁剪机制

**触发条件**：当前上下文超过 80,000 tokens

**裁剪策略**：
1. 统计当前 token 数
2. 如果超过限制，按重要性从低到高逻辑删除
3. 删除操作：`UPDATE memories SET is_deleted = 1 WHERE id = ?`
4. 保留：importance >= 1 的记忆（买卖记录、主线判断）

**保留规则**：
- 买入/卖出记录 → 永不删除
- 主线判断（importance=2）→ 永不删除
- 普通对话（importance=0）→ 可删除

-- 索引
CREATE INDEX idx_memories_session ON memories(session_id);
CREATE INDEX idx_memories_importance ON memories(session_id, importance);
```

#### 3.3.3 裁剪机制

**触发条件**：当前上下文超过 80,000 tokens

**裁剪策略**：
1. 统计当前 token 数
2. 如果超过限制，按重要性从低到高删除
3. 删除前将原始内容存入数据库（只保留 importance >= 1）
4. 生成记忆摘要，更新 session.memory_summary

**保留规则**：
- 买入/卖出记录（importance=1）→ 永不删除
- 主线判断（importance=2）→ 永不删除
- 系统提示词 → 永不删除

### 3.4 交易记录

#### 3.4.1 交易表结构

```sql
-- 交易记录表
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ticker TEXT NOT NULL,      -- 股票代码
    action TEXT NOT NULL,     -- buy/sell
    price REAL NOT NULL,       -- 成交价格
    quantity INT NOT NULL,     -- 成交数量
    amount REAL NOT NULL,      -- 成交金额
    reason TEXT,              -- 操作原因
    timestamp DATETIME NOT NULL,  -- 时间戳
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_trades_session ON trades(session_id);
CREATE INDEX idx_trades_ticker ON trades(ticker);
```

#### 3.4.2 盈利计算

**功能**：输入 session_id，计算当前盈利

**公式**：
```
盈利 = (当前价 - 买入价) × 数量
盈利率 = (当前价 - 买入价) / 买入价 × 100%
```

**Tool 接口**：
```
Tool: calculate_profit
参数: session_id (必需)
返回:
{
    "total_profit": 1234.56,
    "total_profit_rate": 5.23,
    "positions": [
        {"ticker": "600519", "profit": 500, "profit_rate": 2.1},
        {"ticker": "000858", "profit": 734.56, "profit_rate": 8.5}
    ]
}
```

### 3.5 时间标记

**规则**：每次请求必须携带时间戳

**格式**：`【时间】YYYY-MM-DD HH:MM:SS`

**使用场景**：
- 新闻事件触发
- 每日复盘触发
- 用户主动查询
- 所有 Agent 请求

**示例**：
```
【时间】2025-01-01 09:35:00
【事件】用户查询
【用户问题】查看贵州茅台的行情

【时间】2025-01-01 15:00:00
【事件】每日复盘
【持仓】...
```

---

## 4. 工具清单

### 4.1 股票查询工具（已有）

| Tool | 功能 | 参数 |
|------|------|------|
| AShareQuoteTool | 实时行情 | ticker, timestamp |
| AShareFundamentalsTool | 基本面数据 | ticker, timestamp, period |
| AShareScreenTool | 选股筛选 | strategy, timestamp, max_results |

> 注：暂不包含新闻工具（散户获取新闻存在信息差劣势）

### 4.2 交易工具（已有/需扩展）

| Tool | 功能 | 参数 |
|------|------|------|
| TradeRecordTool | 记录买卖 | session_id, ticker, action, price, quantity, reason |
| TradeHistoryTool | 查询历史 | session_id, ticker |
| PositionTool | 查询持仓 | session_id, ticker |
| CalculateProfitTool | 盈利计算 | session_id (新增) |

### 4.3 工具（需新增）

| Tool | 功能 | 参数 |
|------|------|------|
| MarketSummaryTool | 市场总览 | timestamp |
| TopGainersTool | 涨幅榜 | timestamp, limit |
| SectorAnalysisTool | 板块分析 | timestamp |
| MainlineAnalysisTool | 主线分析 | timestamp |

### 4.4 数据同步

| 数据 | 来源 | 同步频率 |
|------|------|----------|
| 日线数据 | 腾讯财经API / AkShare | 每日收盘后 |
| 基本面 | AkShare | 每周 |
| 北向资金 | AkShare | 每日 |
| 龙虎榜 | AkShare | 每日 |

---

## 5. 数据流设计

### 5.1 日线数据存储

```python
# 数据库表结构
CREATE TABLE daily_kline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    amount REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_kline_ticker ON daily_kline(ticker);
CREATE INDEX idx_kline_date ON daily_kline(date);
```

### 5.2 数据同步流程

```
1. 每日 16:00 触发同步任务
2. 获取全市场股票列表
3. 增量同步近 N 天的日线数据
4. 存储到 SQLite
5. Agent 查询时直接读库，不调用外部API
```

---

## 6. 事件配置

### 6.1 Cron 任务

| 任务 | 时间 | 说明 |
|------|------|------|
| 每日复盘 | 0 15 * * 1-5 | 每周一到周五 15:00 |
| 上涨监控 | 5 15 * * 1-5 | 每周一到周五 15:05 |
| 数据同步 | 0 16 * * 1-5 | 每周一到周五 16:00 |

---

## 7. 安全与限制

### 7.1 操作限制

- **不直接下单**：Agent 只给出建议，人确认后执行
- **买入前校验**：查询当前仓位，避免超仓
- **操作冷却**：同一股票 24 小时内不重复操作

### 7.2 风控规则

- 单只股票最大仓位：20%
- 单日最大亏损：-5% 报警，-8% 禁止新开仓
- 止损线：-7% 自动提醒

---

## 8. 验收标准

### 8.1 功能验收

| 功能 | 验收条件 |
|------|----------|
| Session 管理 | session_id 传入后能正确加载历史记忆 |
| 每日复盘 | 15:00 自动触发，生成复盘报告 |
| 新闻事件 | 监听到相关新闻后触发分析 |
| 记忆裁剪 | 超过 80k token 时自动裁剪普通记忆 |
| 盈利计算 | 输入 session_id 返回正确盈利数据 |
| 主线分析 | 收盘后分析涨幅榜，识别主线并存储 |

### 8.2 数据验收

| 数据 | 验收条件 |
|------|----------|
| 日线数据 | 查询历史K线返回正确数据 |
| 实时行情 | 盘中日线、涨跌幅正确 |
| 基本面 | PE、ROE 等指标正确 |

---

## 9. 实施计划

### Phase 1: 基础能力（1周）
- [ ] session_id 传入机制
- [ ] 记忆模块（存储 + 裁剪）
- [ ] 盈利计算工具
- [ ] 交易记录工具

### Phase 2: 事件驱动（1周）
- [ ] 每日复盘事件（cron）
- [ ] 上涨股票监控

### Phase 3: 数据完善（1周）
- [ ] 日线数据同步
- [ ] 基本面数据同步
- [ ] 主线分析逻辑

### Phase 4: 优化（持续）
- [ ] 风控规则
- [ ] 信号评分
- [ ] 性能优化

---

## 10. 附录

### 10.1 提示词示例

**系统提示词**：
```
你是一个专业的股票投资助手。

【重要规则】
1. 每次回答必须基于数据和逻辑
2. 不推荐买入当前涨幅超过 5% 的股票
3. 建议仓位不超过 20% 单只股票
4. 止损线为 -7%

【时间标记】
所有分析必须标注当前时间：{timestamp}
```

**复盘提示词**：
```
【时间】{timestamp}
【事件】每日复盘

【持仓情况】
{positions}

【今日市场】
{market_summary}

请分析：
1. 持仓股票今日表现
2. 是否需要调仓
3. 明日操作建议
```

### 10.2 数据库 ER 图

```
┌──────────────┐       ┌──────────────┐
│   sessions   │       │   memories   │
├──────────────┤       ├──────────────┤
│ session_id   │◄──────│ session_id   │
│ create_time  │       │ role         │
│ last_active  │       │ content      │
│ memory_sum   │       │ importance   │
└──────────────┘       │ created_at   │
                       └──────────────┘

       ┌──────────────┐
       │    trades    │
       ├──────────────┤
       │ session_id   │
       │ ticker       │
       │ action       │
       │ price        │
       │ quantity     │
       │ reason       │
       │ timestamp    │
       └──────────────┘

       ┌──────────────┐
       │  daily_kline │
       ├──────────────┤
       │ ticker       │
       │ date         │
       │ open/high/   │
       │ low/close    │
       │ volume       │
       └──────────────┘
```

---

**版本**：v1.0
**作者**：
**日期**：2025-01-01
