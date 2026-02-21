# Mini-Agent 回测系统 PRD

## 1. 产品定位

**产品名称**：Mini-Agent 回测系统

**核心功能**：基于历史数据模拟交易策略，验证策略有效性，生成绩效分析报告

**目标用户**：希望验证自己投资策略有效性的个人投资者

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        回测引擎                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  数据加载器   │  │  模拟交易引擎 │  │   绩效分析器  │      │
│  │ - 历史K线    │  │ - 订单管理    │  │ - 收益率统计  │      │
│  │ - 资金流向   │  │ - 仓位管理    │  │ - 风险指标    │      │
│  │ - 基本面     │  │ - 滑点/手续费 │  │ - 图表生成    │      │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘      │
│          │                  │                  │              │
│          ▼                  ▼                  ▼              │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                    数据存储层                           │    │
│  │  daily_kline | fundamentals | north_money | lhb      │    │
│  └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心功能

### 3.1 数据加载器

#### 3.1.1 数据源

| 数据类型 | 来源 | 频率 |存储 |
 |----------|------|------|------|
| 日线数据 | 腾讯财经/AkShare | 每日 | daily_kline 表 |
| 资金流向 | AkShare | 每日 | capital_flow 表 |
| 北向资金 | AkShare | 每日 | north_money 表 |
| 龙虎榜 | AkShare | 每日 | lhb 表 |
| 板块涨跌 | AkShare | 每日 | sector_daily 表 |

#### 3.1.2 数据时间范围

- **起始时间**：2020-01-01（5年数据）
- **结束时间**：当前日期 - 1
- **更新频率**：每日收盘后增量同步

#### 3.1.3 数据接口

```python
class DataLoader:
    """历史数据加载器"""
    
    def load_kline(self, ticker: str, start: str, end: str) -> pd.DataFrame
    def load_batch_kline(self, tickers: list, start: str, end: str) -> pd.DataFrame
    def load_capital_flow(self, date: str) -> pd.DataFrame
    def load_north_money(self, date: str) -> pd.DataFrame
    def load_sector(self, date: str) -> pd.DataFrame
    def get_price(self, ticker: str, date: str) -> float  # 获取某日收盘价
```

### 3.2 模拟交易引擎

#### 3.2.1 核心类

```python
class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.0003,  # 佣金万三
        stamp_tax: float = 0.001,          # 印花税千一（卖出）
        slippage: float = 0.001            # 滑点千一
    ):
        self.cash = initial_capital         # 可用资金
        self.initial_capital = initial_capital
        self.positions = {}                 # 持仓 {ticker: Position}
        self.orders = []                    # 订单记录
        self.trades = []                    # 成交记录
    
    def can_buy(self, ticker: str, price: float, quantity: int) -> bool:
        """检查是否可以买入"""
        cost = self._calculate_cost(price, quantity, is_buy=True)
        return self.cash >= cost
    
    def buy(self, ticker: str, price: float, quantity: int, date: str, reason: str = "") -> bool:
        """买入"""
        cost = self._calculate_cost(price, quantity, is_buy=True)
        if self.cash < cost:
            return False
        
        self.cash -= cost
        if ticker in self.positions:
            self.positions[ticker].add(price, quantity)
        else:
            self.positions[ticker] = Position(ticker, price, quantity)
        
        self._record_order(ticker, "buy", price, quantity, date, reason)
        return True
    
    def sell(self, ticker: str, price: float, quantity: int, date: str, reason: str = "") -> bool:
        """卖出"""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        if position.quantity < quantity:
            return False
        
        revenue = self._calculate_revenue(price, quantity)
        self.cash += revenue
        position.reduce(quantity)
        
        if position.quantity == 0:
            del self.positions[ticker]
        
        self._record_order(ticker, "sell", price, quantity, date, reason)
        return True
    
    def get_portfolio_value(self, prices: dict) -> float:
        """获取组合市值"""
        position_value = sum(
            p.quantity * prices.get(p.ticker, p.avg_cost)
            for p in self.positions.values()
        )
        return self.cash + position_value
    
    def _calculate_cost(self, price: float, quantity: int, is_buy: bool) -> float:
        """计算买入成本"""
        base = price * quantity
        commission = base * self.commission_rate
        slippage_cost = base * self.slippage
        if is_buy:
            return base + commission + slippage_cost
        return base  # 卖出时成本不含佣金
    
    def _calculate_revenue(self, price: float, quantity: int) -> float:
        """计算卖出收入"""
        base = price * quantity
        commission = base * self.commission_rate
        stamp = base * self.stamp_tax
        slippage_cost = base * self.slippage
        return base - commission - stamp - slippage_cost
```

#### 3.2.2 仓位管理

```python
class Position:
    """持仓"""
    
    def __init__(self, ticker: str, price: float, quantity: int):
        self.ticker = ticker
        self.total_cost = price * quantity
        self.quantity = quantity
        self.avg_cost = price
    
    def add(self, price: float, quantity: int):
        """加仓"""
        self.total_cost += price * quantity
        self.quantity += quantity
        self.avg_cost = self.total_cost / self.quantity
    
    def reduce(self, quantity: int):
        """减仓"""
        self.quantity -= quantity
    
    def get_profit(self, current_price: float) -> float:
        """当前盈亏"""
        return (current_price - self.avg_cost) * self.quantity
    
    def get_profit_rate(self, current_price: float) -> float:
        """盈亏比例"""
        return (current_price - self.avg_cost) / self.avg_cost
```

### 3.3 绩效分析器

#### 3.3.1 指标计算

```python
class PerformanceAnalyzer:
    """绩效分析器"""
    
    def analyze(self, trades: list, equity_curve: list) -> dict:
        """生成绩效报告"""
        return {
            "total_return": self.calc_total_return(equity_curve),
            "annual_return": self.calc_annual_return(equity_curve),
            "sharpe_ratio": self.calc_sharpe_ratio(equity_curve),
            "max_drawdown": self.calc_max_drawdown(equity_curve),
            "win_rate": self.calc_win_rate(trades),
            "profit_factor": self.calc_profit_factor(trades),
            "total_trades": len(trades),
            "avg_holding_days": self.calc_avg_holding_days(trades),
        }
    
    def calc_total_return(self, equity_curve: list) -> float:
        """总收益率 (最终价值 - 初始资金) / 初始资金"""
        if not equity_curve:
            return 0
        return (equity_curve[-1]["value"] - equity_curve[0]["value"]) / equity_curve[0]["value"]
    
    def calc_annual_return(self, equity_curve: list) -> float:
        """年化收益率"""
        total_return = self.calc_total_return(equity_curve)
        if len(equity_curve) < 2:
            return 0
        days = (equity_curve[-1]["date"] - equity_curve[0]["date"]).days
        years = days / 365
        if years == 0:
            return 0
        return (1 + total_return) ** (1 / years) - 1
    
    def calc_sharpe_ratio(self, equity_curve: list) -> float:
        """夏普比率 = (年化收益率 - 无风险利率) / 年化波动率"""
        if len(equity_curve) < 2:
            return 0
        returns = self._calc_returns(equity_curve)
        if not returns:
            return 0
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        annual_return = avg_return * 252
        annual_std = std_return * np.sqrt(252)
        return (annual_return - 0.03) / annual_std  # 假设无风险利率 3%
    
    def calc_max_drawdown(self, equity_curve: list) -> float:
        """最大回撤"""
        if not equity_curve:
            return 0
        values = [e["value"] for e in equity_curve]
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        return -max_dd
    
    def calc_win_rate(self, trades: list) -> float:
        """胜率"""
        if not trades:
            return 0
        wins = sum(1 for t in trades if t.get("profit", 0) > 0)
        return wins / len(trades)
    
    def calc_profit_factor(self, trades: list) -> float:
        """盈利因子 = 总盈利 / 总亏损"""
        profits = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) > 0)
        losses = abs(sum(t.get("profit", 0) for t in trades if t.get("profit", 0) < 0))
        if losses == 0:
            return 0
        return profits / losses
    
    def calc_avg_holding_days(self, trades: list) -> float:
        """平均持仓天数"""
        sell_trades = [t for t in trades if t["action"] == "sell"]
        if not sell_trades:
            return 0
        holding_days = [
            (t["date"] - t["buy_date"]).days 
            for t in sell_trades if "buy_date" in t
        ]
        return sum(holding_days) / len(holding_days) if holding_days else 0
```

#### 3.3.2 权益曲线

```python
def build_equity_curve(self, engine: BacktestEngine, dates: list, prices: dict) -> list:
    """构建权益曲线"""
    curve = []
    for date in dates:
        value = engine.get_portfolio_value(prices.get(date, {}))
        curve.append({"date": date, "value": value})
    return curve
```

### 3.4 回测接口（Tool）

```python
class BacktestTool(Tool):
    """回测工具"""
    
    name = "stock_backtest"
    description = "基于历史数据进行策略回测"
    
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "策略/会话ID"
            },
            "start_date": {
                "type": "string",
                "description": "开始日期 YYYY-MM-DD"
            },
            "end_date": {
                "type": "string",
                "description": "结束日期 YYYY-MM-DD"
            },
            "initial_capital": {
                "type": "number",
                "description": "初始资金，默认100000"
            },
            "strategy_prompt": {
                "type": "string",
                "description": "策略描述，Agent将根据此策略生成交易信号"
            }
        },
        "required": ["session_id", "start_date", "end_date"]
    }
    
    async def execute(
        self,
        session_id: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        strategy_prompt: str = ""
    ) -> ToolResult:
        """执行回测"""
        # 1. 加载历史数据
        data_loader = DataLoader()
        
        # 2. 初始化引擎
        engine = BacktestEngine(initial_capital=initial_capital)
        
        # 3. 逐日回测
        dates = data_loader.get_trading_days(start_date, end_date)
        for date in dates:
            # 获取当日数据
            prices = data_loader.get_day_prices(date)
            market_data = data_loader.get_market_data(date)
            
            # 触发策略判断
            signal = await self._get_strategy_signal(
                engine, date, market_data, strategy_prompt
            )
            
            # 执行信号
            if signal.get("action") == "buy":
                engine.buy(signal["ticker"], signal["price"], signal["quantity"], date, signal.get("reason", ""))
            elif signal.get("action") == "sell":
                engine.sell(signal["ticker"], signal["price"], signal["quantity"], date, signal.get("reason", ""))
            
            # 记录权益
            engine.record_equity(date)
        
        # 4. 绩效分析
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(engine.trades, engine.equity_curve)
        
        return ToolResult(success=True, content=json.dumps(report, ensure_ascii=False))
    
    async def _get_strategy_signal(
        self,
        engine: BacktestEngine,
        date: str,
        market_data: dict,
        strategy_prompt: str
    ) -> dict:
        """获取策略信号 - 调用 Agent 判断"""
        # 构建上下文
        context = f"""
【时间】{date}
【当前持仓】{engine.get_positions()}
【资金】{engine.cash}
【市场数据】{market_data}
【策略】{strategy_prompt}
请判断是否需要操作，返回JSON格式：
{{"action": "buy/sell/hold", "ticker": "股票代码", "price": 价格, "quantity": 数量, "reason": "原因"}}
"""
        # 调用 Agent
        # ...
```

---

## 4. 数据库设计

### 4.1 交易记录表（扩展）

```sql
-- 回测交易记录
CREATE TABLE backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,      -- 策略ID
    backtest_id TEXT NOT NULL,     -- 回测ID
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,          -- buy/sell
    price REAL NOT NULL,
    quantity INT NOT NULL,
    amount REAL NOT NULL,
    profit REAL,                   -- 盈亏（仅sell记录）
    holding_days INT,              -- 持仓天数
    reason TEXT,
    signal_date TEXT NOT NULL,     -- 信号日期
    execute_date TEXT NOT NULL,    -- 执行日期
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 回测报告
CREATE TABLE backtest_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    backtest_id TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital REAL NOT NULL,
    final_value REAL NOT NULL,
    total_return REAL,
    annual_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    profit_factor REAL,
    total_trades INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 权益曲线
CREATE TABLE backtest_equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id TEXT NOT NULL,
    date DATE NOT NULL,
    value REAL NOT NULL,
    position_value REAL,
    cash REAL,
    UNIQUE(backtest_id, date)
);
```

---

## 5. 与现有系统的整合

### 5.1 复用现有模块

| 现有模块 | 回测复用方式 |
|----------|-------------|
| StockTools | 用 DataLoader 替代，读取历史数据 |
| TradeRecordTool | 记录回测交易 |
| 事件机制 | 模拟历史事件触发 |
| 记忆模块 | 记录回测过程 |

### 5.2 数据流

```
用户请求回测
    │
    ▼
BacktestTool
    │
    ├─ DataLoader ──► 读取历史K线/资金流向/北向资金
    │
    ├─ BacktestEngine ──► 模拟交易
    │       │
    │       ├─ 每日触发策略 Agent
    │       ├─ 执行买卖
    │       └─ 记录权益
    │
    ├─ PerformanceAnalyzer ──► 绩效计算
    │
    └─ 返回报告
```

---

## 6. 回测示例

### 6.1 示例1：均线策略

**策略描述**：买入5日均线上穿10日均线的股票，跌破均线卖出

```python
strategy_prompt = """
策略：均线突破
1. 当MA5上穿MA10时买入
2. 当MA5下穿MA10时卖出
3. 单只股票仓位不超过20%
4. 止损线-7%
"""
```

**回测结果**：
```json
{
    "backtest_id": "bt_20250101_001",
    "period": "2020-01-01 ~ 2024-12-31",
    "total_return": 0.85,
    "annual_return": 0.145,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.12,
    "win_rate": 0.58,
    "total_trades": 126
}
```

### 6.2 示例2：主线跟踪策略

**策略描述**：每日复盘跟踪市场主线，主线板块龙头回调买入

```python
strategy_prompt = """
策略：主线跟踪
1. 每日收盘后分析涨幅榜，识别主线板块
2. 主线板块回调-3%时买入龙头
3. 持有不超过5天
4. 涨幅超过15%自动止盈
"""
```

---

## 7. 约束与限制

### 7.1 避免未来函数

| 允许 | 禁止 |
|------|------|
| T-1 及之前的数据 | T+0 及未来的数据 |
| 收盘价 | 当日盘中价格 |
| 历史公告 | 未来公告 |

### 7.2 滑点与手续费

| 项目 | 默认值 | 可配置 |
|------|--------|--------|
| 佣金 | 万三 | ✓ |
| 印花税 | 千一（卖出） | ✓ |
| 滑点 | 千一 | ✓ |

### 7.3 限制

| 限制项 | 默认值 |
|--------|--------|
| 最大持仓股票数 | 10 |
| 单只股票最大仓位 | 20% |
| 单日最大交易次数 | 10 |

---

## 8. 验收标准

### 8.1 功能验收

| 功能 | 验收条件 |
|------|----------|
| 数据加载 | 能加载2020-至今的历史数据 |
| 模拟交易 | 买入/卖出/持仓计算正确 |
| 绩效分析 | 夏普比率/最大回撤/胜率计算正确 |
| 报告生成 | 返回完整绩效报告 |

### 8.2 准确性验收

| 指标 | 验收条件 |
|------|----------|
| 收益率误差 | < 0.1% |
| 手续费计算 | 与券商一致 |
| 持仓计算 | 与实际成交一致 |

---

## 9. 实施计划

### Phase 1: 数据层（3天）
- [ ] 完善历史数据库（2020-至今）
- [ ] 实现 DataLoader
- [ ] 数据校验

### Phase 2: 引擎（3天）
- [ ] 实现 BacktestEngine
- [ ] 实现仓位管理
- [ ] 实现滑点/手续费计算

### Phase 3: 分析（2天）
- [ ] 实现 PerformanceAnalyzer
- [ ] 权益曲线生成
- [ ] 图表生成

### Phase 4: 接口（2天）
- [ ] 封装 BacktestTool
- [ ] 接入 Agent
- [ ] 测试

---

## 10. 附录

### 10.1 绩效指标说明

| 指标 | 说明 | 优秀/良好/一般 |
|------|------|---------------|
| 总收益率 | 策略累计收益 | >100% / 50-100% / <50% |
| 年化收益率 | 年化后的收益 | >20% / 10-20% / <10% |
| 夏普比率 | 风险调整收益 | >1.5 / 1-1.5 / <1 |
| 最大回撤 | 最大亏损幅度 | <10% / 10-20% / >20% |
| 胜率 | 盈利交易占比 | >60% / 50-60% / <50% |
| 盈利因子 | 盈利/亏损比 | >2 / 1.5-2 / <1.5 |

### 10.2 回测报告示例

```json
{
    "backtest_id": "bt_001",
    "strategy": "均线突破",
    "period": {
        "start": "2020-01-01",
        "end": "2024-12-31",
        "trading_days": 1218
    },
    "capital": {
        "initial": 100000,
        "final": 185000,
        "profit": 85000
    },
    "returns": {
        "total_return": 0.85,
        "annual_return": 0.145,
        "sharpe_ratio": 1.2,
        "calmar_ratio": 1.2
    },
    "risk": {
        "max_drawdown": -0.12,
        "max_drawdown_date": "2022-04-11",
        "volatility": 0.18
    },
    "trading": {
        "total_trades": 126,
        "win_trades": 73,
        "loss_trades": 53,
        "win_rate": 0.58,
        "avg_profit": 1200,
        "avg_loss": -800,
        "profit_factor": 2.1,
        "avg_holding_days": 4.5
    },
    "equity_curve": [
        {"date": "2020-01-02", "value": 100000},
        {"date": "2020-01-03", "value": 100200}
    ]
}
```

---

**版本**：v1.0
**日期**：2025-01-01
