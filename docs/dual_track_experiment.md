# Strict vs Loose 双轨实验（AI能力边界）

目标：在不改交易执行模式（仍为“Agent提醒 + 人工下单”）的前提下，比较“强约束”与“弱约束”两种提示词策略的表现差异。

## 1. 实验设计

- `Strict轨道`：流程和输出模板强约束，保证稳定与一致性。
- `Loose轨道`：仅保留硬风控约束，其余由模型自主判断，观察创造性与泛化能力。

两轨都必须共享同一组硬参数，避免“参数差异”污染结论：

- `max_single_loss_pct=1.5`
- `single_position_cap_pct=15`
- `stop_loss_pct=6`
- `take_profit_pct=12`
- `trade_notice_enabled=true`

## 2. 一键创建

```bash
bash scripts/create_dual_track_sessions.sh
```

可指定 workspace：

```bash
bash scripts/create_dual_track_sessions.sh /home/lijiang/workspace/Mini-Agent
```

创建后查看：

```bash
uv run python -m mini_agent.cli session list
```

## 3. 对照执行

对两个 session 触发同一事件（例如 daily_review），保证输入一致：

```bash
uv run python -m mini_agent.cli event trigger daily_review --session <strict_id>
uv run python -m mini_agent.cli event trigger daily_review --session <loose_id>
```

## 4. 对照指标（建议至少跑 2~4 周）

- 可执行性：建议是否明确到“动作+仓位+风险位+失效条件”
- 稳定性：相似市场下建议是否一致、是否漂移
- 风险表现：最大回撤、连续亏损次数
- 收益质量：胜率、盈亏比、profit factor
- 人工体验：你是否更容易信任并执行

## 5. 判定建议

- 若 `Loose` 的收益明显更好且回撤未恶化，可逐步吸收其自由判断逻辑。
- 若 `Loose` 漂移大、解释不稳定，则保留 `Strict` 为主，仅开放少量自主区间。
- 通常最优是“硬约束固定 + 分析逻辑适度放开”。
