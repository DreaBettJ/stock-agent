"""A-share stock tools skeleton.

This module provides a conservative, extensible tool framework for:
- stock selection
- stock analysis
- trade action planning

Data acquisition methods are intentionally left as placeholders so callers can
plug in real market/fundamental/news providers later.
"""

from __future__ import annotations

import json
from abc import ABC
from typing import Any

from .base import Tool, ToolResult


class AShareDataBackend(ABC):
    """Pluggable backend for A-share data.

    Replace or subclass these methods to connect your own data source.
    """

    async def get_quote(self, ticker: str) -> dict[str, Any]:
        raise NotImplementedError("TODO: implement quote fetch logic")

    async def get_fundamentals(self, ticker: str, period: str = "ttm") -> dict[str, Any]:
        raise NotImplementedError("TODO: implement fundamentals fetch logic")

    async def get_news(self, ticker: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError("TODO: implement news fetch logic")

    async def screen_stocks(self, strategy: str, max_results: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError("TODO: implement stock screening logic")


class PlaceholderAShareDataBackend(AShareDataBackend):
    """Default backend used before real data providers are wired in."""


class AShareQuoteTool(Tool):
    """Get real-time or latest quote data for a ticker."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_a_share_quote"

    @property
    def description(self) -> str:
        return "Get quote data for an A-share ticker (e.g., 600519.SH, 000001.SZ)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "A-share ticker symbol, e.g. 600519.SH or 000001.SZ",
                }
            },
            "required": ["ticker"],
        }

    async def execute(self, ticker: str) -> ToolResult:
        try:
            data = await self.backend.get_quote(ticker=ticker)
            return ToolResult(success=True, content=json.dumps(data, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get quote: {e}")


class AShareFundamentalsTool(Tool):
    """Get fundamental metrics for a ticker."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_a_share_fundamentals"

    @property
    def description(self) -> str:
        return "Get fundamental data for an A-share ticker (valuation, profitability, leverage, cashflow)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "A-share ticker symbol, e.g. 600519.SH or 000001.SZ",
                },
                "period": {
                    "type": "string",
                    "description": "Financial period preference (ttm, annual, quarterly)",
                    "default": "ttm",
                },
            },
            "required": ["ticker"],
        }

    async def execute(self, ticker: str, period: str = "ttm") -> ToolResult:
        try:
            data = await self.backend.get_fundamentals(ticker=ticker, period=period)
            return ToolResult(success=True, content=json.dumps(data, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get fundamentals: {e}")


class AShareNewsTool(Tool):
    """Get market or ticker-specific news/events."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_a_share_news"

    @property
    def description(self) -> str:
        return "Get recent A-share news. Can query market-wide or by ticker."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker symbol for stock-specific news.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of items to return.",
                    "default": 10,
                },
            },
        }

    async def execute(self, ticker: str | None = None, limit: int = 10) -> ToolResult:
        try:
            items = await self.backend.get_news(ticker=ticker, limit=limit)
            return ToolResult(success=True, content=json.dumps(items, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get news: {e}")


class AShareScreenTool(Tool):
    """Screen candidate stocks by conservative strategies."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "screen_a_share_stocks"

    @property
    def description(self) -> str:
        return "Screen A-share candidates using conservative strategies (quality, dividend, low_vol, value)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "description": "Screening strategy.",
                    "enum": ["quality", "dividend", "low_vol", "value"],
                    "default": "quality",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max number of candidate stocks.",
                    "default": 20,
                },
            },
        }

    async def execute(self, strategy: str = "quality", max_results: int = 20) -> ToolResult:
        try:
            result = await self.backend.screen_stocks(strategy=strategy, max_results=max_results)
            return ToolResult(success=True, content=json.dumps(result, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to screen stocks: {e}")


class ConservativeTradePlanTool(Tool):
    """Build a conservative trade plan based on supplied analysis inputs."""

    @property
    def name(self) -> str:
        return "build_conservative_trade_plan"

    @property
    def description(self) -> str:
        return "Build a conservative A-share trade plan with staged entries, stop, and risk controls."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
                "current_price": {"type": "number", "description": "Latest known price"},
                "support_price": {"type": "number", "description": "Technical support level"},
                "resistance_price": {"type": "number", "description": "Technical resistance level"},
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "low",
                    "description": "User risk tolerance. Conservative default is low.",
                },
                "account_size": {"type": "number", "description": "Optional account size for sizing hints"},
            },
            "required": ["ticker", "current_price", "support_price"],
        }

    async def execute(
        self,
        ticker: str,
        current_price: float,
        support_price: float,
        resistance_price: float | None = None,
        risk_level: str = "low",
        account_size: float | None = None,
    ) -> ToolResult:
        try:
            stop_price = round(support_price * 0.98, 3)
            tranche_weights = [0.4, 0.35, 0.25]

            entry_band = [
                round(current_price * 0.995, 3),
                round(current_price * 0.98, 3),
                round(max(support_price, current_price * 0.965), 3),
            ]

            risk_per_share = max(current_price - stop_price, 0.0001)
            position_note = "Use small size (e.g., <= 10% of total capital for single stock)."
            if account_size and risk_level == "low":
                # Conservative: risk 0.5% of account per position idea.
                max_risk_budget = account_size * 0.005
                max_shares = int(max_risk_budget / risk_per_share)
                position_note = (
                    f"Max risk budget: {max_risk_budget:.2f}, suggested max shares: {max_shares} "
                    f"(computed by risk budget / (entry-stop))."
                )

            take_profit = None
            if resistance_price is not None:
                take_profit = round(resistance_price * 0.98, 3)
            else:
                take_profit = round(current_price * 1.08, 3)

            plan = {
                "ticker": ticker,
                "style": "conservative",
                "risk_level": risk_level,
                "entry_plan": {
                    "tranches": [
                        {"weight": tranche_weights[0], "entry_price": entry_band[0]},
                        {"weight": tranche_weights[1], "entry_price": entry_band[1]},
                        {"weight": tranche_weights[2], "entry_price": entry_band[2]},
                    ],
                    "rule": "Only add next tranche if price stabilizes near support with no major negative catalyst.",
                },
                "risk_control": {
                    "stop_loss": stop_price,
                    "invalidation": "Break below support with expanding volume or adverse fundamental event.",
                },
                "profit_taking": {
                    "first_target": take_profit,
                    "de_risk_rule": "Trim 30-50% near first target or on momentum divergence.",
                },
                "position_sizing": position_note,
                "disclaimer": "For research/education only. Final decision and risk are user's responsibility.",
            }
            return ToolResult(success=True, content=json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to build trade plan: {e}")


def create_a_share_tools(backend: AShareDataBackend | None = None) -> list[Tool]:
    """Create all A-share related tools.

    Args:
        backend: Optional concrete backend implementation. If omitted,
            PlaceholderAShareDataBackend is used.
    """
    data_backend = backend or PlaceholderAShareDataBackend()
    return [
        AShareScreenTool(data_backend),
        AShareQuoteTool(data_backend),
        AShareFundamentalsTool(data_backend),
        AShareNewsTool(data_backend),
        ConservativeTradePlanTool(),
    ]
