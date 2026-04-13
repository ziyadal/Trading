"""
models.py — Pydantic output models for each agent in the trading pipeline.

Every agent fills in both a free-text 'report' and structured prediction fields.
These models are used with LangChain's structured output to get reliable data
for performance tracking.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TAOutput(BaseModel):
    """Technical analysis agent output."""

    report: str = Field(description="Full technical analysis report in markdown format")
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Expected price direction over the next 1 week"
    )
    current_price: float = Field(description="Current BTC/USDT price")
    target_low: float = Field(description="Lower bound of 1-week price target")
    target_high: float = Field(description="Upper bound of 1-week price target")
    confidence: float = Field(ge=0, le=1, description="Confidence in prediction 0.0-1.0")
    key_support: float = Field(description="Key support level")
    key_resistance: float = Field(description="Key resistance level")


class NewsOutput(BaseModel):
    """News research agent output."""

    report: str = Field(description="Full news research report in markdown format")
    direction: Literal["BULLISH", "NEUTRAL", "BEARISH"] = Field(
        description="Directional bias based on news"
    )
    price_prediction: float = Field(description="BTC price prediction for 1 week from now")
    confidence: float = Field(ge=0, le=1, description="Confidence in prediction 0.0-1.0")
    key_catalyst: str = Field(description="The single most important catalyst identified")


class BullOutput(BaseModel):
    """Bull agent final structured assessment after debate."""

    report: str = Field(description="Final bullish argument summary after full debate")
    entry: float = Field(description="Recommended entry price")
    stop_loss: float = Field(description="Recommended stop loss price")
    target: float = Field(description="Recommended target price")
    confidence: float = Field(ge=0, le=1, description="Confidence in bullish thesis 0.0-1.0")
    key_argument: str = Field(description="Single strongest bullish argument")


class BearOutput(BaseModel):
    """Bear agent final structured assessment after debate."""

    report: str = Field(description="Final bearish argument summary after full debate")
    entry: float = Field(description="Recommended short entry price")
    stop_loss: float = Field(description="Recommended stop loss price")
    target: float = Field(description="Recommended downside target price")
    confidence: float = Field(ge=0, le=1, description="Confidence in bearish thesis 0.0-1.0")
    key_argument: str = Field(description="Single strongest bearish argument")


class PMOutput(BaseModel):
    """Portfolio manager final decision."""

    report: str = Field(
        description="Full PM evaluation covering debate dynamics, data merit, and risk analysis"
    )
    decision: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(description="Final trading decision")
    entry: float | None = Field(default=None, description="Entry price, null if NEUTRAL")
    stop_loss: float | None = Field(default=None, description="Stop loss price, null if NEUTRAL")
    target: float | None = Field(default=None, description="Target price, null if NEUTRAL")
    position_size: float | None = Field(
        default=None, description="Position size as percent of portfolio, null if NEUTRAL"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in decision 0.0-1.0")
    winning_side: Literal["BULL", "BEAR", "NEITHER"] = Field(
        default="NEITHER", description="Which side argument was stronger"
    )
    key_reason: str = Field(default="", description="The decisive argument or data point")
