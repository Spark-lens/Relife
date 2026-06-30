"""Multi-symbol Bollinger band reversion paper-trading strategy."""

from .config import load_config
from .engine import StrategyEngine

__all__ = ["StrategyEngine", "load_config"]
