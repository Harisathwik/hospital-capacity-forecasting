"""src.governance module — promotion gates and model governance."""
from src.governance.promotion import (
    PromotionBlockedError,
    PromotionGateRunner,
    GateResult,
)

__all__ = [
    "PromotionBlockedError",
    "PromotionGateRunner",
    "GateResult",
]
