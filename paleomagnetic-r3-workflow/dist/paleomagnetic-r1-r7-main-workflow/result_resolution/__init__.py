"""R3/R5/R7 API result resolution."""

from .r3 import resolve_api_result as resolve_r3_api_result
from .r5 import resolve_api_result as resolve_r5_api_result
from .r7 import resolve_api_result as resolve_r7_api_result

__all__ = (
    "resolve_r3_api_result",
    "resolve_r5_api_result",
    "resolve_r7_api_result",
)
