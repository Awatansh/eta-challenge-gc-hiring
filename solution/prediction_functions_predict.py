"""Light wrapper exposing a stable `predict(request: dict) -> float` function."""

from __future__ import annotations

from typing import Any

from solution.eta_pipeline import predict as _predict


def predict(request: dict[str, Any]) -> float:
    return float(_predict(request))
