from __future__ import annotations

from pathlib import Path

__doc__ = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

from scripts import (
    data_processing,
    rl,
    simulation,
    training,
)

__all__ = [
    "data_processing",
    "rl",
    "simulation",
    "training",
]
