from __future__ import annotations

from pathlib import Path

__doc__ = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

from planner.core import (
    base,
    config,
    schema,
)

__all__ = [
    "base",
    "config",
    "schema",
]
