from __future__ import annotations

from pathlib import Path

__doc__ = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

from . import (
    algorithms,
    core,
    domain,
    integration,
    rl,
    routing,
)

__all__ = [
    "algorithms",
    "core",
    "domain",
    "integration",
    "rl",
    "routing",
]
