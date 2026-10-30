from __future__ import annotations

from pathlib import Path

__doc__ = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

from planner.algorithms import (
    ga,
    greedy,
    hybrid_rl_ga,
    random_feasible,
)

__all__ = [
    "ga",
    "greedy",
    "hybrid_rl_ga",
    "random_feasible",
]
