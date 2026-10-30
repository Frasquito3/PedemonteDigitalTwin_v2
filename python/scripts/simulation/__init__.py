from __future__ import annotations

from pathlib import Path

__doc__ = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

from scripts.simulation import run_simulated_comparison, run_single_simulated_execution

__all__ = [
    "run_simulated_comparison",
    "run_single_simulated_execution",
]
