from __future__ import annotations

from pathlib import Path

__doc__ = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

from scripts.rl import train_rl_v2

__all__ = ["train_rl_v2"]
