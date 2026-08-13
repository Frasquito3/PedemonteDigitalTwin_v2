"""
Punto de entrada público para el planificador basado en RL.

La implementación original se conserva en ``rl_planner.py`` para evitar
romper imports existentes. Los nuevos módulos deben importar RLPlanner
mediante:

    from planner.rl.planner import RLPlanner
"""

from .rl_planner import RLPlanner


__all__ = [
    "RLPlanner",
]