"""Definición del protocolo base para algoritmos de planificación."""

from typing import Protocol

from planner.core.schema import InstanciaTurno, PlanTurno

__all__ = ["PlanificadorTurno"]


class PlanificadorTurno(Protocol):
    """Interfaz que debe implementar cualquier algoritmo de planificación.

    Garantiza una interfaz homogénea para heurísticas constructivas,
    metaheurísticas (algoritmos genéticos), RL y generadores aleatorios o de prueba.
    """

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        """Genera un plan operativo de ruteo y asignación para el turno.

        Args:
            instancia: Definición del turno con sus pedidos, flota y restricciones.

        Returns:
            Plan con las rutas y pedidos asignados por camión.
        """
        ...
