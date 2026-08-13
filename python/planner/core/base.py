from typing import Protocol

from planner.core.schema import InstanciaTurno, PlanTurno


class PlanificadorTurno(Protocol):
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        ...