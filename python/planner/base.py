from typing import Protocol

from .schema import InstanciaTurno, PlanTurno


class PlanificadorTurno(Protocol):
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        ...