"""Definición y cálculo de señales de recompensa (reward) para Aprendizaje por Refuerzo."""

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ModoRewardRL",
    "ConfiguracionRewardRL",
]


class ModoRewardRL(str, Enum):
    """Modalidad de cómputo para la recompensa escalar de la política."""

    ABSOLUTO = "ABSOLUTO"
    VENTAJA_GREEDY_RELATIVA = "VENTAJA_GREEDY_RELATIVA"


@dataclass(frozen=True)
class ConfiguracionRewardRL:
    """Configuración de escala y normalización para la función de recompensa.

    Args:
        modo: Estrategia de cómputo (costo absoluto escalado o ventaja relativa vs Greedy).
        escala_absoluta: Factor de escala divisor para el costo absoluto.
        denominador_relativo_minimo: Cota inferior para evitar divisiones por cero en modo relativo.
    """

    modo: ModoRewardRL = ModoRewardRL.ABSOLUTO
    escala_absoluta: float = 100.0
    denominador_relativo_minimo: float = 1.0

    def __post_init__(self) -> None:
        if self.escala_absoluta <= 0.0:
            raise ValueError("escala_absoluta debe ser > 0.")
        if self.denominador_relativo_minimo <= 0.0:
            raise ValueError("denominador_relativo_minimo debe ser > 0.")

    def calcular_reward(
        self,
        costo_plan: float,
        costo_referencia: float | None = None,
    ) -> float:
        """Calcula el valor escalar de recompensa dado el costo del plan."""
        if costo_plan < 0.0:
            raise ValueError("costo_plan no puede ser negativo.")

        if self.modo == ModoRewardRL.ABSOLUTO:
            return -costo_plan / self.escala_absoluta

        if self.modo == ModoRewardRL.VENTAJA_GREEDY_RELATIVA:
            if costo_referencia is None:
                raise ValueError("El reward relativo requiere costo_referencia.")
            denominador = max(abs(costo_referencia), self.denominador_relativo_minimo)
            return (costo_referencia - costo_plan) / denominador

        raise ValueError(f"Modo de reward desconocido: {self.modo}")

    def calcular_gap_relativo(
        self,
        costo_plan: float,
        costo_referencia: float,
    ) -> float:
        """Calcula la brecha porcentual de costo frente a una solución de referencia."""
        if costo_plan < 0.0:
            raise ValueError("costo_plan no puede ser negativo.")
        if costo_referencia < 0.0:
            raise ValueError("costo_referencia no puede ser negativo.")

        denominador = max(abs(costo_referencia), self.denominador_relativo_minimo)
        return (costo_plan - costo_referencia) / denominador
