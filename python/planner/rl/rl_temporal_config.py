from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionTemporalRL:
    """
    Parámetros de la formulación temporal v3.

    La recompensa intermedia utiliza shaping basado en potencial. El
    potencial resume tardanza, pérdida de holgura y esperas largas del
    prefijo actual y de los pedidos que todavía podrían elegirse.

    La máscara temporal dura queda desactivada por defecto. La auditoría
    es determinística y no debe confundirse con una certeza absoluta
    sobre la aceptación estocástica del cliente en AnyLogic.
    """

    margen_critico_min: float = 15.0
    umbral_espera_larga_min: float = 30.0

    peso_tardanza: float = 3.0
    peso_margen_critico: float = 0.50
    peso_espera_larga: float = 0.10

    escala_potencial: float = 5.0
    coeficiente_shaping: float = 1.0

    bonificacion_terminal_sin_riesgo: float = 0.50
    penalizacion_terminal_por_riesgo: float = 3.0

    usar_mascara_temporal_dura: bool = False
    margen_mascara_dura_min: float = 30.0

    def __post_init__(self) -> None:
        positivos = {
            "margen_critico_min": self.margen_critico_min,
            "umbral_espera_larga_min": self.umbral_espera_larga_min,
            "escala_potencial": self.escala_potencial,
        }

        for nombre, valor in positivos.items():
            if valor <= 0.0:
                raise ValueError(
                    f"{nombre} debe ser > 0."
                )

        no_negativos = {
            "peso_tardanza": self.peso_tardanza,
            "peso_margen_critico": self.peso_margen_critico,
            "peso_espera_larga": self.peso_espera_larga,
            "coeficiente_shaping": self.coeficiente_shaping,
            "bonificacion_terminal_sin_riesgo": (
                self.bonificacion_terminal_sin_riesgo
            ),
            "penalizacion_terminal_por_riesgo": (
                self.penalizacion_terminal_por_riesgo
            ),
            "margen_mascara_dura_min": (
                self.margen_mascara_dura_min
            ),
        }

        for nombre, valor in no_negativos.items():
            if valor < 0.0:
                raise ValueError(
                    f"{nombre} no puede ser negativo."
                )
