from dataclasses import dataclass
from enum import Enum


class ModoRewardRL(str, Enum):
    ABSOLUTO = "ABSOLUTO"

    VENTAJA_GREEDY_RELATIVA = (
        "VENTAJA_GREEDY_RELATIVA"
    )


@dataclass(frozen=True)
class ConfiguracionRewardRL:
    modo: ModoRewardRL = (
        ModoRewardRL.ABSOLUTO
    )

    escala_absoluta: float = 100.0

    denominador_relativo_minimo: float = 1.0

    def __post_init__(self) -> None:
        if self.escala_absoluta <= 0.0:
            raise ValueError(
                "escala_absoluta debe ser > 0."
            )

        if (
            self.denominador_relativo_minimo
            <= 0.0
        ):
            raise ValueError(
                "denominador_relativo_minimo "
                "debe ser > 0."
            )

    def calcular_reward(
        self,
        costo_plan: float,
        costo_referencia: float | None = None,
    ) -> float:
        if costo_plan < 0.0:
            raise ValueError(
                "costo_plan no puede ser negativo."
            )

        if self.modo == ModoRewardRL.ABSOLUTO:
            return (
                -costo_plan
                / self.escala_absoluta
            )

        if (
            self.modo
            ==
            ModoRewardRL
            .VENTAJA_GREEDY_RELATIVA
        ):
            if costo_referencia is None:
                raise ValueError(
                    "El reward relativo requiere "
                    "costo_referencia."
                )

            denominador = max(
                abs(costo_referencia),
                self.denominador_relativo_minimo,
            )

            return (
                costo_referencia
                - costo_plan
            ) / denominador

        raise ValueError(
            "Modo de reward desconocido: "
            f"{self.modo}"
        )

    def calcular_gap_relativo(
        self,
        costo_plan: float,
        costo_referencia: float,
    ) -> float:
        if costo_plan < 0.0:
            raise ValueError(
                "costo_plan no puede ser negativo."
            )

        if costo_referencia < 0.0:
            raise ValueError(
                "costo_referencia no puede ser "
                "negativo."
            )

        denominador = max(
            abs(costo_referencia),
            self.denominador_relativo_minimo,
        )

        return (
            costo_plan
            - costo_referencia
        ) / denominador