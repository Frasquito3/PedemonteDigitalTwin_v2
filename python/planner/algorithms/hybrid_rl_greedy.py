from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from time import perf_counter
from typing import Callable, Protocol

from ..core.schema import (
    InstanciaTurno,
    PlanTurno,
)

from ..domain.validator import (
    validar_plan,
)

from .greedy import (
    generar_plan_greedy,
)


class PlanificadorCompatible(
    Protocol
):
    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        ...


GeneradorPlanGreedy = Callable[
    [InstanciaTurno],
    PlanTurno,
]


class FuentePlanHibrido(
    str,
    Enum,
):
    RL = "RL"

    GREEDY = "GREEDY"


class MotivoSeleccionHibrida(
    str,
    Enum,
):
    RL_MENOR_COSTO = (
        "RL_MENOR_COSTO"
    )

    GREEDY_MENOR_O_IGUAL = (
        "GREEDY_MENOR_O_IGUAL"
    )

    RL_INVALIDO = (
        "RL_INVALIDO"
    )

    RL_EXCEPCION = (
        "RL_EXCEPCION"
    )


@dataclass(frozen=True)
class ConfiguracionHibrida:
    tolerancia_empate: float = 1e-9

    preferir_greedy_en_empate: bool = True

    permitir_fallback_rl_invalido: bool = True

    permitir_fallback_excepcion_rl: bool = True

    def __post_init__(
        self,
    ) -> None:
        if self.tolerancia_empate < 0.0:
            raise ValueError(
                "tolerancia_empate no puede "
                "ser negativa."
            )


@dataclass(frozen=True)
class DecisionHibrida:
    instancia_id: str

    fuente_seleccionada: FuentePlanHibrido

    motivo: MotivoSeleccionHibrida

    costo_rl: float | None

    costo_greedy: float

    gap_relativo_rl_vs_greedy: float | None

    tiempo_rl_ms: float

    tiempo_greedy_ms: float

    tiempo_total_ms: float

    errores_rl: tuple[str, ...] = ()


class HybridRLGreedyPlanner:
    """
    Planificador de despliegue que compara una solución RL
    contra una solución Greedy y devuelve la de menor costo.

    La comparación utiliza exactamente el costo estimado común
    registrado dentro de PlanTurno.

    Garantía:

        costo_seleccionado <= costo_greedy

    siempre que el plan Greedy sea válido.
    """

    def __init__(
        self,
        planner_rl: PlanificadorCompatible,
        configuracion:
            ConfiguracionHibrida
            | None = None,
        generador_greedy:
            GeneradorPlanGreedy
            | None = None,
    ) -> None:
        self.planner_rl = planner_rl

        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionHibrida()
        )

        self.generador_greedy = (
            generador_greedy
            if generador_greedy is not None
            else generar_plan_greedy
        )

        self.ultima_decision: DecisionHibrida | None = None

    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        inicio_total = perf_counter()

        inicio_greedy = perf_counter()

        plan_greedy = (
            self.generador_greedy(
                instancia
            )
        )

        tiempo_greedy_ms = (
            perf_counter()
            - inicio_greedy
        ) * 1000.0

        validacion_greedy = validar_plan(
            instancia,
            plan_greedy,
        )

        if not validacion_greedy.valido:
            raise RuntimeError(
                "El plan Greedy de referencia "
                "es inválido: "
                + " | ".join(
                    validacion_greedy.errores
                )
            )

        self._validar_costo(
            nombre="Greedy",
            costo=(
                plan_greedy
                .costo_estimado
            ),
        )

        inicio_rl = perf_counter()

        try:
            plan_rl = (
                self.planner_rl
                .generar_plan(
                    instancia
                )
            )

        except Exception as exc:
            tiempo_rl_ms = (
                perf_counter()
                - inicio_rl
            ) * 1000.0

            if not (
                self.configuracion
                .permitir_fallback_excepcion_rl
            ):
                raise

            return self._seleccionar_greedy_por_fallback(
                instancia=instancia,

                plan_greedy=plan_greedy,

                motivo=(
                    MotivoSeleccionHibrida
                    .RL_EXCEPCION
                ),

                costo_rl=None,

                tiempo_rl_ms=(
                    tiempo_rl_ms
                ),

                tiempo_greedy_ms=(
                    tiempo_greedy_ms
                ),

                inicio_total=(
                    inicio_total
                ),

                errores_rl=(
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                ),
            )

        tiempo_rl_ms = (
            perf_counter()
            - inicio_rl
        ) * 1000.0

        validacion_rl = validar_plan(
            instancia,
            plan_rl,
        )

        if not validacion_rl.valido:
            if not (
                self.configuracion
                .permitir_fallback_rl_invalido
            ):
                raise RuntimeError(
                    "El plan RL es inválido: "
                    + " | ".join(
                        validacion_rl.errores
                    )
                )

            return self._seleccionar_greedy_por_fallback(
                instancia=instancia,

                plan_greedy=plan_greedy,

                motivo=(
                    MotivoSeleccionHibrida
                    .RL_INVALIDO
                ),

                costo_rl=(
                    self._obtener_costo_opcional(
                        plan_rl
                    )
                ),

                tiempo_rl_ms=(
                    tiempo_rl_ms
                ),

                tiempo_greedy_ms=(
                    tiempo_greedy_ms
                ),

                inicio_total=(
                    inicio_total
                ),

                errores_rl=tuple(
                    validacion_rl.errores
                ),
            )

        self._validar_costo(
            nombre="RL",
            costo=(
                plan_rl
                .costo_estimado
            ),
        )

        costo_rl = (
            plan_rl
            .costo_estimado
        )

        costo_greedy = (
            plan_greedy
            .costo_estimado
        )

        diferencia = (
            costo_rl
            - costo_greedy
        )

        rl_es_estrictamente_mejor = (
            diferencia
            <
            -self.configuracion
            .tolerancia_empate
        )

        empate = (
            abs(
                diferencia
            )
            <=
            self.configuracion
            .tolerancia_empate
        )

        seleccionar_rl = (
            rl_es_estrictamente_mejor

            or (
                empate
                and not (
                    self.configuracion
                    .preferir_greedy_en_empate
                )
            )
        )

        tiempo_total_ms = (
            perf_counter()
            - inicio_total
        ) * 1000.0

        gap_relativo = (
            costo_rl
            - costo_greedy
        ) / max(
            abs(
                costo_greedy
            ),
            1.0,
        )

        if seleccionar_rl:
            plan_seleccionado = (
                plan_rl
            )

            fuente = (
                FuentePlanHibrido.RL
            )

            motivo = (
                MotivoSeleccionHibrida
                .RL_MENOR_COSTO
            )

        else:
            plan_seleccionado = (
                plan_greedy
            )

            fuente = (
                FuentePlanHibrido
                .GREEDY
            )

            motivo = (
                MotivoSeleccionHibrida
                .GREEDY_MENOR_O_IGUAL
            )

        plan_seleccionado.tiempo_computo_ms = (
            tiempo_total_ms
        )

        self.ultima_decision = (
            DecisionHibrida(
                instancia_id=(
                    instancia
                    .instancia_id
                ),

                fuente_seleccionada=(
                    fuente
                ),

                motivo=motivo,

                costo_rl=costo_rl,

                costo_greedy=(
                    costo_greedy
                ),

                gap_relativo_rl_vs_greedy=(
                    gap_relativo
                ),

                tiempo_rl_ms=(
                    tiempo_rl_ms
                ),

                tiempo_greedy_ms=(
                    tiempo_greedy_ms
                ),

                tiempo_total_ms=(
                    tiempo_total_ms
                ),

                errores_rl=(),
            )
        )

        return plan_seleccionado

    def _seleccionar_greedy_por_fallback(
        self,
        instancia: InstanciaTurno,
        plan_greedy: PlanTurno,
        motivo: MotivoSeleccionHibrida,
        costo_rl: float | None,
        tiempo_rl_ms: float,
        tiempo_greedy_ms: float,
        inicio_total: float,
        errores_rl: tuple[str, ...],
    ) -> PlanTurno:
        tiempo_total_ms = (
            perf_counter()
            - inicio_total
        ) * 1000.0

        plan_greedy.tiempo_computo_ms = (
            tiempo_total_ms
        )

        gap_relativo: (
            float | None
        ) = None

        if costo_rl is not None:
            gap_relativo = (
                costo_rl
                -
                plan_greedy.costo_estimado
            ) / max(
                abs(
                    plan_greedy
                    .costo_estimado
                ),
                1.0,
            )

        self.ultima_decision = (
            DecisionHibrida(
                instancia_id=(
                    instancia
                    .instancia_id
                ),

                fuente_seleccionada=(
                    FuentePlanHibrido
                    .GREEDY
                ),

                motivo=motivo,

                costo_rl=costo_rl,

                costo_greedy=(
                    plan_greedy
                    .costo_estimado
                ),

                gap_relativo_rl_vs_greedy=(
                    gap_relativo
                ),

                tiempo_rl_ms=(
                    tiempo_rl_ms
                ),

                tiempo_greedy_ms=(
                    tiempo_greedy_ms
                ),

                tiempo_total_ms=(
                    tiempo_total_ms
                ),

                errores_rl=errores_rl,
            )
        )

        return plan_greedy

    @staticmethod
    def _validar_costo(
        nombre: str,
        costo: float,
    ) -> None:
        if not isfinite(
            costo
        ):
            raise RuntimeError(
                f"El costo de {nombre} "
                f"no es finito: {costo}"
            )

        if costo < 0.0:
            raise RuntimeError(
                f"El costo de {nombre} "
                f"es negativo: {costo}"
            )

    @staticmethod
    def _obtener_costo_opcional(
        plan: PlanTurno,
    ) -> float | None:
        costo = (
            plan.costo_estimado
        )

        if not isfinite(
            costo
        ):
            return None

        if costo < 0.0:
            return None

        return costo