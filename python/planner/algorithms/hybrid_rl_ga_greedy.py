from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from time import perf_counter
from typing import Callable, Protocol

from planner.algorithms.ga import (
    ConfiguracionGA,
    GeneticAlgorithmPlanner,
)
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.routing.decoder import Cromosoma, validar_permutacion
from planner.routing.travel import ProveedorViaje


class PlanificadorCompatible(Protocol):
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        ...


GeneradorGARefinado = Callable[[InstanciaTurno, Cromosoma], PlanTurno]


class FuenteResultadoHibrido(str, Enum):
    SEMILLA_RL = "SEMILLA_RL"
    REFINADO_GA = "REFINADO_GA"


class MotivoResultadoHibrido(str, Enum):
    GA_MEJORA_SEMILLA_RL = "GA_MEJORA_SEMILLA_RL"
    SEMILLA_RL_CONSERVADA = "SEMILLA_RL_CONSERVADA"
    GA_NO_EJECUTABLE = "GA_NO_EJECUTABLE"


@dataclass(frozen=True)
class ConfiguracionHibridaRLGA:
    tolerancia_mejora: float = 1e-9
    fraccion_variantes_semilla_rl: float = 0.50
    incluir_semilla_greedy_en_refinamiento: bool = False

    def __post_init__(self) -> None:
        if self.tolerancia_mejora < 0.0:
            raise ValueError("tolerancia_mejora no puede ser negativa.")
        if not 0.0 <= self.fraccion_variantes_semilla_rl <= 1.0:
            raise ValueError(
                "fraccion_variantes_semilla_rl debe estar entre 0 y 1."
            )


@dataclass(frozen=True)
class DecisionHibridaRLGA:
    instancia_id: str
    fuente_rl: str
    resultado: FuenteResultadoHibrido
    motivo: MotivoResultadoHibrido
    costo_semilla_rl: float
    costo_refinado_ga: float | None
    costo_final: float
    mejora_absoluta: float
    mejora_porcentual: float
    tiempo_rl_ms: float
    tiempo_ga_ms: float
    tiempo_total_ms: float
    seed_ga: int
    generaciones_ga: int
    error_ga: tuple[str, ...] = ()

    @property
    def mejoro_semilla_rl(self) -> bool:
        return self.resultado == FuenteResultadoHibrido.REFINADO_GA


class HybridRLGAPlanner:
    """
    Híbrido secuencial RL -> GA.

    1. Obtiene un plan del modo RL puro.
    2. Convierte el orden de ese plan en un cromosoma.
    3. Inyecta ese cromosoma y variantes cercanas en la población del GA.
    4. Conserva la semilla RL si el refinamiento no la mejora.

    GREEDY no se ejecuta como candidato independiente y, por defecto, tampoco
    se inyecta como semilla en el refinamiento. El resultado es por lo tanto un
    refinamiento genuinamente guiado por RL, no un selector min(Greedy, GA, RL).
    """

    VERSION_PLANIFICADOR = "HIBRIDO_RL_GA_SEEDED_V1"

    def __init__(
        self,
        planner_rl: PlanificadorCompatible,
        configuracion: ConfiguracionHibridaRLGA | None = None,
        configuracion_ga: ConfiguracionGA | None = None,
        generador_ga_refinado: GeneradorGARefinado | None = None,
        configuracion_planificacion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
    ) -> None:
        self.planner_rl = planner_rl
        self.configuracion = configuracion or ConfiguracionHibridaRLGA()
        self.configuracion_ga = configuracion_ga or ConfiguracionGA()
        self.generador_ga_refinado = generador_ga_refinado
        self.configuracion_planificacion = (
            configuracion_planificacion or ConfiguracionPlanificacion()
        )
        self.proveedor_viaje = proveedor_viaje
        self.ultima_decision: DecisionHibridaRLGA | None = None

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        inicio_total = perf_counter()

        inicio_rl = perf_counter()
        try:
            plan_rl = self.planner_rl.generar_plan(instancia)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo obtener la semilla RL del híbrido: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        tiempo_rl_ms = (perf_counter() - inicio_rl) * 1000.0

        self._exigir_plan_ejecutable(
            instancia=instancia,
            plan=plan_rl,
            nombre="semilla RL",
        )

        cromosoma_rl = self._plan_a_cromosoma(instancia, plan_rl)
        fuente_rl = self._fuente_rl_seleccionada()
        costo_semilla_rl = float(plan_rl.costo_estimado)

        seed_ga = instancia.seed_escenario + 9001
        tiempo_ga_ms = 0.0
        generaciones_ga = 0
        plan_ga: PlanTurno | None = None
        costo_ga: float | None = None
        errores_ga: tuple[str, ...] = ()

        inicio_ga = perf_counter()
        try:
            if self.generador_ga_refinado is not None:
                plan_ga = self.generador_ga_refinado(
                    instancia,
                    cromosoma_rl,
                )
            else:
                planner_ga = GeneticAlgorithmPlanner(
                    configuracion=self.configuracion_planificacion,
                    configuracion_ga=self.configuracion_ga,
                    seed=seed_ga,
                    proveedor_viaje=self.proveedor_viaje,
                    semillas_iniciales=(cromosoma_rl,),
                    incluir_semilla_greedy=(
                        self.configuracion.incluir_semilla_greedy_en_refinamiento
                    ),
                    fraccion_variantes_semilla=(
                        self.configuracion.fraccion_variantes_semilla_rl
                    ),
                )
                plan_ga = planner_ga.generar_plan(instancia)
                generaciones_ga = planner_ga.generaciones_ejecutadas

            self._exigir_plan_ejecutable(
                instancia=instancia,
                plan=plan_ga,
                nombre="refinamiento GA",
            )
            costo_ga = float(plan_ga.costo_estimado)

        except Exception as exc:  # noqa: BLE001 - queda auditado y conserva RL
            errores_ga = (f"{type(exc).__name__}: {exc}",)
            plan_ga = None
            costo_ga = None

        tiempo_ga_ms = (perf_counter() - inicio_ga) * 1000.0

        if (
            plan_ga is not None
            and costo_ga is not None
            and costo_ga
            < costo_semilla_rl - self.configuracion.tolerancia_mejora
        ):
            seleccionado = plan_ga
            resultado = FuenteResultadoHibrido.REFINADO_GA
            motivo = MotivoResultadoHibrido.GA_MEJORA_SEMILLA_RL
            mejora_absoluta = costo_semilla_rl - costo_ga
        else:
            seleccionado = plan_rl
            resultado = FuenteResultadoHibrido.SEMILLA_RL
            motivo = (
                MotivoResultadoHibrido.GA_NO_EJECUTABLE
                if errores_ga
                else MotivoResultadoHibrido.SEMILLA_RL_CONSERVADA
            )
            mejora_absoluta = 0.0

        mejora_porcentual = (
            mejora_absoluta / costo_semilla_rl * 100.0
            if costo_semilla_rl > 0.0
            else 0.0
        )
        tiempo_total_ms = (perf_counter() - inicio_total) * 1000.0

        seleccionado.tiempo_computo_ms = tiempo_total_ms
        seleccionado.warnings.append(self.VERSION_PLANIFICADOR)
        seleccionado.warnings.append(f"SEMILLA_RL={fuente_rl}")
        seleccionado.warnings.append(f"RESULTADO_HIBRIDO={resultado.value}")

        self.ultima_decision = DecisionHibridaRLGA(
            instancia_id=instancia.instancia_id,
            fuente_rl=fuente_rl,
            resultado=resultado,
            motivo=motivo,
            costo_semilla_rl=costo_semilla_rl,
            costo_refinado_ga=costo_ga,
            costo_final=float(seleccionado.costo_estimado),
            mejora_absoluta=mejora_absoluta,
            mejora_porcentual=mejora_porcentual,
            tiempo_rl_ms=tiempo_rl_ms,
            tiempo_ga_ms=tiempo_ga_ms,
            tiempo_total_ms=tiempo_total_ms,
            seed_ga=seed_ga,
            generaciones_ga=generaciones_ga,
            error_ga=errores_ga,
        )

        return seleccionado

    def _fuente_rl_seleccionada(self) -> str:
        decision_rl = getattr(self.planner_rl, "ultima_decision", None)
        fuente = getattr(decision_rl, "fuente_seleccionada", None)
        if fuente is None:
            return "RL"
        return str(getattr(fuente, "value", fuente)).upper()

    @staticmethod
    def _plan_a_cromosoma(
        instancia: InstanciaTurno,
        plan: PlanTurno,
    ) -> Cromosoma:
        cromosoma = tuple(
            pedido_id
            for camion in plan.camiones
            for viaje in camion.viajes
            for pedido_id in viaje.pedido_ids
        )
        pedidos_por_id = {
            pedido.pedido_id: pedido
            for pedido in instancia.pedidos
        }
        validar_permutacion(pedidos_por_id, cromosoma)
        return cromosoma

    @staticmethod
    def _exigir_plan_ejecutable(
        *,
        instancia: InstanciaTurno,
        plan: PlanTurno,
        nombre: str,
    ) -> None:
        if plan is None:
            raise RuntimeError(f"El {nombre} es nulo.")

        validacion = validar_plan(instancia, plan)
        errores = list(validacion.errores)

        if not isfinite(plan.costo_estimado) or plan.costo_estimado < 0.0:
            errores.append(
                f"Costo inválido de {nombre}: {plan.costo_estimado}."
            )

        if errores:
            raise RuntimeError(
                f"El {nombre} no es ejecutable: " + " | ".join(errores)
            )


# Alias temporal para que imports históricos no fallen durante la migración.
HybridRLGAGreedyPlanner = HybridRLGAPlanner
