"""Hibridación secuencial RL -> GA: sembrado de poblaciones genéticas mediante Aprendizaje por Refuerzo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from time import perf_counter
from typing import Protocol

from planner.algorithms.ga import ConfiguracionGA, GeneticAlgorithmPlanner
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.routing.decoder import Cromosoma, validar_permutacion
from planner.routing.travel import ProveedorViaje

__all__ = [
    "PlanificadorCompatible",
    "GeneradorGARefinado",
    "FuenteResultadoHibrido",
    "MotivoResultadoHibrido",
    "ConfiguracionHibridaRLGA",
    "DecisionHibridaRLGA",
    "HybridRLGAPlanner",
]


class PlanificadorCompatible(Protocol):
    """Protocolo que debe satisfacer cualquier modelo o wrapper RL para sembrar el algoritmo genético."""

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno: ...


GeneradorGARefinado = Callable[[InstanciaTurno, Cromosoma], PlanTurno]


class FuenteResultadoHibrido(str, Enum):
    """Origen de la solución final seleccionada por el optimizador híbrido."""

    SEMILLA_RL = "SEMILLA_RL"
    REFINADO_GA = "REFINADO_GA"


class MotivoResultadoHibrido(str, Enum):
    """Causa por la cual se adoptó o descartó el refinamiento genético."""

    GA_MEJORA_SEMILLA_RL = "GA_MEJORA_SEMILLA_RL"
    SEMILLA_RL_CONSERVADA = "SEMILLA_RL_CONSERVADA"
    GA_NO_EJECUTABLE = "GA_NO_EJECUTABLE"


@dataclass(frozen=True)
class ConfiguracionHibridaRLGA:
    """Parámetros de tolerancia y configuración de la hibridación RL -> GA.

    Attributes:
        tolerancia_mejora: Umbral mínimo absoluto que el GA debe superar respecto al RL para reemplazarlo.
        fraccion_variantes_semilla_rl: Proporción de la población de GA poblada con variantes de la semilla RL.
        incluir_semilla_greedy_en_refinamiento: Si es `True`, agrega también la semilla voraz en el GA.
    """

    tolerancia_mejora: float = 1e-9
    fraccion_variantes_semilla_rl: float = 0.50
    incluir_semilla_greedy_en_refinamiento: bool = False

    def __post_init__(self) -> None:
        if self.tolerancia_mejora < 0.0:
            raise ValueError("tolerancia_mejora no puede ser negativa.")
        if not 0.0 <= self.fraccion_variantes_semilla_rl <= 1.0:
            raise ValueError("fraccion_variantes_semilla_rl debe estar entre 0 y 1.")


@dataclass(frozen=True)
class DecisionHibridaRLGA:
    """Diagnóstico detallado y comparativa de costos entre la solución RL y el refinamiento GA."""

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
        """Indica si el refinamiento genético logró reducir estrictamente el costo del plan RL."""
        return self.resultado == FuenteResultadoHibrido.REFINADO_GA


class HybridRLGAPlanner:
    """Planificador híbrido en dos fases: RL para arranque en caliente (warm-start) y GA para refinamiento.

    Flujo de ejecución:
    1. Obtiene una solución inicial completa mediante la política de Aprendizaje por Refuerzo.
    2. Transforma la secuencia de entregas del plan RL en un cromosoma de permutación.
    3. Inyecta el cromosoma y sus variantes vecinales en la población inicial del GA.
    4. Ejecuta la evolución metaheurística. Si el GA supera al RL por más de la tolerancia,
       se adopta el plan refinado; en caso contrario, se preserva de forma segura el plan RL.
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
        """Ejecuta la resolución híbrida evaluando y refinando la propuesta de RL.

        Args:
            instancia: Escenario del turno a planificar.

        Returns:
            PlanTurno óptimo consolidado, con advertencias diagnósticas y tiempos de cómputo.
        """
        inicio_total = perf_counter()

        inicio_rl = perf_counter()
        try:
            plan_rl = self.planner_rl.generar_plan(instancia)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo obtener la semilla RL del híbrido: {type(exc).__name__}: {exc}"
            ) from exc
        tiempo_rl_ms = (perf_counter() - inicio_rl) * 1000.0

        self._exigir_plan_ejecutable(instancia=instancia, plan=plan_rl, nombre="semilla RL")

        cromosoma_rl = self._plan_a_cromosoma(instancia, plan_rl)
        semillas_rl: list[Cromosoma] = [cromosoma_rl]
        for perm_extra in getattr(self.planner_rl, "ultimas_permutaciones", ()):
            if perm_extra and perm_extra not in semillas_rl:
                semillas_rl.append(tuple(perm_extra))

        fuente_rl = self._fuente_rl_seleccionada()
        costo_semilla_rl = float(plan_rl.costo_estimado)

        seed_base = (
            instancia.seed_ejecucion if instancia.seed_ejecucion > 0 else instancia.seed_escenario
        )
        seed_ga = seed_base + 9001
        tiempo_ga_ms = 0.0
        generaciones_ga = 0
        plan_ga: PlanTurno | None = None
        costo_ga: float | None = None
        errores_ga: tuple[str, ...] = ()

        inicio_ga = perf_counter()
        try:
            if self.generador_ga_refinado is not None:
                plan_ga = self.generador_ga_refinado(instancia, cromosoma_rl)
            else:
                planner_ga = GeneticAlgorithmPlanner(
                    configuracion=self.configuracion_planificacion,
                    configuracion_ga=self.configuracion_ga,
                    seed=seed_ga,
                    proveedor_viaje=self.proveedor_viaje,
                    semillas_iniciales=tuple(semillas_rl),
                    incluir_semilla_greedy=(
                        self.configuracion.incluir_semilla_greedy_en_refinamiento
                    ),
                    fraccion_variantes_semilla=(self.configuracion.fraccion_variantes_semilla_rl),
                )
                plan_ga = planner_ga.generar_plan(instancia)
                generaciones_ga = planner_ga.generaciones_ejecutadas

            self._exigir_plan_ejecutable(
                instancia=instancia, plan=plan_ga, nombre="refinamiento GA"
            )
            costo_ga = float(plan_ga.costo_estimado)

        except Exception as exc:  # noqa: BLE001 - queda auditado y preserva RL
            errores_ga = (f"{type(exc).__name__}: {exc}",)
            plan_ga = None
            costo_ga = None

        tiempo_ga_ms = (perf_counter() - inicio_ga) * 1000.0

        if (
            plan_ga is not None
            and costo_ga is not None
            and costo_ga < costo_semilla_rl - self.configuracion.tolerancia_mejora
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
            mejora_absoluta / costo_semilla_rl * 100.0 if costo_semilla_rl > 0.0 else 0.0
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
    def _plan_a_cromosoma(instancia: InstanciaTurno, plan: PlanTurno) -> Cromosoma:
        cromosoma = tuple(
            p_id for cam in plan.camiones for v in cam.viajes for p_id in v.pedido_ids
        )
        pedidos_dict = {p.pedido_id: p for p in instancia.pedidos}
        validar_permutacion(pedidos_dict, cromosoma)
        return cromosoma

    @staticmethod
    def _exigir_plan_ejecutable(*, instancia: InstanciaTurno, plan: PlanTurno, nombre: str) -> None:
        if plan is None:
            raise RuntimeError(f"El {nombre} es nulo.")

        validacion = validar_plan(instancia, plan)
        errores = list(validacion.errores)

        if not isfinite(plan.costo_estimado) or plan.costo_estimado < 0.0:
            errores.append(f"Costo inválido de {nombre}: {plan.costo_estimado}.")

        if errores:
            raise RuntimeError(f"El {nombre} no es ejecutable: " + " | ".join(errores))
