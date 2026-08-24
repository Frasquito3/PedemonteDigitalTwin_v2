from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from time import perf_counter
from typing import Callable, Protocol

from planner.algorithms.ga import generar_plan_ga
from planner.algorithms.greedy import generar_plan_greedy
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.routing.travel import ProveedorViaje


class PlanificadorCompatible(Protocol):
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        ...


GeneradorPlan = Callable[[InstanciaTurno], PlanTurno]


class FuentePlanHibridoRobusto(str, Enum):
    GREEDY = "GREEDY"
    GA = "GA"
    RL = "RL"


class MotivoSeleccionHibridaRobusta(str, Enum):
    GREEDY_MENOR_O_IGUAL = "GREEDY_MENOR_O_IGUAL"
    GA_MENOR_COSTO = "GA_MENOR_COSTO"
    RL_MENOR_COSTO = "RL_MENOR_COSTO"


@dataclass(frozen=True)
class ConfiguracionHibridaRobusta:
    tolerancia_empate: float = 1e-9
    permitir_fallback_ga: bool = True
    permitir_fallback_rl: bool = True

    def __post_init__(self) -> None:
        if self.tolerancia_empate < 0.0:
            raise ValueError(
                "tolerancia_empate no puede ser negativa."
            )


@dataclass(frozen=True)
class DecisionHibridaRobusta:
    instancia_id: str
    fuente_seleccionada: FuentePlanHibridoRobusto
    motivo: MotivoSeleccionHibridaRobusta
    costo_greedy: float
    costo_ga: float | None
    costo_rl: float | None
    tiempo_greedy_ms: float
    tiempo_ga_ms: float
    tiempo_rl_ms: float
    tiempo_total_ms: float
    seed_ga: int
    errores_ga: tuple[str, ...] = ()
    errores_rl: tuple[str, ...] = ()

    @property
    def cumple_garantia_greedy(self) -> bool:
        costo_seleccionado = {
            FuentePlanHibridoRobusto.GREEDY: self.costo_greedy,
            FuentePlanHibridoRobusto.GA: self.costo_ga,
            FuentePlanHibridoRobusto.RL: self.costo_rl,
        }[self.fuente_seleccionada]

        return (
            costo_seleccionado is not None
            and costo_seleccionado <= self.costo_greedy + 1e-9
        )

    @property
    def cumple_garantia_ga(self) -> bool | None:
        if self.costo_ga is None:
            return None

        costo_seleccionado = {
            FuentePlanHibridoRobusto.GREEDY: self.costo_greedy,
            FuentePlanHibridoRobusto.GA: self.costo_ga,
            FuentePlanHibridoRobusto.RL: self.costo_rl,
        }[self.fuente_seleccionada]

        return (
            costo_seleccionado is not None
            and costo_seleccionado <= self.costo_ga + 1e-9
        )


@dataclass(frozen=True)
class _CandidatoValido:
    fuente: FuentePlanHibridoRobusto
    plan: PlanTurno
    costo: float


class HybridRLGAGreedyPlanner:
    """
    Selector robusto que evalúa GREEDY, GA y RL sobre la misma instancia
    y devuelve el plan válido de menor costo estimado.

    Prioridad de desempate:

        GREEDY -> GA -> RL

    Con esa prioridad se conserva la solución más simple cuando los costos
    son equivalentes. Si GA es válido, la solución seleccionada nunca es
    peor que GA. Si GA falla, se conserva como mínimo la garantía frente a
    GREEDY. Los errores de GA o RL quedan auditados en ultima_decision.
    """

    def __init__(
        self,
        planner_rl: PlanificadorCompatible,
        configuracion: ConfiguracionHibridaRobusta | None = None,
        generador_greedy: GeneradorPlan | None = None,
        generador_ga: GeneradorPlan | None = None,
        configuracion_planificacion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
    ) -> None:
        self.planner_rl = planner_rl
        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionHibridaRobusta()
        )
        self.generador_greedy = generador_greedy
        self.generador_ga = generador_ga
        self.configuracion_planificacion = (
            configuracion_planificacion
            if configuracion_planificacion is not None
            else ConfiguracionPlanificacion()
        )
        self.proveedor_viaje = proveedor_viaje
        self.ultima_decision: DecisionHibridaRobusta | None = None

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        inicio_total = perf_counter()

        plan_greedy, tiempo_greedy_ms = self._generar_greedy(instancia)
        self._exigir_plan_base_valido(
            instancia=instancia,
            plan=plan_greedy,
            nombre="Greedy",
        )

        candidatos = [
            _CandidatoValido(
                fuente=FuentePlanHibridoRobusto.GREEDY,
                plan=plan_greedy,
                costo=plan_greedy.costo_estimado,
            )
        ]

        seed_ga = instancia.seed_escenario + 8001
        plan_ga, costo_ga, tiempo_ga_ms, errores_ga = (
            self._intentar_generar_candidato(
                instancia=instancia,
                nombre="GA",
                fuente=FuentePlanHibridoRobusto.GA,
                generador=lambda: self._generar_ga(instancia),
                permitir_fallback=self.configuracion.permitir_fallback_ga,
            )
        )
        if plan_ga is not None and costo_ga is not None:
            candidatos.append(
                _CandidatoValido(
                    fuente=FuentePlanHibridoRobusto.GA,
                    plan=plan_ga,
                    costo=costo_ga,
                )
            )

        plan_rl, costo_rl, tiempo_rl_ms, errores_rl = (
            self._intentar_generar_candidato(
                instancia=instancia,
                nombre="RL",
                fuente=FuentePlanHibridoRobusto.RL,
                generador=lambda: self.planner_rl.generar_plan(instancia),
                permitir_fallback=self.configuracion.permitir_fallback_rl,
            )
        )
        if plan_rl is not None and costo_rl is not None:
            candidatos.append(
                _CandidatoValido(
                    fuente=FuentePlanHibridoRobusto.RL,
                    plan=plan_rl,
                    costo=costo_rl,
                )
            )

        seleccionado = self._seleccionar_candidato(candidatos)
        tiempo_total_ms = (perf_counter() - inicio_total) * 1000.0
        seleccionado.plan.tiempo_computo_ms = tiempo_total_ms

        motivo = {
            FuentePlanHibridoRobusto.GREEDY: (
                MotivoSeleccionHibridaRobusta.GREEDY_MENOR_O_IGUAL
            ),
            FuentePlanHibridoRobusto.GA: (
                MotivoSeleccionHibridaRobusta.GA_MENOR_COSTO
            ),
            FuentePlanHibridoRobusto.RL: (
                MotivoSeleccionHibridaRobusta.RL_MENOR_COSTO
            ),
        }[seleccionado.fuente]

        self.ultima_decision = DecisionHibridaRobusta(
            instancia_id=instancia.instancia_id,
            fuente_seleccionada=seleccionado.fuente,
            motivo=motivo,
            costo_greedy=plan_greedy.costo_estimado,
            costo_ga=costo_ga,
            costo_rl=costo_rl,
            tiempo_greedy_ms=tiempo_greedy_ms,
            tiempo_ga_ms=tiempo_ga_ms,
            tiempo_rl_ms=tiempo_rl_ms,
            tiempo_total_ms=tiempo_total_ms,
            seed_ga=seed_ga,
            errores_ga=errores_ga,
            errores_rl=errores_rl,
        )

        if not self.ultima_decision.cumple_garantia_greedy:
            raise RuntimeError(
                "El híbrido robusto violó la garantía frente a Greedy."
            )

        if self.ultima_decision.cumple_garantia_ga is False:
            raise RuntimeError(
                "El híbrido robusto violó la garantía frente a GA."
            )

        return seleccionado.plan

    def _generar_greedy(
        self,
        instancia: InstanciaTurno,
    ) -> tuple[PlanTurno, float]:
        inicio = perf_counter()

        if self.generador_greedy is not None:
            plan = self.generador_greedy(instancia)
        else:
            plan = generar_plan_greedy(
                instancia,
                configuracion=self.configuracion_planificacion,
                proveedor_viaje=self.proveedor_viaje,
            )

        return plan, (perf_counter() - inicio) * 1000.0

    def _generar_ga(self, instancia: InstanciaTurno) -> PlanTurno:
        if self.generador_ga is not None:
            return self.generador_ga(instancia)

        return generar_plan_ga(
            instancia,
            seed=instancia.seed_escenario + 8001,
            configuracion=self.configuracion_planificacion,
            proveedor_viaje=self.proveedor_viaje,
        )

    def _intentar_generar_candidato(
        self,
        *,
        instancia: InstanciaTurno,
        nombre: str,
        fuente: FuentePlanHibridoRobusto,
        generador: Callable[[], PlanTurno],
        permitir_fallback: bool,
    ) -> tuple[
        PlanTurno | None,
        float | None,
        float,
        tuple[str, ...],
    ]:
        inicio = perf_counter()

        try:
            plan = generador()
        except Exception as exc:
            tiempo_ms = (perf_counter() - inicio) * 1000.0
            if not permitir_fallback:
                raise
            return (
                None,
                None,
                tiempo_ms,
                (f"{type(exc).__name__}: {exc}",),
            )

        tiempo_ms = (perf_counter() - inicio) * 1000.0
        validacion = validar_plan(instancia, plan)
        errores = list(validacion.errores)

        if not self._costo_valido(plan.costo_estimado):
            errores.append(
                f"Costo inválido de {nombre}: {plan.costo_estimado}."
            )

        if errores:
            if not permitir_fallback:
                raise RuntimeError(
                    f"El plan {nombre} es inválido: " + " | ".join(errores)
                )
            return None, None, tiempo_ms, tuple(errores)

        return plan, plan.costo_estimado, tiempo_ms, ()

    def _exigir_plan_base_valido(
        self,
        *,
        instancia: InstanciaTurno,
        plan: PlanTurno,
        nombre: str,
    ) -> None:
        validacion = validar_plan(instancia, plan)
        errores = list(validacion.errores)

        if not self._costo_valido(plan.costo_estimado):
            errores.append(
                f"Costo inválido de {nombre}: {plan.costo_estimado}."
            )

        if errores:
            raise RuntimeError(
                f"El plan {nombre} de referencia es inválido: "
                + " | ".join(errores)
            )

    def _seleccionar_candidato(
        self,
        candidatos: list[_CandidatoValido],
    ) -> _CandidatoValido:
        mejor_costo = min(candidato.costo for candidato in candidatos)
        tolerancia = self.configuracion.tolerancia_empate

        elegibles = {
            candidato.fuente: candidato
            for candidato in candidatos
            if candidato.costo <= mejor_costo + tolerancia
        }

        for fuente in (
            FuentePlanHibridoRobusto.GREEDY,
            FuentePlanHibridoRobusto.GA,
            FuentePlanHibridoRobusto.RL,
        ):
            candidato = elegibles.get(fuente)
            if candidato is not None:
                return candidato

        raise RuntimeError(
            "No se pudo seleccionar un candidato híbrido válido."
        )

    @staticmethod
    def _costo_valido(costo: float) -> bool:
        return isfinite(costo) and costo >= 0.0
