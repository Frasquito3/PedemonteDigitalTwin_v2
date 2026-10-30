"""Selector unificado y fachada de algoritmos de planificación.

Este módulo implementa el despachador (`SelectorPlanificadores`) que orquesta la
generación de planes para cada modo solicitado (`GREEDY`, `RANDOM`, `GA`, `RL`, `HIBRIDO`),
gestionando la carga diferida (lazy-loading) de redes neuronales y manteniendo trazabilidad
completa de las decisiones y tiempos de procesamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Protocol

from planner.algorithms.ga import generar_plan_ga
from planner.algorithms.greedy import generar_plan_greedy
from planner.algorithms.hybrid_rl_ga import HybridRLGAPlanner
from planner.algorithms.random_feasible import generar_plan_random
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import AlgoritmoPlanificacion, InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.routing.travel import ProveedorViaje

__all__ = [
    "DecisionSelector",
    "ModoPlanificacion",
    "PlanificadorCompatible",
    "SelectorPlanificadores",
    "normalizar_modo",
]


class PlanificadorCompatible(Protocol):
    """Protocolo estructural que deben satisfacer los planificadores autónomos."""

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        """Genera un plan de ruteo para la instancia dada."""
        ...


class ModoPlanificacion(str, Enum):
    """Estrategias de resolución disponibles para el selector."""

    GREEDY = "GREEDY"
    RANDOM = "RANDOM"
    GA = "GA"
    RL = "RL"
    HIBRIDO = "HIBRIDO"


@dataclass(frozen=True)
class DecisionSelector:
    """Registro de trazabilidad y auditoría de una invocación del selector."""

    instancia_id: str
    modo_solicitado: ModoPlanificacion
    algoritmo_resultante: AlgoritmoPlanificacion
    costo_estimado: float
    tiempo_plan_ms: float
    tiempo_selector_ms: float
    detalle: str


class SelectorPlanificadores:
    """Fachada unificada para seleccionar y ejecutar algoritmos de ruteo.

    Los modelos de Aprendizaje por Refuerzo e Híbridos se inicializan de forma
    perezosa (lazy) desde disco cuando se solicita `RL` o `HIBRIDO`, evitando
    sobrecargar la memoria si solo se usan métodos heurísticos clásicos.
    """

    def __init__(
        self,
        model_path_rl: str | Path | None = None,
        planner_rl: PlanificadorCompatible | None = None,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
    ) -> None:
        if max_pedidos <= 0:
            raise ValueError("max_pedidos debe ser > 0.")

        self.configuracion = (
            configuracion if configuracion is not None else ConfiguracionPlanificacion()
        )
        self.proveedor_viaje = proveedor_viaje
        self.max_pedidos = max_pedidos
        self.deterministic = deterministic

        self.model_path_rl: Path | None = None
        if model_path_rl is not None:
            ruta = Path(model_path_rl).expanduser().resolve()
            if not ruta.is_file():
                raise FileNotFoundError(f"No existe el modelo RL: {ruta}")
            self.model_path_rl = ruta

        self._planner_rl = planner_rl
        self._planner_hibrido: HybridRLGAPlanner | None = None
        self.ultima_decision: DecisionSelector | None = None

    def precargar_rl(self) -> None:
        """Carga anticipadamente el modelo RL y el planificador híbrido."""
        self._obtener_planner_rl()
        self._obtener_planner_hibrido()

    def generar_plan(
        self,
        instancia: InstanciaTurno,
        modo: ModoPlanificacion | str,
    ) -> PlanTurno:
        """Genera un plan de ruteo para la instancia según el modo seleccionado.

        Args:
            instancia: Instancia del turno con sus pedidos y restricciones.
            modo: Estrategia de planificación solicitada.

        Returns:
            PlanTurno validado y con costo calculado.

        Raises:
            RuntimeError: Si el plan resultante es inválido o inviable.
        """
        modo_normalizado = normalizar_modo(modo)
        inicio_selector = perf_counter()

        seed_base = (
            instancia.seed_ejecucion if instancia.seed_ejecucion > 0 else instancia.seed_escenario
        )
        detalle = f"seed={seed_base}"

        if modo_normalizado == ModoPlanificacion.GREEDY:
            plan = generar_plan_greedy(
                instancia,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )
        elif modo_normalizado == ModoPlanificacion.RANDOM:
            seed = seed_base + 7001
            plan = generar_plan_random(
                instancia,
                seed=seed,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )
            detalle = f"seed={seed}"
        elif modo_normalizado == ModoPlanificacion.GA:
            seed = seed_base + 8001
            plan = generar_plan_ga(
                instancia,
                seed=seed,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )
            detalle = f"seed={seed}"
        elif modo_normalizado == ModoPlanificacion.RL:
            planner_rl = self._obtener_planner_rl()
            plan = planner_rl.generar_plan(instancia)
            detalle_operacional = getattr(planner_rl, "ultimo_detalle", "")
            if detalle_operacional:
                detalle = f"{detalle_operacional}|seed={seed_base}"
        else:
            planner_hibrido = self._obtener_planner_hibrido()
            plan = planner_hibrido.generar_plan(instancia)
            decision_hibrida = planner_hibrido.ultima_decision

            if decision_hibrida is not None:
                error_ga = (
                    ";".join(decision_hibrida.error_ga) if decision_hibrida.error_ga else "NINGUNO"
                )
                costo_refinado = (
                    "NO_DISPONIBLE"
                    if decision_hibrida.costo_refinado_ga is None
                    else str(decision_hibrida.costo_refinado_ga)
                )
                detalle = (
                    f"arquitectura=RL_GA_SEEDED|fuente_rl={decision_hibrida.fuente_rl}"
                    f"|resultado={decision_hibrida.resultado.value}|motivo={decision_hibrida.motivo.value}"
                    f"|seed_ga={decision_hibrida.seed_ga}|costo_semilla_rl={decision_hibrida.costo_semilla_rl}"
                    f"|costo_refinado_ga={costo_refinado}|costo_final={decision_hibrida.costo_final}"
                    f"|mejora_abs={decision_hibrida.mejora_absoluta}|mejora_pct={decision_hibrida.mejora_porcentual}"
                    f"|generaciones_ga={decision_hibrida.generaciones_ga}|error_ga={error_ga}"
                )

        validacion = validar_plan(instancia, plan)
        if not validacion.valido:
            raise RuntimeError(
                "El selector recibió un plan inválido: " + " | ".join(validacion.errores)
            )

        if not isfinite(plan.costo_estimado) or plan.costo_estimado < 0.0:
            raise RuntimeError(
                f"El plan seleccionado tiene un costo estimado inválido: {plan.costo_estimado}."
            )

        tiempo_selector_ms = (perf_counter() - inicio_selector) * 1000.0
        self.ultima_decision = DecisionSelector(
            instancia_id=instancia.instancia_id,
            modo_solicitado=modo_normalizado,
            algoritmo_resultante=plan.algoritmo,
            costo_estimado=plan.costo_estimado,
            tiempo_plan_ms=plan.tiempo_computo_ms,
            tiempo_selector_ms=tiempo_selector_ms,
            detalle=detalle,
        )

        return plan

    def _obtener_planner_rl(self) -> PlanificadorCompatible:
        if self._planner_rl is not None:
            return self._planner_rl

        if self.model_path_rl is None:
            raise RuntimeError("El modo RL requiere model_path_rl o un planner_rl inyectado.")

        if self.model_path_rl.suffix.lower() != ".json":
            raise ValueError(
                f"El modo RL requiere el manifiesto JSON de políticas: {self.model_path_rl}"
            )

        from planner.rl.policy_runtime import PlanificadorOperativoRL

        self._planner_rl = PlanificadorOperativoRL(
            manifest_path=self.model_path_rl,
            max_pedidos=self.max_pedidos,
            deterministic=self.deterministic,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
        )
        return self._planner_rl

    def _obtener_planner_hibrido(self) -> HybridRLGAPlanner:
        if self._planner_hibrido is None:
            self._planner_hibrido = HybridRLGAPlanner(
                planner_rl=self._obtener_planner_rl(),
                configuracion_planificacion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )
        return self._planner_hibrido


def normalizar_modo(modo: ModoPlanificacion | str) -> ModoPlanificacion:
    """Normaliza cadenas o instancias a su correspondiente enum `ModoPlanificacion`."""
    if isinstance(modo, ModoPlanificacion):
        return modo

    texto = str(modo).strip().upper()
    alias = {
        "HYBRID": "HIBRIDO",
        "HÍBRIDO": "HIBRIDO",
        "GENETIC": "GA",
        "GENETICO": "GA",
        "GENÉTICO": "GA",
    }
    texto = alias.get(texto, texto)

    try:
        return ModoPlanificacion(texto)
    except ValueError as exc:
        disponibles = ", ".join(item.value for item in ModoPlanificacion)
        raise ValueError(
            f"Modo de planificación no soportado: {modo!r}. Disponibles: {disponibles}."
        ) from exc
