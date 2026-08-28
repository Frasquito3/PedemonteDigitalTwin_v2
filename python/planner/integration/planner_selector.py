from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Protocol

from planner.algorithms.ga import (
    generar_plan_ga,
)
from planner.algorithms.greedy import (
    generar_plan_greedy,
)
from planner.algorithms.hybrid_rl_ga_greedy import (
    HybridRLGAPlanner,
)
from planner.algorithms.random_feasible import (
    generar_plan_random,
)
from planner.core.config import ConfiguracionPlanificacion

from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PlanTurno,
)
from planner.routing.travel import ProveedorViaje

from planner.domain.validator import (
    validar_plan,
)


class PlanificadorCompatible(
    Protocol
):
    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        ...


class ModoPlanificacion(
    str,
    Enum,
):
    GREEDY = "GREEDY"
    RANDOM = "RANDOM"
    GA = "GA"
    RL = "RL"
    HIBRIDO = "HIBRIDO"


@dataclass(frozen=True)
class DecisionSelector:
    instancia_id: str

    modo_solicitado: ModoPlanificacion

    algoritmo_resultante: (
        AlgoritmoPlanificacion
    )

    costo_estimado: float

    tiempo_plan_ms: float

    tiempo_selector_ms: float

    detalle: str


class SelectorPlanificadores:
    """
    Fachada común para seleccionar el algoritmo
    encargado de generar el plan.

    El modelo RL se carga de forma diferida. Si model_path_rl apunta
    a un manifiesto JSON, se utiliza la política operacional temporal v4;
    si apunta a un ZIP, se conserva el planificador RL histórico.
    GREEDY, RANDOM y GA no necesitan cargarlo.
    """

    def __init__(
        self,
        model_path_rl:
            str
            | Path
            | None = None,
        planner_rl:
            PlanificadorCompatible
            | None = None,
        configuracion:
            ConfiguracionPlanificacion
            | None = None,
        proveedor_viaje:
            ProveedorViaje
            | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
    ) -> None:
        if max_pedidos <= 0:
            raise ValueError(
                "max_pedidos debe ser > 0."
            )

        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )

        self.proveedor_viaje = proveedor_viaje

        self.max_pedidos = (
            max_pedidos
        )

        self.deterministic = (
            deterministic
        )

        self.model_path_rl: (
            Path
            | None
        ) = None

        if model_path_rl is not None:
            ruta = (
                Path(
                    model_path_rl
                )
                .expanduser()
                .resolve()
            )

            if not ruta.is_file():
                raise FileNotFoundError(
                    "No existe el modelo RL: "
                    f"{ruta}"
                )

            self.model_path_rl = ruta

        self._planner_rl = (
            planner_rl
        )

        self._planner_hibrido: (
            HybridRLGAPlanner
            | None
        ) = None

        self.ultima_decision: (
            DecisionSelector
            | None
        ) = None

    def precargar_rl(
        self,
    ) -> None:
        """
        Carga anticipadamente el modelo RL.

        Pypeline la utiliza durante la inicialización
        para que un error de modelo aparezca antes
        de solicitar el primer plan.
        """
        self._obtener_planner_rl()

        self._obtener_planner_hibrido()

    def generar_plan(
        self,
        instancia: InstanciaTurno,
        modo:
            ModoPlanificacion
            | str,
    ) -> PlanTurno:
        modo_normalizado = (
            normalizar_modo(
                modo
            )
        )

        inicio_selector = (
            perf_counter()
        )

        detalle = ""

        if (
            modo_normalizado
            == ModoPlanificacion.GREEDY
        ):
            plan = generar_plan_greedy(
                instancia,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )

        elif (
            modo_normalizado
            == ModoPlanificacion.RANDOM
        ):
            seed = (
                instancia.seed_escenario
                + 7001
            )

            plan = generar_plan_random(
                instancia,
                seed=seed,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )

            detalle = (
                f"seed={seed}"
            )

        elif (
            modo_normalizado
            == ModoPlanificacion.GA
        ):
            seed = (
                instancia.seed_escenario
                + 8001
            )

            plan = generar_plan_ga(
                instancia,
                seed=seed,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )

            detalle = (
                f"seed={seed}"
            )

        elif (
            modo_normalizado
            == ModoPlanificacion.RL
        ):
            planner_rl = (
                self
                ._obtener_planner_rl()
            )

            plan = (
                planner_rl
                .generar_plan(
                    instancia
                )
            )

            detalle_operacional = getattr(
                planner_rl,
                "ultimo_detalle",
                "",
            )

            if detalle_operacional:
                detalle = str(
                    detalle_operacional
                )

        else:
            planner_hibrido = (
                self
                ._obtener_planner_hibrido()
            )

            plan = (
                planner_hibrido
                .generar_plan(
                    instancia
                )
            )

            decision_hibrida = (
                planner_hibrido
                .ultima_decision
            )

            if (
                decision_hibrida
                is not None
            ):
                error_ga = (
                    ";".join(
                        decision_hibrida.error_ga
                    )
                    if decision_hibrida.error_ga
                    else "NINGUNO"
                )

                costo_refinado = (
                    "NO_DISPONIBLE"
                    if decision_hibrida.costo_refinado_ga is None
                    else str(decision_hibrida.costo_refinado_ga)
                )

                detalle = (
                    "arquitectura=RL_GA_SEEDED"
                    "|fuente_rl="
                    f"{decision_hibrida.fuente_rl}"
                    "|resultado="
                    f"{decision_hibrida.resultado.value}"
                    "|motivo="
                    f"{decision_hibrida.motivo.value}"
                    "|seed_ga="
                    f"{decision_hibrida.seed_ga}"
                    "|costo_semilla_rl="
                    f"{decision_hibrida.costo_semilla_rl}"
                    "|costo_refinado_ga="
                    f"{costo_refinado}"
                    "|costo_final="
                    f"{decision_hibrida.costo_final}"
                    "|mejora_abs="
                    f"{decision_hibrida.mejora_absoluta}"
                    "|mejora_pct="
                    f"{decision_hibrida.mejora_porcentual}"
                    "|generaciones_ga="
                    f"{decision_hibrida.generaciones_ga}"
                    "|error_ga="
                    f"{error_ga}"
                )

        validacion = validar_plan(
            instancia,
            plan,
        )

        if not validacion.valido:
            raise RuntimeError(
                "El selector recibió un "
                "plan inválido: "
                + " | ".join(
                    validacion.errores
                )
            )

        if (
            not isfinite(
                plan.costo_estimado
            )
            or plan.costo_estimado
            < 0.0
        ):
            raise RuntimeError(
                "El plan seleccionado tiene "
                "un costo estimado inválido: "
                f"{plan.costo_estimado}."
            )

        tiempo_selector_ms = (
            (
                perf_counter()
                - inicio_selector
            )
            * 1000.0
        )

        self.ultima_decision = (
            DecisionSelector(
                instancia_id=(
                    instancia.instancia_id
                ),
                modo_solicitado=(
                    modo_normalizado
                ),
                algoritmo_resultante=(
                    plan.algoritmo
                ),
                costo_estimado=(
                    plan.costo_estimado
                ),
                tiempo_plan_ms=(
                    plan.tiempo_computo_ms
                ),
                tiempo_selector_ms=(
                    tiempo_selector_ms
                ),
                detalle=detalle,
            )
        )

        return plan

    def _obtener_planner_rl(
        self,
    ) -> PlanificadorCompatible:
        if self._planner_rl is not None:
            return self._planner_rl

        if self.model_path_rl is None:
            raise RuntimeError(
                "El modo RL requiere "
                "model_path_rl o un "
                "planner_rl inyectado."
            )

        if (
            self.model_path_rl.suffix.lower()
            == ".json"
        ):
            from planner.rl.rl_temporal_v4_operational import (
                RLTemporalV4OperationalPlanner,
            )

            self._planner_rl = (
                RLTemporalV4OperationalPlanner(
                    manifest_path=(
                        self.model_path_rl
                    ),
                    max_pedidos=(
                        self.max_pedidos
                    ),
                    deterministic=(
                        self.deterministic
                    ),
                    configuracion=(
                        self.configuracion
                    ),
                    proveedor_viaje=(
                        self.proveedor_viaje
                    ),
                )
            )

        else:
            from planner.rl.planner import (
                RLPlanner,
            )

            self._planner_rl = RLPlanner(
                model_path=(
                    self.model_path_rl
                ),
                max_pedidos=(
                    self.max_pedidos
                ),
                deterministic=(
                    self.deterministic
                ),
                configuracion=(
                    self.configuracion
                ),
                proveedor_viaje=(
                    self.proveedor_viaje
                ),
            )

        return self._planner_rl

    def _obtener_planner_hibrido(
        self,
    ) -> HybridRLGAPlanner:
        if (
            self._planner_hibrido
            is None
        ):
            self._planner_hibrido = (
                HybridRLGAPlanner(
                    planner_rl=(
                        self
                        ._obtener_planner_rl()
                    ),
                    configuracion_planificacion=(
                        self.configuracion
                    ),
                    proveedor_viaje=(
                        self.proveedor_viaje
                    ),
                )
            )

        return self._planner_hibrido


def normalizar_modo(
    modo:
        ModoPlanificacion
        | str,
) -> ModoPlanificacion:
    if isinstance(
        modo,
        ModoPlanificacion,
    ):
        return modo

    texto = (
        str(
            modo
        )
        .strip()
        .upper()
    )

    alias = {
        "HYBRID": "HIBRIDO",
        "HÍBRIDO": "HIBRIDO",
        "GENETIC": "GA",
        "GENETICO": "GA",
        "GENÉTICO": "GA",
    }

    texto = alias.get(
        texto,
        texto,
    )

    try:
        return ModoPlanificacion(
            texto
        )

    except ValueError as exc:
        disponibles = ", ".join(
            item.value
            for item
            in ModoPlanificacion
        )

        raise ValueError(
            "Modo de planificación "
            "no soportado: "
            f"{modo!r}. "
            "Disponibles: "
            f"{disponibles}."
        ) from exc