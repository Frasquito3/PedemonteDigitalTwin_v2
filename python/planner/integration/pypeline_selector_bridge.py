from __future__ import annotations

from pathlib import Path
from typing import Sequence

from planner.integration.alpyne_codec import (
    codificar_plan_alpyne,
)
from planner.integration.planner_selector import (
    ModoPlanificacion,
    SelectorPlanificadores,
)
from planner.integration.pypeline_bridge import (
    decodificar_instancia_vector,
)


_selector: (
    SelectorPlanificadores
    | None
) = None

_modelo: (
    Path
    | None
) = None

_ultima_decision = (
    "SIN_DECISION"
)


def inicializar(
    model_path: str,
    max_pedidos: int = 30,
    deterministic: bool = True,
) -> str:
    global _selector
    global _modelo
    global _ultima_decision

    ruta = (
        Path(
            model_path
        )
        .expanduser()
        .resolve()
    )

    if not ruta.is_file():
        raise FileNotFoundError(
            "No existe el modelo RL: "
            f"{ruta}"
        )

    if (
        _selector is not None
        and _modelo == ruta
    ):
        return (
            "OK|REUTILIZADO|"
            f"modelo={ruta}"
        )

    selector = (
        SelectorPlanificadores(
            model_path_rl=ruta,
            max_pedidos=(
                max_pedidos
            ),
            deterministic=(
                deterministic
            ),
        )
    )

    selector.precargar_rl()

    _selector = selector

    _modelo = ruta

    _ultima_decision = (
        "SIN_DECISION"
    )

    return (
        "OK|CARGADO|"
        f"modelo={ruta}"
    )


def reiniciar() -> str:
    global _selector
    global _modelo
    global _ultima_decision

    _selector = None

    _modelo = None

    _ultima_decision = (
        "SIN_DECISION"
    )

    return "OK|REINICIADO"


def obtener_estado() -> str:
    if _selector is None:
        return "NO_INICIALIZADO"

    return (
        "INICIALIZADO|"
        f"modelo={_modelo}"
    )


def obtener_modos_disponibles() -> str:
    return "|".join(
        modo.value
        for modo
        in ModoPlanificacion
    )


def obtener_ultima_decision() -> str:
    return _ultima_decision


def planificar_vector(
    instancia_vector:
        Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
    modo_planificacion: str,
) -> list[float]:
    global _ultima_decision

    if _selector is None:
        raise RuntimeError(
            "El selector Pypeline "
            "no fue inicializado."
        )

    instancia = (
        decodificar_instancia_vector(
            instancia_vector,
            seed_escenario,
            seed_ejecucion,
        )
    )

    plan = _selector.generar_plan(
        instancia,
        modo_planificacion,
    )

    decision = (
        _selector
        .ultima_decision
    )

    if decision is None:
        _ultima_decision = (
            "SIN_DECISION"
        )

    else:
        _ultima_decision = (
            "modo="
            f"{decision.modo_solicitado.value}"
            "|algoritmo="
            f"{decision.algoritmo_resultante.value}"
            "|costo="
            f"{decision.costo_estimado}"
            "|tiempo_plan_ms="
            f"{decision.tiempo_plan_ms}"
            "|tiempo_selector_ms="
            f"{decision.tiempo_selector_ms}"
            "|detalle="
            f"{decision.detalle}"
        )

    return codificar_plan_alpyne(
        instancia,
        plan,
    )