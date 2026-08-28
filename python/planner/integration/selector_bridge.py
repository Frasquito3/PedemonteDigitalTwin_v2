from __future__ import annotations

from math import isclose
from pathlib import Path
from typing import Sequence

from planner.core.schema import (
    InstanciaTurno,
    PlanTurno,
)
from planner.integration.alpyne_codec import (
    codificar_plan_alpyne,
)
from planner.integration.planner_selector import (
    ModoPlanificacion,
    SelectorPlanificadores,
)
from planner.integration.instance_vector_codec import (
    decodificar_instancia_vector,
)
from planner.routing.objective import (
    evaluar_plan_estimado,
    serializar_auditoria_estimacion,
)
from planner.routing.travel import (
    ProveedorHaversineAjustado,
    ProveedorViaje,
    construir_matriz_viaje,
)
from planner.routing.vial_cache import (
    ProveedorVialCachePersistente,
)


_selector: (
    SelectorPlanificadores
    | None
) = None

_modelo: (
    Path
    | None
) = None

_proveedor_viaje: (
    ProveedorViaje
    | None
) = None

_firma_inicializacion: (
    tuple[
        str,
        int,
        bool,
        str,
        str,
        bool,
    ]
    | None
) = None

_resumen_proveedor = (
    "proveedor=NO_INICIALIZADO"
)

_ultima_decision = (
    "SIN_DECISION"
)

_ultima_auditoria_estimacion = (
    "SIN_AUDITORIA"
)


def inicializar(
    model_path: str,
    max_pedidos: int = 30,
    deterministic: bool = True,
    cache_vial_path: str | None = None,
    version_cache_vial: str = "pedemonte-vial-v1",
    permitir_fallback_vial: bool = False,
) -> str:
    """
    Inicializa el selector y su proveedor común de viajes.

    Si cache_vial_path está informado, todos los algoritmos reciben
    una misma instancia de ProveedorVialCachePersistente.

    La firma de inicialización incluye la huella SHA-256 de la caché.
    Si el CSV cambia, el selector se reconstruye aunque la ruta del
    archivo y el modelo RL permanezcan iguales.
    """
    global _selector
    global _modelo
    global _proveedor_viaje
    global _firma_inicializacion
    global _resumen_proveedor
    global _ultima_decision
    global _ultima_auditoria_estimacion

    if max_pedidos <= 0:
        raise ValueError(
            "max_pedidos debe ser > 0."
        )

    ruta_modelo = (
        Path(model_path)
        .expanduser()
        .resolve()
    )

    if not ruta_modelo.is_file():
        raise FileNotFoundError(
            "No existe el modelo RL: "
            f"{ruta_modelo}"
        )

    proveedor_viaje, resumen_proveedor = (
        _crear_proveedor_viaje(
            cache_vial_path=cache_vial_path,
            version_cache_vial=version_cache_vial,
            permitir_fallback_vial=(
                permitir_fallback_vial
            ),
        )
    )

    firma = (
        str(ruta_modelo),
        max_pedidos,
        deterministic,
        proveedor_viaje.fuente.value,
        proveedor_viaje.version,
        permitir_fallback_vial,
    )

    if (
        _selector is not None
        and _firma_inicializacion == firma
    ):
        return (
            "OK|REUTILIZADO|"
            f"modelo={ruta_modelo}|"
            f"{resumen_proveedor}"
        )

    selector = SelectorPlanificadores(
        model_path_rl=ruta_modelo,
        max_pedidos=max_pedidos,
        deterministic=deterministic,
        proveedor_viaje=proveedor_viaje,
    )

    selector.precargar_rl()

    _selector = selector
    _modelo = ruta_modelo
    _proveedor_viaje = proveedor_viaje
    _firma_inicializacion = firma
    _resumen_proveedor = resumen_proveedor
    _ultima_decision = "SIN_DECISION"
    _ultima_auditoria_estimacion = (
        "SIN_AUDITORIA"
    )

    return (
        "OK|CARGADO|"
        f"modelo={ruta_modelo}|"
        f"{resumen_proveedor}"
    )


def reiniciar() -> str:
    global _selector
    global _modelo
    global _proveedor_viaje
    global _firma_inicializacion
    global _resumen_proveedor
    global _ultima_decision
    global _ultima_auditoria_estimacion

    _selector = None
    _modelo = None
    _proveedor_viaje = None
    _firma_inicializacion = None
    _resumen_proveedor = (
        "proveedor=NO_INICIALIZADO"
    )
    _ultima_decision = "SIN_DECISION"
    _ultima_auditoria_estimacion = (
        "SIN_AUDITORIA"
    )

    return "OK|REINICIADO"


def obtener_estado() -> str:
    if _selector is None:
        return "NO_INICIALIZADO"

    return (
        "INICIALIZADO|"
        f"modelo={_modelo}|"
        f"{_resumen_proveedor}"
    )


def obtener_fuente_viaje() -> str:
    return _resumen_proveedor


def obtener_modos_disponibles() -> str:
    return "|".join(
        modo.value
        for modo
        in ModoPlanificacion
    )


def obtener_ultima_decision() -> str:
    return _ultima_decision


def obtener_ultima_auditoria_estimacion() -> str:
    return _ultima_auditoria_estimacion


def planificar_vector(
    instancia_vector: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
    modo_planificacion: str,
) -> list[float]:
    global _ultima_decision
    global _ultima_auditoria_estimacion

    if _selector is None:
        raise RuntimeError(
            "El selector Pypeline "
            "no fue inicializado."
        )

    instancia = decodificar_instancia_vector(
        instancia_vector,
        seed_escenario,
        seed_ejecucion,
    )

    _ultima_auditoria_estimacion = (
        "SIN_AUDITORIA"
    )

    plan = _selector.generar_plan(
        instancia,
        modo_planificacion,
    )

    _ultima_auditoria_estimacion = (
        _auditar_plan_estimado(
            instancia,
            plan,
        )
    )

    decision = _selector.ultima_decision

    if decision is None:
        _ultima_decision = (
            "SIN_DECISION|"
            f"{_resumen_proveedor}|"
            "auditoria_estimacion="
            f"{_ultima_auditoria_estimacion}"
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
            "|"
            f"{_resumen_proveedor}"
            "|auditoria_estimacion="
            f"{_ultima_auditoria_estimacion}"
        )

    return codificar_plan_alpyne(
        instancia,
        plan,
    )


def _auditar_plan_estimado(
    instancia: InstanciaTurno,
    plan: PlanTurno,
) -> str:
    if _selector is None:
        raise RuntimeError(
            "No existe selector para auditar el plan."
        )

    if _proveedor_viaje is None:
        raise RuntimeError(
            "No existe proveedor de viaje para "
            "auditar el plan."
        )

    matriz = construir_matriz_viaje(
        instancia,
        _selector.configuracion,
        proveedor=_proveedor_viaje,
    )

    estimacion = evaluar_plan_estimado(
        instancia,
        plan,
        matriz,
        _selector.configuracion,
    )

    if not isclose(
        plan.costo_estimado,
        estimacion.costo_total,
        rel_tol=1e-10,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "El costo estimado del plan no coincide "
            "con su auditoría. "
            f"plan={plan.costo_estimado}, "
            f"auditoria={estimacion.costo_total}."
        )

    return serializar_auditoria_estimacion(
        estimacion
    )


def _crear_proveedor_viaje(
    *,
    cache_vial_path: str | None,
    version_cache_vial: str,
    permitir_fallback_vial: bool,
) -> tuple[ProveedorViaje, str]:
    texto_cache = (
        ""
        if cache_vial_path is None
        else str(cache_vial_path).strip()
    )

    if not texto_cache:
        proveedor = (
            ProveedorHaversineAjustado()
        )

        return (
            proveedor,
            "proveedor="
            f"{proveedor.fuente.value}|"
            "version_viaje="
            f"{proveedor.version}|"
            "cache_tramos=NA|"
            "fallback=NA",
        )

    ruta_cache = (
        Path(texto_cache)
        .expanduser()
        .resolve()
    )

    proveedor_cache = (
        ProveedorVialCachePersistente(
            ruta_cache,
            version_cache_esperada=(
                version_cache_vial
            ),
            permitir_fallback=(
                permitir_fallback_vial
            ),
        )
    )

    estadisticas = (
        proveedor_cache.estadisticas
    )

    modo_fallback = (
        "PERMITIDO"
        if permitir_fallback_vial
        else "ESTRICTO"
    )

    return (
        proveedor_cache,
        "proveedor="
        f"{proveedor_cache.fuente.value}|"
        "version_viaje="
        f"{proveedor_cache.version}|"
        "cache_tramos="
        f"{estadisticas.cantidad_tramos}|"
        "cache_archivo="
        f"{ruta_cache}|"
        "fallback="
        f"{modo_fallback}",
    )
