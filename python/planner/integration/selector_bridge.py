"""Puente de ejecución y estado global para AnyLogic vía Pypeline.

Este módulo expone los puntos de entrada serializados que AnyLogic invoca durante
la simulación para inicializar el motor de planificación, despachar planes de ruteo
en tiempo real y ejecutar comparaciones multialgoritmo conservando el estado del modelo.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isclose
from pathlib import Path

from planner.core.schema import InstanciaTurno, PlanTurno
from planner.integration.alpyne_codec import codificar_plan_alpyne
from planner.integration.estimated_comparison import (
    ComparacionEstimada,
    codificar_comparacion_estimada,
    ejecutar_comparacion_estimada,
    firmar_instancia_vector,
    serializar_resultado_metodo,
    serializar_resumen_comparacion,
)
from planner.integration.instance_vector_codec import decodificar_instancia_vector
from planner.integration.planner_selector import ModoPlanificacion, SelectorPlanificadores
from planner.routing.objective import evaluar_plan_estimado, serializar_auditoria_estimacion
from planner.routing.travel import (
    ProveedorHaversineAjustado,
    ProveedorViaje,
    construir_matriz_viaje,
)
from planner.routing.vial_cache import ProveedorVialCachePersistente

__all__ = [
    "comparar_estimado_vector",
    "inicializar",
    "limpiar_comparacion_estimada",
    "obtener_estado",
    "obtener_firma_comparacion_estimada",
    "obtener_fuente_viaje",
    "obtener_modos_disponibles",
    "obtener_plan_comparacion_vector",
    "obtener_resultado_comparacion_estimado",
    "obtener_resumen_comparacion_estimada",
    "obtener_ultima_auditoria_estimacion",
    "obtener_ultima_decision",
    "planificar_vector",
    "reiniciar",
]

_selector: SelectorPlanificadores | None = None
_modelo: Path | None = None
_proveedor_viaje: ProveedorViaje | None = None
_firma_inicializacion: tuple[str, int, bool, str, str, bool] | None = None
_resumen_proveedor: str = "proveedor=NO_INICIALIZADO"
_ultima_decision: str = "SIN_DECISION"
_ultima_auditoria_estimacion: str = "SIN_AUDITORIA"
_ultima_comparacion_estimada: ComparacionEstimada | None = None


def inicializar(
    model_path: str,
    max_pedidos: int = 30,
    deterministic: bool = True,
    cache_vial_path: str | None = None,
    version_cache_vial: str = "pedemonte-vial-v1",
    permitir_fallback_vial: bool = False,
) -> str:
    """Inicializa el selector global y el proveedor común de viajes."""
    global _selector, _modelo, _proveedor_viaje, _firma_inicializacion
    global _resumen_proveedor, _ultima_decision, _ultima_auditoria_estimacion
    global _ultima_comparacion_estimada

    if max_pedidos <= 0:
        raise ValueError("max_pedidos debe ser > 0.")

    ruta_modelo = Path(model_path).expanduser().resolve()
    if not ruta_modelo.is_file():
        raise FileNotFoundError(f"No existe el modelo RL: {ruta_modelo}")

    proveedor_viaje, resumen_proveedor = _crear_proveedor_viaje(
        cache_vial_path=cache_vial_path,
        version_cache_vial=version_cache_vial,
        permitir_fallback_vial=permitir_fallback_vial,
    )

    firma = (
        str(ruta_modelo),
        max_pedidos,
        deterministic,
        proveedor_viaje.fuente.value,
        proveedor_viaje.version,
        permitir_fallback_vial,
    )

    if _selector is not None and _firma_inicializacion == firma:
        _ultima_comparacion_estimada = None
        return f"OK|REUTILIZADO|modelo={ruta_modelo}|{resumen_proveedor}"

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
    _ultima_auditoria_estimacion = "SIN_AUDITORIA"
    _ultima_comparacion_estimada = None

    return f"OK|CARGADO|modelo={ruta_modelo}|{resumen_proveedor}"


def reiniciar() -> str:
    """Libera el selector y restablece el estado global a no inicializado."""
    global _selector, _modelo, _proveedor_viaje, _firma_inicializacion
    global _resumen_proveedor, _ultima_decision, _ultima_auditoria_estimacion
    global _ultima_comparacion_estimada

    _selector = None
    _modelo = None
    _proveedor_viaje = None
    _firma_inicializacion = None
    _resumen_proveedor = "proveedor=NO_INICIALIZADO"
    _ultima_decision = "SIN_DECISION"
    _ultima_auditoria_estimacion = "SIN_AUDITORIA"
    _ultima_comparacion_estimada = None
    return "OK|REINICIADO"


def obtener_estado() -> str:
    """Retorna una cadena descriptiva con el estado de inicialización actual."""
    if _selector is None:
        return "NO_INICIALIZADO"
    return f"INICIALIZADO|modelo={_modelo}|{_resumen_proveedor}"


def obtener_fuente_viaje() -> str:
    """Retorna los metadatos del proveedor de distancias y velocidades configurado."""
    return _resumen_proveedor


def obtener_modos_disponibles() -> str:
    """Retorna la lista de modos soportados separados por tuberías (|)."""
    return "|".join(modo.value for modo in ModoPlanificacion)


def obtener_ultima_decision() -> str:
    """Retorna los detalles y métricas de la última decisión tomada por el selector."""
    return _ultima_decision


def obtener_ultima_auditoria_estimacion() -> str:
    """Retorna la auditoría desglosada del último plan evaluado."""
    return _ultima_auditoria_estimacion


def planificar_vector(
    instancia_vector: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
    modo_planificacion: str,
) -> list[float]:
    """Decodifica una instancia, genera el plan solicitado y lo retorna codificado."""
    global _ultima_decision, _ultima_auditoria_estimacion, _ultima_comparacion_estimada

    if _selector is None:
        raise RuntimeError("El selector Pypeline no fue inicializado.")

    instancia = decodificar_instancia_vector(instancia_vector, seed_escenario, seed_ejecucion)
    _ultima_comparacion_estimada = None
    _ultima_auditoria_estimacion = "SIN_AUDITORIA"

    plan = _selector.generar_plan(instancia, modo_planificacion)
    _ultima_auditoria_estimacion = _auditar_plan_estimado(instancia, plan)
    decision = _selector.ultima_decision

    if decision is None:
        _ultima_decision = (
            f"SIN_DECISION|{_resumen_proveedor}|auditoria_estimacion={_ultima_auditoria_estimacion}"
        )
    else:
        _ultima_decision = (
            f"modo={decision.modo_solicitado.value}"
            f"|algoritmo={decision.algoritmo_resultante.value}"
            f"|costo={decision.costo_estimado}"
            f"|tiempo_plan_ms={decision.tiempo_plan_ms}"
            f"|tiempo_selector_ms={decision.tiempo_selector_ms}"
            f"|detalle={decision.detalle}"
            f"|{_resumen_proveedor}"
            f"|auditoria_estimacion={_ultima_auditoria_estimacion}"
        )

    return codificar_plan_alpyne(instancia, plan)


def comparar_estimado_vector(
    instancia_vector: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
) -> list[float]:
    """Ejecuta todos los algoritmos disponibles sobre la instancia y almacena la comparación."""
    global _ultima_comparacion_estimada

    _ultima_comparacion_estimada = None
    if _selector is None:
        raise RuntimeError("El selector Pypeline no fue inicializado.")
    if _proveedor_viaje is None:
        raise RuntimeError("El proveedor común de viajes no fue inicializado.")

    instancia = decodificar_instancia_vector(instancia_vector, seed_escenario, seed_ejecucion)
    firma = firmar_instancia_vector(instancia_vector, seed_escenario, seed_ejecucion)

    comparacion = ejecutar_comparacion_estimada(
        instancia=instancia,
        selector=_selector,
        proveedor_viaje=_proveedor_viaje,
        firma_instancia=firma,
    )
    _ultima_comparacion_estimada = comparacion
    return codificar_comparacion_estimada(comparacion)


def obtener_resumen_comparacion_estimada() -> str:
    """Retorna el resumen textual de la última comparación estimada ejecutada."""
    if _ultima_comparacion_estimada is None:
        return "SIN_COMPARACION"
    return serializar_resumen_comparacion(_ultima_comparacion_estimada)


def obtener_resultado_comparacion_estimado(modo_planificacion: str) -> str:
    """Retorna las métricas del método solicitado dentro de la última comparación."""
    if _ultima_comparacion_estimada is None:
        raise RuntimeError("No existe una comparación estimada disponible.")
    resultado = _ultima_comparacion_estimada.obtener_resultado(modo_planificacion)
    return serializar_resultado_metodo(resultado)


def obtener_plan_comparacion_vector(modo_planificacion: str) -> list[float]:
    """Retorna el vector codificado del plan generado por un método en la última comparación."""
    if _ultima_comparacion_estimada is None:
        raise RuntimeError("No existe una comparación estimada disponible.")

    resultado = _ultima_comparacion_estimada.obtener_resultado(modo_planificacion)
    if not resultado.factible:
        raise RuntimeError(
            f"El método solicitado no produjo un plan factible: {resultado.error or 'error no especificado'}."
        )
    if not resultado.plan_vector:
        raise RuntimeError("El método solicitado no tiene un plan almacenado.")

    return list(resultado.plan_vector)


def obtener_firma_comparacion_estimada() -> str:
    """Retorna la firma hash de la instancia en la última comparación."""
    if _ultima_comparacion_estimada is None:
        return "SIN_COMPARACION"
    return _ultima_comparacion_estimada.firma_instancia


def limpiar_comparacion_estimada() -> str:
    """Limpia los resultados almacenados de la última comparación."""
    global _ultima_comparacion_estimada
    _ultima_comparacion_estimada = None
    return "OK|COMPARACION_LIMPIA"


def _auditar_plan_estimado(instancia: InstanciaTurno, plan: PlanTurno) -> str:
    if _selector is None or _proveedor_viaje is None:
        raise RuntimeError("No existe selector o proveedor de viaje para auditar el plan.")

    matriz = construir_matriz_viaje(instancia, _selector.configuracion, proveedor=_proveedor_viaje)
    estimacion = evaluar_plan_estimado(instancia, plan, matriz, _selector.configuracion)

    if not isclose(plan.costo_estimado, estimacion.costo_total, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError(
            f"El costo estimado del plan no coincide con su auditoría. "
            f"plan={plan.costo_estimado}, auditoria={estimacion.costo_total}."
        )

    return serializar_auditoria_estimacion(estimacion)


def _crear_proveedor_viaje(
    *,
    cache_vial_path: str | None,
    version_cache_vial: str,
    permitir_fallback_vial: bool,
) -> tuple[ProveedorViaje, str]:
    texto_cache = "" if cache_vial_path is None else str(cache_vial_path).strip()
    if not texto_cache:
        proveedor = ProveedorHaversineAjustado()
        return (
            proveedor,
            f"proveedor={proveedor.fuente.value}|version_viaje={proveedor.version}|cache_tramos=NA|fallback=NA",
        )

    ruta_cache = Path(texto_cache).expanduser().resolve()
    proveedor_cache = ProveedorVialCachePersistente(
        ruta_cache,
        version_cache_esperada=version_cache_vial,
        permitir_fallback=permitir_fallback_vial,
    )
    estadisticas = proveedor_cache.estadisticas
    modo_fallback = "PERMITIDO" if permitir_fallback_vial else "ESTRICTO"

    return (
        proveedor_cache,
        f"proveedor={proveedor_cache.fuente.value}|version_viaje={proveedor_cache.version}"
        f"|cache_tramos={estadisticas.cantidad_tramos}|cache_archivo={ruta_cache}|fallback={modo_fallback}",
    )
