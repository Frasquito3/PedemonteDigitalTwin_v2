"""Evaluación de la función objetivo, cálculo de costos y auditoría de planes operativos."""

from __future__ import annotations

from dataclasses import dataclass

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.routing.operations import (
    EstimacionCargaViaje,
    EstimacionEsperaCliente,
    ReservaEmpleadosCorralon,
    ResultadoOperacionEstimada,
    estimar_espera_cliente,
    simular_plan_operativo_estimado,
    tiempo_carga_estimado_min,
    tiempo_descarga_estimado_min,
    tiempo_espera_respuesta_cliente_esperado_min,
)
from planner.routing.travel import MatrizViaje

__all__ = [
    "VERSION_AUDITORIA_COSTO",
    "EstimacionPlan",
    "evaluar_plan_estimado",
    "serializar_auditoria_estimacion",
    "EstimacionCargaViaje",
    "EstimacionEsperaCliente",
    "ReservaEmpleadosCorralon",
    "estimar_espera_cliente",
    "simular_plan_operativo_estimado",
    "tiempo_carga_estimado_min",
    "tiempo_descarga_estimado_min",
    "tiempo_espera_respuesta_cliente_esperado_min",
]

VERSION_AUDITORIA_COSTO = "estimacion-costo-v3"


@dataclass(frozen=True)
class EstimacionPlan:
    """Balance integral de indicadores operativos, tiempos acumulados y costos del plan.

    Agrupa las métricas cinemáticas (distancia, tiempos de carga, viaje, espera y descarga),
    la puntualidad del servicio, el cierre temporal de la flota y el desglose monetario
    de la función objetivo de optimización.
    """

    distancia_total_km: float

    tiempo_carga_total_min: float
    cargas_estimadas: tuple[EstimacionCargaViaje, ...]

    tiempo_viaje_total_min: float
    tiempo_espera_ventana_total_min: float
    tiempo_espera_respuesta_cliente_total_min: float
    tiempo_descarga_total_min: float
    tardanza_total_min: float
    pedidos_tardios: int

    minuto_fin_estimado: float
    duracion_operacion_min: float
    exceso_tolerancia_min: float
    diferencia_fin_camiones_min: float
    viajes_totales: int

    costo_tardanza: float
    costo_exceso_tolerancia: float
    costo_operacion: float
    costo_distancia: float
    costo_viajes: float
    costo_desbalance: float
    costo_total: float


def _calcular_costos_plan(
    operacion: ResultadoOperacionEstimada,
    minuto_fin_estimado: float,
    minuto_fin_minimo: float,
    instancia: InstanciaTurno,
    configuracion: ConfiguracionPlanificacion,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Computa los lapsos horarios, distancias y penalizaciones de la función objetivo.

    Returns:
        Tupla con (duracion_operacion_min, exceso_tolerancia_min, diferencia_fin_camiones_min,
        distancia_total_km, costo_tardanza, costo_exceso_tolerancia, costo_operacion,
        costo_distancia, costo_viajes, costo_desbalance, costo_total).
    """
    duracion_operacion_min = max(0.0, minuto_fin_estimado - instancia.hora_inicio_turno_min)
    exceso_tolerancia_min = max(0.0, minuto_fin_estimado - instancia.hora_fin_tolerancia_min)
    diferencia_fin_camiones_min = minuto_fin_estimado - minuto_fin_minimo
    distancia_total_km = operacion.distancia_total_m / 1000.0

    costo_tardanza = operacion.tardanza_total_min * configuracion.costo_por_min_tardanza
    costo_exceso_tolerancia = exceso_tolerancia_min * configuracion.costo_por_min_exceso_tolerancia
    costo_operacion = duracion_operacion_min * configuracion.costo_por_min_operacion
    costo_distancia = distancia_total_km * configuracion.costo_por_km
    costo_viajes = operacion.viajes_totales * configuracion.costo_por_viaje
    costo_desbalance = diferencia_fin_camiones_min * configuracion.costo_por_min_desbalance_fin

    costo_total = (
        costo_tardanza
        + costo_exceso_tolerancia
        + costo_operacion
        + costo_distancia
        + costo_viajes
        + costo_desbalance
    )

    return (
        duracion_operacion_min,
        exceso_tolerancia_min,
        diferencia_fin_camiones_min,
        distancia_total_km,
        costo_tardanza,
        costo_exceso_tolerancia,
        costo_operacion,
        costo_distancia,
        costo_viajes,
        costo_desbalance,
        costo_total,
    )


def evaluar_plan_estimado(
    instancia: InstanciaTurno,
    plan: PlanTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    reservas_empleados: tuple[ReservaEmpleadosCorralon, ...] = (),
) -> EstimacionPlan:
    """Evalúa determinísticamente un plan de turno calculando su costo global y métricas operativas.

    Valida la factibilidad del plan, simula la cronología de eventos con concurrencia
    en corralón y computa las penalizaciones por demoras, kilometraje y desbalance.

    Args:
        instancia: Escenario del turno.
        plan: Rutas y asignaciones por vehículo.
        matriz: Matriz vial de tiempos y distancias.
        configuracion: Parámetros operativos y costos unitarios.
        reservas_empleados: Ocupaciones previas o externas de personal de corralón.

    Returns:
        Instancia de `EstimacionPlan` con todas las métricas y costos computados.

    Raises:
        ValueError: Si el plan incumple restricciones duras de la instancia.
    """
    validacion = validar_plan(instancia, plan)
    if not validacion.valido:
        raise ValueError("No se puede evaluar un plan inválido: " + " | ".join(validacion.errores))

    operacion = simular_plan_operativo_estimado(
        instancia,
        plan,
        matriz,
        configuracion,
        reservas_empleados=reservas_empleados,
    )

    finales_camiones = list(operacion.finales_camiones_min)
    minuto_fin_estimado = max(finales_camiones, default=instancia.hora_inicio_turno_min)
    minuto_fin_minimo = min(finales_camiones, default=minuto_fin_estimado)

    (
        duracion_operacion_min,
        exceso_tolerancia_min,
        diferencia_fin_camiones_min,
        distancia_total_km,
        costo_tardanza,
        costo_exceso_tolerancia,
        costo_operacion,
        costo_distancia,
        costo_viajes,
        costo_desbalance,
        costo_total,
    ) = _calcular_costos_plan(
        operacion,
        minuto_fin_estimado,
        minuto_fin_minimo,
        instancia,
        configuracion,
    )

    return EstimacionPlan(
        distancia_total_km=distancia_total_km,
        tiempo_carga_total_min=operacion.tiempo_carga_total_min,
        cargas_estimadas=operacion.cargas,
        tiempo_viaje_total_min=operacion.tiempo_viaje_total_min,
        tiempo_espera_ventana_total_min=operacion.tiempo_espera_ventana_total_min,
        tiempo_espera_respuesta_cliente_total_min=operacion.tiempo_espera_respuesta_cliente_total_min,
        tiempo_descarga_total_min=operacion.tiempo_descarga_total_min,
        tardanza_total_min=operacion.tardanza_total_min,
        pedidos_tardios=operacion.pedidos_tardios,
        minuto_fin_estimado=minuto_fin_estimado,
        duracion_operacion_min=duracion_operacion_min,
        exceso_tolerancia_min=exceso_tolerancia_min,
        diferencia_fin_camiones_min=diferencia_fin_camiones_min,
        viajes_totales=operacion.viajes_totales,
        costo_tardanza=costo_tardanza,
        costo_exceso_tolerancia=costo_exceso_tolerancia,
        costo_operacion=costo_operacion,
        costo_distancia=costo_distancia,
        costo_viajes=costo_viajes,
        costo_desbalance=costo_desbalance,
        costo_total=costo_total,
    )


def _serializar_cargas(cargas: tuple[EstimacionCargaViaje, ...]) -> str:
    """Serializa la secuencia de cargas en una representación compacta para auditoría."""
    if not cargas:
        return "NINGUNA"

    return ";".join(
        f"c{c.camion_id}-v{c.numero_viaje}-u{c.unidades}-t{c.minuto_inicio:.6f}"
        f"-p{c.personas_estimadas}-d{c.duracion_min:.6f}"
        for c in cargas
    )


def serializar_auditoria_estimacion(estimacion: EstimacionPlan) -> str:
    """Genera una cadena diagnóstica determinística y estable para comparación y benchmarking."""
    return (
        f"version={VERSION_AUDITORIA_COSTO}"
        f"|distancia_km={estimacion.distancia_total_km:.6f}"
        f"|carga_min={estimacion.tiempo_carga_total_min:.6f}"
        f"|cargas={_serializar_cargas(estimacion.cargas_estimadas)}"
        f"|viaje_min={estimacion.tiempo_viaje_total_min:.6f}"
        f"|espera_ventana_min={estimacion.tiempo_espera_ventana_total_min:.6f}"
        f"|espera_respuesta_cliente_min={estimacion.tiempo_espera_respuesta_cliente_total_min:.6f}"
        f"|descarga_min={estimacion.tiempo_descarga_total_min:.6f}"
        f"|duracion_operacion_min={estimacion.duracion_operacion_min:.6f}"
        f"|fin_estimado_min={estimacion.minuto_fin_estimado:.6f}"
        f"|tardanza_min={estimacion.tardanza_total_min:.6f}"
        f"|pedidos_tardios={estimacion.pedidos_tardios}"
        f"|exceso_tolerancia_min={estimacion.exceso_tolerancia_min:.6f}"
        f"|desbalance_fin_min={estimacion.diferencia_fin_camiones_min:.6f}"
        f"|viajes={estimacion.viajes_totales}"
        f"|costo_tardanza={estimacion.costo_tardanza:.6f}"
        f"|costo_exceso_tolerancia={estimacion.costo_exceso_tolerancia:.6f}"
        f"|costo_operacion={estimacion.costo_operacion:.6f}"
        f"|costo_distancia={estimacion.costo_distancia:.6f}"
        f"|costo_viajes={estimacion.costo_viajes:.6f}"
        f"|costo_desbalance={estimacion.costo_desbalance:.6f}"
        f"|costo_total={estimacion.costo_total:.6f}"
    )
