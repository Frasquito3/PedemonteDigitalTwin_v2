from __future__ import annotations

from dataclasses import dataclass

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    InstanciaTurno,
    PlanTurno,
)
from planner.domain.validator import validar_plan
from planner.routing.operations import (
    EstimacionCargaViaje,
    EstimacionEsperaCliente,
    ReservaEmpleadosCorralon,
    estimar_espera_cliente,
    simular_plan_operativo_estimado,
    tiempo_carga_estimado_min,
    tiempo_descarga_estimado_min,
    tiempo_espera_respuesta_cliente_esperado_min,
)
from planner.routing.travel import MatrizViaje


VERSION_AUDITORIA_COSTO = "estimacion-costo-v3"


@dataclass(frozen=True)
class EstimacionPlan:
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


def evaluar_plan_estimado(
    instancia: InstanciaTurno,
    plan: PlanTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    reservas_empleados: tuple[
        ReservaEmpleadosCorralon,
        ...
    ] = (),
) -> EstimacionPlan:
    validacion = validar_plan(
        instancia,
        plan,
    )

    if not validacion.valido:
        raise ValueError(
            "No se puede evaluar un plan inválido: "
            + " | ".join(validacion.errores)
        )

    operacion = simular_plan_operativo_estimado(
        instancia,
        plan,
        matriz,
        configuracion,
        reservas_empleados=reservas_empleados,
    )

    finales_camiones = list(
        operacion.finales_camiones_min
    )

    minuto_fin_estimado = max(
        finales_camiones,
        default=instancia.hora_inicio_turno_min,
    )

    minuto_fin_minimo = min(
        finales_camiones,
        default=minuto_fin_estimado,
    )

    duracion_operacion_min = max(
        0.0,
        minuto_fin_estimado
        - instancia.hora_inicio_turno_min,
    )

    exceso_tolerancia_min = max(
        0.0,
        minuto_fin_estimado
        - instancia.hora_fin_tolerancia_min,
    )

    diferencia_fin_camiones_min = (
        minuto_fin_estimado
        - minuto_fin_minimo
    )

    distancia_total_km = (
        operacion.distancia_total_m / 1000.0
    )

    costo_tardanza = (
        operacion.tardanza_total_min
        * configuracion.costo_por_min_tardanza
    )

    costo_exceso_tolerancia = (
        exceso_tolerancia_min
        * configuracion
        .costo_por_min_exceso_tolerancia
    )

    costo_operacion = (
        duracion_operacion_min
        * configuracion.costo_por_min_operacion
    )

    costo_distancia = (
        distancia_total_km
        * configuracion.costo_por_km
    )

    costo_viajes = (
        operacion.viajes_totales
        * configuracion.costo_por_viaje
    )

    costo_desbalance = (
        diferencia_fin_camiones_min
        * configuracion
        .costo_por_min_desbalance_fin
    )

    costo_total = (
        costo_tardanza
        + costo_exceso_tolerancia
        + costo_operacion
        + costo_distancia
        + costo_viajes
        + costo_desbalance
    )

    return EstimacionPlan(
        distancia_total_km=distancia_total_km,
        tiempo_carga_total_min=(
            operacion.tiempo_carga_total_min
        ),
        cargas_estimadas=operacion.cargas,
        tiempo_viaje_total_min=(
            operacion.tiempo_viaje_total_min
        ),
        tiempo_espera_ventana_total_min=(
            operacion
            .tiempo_espera_ventana_total_min
        ),
        tiempo_espera_respuesta_cliente_total_min=(
            operacion
            .tiempo_espera_respuesta_cliente_total_min
        ),
        tiempo_descarga_total_min=(
            operacion.tiempo_descarga_total_min
        ),
        tardanza_total_min=(
            operacion.tardanza_total_min
        ),
        pedidos_tardios=operacion.pedidos_tardios,
        minuto_fin_estimado=minuto_fin_estimado,
        duracion_operacion_min=(
            duracion_operacion_min
        ),
        exceso_tolerancia_min=(
            exceso_tolerancia_min
        ),
        diferencia_fin_camiones_min=(
            diferencia_fin_camiones_min
        ),
        viajes_totales=operacion.viajes_totales,
        costo_tardanza=costo_tardanza,
        costo_exceso_tolerancia=(
            costo_exceso_tolerancia
        ),
        costo_operacion=costo_operacion,
        costo_distancia=costo_distancia,
        costo_viajes=costo_viajes,
        costo_desbalance=costo_desbalance,
        costo_total=costo_total,
    )


def _serializar_cargas(
    cargas: tuple[EstimacionCargaViaje, ...],
) -> str:
    if not cargas:
        return "NINGUNA"

    return ";".join(
        (
            f"c{carga.camion_id}"
            f"-v{carga.numero_viaje}"
            f"-u{carga.unidades}"
            f"-t{carga.minuto_inicio:.6f}"
            f"-p{carga.personas_estimadas}"
            f"-d{carga.duracion_min:.6f}"
        )
        for carga in cargas
    )


def serializar_auditoria_estimacion(
    estimacion: EstimacionPlan,
) -> str:
    """Devuelve un resumen compacto y estable para consola."""
    return (
        f"version={VERSION_AUDITORIA_COSTO}"
        f"|distancia_km={estimacion.distancia_total_km:.6f}"
        f"|carga_min={estimacion.tiempo_carga_total_min:.6f}"
        f"|cargas={_serializar_cargas(estimacion.cargas_estimadas)}"
        f"|viaje_min={estimacion.tiempo_viaje_total_min:.6f}"
        "|espera_ventana_min="
        f"{estimacion.tiempo_espera_ventana_total_min:.6f}"
        "|espera_respuesta_cliente_min="
        f"{estimacion.tiempo_espera_respuesta_cliente_total_min:.6f}"
        f"|descarga_min={estimacion.tiempo_descarga_total_min:.6f}"
        f"|duracion_operacion_min={estimacion.duracion_operacion_min:.6f}"
        f"|fin_estimado_min={estimacion.minuto_fin_estimado:.6f}"
        f"|tardanza_min={estimacion.tardanza_total_min:.6f}"
        f"|pedidos_tardios={estimacion.pedidos_tardios}"
        f"|exceso_tolerancia_min={estimacion.exceso_tolerancia_min:.6f}"
        "|desbalance_fin_min="
        f"{estimacion.diferencia_fin_camiones_min:.6f}"
        f"|viajes={estimacion.viajes_totales}"
        f"|costo_tardanza={estimacion.costo_tardanza:.6f}"
        "|costo_exceso_tolerancia="
        f"{estimacion.costo_exceso_tolerancia:.6f}"
        f"|costo_operacion={estimacion.costo_operacion:.6f}"
        f"|costo_distancia={estimacion.costo_distancia:.6f}"
        f"|costo_viajes={estimacion.costo_viajes:.6f}"
        f"|costo_desbalance={estimacion.costo_desbalance:.6f}"
        f"|costo_total={estimacion.costo_total:.6f}"
    )
