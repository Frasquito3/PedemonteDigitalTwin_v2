from dataclasses import dataclass

from .config import ConfiguracionPlanificacion

from .schema import (
    InstanciaTurno,
    PedidoInput,
    PlanTurno,
)

from .travel import (
    MatrizViaje,
    tiempo_viaje_esperado_min,
)

from .validator import validar_plan


@dataclass(frozen=True)
class EstimacionPlan:
    distancia_total_km: float

    tardanza_total_min: float

    minuto_fin_estimado: float

    duracion_operacion_min: float

    exceso_tolerancia_min: float

    diferencia_fin_camiones_min: float

    viajes_totales: int

    costo_total: float


def tiempo_carga_estimado_min(
    unidades: int,
    configuracion: ConfiguracionPlanificacion,
) -> float:
    personas_efectivas = (
        1.0
        + configuracion
        .carga_eficiencia_persona_adicional
        * (
            configuracion
            .personas_carga_estimadas
            - 1
        )
    )

    return (
        configuracion.carga_setup_min
        + (
            configuracion
            .carga_min_por_unidad_1_persona
            * unidades
        )
        / personas_efectivas
    )


def tiempo_descarga_estimado_min(
    pedido: PedidoInput,
    configuracion: ConfiguracionPlanificacion,
) -> float:
    return (
        configuracion.descarga_setup_min
        + configuracion.descarga_min_por_unidad
        * pedido.unidades_capacidad
    )


def evaluar_plan_estimado(
    instancia: InstanciaTurno,
    plan: PlanTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
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

    pedidos_por_id = {
        pedido.pedido_id: pedido
        for pedido in instancia.pedidos
    }

    distancia_total_m = 0.0

    tardanza_total_min = 0.0

    viajes_totales = 0

    finales_camiones: list[float] = []

    for plan_camion in plan.camiones:
        minuto_actual = (
            instancia.hora_inicio_turno_min
        )

        for viaje in plan_camion.viajes:
            viajes_totales += 1

            unidades_viaje = sum(
                pedidos_por_id[
                    pedido_id
                ].unidades_capacidad
                for pedido_id
                in viaje.pedido_ids
            )

            minuto_actual += (
                tiempo_carga_estimado_min(
                    unidades_viaje,
                    configuracion,
                )
            )

            nodo_actual = (
                configuracion.id_nodo_corralon
            )

            for pedido_id in viaje.pedido_ids:
                pedido = pedidos_por_id[
                    pedido_id
                ]

                distancia_total_m += (
                    matriz.distancia(
                        nodo_actual,
                        pedido_id,
                    )
                )

                minuto_actual += (
                    tiempo_viaje_esperado_min(
                        matriz,
                        nodo_actual,
                        pedido_id,
                        minuto_actual,
                        configuracion,
                    )
                )

                tardanza_total_min += max(
                    0.0,
                    minuto_actual
                    - pedido.hora_hasta_min,
                )

                if (
                    minuto_actual
                    < pedido.hora_desde_min
                ):
                    minuto_actual = (
                        pedido.hora_desde_min
                    )

                minuto_actual += (
                    tiempo_descarga_estimado_min(
                        pedido,
                        configuracion,
                    )
                )

                nodo_actual = pedido_id

            distancia_total_m += (
                matriz.distancia(
                    nodo_actual,
                    configuracion
                    .id_nodo_corralon,
                )
            )

            minuto_actual += (
                tiempo_viaje_esperado_min(
                    matriz,
                    nodo_actual,
                    configuracion
                    .id_nodo_corralon,
                    minuto_actual,
                    configuracion,
                )
            )

        finales_camiones.append(
            minuto_actual
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
        distancia_total_m / 1000.0
    )

    costo_total = (
        tardanza_total_min
        * configuracion.costo_por_min_tardanza

        + exceso_tolerancia_min
        * configuracion
        .costo_por_min_exceso_tolerancia

        + duracion_operacion_min
        * configuracion
        .costo_por_min_operacion

        + distancia_total_km
        * configuracion.costo_por_km

        + viajes_totales
        * configuracion.costo_por_viaje

        + diferencia_fin_camiones_min
        * configuracion
        .costo_por_min_desbalance_fin
    )

    return EstimacionPlan(
        distancia_total_km=distancia_total_km,

        tardanza_total_min=(
            tardanza_total_min
        ),

        minuto_fin_estimado=(
            minuto_fin_estimado
        ),

        duracion_operacion_min=(
            duracion_operacion_min
        ),

        exceso_tolerancia_min=(
            exceso_tolerancia_min
        ),

        diferencia_fin_camiones_min=(
            diferencia_fin_camiones_min
        ),

        viajes_totales=viajes_totales,

        costo_total=costo_total,
    )