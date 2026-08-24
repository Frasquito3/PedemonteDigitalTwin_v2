from dataclasses import dataclass

from planner.core.config import ConfiguracionPlanificacion

from planner.core.schema import (
    InstanciaTurno,
    PedidoInput,
    PlanTurno,
)

from planner.routing.travel import (
    MatrizViaje,
    tiempo_viaje_esperado_min,
)

from planner.domain.validator import validar_plan


VERSION_AUDITORIA_COSTO = "estimacion-costo-v2"


@dataclass(frozen=True)
class EstimacionEsperaCliente:
    tiempo_espera_ventana_min: float

    tiempo_espera_respuesta_min: float

    minuto_inicio_descarga: float


@dataclass(frozen=True)
class EstimacionPlan:
    distancia_total_km: float

    tiempo_carga_total_min: float

    tiempo_viaje_total_min: float

    tiempo_espera_ventana_total_min: float

    tiempo_espera_respuesta_cliente_total_min: float

    tiempo_descarga_total_min: float

    tardanza_total_min: float

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


def tiempo_espera_respuesta_cliente_esperado_min(
    configuracion: ConfiguracionPlanificacion,
) -> float:
    """
    Valor esperado de la distribución triangular utilizada por
    AnyLogic para la respuesta del cliente.
    """
    return (
        configuracion.cliente_espera_respuesta_min
        + configuracion.cliente_espera_respuesta_moda
        + configuracion.cliente_espera_respuesta_max
    ) / 3.0


def estimar_espera_cliente(
    pedido: PedidoInput,
    minuto_llegada: float,
    configuracion: ConfiguracionPlanificacion,
) -> EstimacionEsperaCliente:
    """
    Estima la espera previa a la descarga con la misma estructura
    temporal del modelo AnyLogic.

    Política determinística del planificador:
    - llegada temprana: espera hasta la apertura y luego la respuesta;
    - llegada dentro de ventana: espera la respuesta;
    - llegada tardía: no agrega espera de respuesta.

    La aceptación probabilística antes o después de la ventana sigue
    siendo variabilidad exclusiva de la simulación en esta fase.
    """
    if minuto_llegada > pedido.hora_hasta_min:
        return EstimacionEsperaCliente(
            tiempo_espera_ventana_min=0.0,
            tiempo_espera_respuesta_min=0.0,
            minuto_inicio_descarga=minuto_llegada,
        )

    tiempo_espera_ventana_min = max(
        0.0,
        pedido.hora_desde_min - minuto_llegada,
    )

    tiempo_espera_respuesta_min = (
        tiempo_espera_respuesta_cliente_esperado_min(
            configuracion
        )
    )

    return EstimacionEsperaCliente(
        tiempo_espera_ventana_min=(
            tiempo_espera_ventana_min
        ),
        tiempo_espera_respuesta_min=(
            tiempo_espera_respuesta_min
        ),
        minuto_inicio_descarga=(
            minuto_llegada
            + tiempo_espera_ventana_min
            + tiempo_espera_respuesta_min
        ),
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

    tiempo_carga_total_min = 0.0

    tiempo_viaje_total_min = 0.0

    tiempo_espera_ventana_total_min = 0.0

    tiempo_espera_respuesta_cliente_total_min = 0.0

    tiempo_descarga_total_min = 0.0

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

            tiempo_carga = (
                tiempo_carga_estimado_min(
                    unidades_viaje,
                    configuracion,
                )
            )

            tiempo_carga_total_min += tiempo_carga

            minuto_actual += tiempo_carga

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

                tiempo_viaje = (
                    tiempo_viaje_esperado_min(
                        matriz,
                        nodo_actual,
                        pedido_id,
                        minuto_actual,
                        configuracion,
                    )
                )

                tiempo_viaje_total_min += tiempo_viaje

                minuto_actual += tiempo_viaje

                tardanza_total_min += max(
                    0.0,
                    minuto_actual
                    - pedido.hora_hasta_min,
                )

                estimacion_espera = (
                    estimar_espera_cliente(
                        pedido,
                        minuto_actual,
                        configuracion,
                    )
                )

                tiempo_espera_ventana_total_min += (
                    estimacion_espera
                    .tiempo_espera_ventana_min
                )

                tiempo_espera_respuesta_cliente_total_min += (
                    estimacion_espera
                    .tiempo_espera_respuesta_min
                )

                minuto_actual = (
                    estimacion_espera
                    .minuto_inicio_descarga
                )

                tiempo_descarga = (
                    tiempo_descarga_estimado_min(
                        pedido,
                        configuracion,
                    )
                )

                tiempo_descarga_total_min += (
                    tiempo_descarga
                )

                minuto_actual += tiempo_descarga

                nodo_actual = pedido_id

            distancia_total_m += (
                matriz.distancia(
                    nodo_actual,
                    configuracion
                    .id_nodo_corralon,
                )
            )

            tiempo_regreso = (
                tiempo_viaje_esperado_min(
                    matriz,
                    nodo_actual,
                    configuracion
                    .id_nodo_corralon,
                    minuto_actual,
                    configuracion,
                )
            )

            tiempo_viaje_total_min += tiempo_regreso

            minuto_actual += tiempo_regreso

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

    costo_tardanza = (
        tardanza_total_min
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
        viajes_totales
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
            tiempo_carga_total_min
        ),

        tiempo_viaje_total_min=(
            tiempo_viaje_total_min
        ),

        tiempo_espera_ventana_total_min=(
            tiempo_espera_ventana_total_min
        ),

        tiempo_espera_respuesta_cliente_total_min=(
            tiempo_espera_respuesta_cliente_total_min
        ),

        tiempo_descarga_total_min=(
            tiempo_descarga_total_min
        ),

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


def serializar_auditoria_estimacion(
    estimacion: EstimacionPlan,
) -> str:
    """
    Devuelve un resumen compacto y estable para consola e integración.

    Expone de forma estable los componentes que intervienen en
    evaluar_plan_estimado(), incluida la espera esperada del cliente.
    """
    return (
        f"version={VERSION_AUDITORIA_COSTO}"
        f"|distancia_km={estimacion.distancia_total_km:.6f}"
        f"|carga_min={estimacion.tiempo_carga_total_min:.6f}"
        f"|viaje_min={estimacion.tiempo_viaje_total_min:.6f}"
        "|espera_ventana_min="
        f"{estimacion.tiempo_espera_ventana_total_min:.6f}"
        "|espera_respuesta_cliente_min="
        f"{estimacion.tiempo_espera_respuesta_cliente_total_min:.6f}"
        f"|descarga_min={estimacion.tiempo_descarga_total_min:.6f}"
        f"|duracion_operacion_min={estimacion.duracion_operacion_min:.6f}"
        f"|fin_estimado_min={estimacion.minuto_fin_estimado:.6f}"
        f"|tardanza_min={estimacion.tardanza_total_min:.6f}"
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
