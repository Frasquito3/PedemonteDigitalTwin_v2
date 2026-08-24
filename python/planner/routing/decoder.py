from planner.core.config import ConfiguracionPlanificacion

from planner.routing.objective import (
    estimar_espera_cliente,
    tiempo_carga_estimado_min,
    tiempo_descarga_estimado_min,
)
from planner.routing.operational import (
    simular_plan_operativo_estimado,
)

from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    ViajePlan,
)

from planner.routing.travel import (
    MatrizViaje,
    tiempo_viaje_esperado_min,
)

from planner.domain.validator import validar_plan


Cromosoma = tuple[str, ...]


def validar_permutacion(
    pedidos_por_id: dict[str, PedidoInput],
    cromosoma: Cromosoma,
) -> None:
    ids_esperados = set(
        pedidos_por_id
    )

    ids_recibidos = set(
        cromosoma
    )

    if (
        len(cromosoma)
        != len(pedidos_por_id)
    ):
        raise ValueError(
            "Longitud incorrecta de la permutación."
        )

    if (
        len(cromosoma)
        != len(ids_recibidos)
    ):
        raise ValueError(
            "La permutación contiene "
            "pedidos repetidos."
        )

    if ids_recibidos != ids_esperados:
        faltantes = sorted(
            ids_esperados - ids_recibidos
        )

        desconocidos = sorted(
            ids_recibidos - ids_esperados
        )

        raise ValueError(
            "Permutación incompatible. "
            f"Faltantes={faltantes}, "
            f"desconocidos={desconocidos}."
        )


def decodificar_viajes_permutacion(
    instancia: InstanciaTurno,
    pedidos_por_id: dict[str, PedidoInput],
    cromosoma: Cromosoma,
) -> list[list[str]]:
    validar_permutacion(
        pedidos_por_id,
        cromosoma,
    )

    viajes: list[list[str]] = []

    viaje_actual: list[str] = []

    carga_actual = 0

    contiene_volcador = False

    def cerrar_viaje() -> None:
        nonlocal viaje_actual
        nonlocal carga_actual
        nonlocal contiene_volcador

        if viaje_actual:
            viajes.append(
                list(viaje_actual)
            )

        viaje_actual = []

        carga_actual = 0

        contiene_volcador = False

    for pedido_id in cromosoma:
        pedido = pedidos_por_id[
            pedido_id
        ]

        # Defensa adicional. Normalmente el volcador
        # ya habrá cerrado el viaje inmediatamente.
        if contiene_volcador:
            cerrar_viaje()

        supera_capacidad = (
            carga_actual
            + pedido.unidades_capacidad
            > instancia.capacidad_camion
        )

        if supera_capacidad:
            cerrar_viaje()

        viaje_actual.append(
            pedido_id
        )

        carga_actual += (
            pedido.unidades_capacidad
        )

        if pedido.requiere_volcador:
            contiene_volcador = True

            # Restricción dura:
            # el volcador queda último.
            cerrar_viaje()

    cerrar_viaje()

    return viajes


def estimar_fin_viaje(
    pedido_ids: list[str],
    minuto_inicio: float,
    pedidos_por_id: dict[str, PedidoInput],
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
) -> float:
    unidades = sum(
        pedidos_por_id[
            pedido_id
        ].unidades_capacidad

        for pedido_id in pedido_ids
    )

    minuto_actual = (
        minuto_inicio
        + tiempo_carga_estimado_min(
            unidades,
            configuracion,
        )
    )

    nodo_actual = (
        configuracion.id_nodo_corralon
    )

    for pedido_id in pedido_ids:
        pedido = pedidos_por_id[
            pedido_id
        ]

        minuto_actual += (
            tiempo_viaje_esperado_min(
                matriz,
                nodo_actual,
                pedido_id,
                minuto_actual,
                configuracion,
            )
        )

        estimacion_espera = (
            estimar_espera_cliente(
                pedido,
                minuto_actual,
                configuracion,
            )
        )

        minuto_actual = (
            estimacion_espera
            .minuto_inicio_descarga
        )

        minuto_actual += (
            tiempo_descarga_estimado_min(
                pedido,
                configuracion,
            )
        )

        nodo_actual = pedido_id

    minuto_actual += (
        tiempo_viaje_esperado_min(
            matriz,
            nodo_actual,
            configuracion.id_nodo_corralon,
            minuto_actual,
            configuracion,
        )
    )

    return minuto_actual


def decodificar_plan_permutacion(
    instancia: InstanciaTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    cromosoma: Cromosoma,
    algoritmo: AlgoritmoPlanificacion,
) -> PlanTurno:
    pedidos_por_id = {
        pedido.pedido_id: pedido
        for pedido in instancia.pedidos
    }

    viajes = decodificar_viajes_permutacion(
        instancia,
        pedidos_por_id,
        cromosoma,
    )

    planes_camion = [
        PlanCamion(
            camion_id=camion_id
        )

        for camion_id in range(
            instancia.cantidad_camiones
        )
    ]

    disponibilidad = [
        float(
            instancia.hora_inicio_turno_min
        )

        for _ in range(
            instancia.cantidad_camiones
        )
    ]

    for pedido_ids in viajes:
        camion_id = min(
            range(
                instancia.cantidad_camiones
            ),

            key=lambda candidato_id: (
                disponibilidad[
                    candidato_id
                ],

                len(
                    planes_camion[
                        candidato_id
                    ].viajes
                ),

                candidato_id,
            ),
        )

        numero_viaje = (
            len(
                planes_camion[
                    camion_id
                ].viajes
            )
            + 1
        )

        planes_camion[
            camion_id
        ].viajes.append(
            ViajePlan(
                numero_viaje=numero_viaje,

                pedido_ids=list(
                    pedido_ids
                ),
            )
        )

        plan_parcial = PlanTurno(
            instancia_id=instancia.instancia_id,
            algoritmo=algoritmo,
            camiones=planes_camion,
        )

        resultado_operativo = (
            simular_plan_operativo_estimado(
                instancia,
                plan_parcial,
                matriz,
                configuracion,
            )
        )

        disponibilidad = list(
            resultado_operativo
            .finales_camiones_min
        )

    plan = PlanTurno(
        instancia_id=instancia.instancia_id,

        algoritmo=algoritmo,

        camiones=planes_camion,
    )

    validacion = validar_plan(
        instancia,
        plan,
    )

    if not validacion.valido:
        raise RuntimeError(
            "El decodificador compartido produjo "
            "un plan inválido: "
            + " | ".join(
                validacion.errores
            )
        )

    return plan