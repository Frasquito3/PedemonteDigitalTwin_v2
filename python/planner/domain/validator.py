from collections import defaultdict

from planner.core.schema import (
    InstanciaTurno,
    PlanTurno,
    ResultadoValidacionPlan,
)


def validar_instancia(
    instancia: InstanciaTurno,
) -> list[str]:
    errores: list[str] = []

    if not instancia.instancia_id.strip():
        errores.append(
            "instancia_id no puede estar vacío."
        )

    if instancia.capacidad_camion <= 0:
        errores.append(
            "capacidad_camion debe ser > 0."
        )

    if instancia.cantidad_camiones <= 0:
        errores.append(
            "cantidad_camiones debe ser > 0."
        )

    if (
        instancia.hora_inicio_turno_min
        >= instancia.hora_fin_objetivo_min
    ):
        errores.append(
            "La hora de inicio debe ser "
            "anterior al fin objetivo."
        )

    if (
        instancia.hora_fin_objetivo_min
        > instancia.hora_fin_tolerancia_min
    ):
        errores.append(
            "El fin objetivo no puede superar "
            "el fin de tolerancia."
        )

    pedidos_por_id = {}

    partes_por_original: dict[
        str,
        list,
    ] = defaultdict(list)

    for pedido in instancia.pedidos:
        if not pedido.pedido_id.strip():
            errores.append(
                "Existe un pedido con "
                "pedido_id vacío."
            )

            continue

        if pedido.pedido_id in pedidos_por_id:
            errores.append(
                "pedido_id duplicado en la "
                f"instancia: {pedido.pedido_id}"
            )

            continue

        pedidos_por_id[pedido.pedido_id] = (
            pedido
        )

        if pedido.turno != instancia.turno:
            errores.append(
                f"Pedido {pedido.pedido_id}: "
                f"turno {pedido.turno.value} "
                "distinto de la instancia "
                f"{instancia.turno.value}."
            )

        if pedido.unidades_capacidad <= 0:
            errores.append(
                f"Pedido {pedido.pedido_id}: "
                "unidades debe ser > 0."
            )

        if (
            pedido.unidades_capacidad
            > instancia.capacidad_camion
        ):
            errores.append(
                f"Pedido {pedido.pedido_id}: "
                "supera la capacidad y debe pasar "
                "por el preprocesador de split."
            )

        if (
            pedido.hora_desde_min
            >= pedido.hora_hasta_min
        ):
            errores.append(
                f"Pedido {pedido.pedido_id}: "
                "ventana horaria inválida."
            )

        original_id = (
            pedido.pedido_original_id
            or pedido.pedido_id
        )

        partes_por_original[
            original_id
        ].append(pedido)

    for (
        original_id,
        partes,
    ) in partes_por_original.items():
        totales_declarados = {
            pedido.total_partes
            for pedido in partes
        }

        if len(totales_declarados) != 1:
            errores.append(
                f"Pedido original {original_id}: "
                "total_partes inconsistente."
            )

            continue

        total_partes = next(
            iter(totales_declarados)
        )

        if total_partes <= 0:
            errores.append(
                f"Pedido original {original_id}: "
                "total_partes debe ser > 0."
            )

            continue

        numeros_reales = {
            pedido.numero_parte
            for pedido in partes
        }

        numeros_esperados = set(
            range(1, total_partes + 1)
        )

        if len(partes) != total_partes:
            errores.append(
                f"Pedido original {original_id}: "
                f"se esperaban {total_partes} partes "
                f"y se recibieron {len(partes)}."
            )

        if numeros_reales != numeros_esperados:
            errores.append(
                f"Pedido original {original_id}: "
                "numeración de partes inválida. "
                f"Recibida={sorted(numeros_reales)}, "
                f"esperada={sorted(numeros_esperados)}."
            )

    return errores


def validar_plan(
    instancia: InstanciaTurno,
    plan: PlanTurno,
) -> ResultadoValidacionPlan:
    errores = validar_instancia(
        instancia
    )

    if plan.instancia_id != instancia.instancia_id:
        errores.append(
            "El plan no corresponde a la instancia. "
            f"Esperada={instancia.instancia_id}, "
            f"recibida={plan.instancia_id}."
        )

    pedidos_por_id = {
        pedido.pedido_id: pedido
        for pedido in instancia.pedidos
    }

    asignaciones = {
        pedido_id: 0
        for pedido_id in pedidos_por_id
    }

    if (
        len(plan.camiones)
        != instancia.cantidad_camiones
    ):
        errores.append(
            "Cantidad incorrecta de camiones "
            "en el plan. "
            f"Esperados={instancia.cantidad_camiones}, "
            f"recibidos={len(plan.camiones)}."
        )

    camiones_vistos: set[int] = set()

    for plan_camion in plan.camiones:
        if (
            plan_camion.camion_id
            in camiones_vistos
        ):
            errores.append(
                "camion_id duplicado: "
                f"{plan_camion.camion_id}."
            )

        camiones_vistos.add(
            plan_camion.camion_id
        )

        if not (
            0
            <= plan_camion.camion_id
            < instancia.cantidad_camiones
        ):
            errores.append(
                "camion_id fuera de rango: "
                f"{plan_camion.camion_id}."
            )

        numeros_viaje: set[int] = set()

        for viaje in plan_camion.viajes:
            if viaje.numero_viaje <= 0:
                errores.append(
                    f"Camión {plan_camion.camion_id}: "
                    "numero_viaje debe ser > 0."
                )

            if (
                viaje.numero_viaje
                in numeros_viaje
            ):
                errores.append(
                    f"Camión {plan_camion.camion_id}: "
                    "numero_viaje repetido "
                    f"{viaje.numero_viaje}."
                )

            numeros_viaje.add(
                viaje.numero_viaje
            )

            if not viaje.pedido_ids:
                errores.append(
                    f"Camión {plan_camion.camion_id}, "
                    f"viaje {viaje.numero_viaje}: "
                    "viaje vacío."
                )

                continue

            carga = 0

            posiciones_volcador: list[int] = []

            for (
                posicion,
                pedido_id,
            ) in enumerate(viaje.pedido_ids):
                pedido = pedidos_por_id.get(
                    pedido_id
                )

                if pedido is None:
                    errores.append(
                        "Pedido inexistente en "
                        f"el plan: {pedido_id}."
                    )

                    continue

                asignaciones[pedido_id] += 1

                carga += (
                    pedido.unidades_capacidad
                )

                if pedido.requiere_volcador:
                    posiciones_volcador.append(
                        posicion
                    )

            if (
                carga
                > instancia.capacidad_camion
            ):
                errores.append(
                    f"Camión {plan_camion.camion_id}, "
                    f"viaje {viaje.numero_viaje}: "
                    f"carga={carga} supera "
                    "capacidad="
                    f"{instancia.capacidad_camion}."
                )

            if len(posiciones_volcador) > 1:
                errores.append(
                    f"Camión {plan_camion.camion_id}, "
                    f"viaje {viaje.numero_viaje}: "
                    "contiene más de un pedido "
                    "con volcador."
                )

            if (
                len(posiciones_volcador) == 1
                and posiciones_volcador[0]
                != len(viaje.pedido_ids) - 1
            ):
                errores.append(
                    f"Camión {plan_camion.camion_id}, "
                    f"viaje {viaje.numero_viaje}: "
                    "el pedido con volcador "
                    "no es el último."
                )

    for (
        pedido_id,
        cantidad,
    ) in asignaciones.items():
        if cantidad == 0:
            errores.append(
                f"Pedido no asignado: {pedido_id}."
            )

        elif cantidad > 1:
            errores.append(
                "Pedido asignado más de una vez: "
                f"{pedido_id}."
            )

    return ResultadoValidacionPlan(
        valido=not errores,
        errores=errores,
    )