"""Validación de restricciones duras, coherencia de instancias y factibilidad de planes."""

from collections import defaultdict

from planner.core.schema import (
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    ResultadoValidacionPlan,
    ViajePlan,
)

__all__ = ["validar_instancia", "validar_plan"]


# ============================================================================
# Validaciones de Instancia de Turno
# ============================================================================


def _validar_parametros_instancia(instancia: InstanciaTurno) -> list[str]:
    """Comprueba identificadores, capacidades y coherencia horaria del turno."""
    errores: list[str] = []

    if not instancia.instancia_id.strip():
        errores.append("instancia_id no puede estar vacío.")

    if instancia.capacidad_camion <= 0:
        errores.append("capacidad_camion debe ser > 0.")

    if instancia.cantidad_camiones <= 0:
        errores.append("cantidad_camiones debe ser > 0.")

    if instancia.hora_inicio_turno_min >= instancia.hora_fin_objetivo_min:
        errores.append("La hora de inicio debe ser anterior al fin objetivo.")

    if instancia.hora_fin_objetivo_min > instancia.hora_fin_tolerancia_min:
        errores.append("El fin objetivo no puede superar el fin de tolerancia.")

    return errores


def _validar_pedidos_instancia(
    instancia: InstanciaTurno,
) -> tuple[list[str], dict[str, list[PedidoInput]]]:
    """Valida individualmente los pedidos del turno y los agrupa por pedido original."""
    errores: list[str] = []
    pedidos_por_id: set[str] = set()
    partes_por_original: dict[str, list[PedidoInput]] = defaultdict(list)

    for pedido in instancia.pedidos:
        pedido_id = pedido.pedido_id

        if not pedido_id.strip():
            errores.append("Existe un pedido con pedido_id vacío.")
            continue

        if pedido_id in pedidos_por_id:
            errores.append(f"pedido_id duplicado en la instancia: {pedido_id}")
            continue

        pedidos_por_id.add(pedido_id)

        if pedido.turno != instancia.turno:
            errores.append(
                f"Pedido {pedido_id}: turno {pedido.turno.value} "
                f"distinto de la instancia {instancia.turno.value}."
            )

        if pedido.unidades_capacidad <= 0:
            errores.append(f"Pedido {pedido_id}: unidades debe ser > 0.")

        if pedido.unidades_capacidad > instancia.capacidad_camion:
            errores.append(
                f"Pedido {pedido_id}: supera la capacidad y debe pasar por el preprocesador de split."
            )

        if pedido.hora_desde_min >= pedido.hora_hasta_min:
            errores.append(f"Pedido {pedido_id}: ventana horaria inválida.")

        original_id = pedido.pedido_original_id or pedido_id
        partes_por_original[original_id].append(pedido)

    return errores, partes_por_original


def _validar_integridad_partes_originales(
    partes_por_original: dict[str, list[PedidoInput]],
) -> list[str]:
    """Verifica que las entregas fraccionadas mantengan una numeración 1..N continua y consistente."""
    errores: list[str] = []

    for original_id, partes in partes_por_original.items():
        totales_declarados = {p.total_partes for p in partes}

        if len(totales_declarados) != 1:
            errores.append(f"Pedido original {original_id}: total_partes inconsistente.")
            continue

        total_partes = next(iter(totales_declarados))

        if total_partes <= 0:
            errores.append(f"Pedido original {original_id}: total_partes debe ser > 0.")
            continue

        numeros_reales = {p.numero_parte for p in partes}
        numeros_esperados = set(range(1, total_partes + 1))

        if len(partes) != total_partes:
            errores.append(
                f"Pedido original {original_id}: se esperaban {total_partes} partes "
                f"y se recibieron {len(partes)}."
            )

        if numeros_reales != numeros_esperados:
            errores.append(
                f"Pedido original {original_id}: numeración de partes inválida. "
                f"Recibida={sorted(numeros_reales)}, esperada={sorted(numeros_esperados)}."
            )

    return errores


def validar_instancia(instancia: InstanciaTurno) -> list[str]:
    """Valida integralmente la coherencia de una instancia de turno.

    Comprueba límites horarios, capacidades de flota, ventanas de entrega y
    la correcta numeración de pedidos fraccionados.

    Args:
        instancia: Escenario de planificación a validar.

    Returns:
        Lista de errores encontrados. Vacía si la instancia es válida.
    """
    errores = _validar_parametros_instancia(instancia)

    errores_pedidos, partes_por_original = _validar_pedidos_instancia(instancia)
    errores.extend(errores_pedidos)

    errores_partes = _validar_integridad_partes_originales(partes_por_original)
    errores.extend(errores_partes)

    return errores


# ============================================================================
# Validaciones de Plan de Turno
# ============================================================================


def _validar_flota_plan(
    camiones: list[PlanCamion],
    cantidad_esperada: int,
) -> list[str]:
    """Verifica que la flota en el plan coincida en cantidad y que sus identificadores sean válidos."""
    errores: list[str] = []

    if len(camiones) != cantidad_esperada:
        errores.append(
            f"Cantidad incorrecta de camiones en el plan. "
            f"Esperados={cantidad_esperada}, recibidos={len(camiones)}."
        )

    camiones_vistos: set[int] = set()

    for plan_camion in camiones:
        camion_id = plan_camion.camion_id

        if camion_id in camiones_vistos:
            errores.append(f"camion_id duplicado: {camion_id}.")

        camiones_vistos.add(camion_id)

        if not (0 <= camion_id < cantidad_esperada):
            errores.append(f"camion_id fuera de rango: {camion_id}.")

    return errores


def _validar_restricciones_viaje(
    viaje: ViajePlan,
    camion_id: int,
    capacidad_camion: int,
    pedidos_por_id: dict[str, PedidoInput],
    asignaciones: dict[str, int],
) -> list[str]:
    """Valida la capacidad de carga y reglas de descarga (volcador) de un viaje individual."""
    errores: list[str] = []
    carga = 0
    posiciones_volcador: list[int] = []

    for posicion, pedido_id in enumerate(viaje.pedido_ids):
        pedido = pedidos_por_id.get(pedido_id)

        if pedido is None:
            errores.append(f"Pedido inexistente en el plan: {pedido_id}.")
            continue

        asignaciones[pedido_id] += 1
        carga += pedido.unidades_capacidad

        if pedido.requiere_volcador:
            posiciones_volcador.append(posicion)

    if carga > capacidad_camion:
        errores.append(
            f"Camión {camion_id}, viaje {viaje.numero_viaje}: "
            f"carga={carga} supera capacidad={capacidad_camion}."
        )

    if len(posiciones_volcador) > 1:
        errores.append(
            f"Camión {camion_id}, viaje {viaje.numero_viaje}: "
            "contiene más de un pedido con volcador."
        )

    if len(posiciones_volcador) == 1 and posiciones_volcador[0] != len(viaje.pedido_ids) - 1:
        errores.append(
            f"Camión {camion_id}, viaje {viaje.numero_viaje}: "
            "el pedido con volcador no es el último."
        )

    return errores


def _validar_viajes_camion(
    plan_camion: PlanCamion,
    capacidad_camion: int,
    pedidos_por_id: dict[str, PedidoInput],
    asignaciones: dict[str, int],
) -> list[str]:
    """Verifica los números de viaje y restricciones operativas para un camión."""
    errores: list[str] = []
    numeros_viaje: set[int] = set()

    for viaje in plan_camion.viajes:
        if viaje.numero_viaje <= 0:
            errores.append(f"Camión {plan_camion.camion_id}: numero_viaje debe ser > 0.")

        if viaje.numero_viaje in numeros_viaje:
            errores.append(
                f"Camión {plan_camion.camion_id}: numero_viaje repetido {viaje.numero_viaje}."
            )

        numeros_viaje.add(viaje.numero_viaje)

        if not viaje.pedido_ids:
            errores.append(
                f"Camión {plan_camion.camion_id}, viaje {viaje.numero_viaje}: viaje vacío."
            )
            continue

        errores_viaje = _validar_restricciones_viaje(
            viaje,
            plan_camion.camion_id,
            capacidad_camion,
            pedidos_por_id,
            asignaciones,
        )
        errores.extend(errores_viaje)

    return errores


def _validar_cobertura_pedidos(asignaciones: dict[str, int]) -> list[str]:
    """Verifica que cada pedido de la instancia haya sido asignado exactamente una vez."""
    errores: list[str] = []

    for pedido_id, cantidad in asignaciones.items():
        if cantidad == 0:
            errores.append(f"Pedido no asignado: {pedido_id}.")
        elif cantidad > 1:
            errores.append(f"Pedido asignado más de una vez: {pedido_id}.")

    return errores


def validar_plan(
    instancia: InstanciaTurno,
    plan: PlanTurno,
) -> ResultadoValidacionPlan:
    """Valida la factibilidad física y operativa de un plan generado para un turno.

    Verifica que la instancia sea válida, que el plan corresponda al escenario,
    que no se sobrecarguen los camiones, que la descarga por volcador sea factible
    y que todos los pedidos queden asignados exactamente una vez.

    Args:
        instancia: Escenario del turno a planificar.
        plan: Asignación de rutas y pedidos por camión.

    Returns:
        Resultado del diagnóstico indicando si el plan es válido y los errores hallados.
    """
    errores = validar_instancia(instancia)

    if plan.instancia_id != instancia.instancia_id:
        errores.append(
            f"El plan no corresponde a la instancia. "
            f"Esperada={instancia.instancia_id}, recibida={plan.instancia_id}."
        )

    errores.extend(_validar_flota_plan(plan.camiones, instancia.cantidad_camiones))

    pedidos_por_id = {p.pedido_id: p for p in instancia.pedidos}
    asignaciones = {p.pedido_id: 0 for p in instancia.pedidos}

    for plan_camion in plan.camiones:
        errores.extend(
            _validar_viajes_camion(
                plan_camion,
                instancia.capacidad_camion,
                pedidos_por_id,
                asignaciones,
            )
        )

    errores.extend(_validar_cobertura_pedidos(asignaciones))

    return ResultadoValidacionPlan(
        valido=not errores,
        errores=errores,
    )
