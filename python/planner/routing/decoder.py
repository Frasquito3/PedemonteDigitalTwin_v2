"""Decodificación de secuencias y permutaciones de pedidos en planes operativos de turnos."""

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    ViajePlan,
)
from planner.domain.validator import validar_plan
from planner.routing.objective import (
    estimar_espera_cliente,
    tiempo_carga_estimado_min,
    tiempo_descarga_estimado_min,
)
from planner.routing.operations import simular_plan_operativo_estimado
from planner.routing.travel import MatrizViaje, tiempo_viaje_esperado_min

__all__ = [
    "Cromosoma",
    "validar_permutacion",
    "decodificar_viajes_permutacion",
    "estimar_fin_viaje",
    "decodificar_plan_permutacion",
]

Cromosoma = tuple[str, ...]


def validar_permutacion(
    pedidos_por_id: dict[str, PedidoInput],
    cromosoma: Cromosoma,
) -> None:
    """Verifica que una tupla represente una permutación estricta y completa de los pedidos.

    Args:
        pedidos_por_id: Diccionario de pedidos válidos de la instancia.
        cromosoma: Secuencia ordenada de identificadores de pedido.

    Raises:
        ValueError: Si la longitud no coincide, hay repetidos o los IDs son incompatibles.
    """
    ids_esperados = set(pedidos_por_id)
    ids_recibidos = set(cromosoma)

    if len(cromosoma) != len(pedidos_por_id):
        raise ValueError("Longitud incorrecta de la permutación.")

    if len(cromosoma) != len(ids_recibidos):
        raise ValueError("La permutación contiene pedidos repetidos.")

    if ids_recibidos != ids_esperados:
        faltantes = sorted(ids_esperados - ids_recibidos)
        desconocidos = sorted(ids_recibidos - ids_esperados)
        raise ValueError(
            f"Permutación incompatible. Faltantes={faltantes}, desconocidos={desconocidos}."
        )


def decodificar_viajes_permutacion(
    instancia: InstanciaTurno,
    pedidos_por_id: dict[str, PedidoInput],
    cromosoma: Cromosoma,
) -> list[list[str]]:
    """Empaqueta una secuencia ordenada de pedidos en viajes respetando capacidad y volcadores.

    Los viajes se cierran cuando una orden supera la capacidad restante del camión
    o cuando un pedido requiere descarga por volcador (que debe situarse al final).

    Args:
        instancia: Escenario del turno con la capacidad de camión.
        pedidos_por_id: Mapeo de pedidos por ID.
        cromosoma: Orden de entrega a evaluar.

    Returns:
        Lista de viajes, donde cada viaje es una lista de identificadores de pedidos.
    """
    validar_permutacion(pedidos_por_id, cromosoma)

    viajes: list[list[str]] = []
    viaje_actual: list[str] = []
    carga_actual = 0
    contiene_volcador = False

    def cerrar_viaje() -> None:
        nonlocal viaje_actual, carga_actual, contiene_volcador
        if viaje_actual:
            viajes.append(list(viaje_actual))
        viaje_actual = []
        carga_actual = 0
        contiene_volcador = False

    for pedido_id in cromosoma:
        pedido = pedidos_por_id[pedido_id]

        if contiene_volcador:
            cerrar_viaje()

        if carga_actual + pedido.unidades_capacidad > instancia.capacidad_camion:
            cerrar_viaje()

        viaje_actual.append(pedido_id)
        carga_actual += pedido.unidades_capacidad

        if pedido.requiere_volcador:
            contiene_volcador = True
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
    """Calcula de forma aislada el minuto proyectado de retorno al corralón de un único viaje.

    Args:
        pedido_ids: Identificadores de los pedidos que componen el viaje.
        minuto_inicio: Momento en el que se inicia la preparación y carga.
        pedidos_por_id: Mapeo de pedidos de la instancia.
        matriz: Matriz cinemática de viajes.
        configuracion: Parámetros operativos y tiempos de atención.

    Returns:
        Minuto del día de regreso del vehículo al depósito.
    """
    unidades = sum(pedidos_por_id[p_id].unidades_capacidad for p_id in pedido_ids)
    minuto_actual = minuto_inicio + tiempo_carga_estimado_min(unidades, configuracion)
    nodo_actual = configuracion.id_nodo_corralon

    for pedido_id in pedido_ids:
        pedido = pedidos_por_id[pedido_id]
        minuto_actual += tiempo_viaje_esperado_min(
            matriz, nodo_actual, pedido_id, minuto_actual, configuracion
        )

        espera = estimar_espera_cliente(pedido, minuto_actual, configuracion)
        minuto_actual = espera.minuto_inicio_descarga
        minuto_actual += tiempo_descarga_estimado_min(pedido, configuracion)
        nodo_actual = pedido_id

    minuto_actual += tiempo_viaje_esperado_min(
        matriz, nodo_actual, configuracion.id_nodo_corralon, minuto_actual, configuracion
    )
    return minuto_actual


def _seleccionar_camion_candidato(
    disponibilidad: list[float],
    planes_camion: list[PlanCamion],
    cantidad_camiones: int,
) -> int:
    """Selecciona el camión que se desocupa más temprano, desempatando por carga de viajes e ID."""
    return min(
        range(cantidad_camiones),
        key=lambda c_id: (
            disponibilidad[c_id],
            len(planes_camion[c_id].viajes),
            c_id,
        ),
    )


def decodificar_plan_permutacion(
    instancia: InstanciaTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    cromosoma: Cromosoma,
    algoritmo: AlgoritmoPlanificacion,
) -> PlanTurno:
    """Transforma una permutación de pedidos en un plan operativo balanceado entre la flota.

    Decodifica los viajes necesarios y los asigna dinámicamente al camión disponible
    más temprano simulando la concurrencia operativa en el corralón.

    Args:
        instancia: Datos de la jornada.
        matriz: Matriz vial del turno.
        configuracion: Parámetros del planificador.
        cromosoma: Ordenamiento de entregas.
        algoritmo: Identificador del algoritmo solicitante.

    Returns:
        PlanTurno válido y factible listo para ser evaluado o ejecutado.

    Raises:
        RuntimeError: Si la decodificación viola alguna restricción dura de la instancia.
    """
    pedidos_por_id = {pedido.pedido_id: pedido for pedido in instancia.pedidos}
    viajes = decodificar_viajes_permutacion(instancia, pedidos_por_id, cromosoma)

    planes_camion = [PlanCamion(camion_id=c_id) for c_id in range(instancia.cantidad_camiones)]
    disponibilidad = [
        float(instancia.hora_inicio_turno_min) for _ in range(instancia.cantidad_camiones)
    ]

    for pedido_ids in viajes:
        camion_id = _seleccionar_camion_candidato(
            disponibilidad, planes_camion, instancia.cantidad_camiones
        )
        numero_viaje = len(planes_camion[camion_id].viajes) + 1
        planes_camion[camion_id].viajes.append(
            ViajePlan(numero_viaje=numero_viaje, pedido_ids=list(pedido_ids))
        )

        plan_parcial = PlanTurno(
            instancia_id=instancia.instancia_id,
            algoritmo=algoritmo,
            camiones=planes_camion,
        )
        resultado_operativo = simular_plan_operativo_estimado(
            instancia, plan_parcial, matriz, configuracion
        )
        disponibilidad = list(resultado_operativo.finales_camiones_min)

    plan = PlanTurno(
        instancia_id=instancia.instancia_id,
        algoritmo=algoritmo,
        camiones=planes_camion,
    )

    validacion = validar_plan(instancia, plan)
    if not validacion.valido:
        raise RuntimeError(
            "El decodificador compartido produjo un plan inválido: "
            + " | ".join(validacion.errores)
        )

    return plan
