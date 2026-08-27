from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
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


_TOLERANCIA_TIEMPO = 1e-9


@dataclass(frozen=True)
class EstimacionEsperaCliente:
    tiempo_espera_ventana_min: float
    tiempo_espera_respuesta_min: float
    minuto_inicio_descarga: float


@dataclass(frozen=True)
class ReservaEmpleadosCorralon:
    """
    Bloqueo determinístico opcional de empleados del corralón.

    Sirve para probar y auditar competencia de recursos, por ejemplo
    cuando un proveedor ya está siendo atendido. No intenta predecir
    las llegadas estocásticas futuras de proveedores.
    """

    minuto_inicio: float
    minuto_fin: float
    empleados_ocupados: int
    motivo: str = "PROVEEDOR"

    def __post_init__(self) -> None:
        if self.minuto_fin < self.minuto_inicio:
            raise ValueError(
                "La reserva de empleados debe cumplir "
                "minuto_inicio <= minuto_fin."
            )

        if self.empleados_ocupados < 0:
            raise ValueError(
                "empleados_ocupados no puede ser negativo."
            )


@dataclass(frozen=True)
class EstimacionCargaViaje:
    camion_id: int
    numero_viaje: int
    unidades: int
    minuto_inicio: float
    personas_estimadas: int
    empleados_corralon_asignados: int
    chofer_ayudante_asignado: bool
    duracion_min: float
    minuto_fin: float


@dataclass(frozen=True)
class ResultadoOperacionEstimada:
    cargas: tuple[EstimacionCargaViaje, ...]
    finales_camiones_min: tuple[float, ...]

    distancia_total_m: float
    tiempo_carga_total_min: float
    tiempo_viaje_total_min: float
    tiempo_espera_ventana_total_min: float
    tiempo_espera_respuesta_cliente_total_min: float
    tiempo_descarga_total_min: float
    tardanza_total_min: float
    pedidos_tardios: int
    viajes_totales: int


@dataclass
class _EstadoCamion:
    plan: PlanCamion
    indice_viaje: int = 0
    estado: str = "FINALIZADO"
    minuto_fin_carga: float = -1.0
    minuto_regreso: float = -1.0
    minuto_fin: float = -1.0

    @property
    def tiene_viaje_pendiente(self) -> bool:
        return self.indice_viaje < len(self.plan.viajes)


@dataclass(frozen=True)
class _ResultadoViaje:
    minuto_regreso: float
    distancia_m: float
    tiempo_viaje_min: float
    tiempo_espera_ventana_min: float
    tiempo_espera_respuesta_min: float
    tiempo_descarga_min: float
    tardanza_min: float
    pedidos_tardios: int


def tiempo_carga_estimado_min(
    unidades: int,
    configuracion: ConfiguracionPlanificacion,
    personas: int | None = None,
) -> float:
    """
    Tiempo esperado de carga sin variación aleatoria.

    Cuando personas es None se conserva el comportamiento histórico
    para heurísticas incrementales. La evaluación final del plan pasa
    siempre la cantidad dinámica calculada por el simulador operativo.
    """
    if unidades <= 0:
        raise ValueError(
            "unidades debe ser > 0."
        )

    personas_estimadas = (
        configuracion.personas_carga_estimadas
        if personas is None
        else personas
    )

    if personas_estimadas <= 0:
        raise ValueError(
            "personas debe ser > 0."
        )

    personas_efectivas = (
        1.0
        + configuracion
        .carga_eficiencia_persona_adicional
        * (personas_estimadas - 1)
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


def _simular_viaje_despues_carga(
    viaje: ViajePlan,
    minuto_salida: float,
    pedidos_por_id: dict[str, PedidoInput],
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
) -> _ResultadoViaje:
    minuto_actual = minuto_salida
    nodo_actual = configuracion.id_nodo_corralon

    distancia_m = 0.0
    tiempo_viaje_min = 0.0
    tiempo_espera_ventana_min = 0.0
    tiempo_espera_respuesta_min = 0.0
    tiempo_descarga_min = 0.0
    tardanza_min = 0.0
    pedidos_tardios = 0

    for pedido_id in viaje.pedido_ids:
        pedido = pedidos_por_id[pedido_id]

        distancia_m += matriz.distancia(
            nodo_actual,
            pedido_id,
        )

        tiempo_tramo = tiempo_viaje_esperado_min(
            matriz,
            nodo_actual,
            pedido_id,
            minuto_actual,
            configuracion,
        )

        tiempo_viaje_min += tiempo_tramo
        minuto_actual += tiempo_tramo

        tardanza_pedido_min = max(
            0.0,
            minuto_actual - pedido.hora_hasta_min,
        )
        tardanza_min += tardanza_pedido_min
        if tardanza_pedido_min > _TOLERANCIA_TIEMPO:
            pedidos_tardios += 1

        espera = estimar_espera_cliente(
            pedido,
            minuto_actual,
            configuracion,
        )

        tiempo_espera_ventana_min += (
            espera.tiempo_espera_ventana_min
        )
        tiempo_espera_respuesta_min += (
            espera.tiempo_espera_respuesta_min
        )

        minuto_actual = espera.minuto_inicio_descarga

        descarga = tiempo_descarga_estimado_min(
            pedido,
            configuracion,
        )

        tiempo_descarga_min += descarga
        minuto_actual += descarga
        nodo_actual = pedido_id

    distancia_m += matriz.distancia(
        nodo_actual,
        configuracion.id_nodo_corralon,
    )

    tiempo_regreso = tiempo_viaje_esperado_min(
        matriz,
        nodo_actual,
        configuracion.id_nodo_corralon,
        minuto_actual,
        configuracion,
    )

    tiempo_viaje_min += tiempo_regreso
    minuto_actual += tiempo_regreso

    return _ResultadoViaje(
        minuto_regreso=minuto_actual,
        distancia_m=distancia_m,
        tiempo_viaje_min=tiempo_viaje_min,
        tiempo_espera_ventana_min=(
            tiempo_espera_ventana_min
        ),
        tiempo_espera_respuesta_min=(
            tiempo_espera_respuesta_min
        ),
        tiempo_descarga_min=tiempo_descarga_min,
        tardanza_min=tardanza_min,
        pedidos_tardios=pedidos_tardios,
    )


def _estado_operativo_en_minuto(
    estado: _EstadoCamion,
    minuto: float,
) -> str:
    if estado.estado == "FINALIZADO":
        return "FINALIZADO"

    if estado.estado == "ESPERANDO_CARGA":
        return "CORRALON_CON_TRABAJO"

    if estado.estado == "OCUPADO":
        if minuto < estado.minuto_fin_carga - _TOLERANCIA_TIEMPO:
            return "CARGANDO"

        if minuto < estado.minuto_regreso - _TOLERANCIA_TIEMPO:
            return "EN_CALLE"

    return "CORRALON_CON_TRABAJO"


def _cantidad_empleados_reservados(
    minuto: float,
    reservas: tuple[ReservaEmpleadosCorralon, ...],
    cantidad_total: int,
) -> int:
    ocupados = sum(
        reserva.empleados_ocupados
        for reserva in reservas
        if (
            reserva.minuto_inicio
            <= minuto
            < reserva.minuto_fin
        )
    )

    return min(
        cantidad_total,
        ocupados,
    )


def simular_plan_operativo_estimado(
    instancia: InstanciaTurno,
    plan: PlanTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    reservas_empleados: tuple[
        ReservaEmpleadosCorralon,
        ...
    ] = (),
) -> ResultadoOperacionEstimada:
    """
    Simula de forma determinística la cronología esperada del plan.

    Replica la política de asignación de trabajadores de AnyLogic para
    dos camiones:
    - el chofer propio siempre carga;
    - si el otro camión tiene trabajo en el corralón, se asigna como
      máximo un empleado;
    - si el otro está en la calle, pueden asignarse dos empleados;
    - si el otro terminó, pueden asignarse dos empleados y su chofer.

    Las cargas que empiezan en el mismo instante se procesan por ID de
    camión, como desempate determinista. Con dos empleados disponibles,
    dos camiones que cargan simultáneamente reciben uno cada uno.
    """
    pedidos_por_id = {
        pedido.pedido_id: pedido
        for pedido in instancia.pedidos
    }

    planes_por_id = {
        plan_camion.camion_id: plan_camion
        for plan_camion in plan.camiones
    }

    estados: dict[int, _EstadoCamion] = {}

    for camion_id in range(instancia.cantidad_camiones):
        plan_camion = planes_por_id.get(
            camion_id,
            PlanCamion(camion_id=camion_id),
        )

        if plan_camion.viajes:
            estado_inicial = "ESPERANDO_CARGA"
            minuto_fin = -1.0
        else:
            estado_inicial = "FINALIZADO"
            minuto_fin = float(
                instancia.hora_inicio_turno_min
            )

        estados[camion_id] = _EstadoCamion(
            plan=plan_camion,
            estado=estado_inicial,
            minuto_fin=minuto_fin,
        )

    empleados_ocupados_hasta = [
        float("-inf")
        for _ in range(
            configuracion.cantidad_empleados_corralon
        )
    ]

    chofer_ayudante_ocupado_hasta = {
        camion_id: float("-inf")
        for camion_id in range(
            instancia.cantidad_camiones
        )
    }

    eventos: list[tuple[float, int, str, int]] = []
    secuencia_evento = count()

    for camion_id, estado in estados.items():
        if estado.estado == "ESPERANDO_CARGA":
            heapq.heappush(
                eventos,
                (
                    float(instancia.hora_inicio_turno_min),
                    next(secuencia_evento),
                    "SOLICITUD_CARGA",
                    camion_id,
                ),
            )

    cargas: list[EstimacionCargaViaje] = []

    distancia_total_m = 0.0
    tiempo_carga_total_min = 0.0
    tiempo_viaje_total_min = 0.0
    tiempo_espera_ventana_total_min = 0.0
    tiempo_espera_respuesta_total_min = 0.0
    tiempo_descarga_total_min = 0.0
    tardanza_total_min = 0.0
    pedidos_tardios = 0
    viajes_totales = 0

    while eventos:
        minuto_evento = eventos[0][0]
        eventos_mismo_instante: list[
            tuple[float, int, str, int]
        ] = []

        while (
            eventos
            and abs(eventos[0][0] - minuto_evento)
            <= _TOLERANCIA_TIEMPO
        ):
            eventos_mismo_instante.append(
                heapq.heappop(eventos)
            )

        solicitudes_carga: set[int] = set()

        # Primero procesamos retornos. Así un camión que termina o pide
        # su siguiente carga queda correctamente visible para el otro.
        for _, _, tipo_evento, camion_id in sorted(
            eventos_mismo_instante,
            key=lambda evento: (
                0 if evento[2] == "RETORNO" else 1,
                evento[3],
                evento[1],
            ),
        ):
            estado = estados[camion_id]

            if tipo_evento == "RETORNO":
                estado.indice_viaje += 1
                estado.minuto_fin_carga = -1.0
                estado.minuto_regreso = -1.0

                if estado.tiene_viaje_pendiente:
                    estado.estado = "ESPERANDO_CARGA"
                    solicitudes_carga.add(camion_id)
                else:
                    estado.estado = "FINALIZADO"
                    estado.minuto_fin = minuto_evento

            elif tipo_evento == "SOLICITUD_CARGA":
                solicitudes_carga.add(camion_id)

        for camion_id in sorted(solicitudes_carga):
            estado = estados[camion_id]

            if not estado.tiene_viaje_pendiente:
                estado.estado = "FINALIZADO"
                estado.minuto_fin = minuto_evento
                continue

            otros_ids = [
                otro_id
                for otro_id in sorted(estados)
                if otro_id != camion_id
            ]

            otro_id = (
                otros_ids[0]
                if otros_ids
                else None
            )

            if otro_id is None:
                estado_otro = "FINALIZADO"
            else:
                estado_otro = _estado_operativo_en_minuto(
                    estados[otro_id],
                    minuto_evento,
                )

            if estado_otro in {
                "FINALIZADO",
                "EN_CALLE",
            }:
                max_empleados = 2
            else:
                max_empleados = 1

            max_empleados = min(
                max_empleados,
                configuracion.cantidad_empleados_corralon,
            )

            reservados = _cantidad_empleados_reservados(
                minuto_evento,
                reservas_empleados,
                configuracion.cantidad_empleados_corralon,
            )

            indices_reservados = set(
                range(reservados)
            )

            empleados_disponibles = [
                empleado_id
                for empleado_id, ocupado_hasta
                in enumerate(empleados_ocupados_hasta)
                if (
                    empleado_id not in indices_reservados
                    and ocupado_hasta
                    <= minuto_evento + _TOLERANCIA_TIEMPO
                )
            ]

            empleados_asignados = empleados_disponibles[
                :max_empleados
            ]

            chofer_ayudante = False

            if (
                otro_id is not None
                and estado_otro == "FINALIZADO"
                and chofer_ayudante_ocupado_hasta[otro_id]
                <= minuto_evento + _TOLERANCIA_TIEMPO
            ):
                chofer_ayudante = True

            personas = (
                1
                + len(empleados_asignados)
                + (1 if chofer_ayudante else 0)
            )

            viaje = estado.plan.viajes[
                estado.indice_viaje
            ]

            unidades = sum(
                pedidos_por_id[pedido_id]
                .unidades_capacidad
                for pedido_id in viaje.pedido_ids
            )

            duracion_carga = tiempo_carga_estimado_min(
                unidades,
                configuracion,
                personas=personas,
            )

            minuto_fin_carga = (
                minuto_evento + duracion_carga
            )

            for empleado_id in empleados_asignados:
                empleados_ocupados_hasta[empleado_id] = (
                    minuto_fin_carga
                )

            if chofer_ayudante and otro_id is not None:
                chofer_ayudante_ocupado_hasta[otro_id] = (
                    minuto_fin_carga
                )

            cargas.append(
                EstimacionCargaViaje(
                    camion_id=camion_id,
                    numero_viaje=viaje.numero_viaje,
                    unidades=unidades,
                    minuto_inicio=minuto_evento,
                    personas_estimadas=personas,
                    empleados_corralon_asignados=(
                        len(empleados_asignados)
                    ),
                    chofer_ayudante_asignado=(
                        chofer_ayudante
                    ),
                    duracion_min=duracion_carga,
                    minuto_fin=minuto_fin_carga,
                )
            )

            resultado_viaje = _simular_viaje_despues_carga(
                viaje,
                minuto_fin_carga,
                pedidos_por_id,
                matriz,
                configuracion,
            )

            estado.estado = "OCUPADO"
            estado.minuto_fin_carga = minuto_fin_carga
            estado.minuto_regreso = (
                resultado_viaje.minuto_regreso
            )

            heapq.heappush(
                eventos,
                (
                    resultado_viaje.minuto_regreso,
                    next(secuencia_evento),
                    "RETORNO",
                    camion_id,
                ),
            )

            viajes_totales += 1
            tiempo_carga_total_min += duracion_carga
            distancia_total_m += resultado_viaje.distancia_m
            tiempo_viaje_total_min += (
                resultado_viaje.tiempo_viaje_min
            )
            tiempo_espera_ventana_total_min += (
                resultado_viaje
                .tiempo_espera_ventana_min
            )
            tiempo_espera_respuesta_total_min += (
                resultado_viaje
                .tiempo_espera_respuesta_min
            )
            tiempo_descarga_total_min += (
                resultado_viaje.tiempo_descarga_min
            )
            tardanza_total_min += (
                resultado_viaje.tardanza_min
            )
            pedidos_tardios += resultado_viaje.pedidos_tardios

    finales_camiones = tuple(
        estados[camion_id].minuto_fin
        for camion_id in range(
            instancia.cantidad_camiones
        )
    )

    return ResultadoOperacionEstimada(
        cargas=tuple(cargas),
        finales_camiones_min=finales_camiones,
        distancia_total_m=distancia_total_m,
        tiempo_carga_total_min=tiempo_carga_total_min,
        tiempo_viaje_total_min=tiempo_viaje_total_min,
        tiempo_espera_ventana_total_min=(
            tiempo_espera_ventana_total_min
        ),
        tiempo_espera_respuesta_cliente_total_min=(
            tiempo_espera_respuesta_total_min
        ),
        tiempo_descarga_total_min=(
            tiempo_descarga_total_min
        ),
        tardanza_total_min=tardanza_total_min,
        pedidos_tardios=pedidos_tardios,
        viajes_totales=viajes_totales,
    )
