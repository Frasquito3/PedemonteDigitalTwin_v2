"""Modelos de datos, enumeraciones y estructuras de resultado del planificador."""

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Turno",
    "EstadoPedido",
    "AlgoritmoPlanificacion",
    "ModoEjecucion",
    "TipoActividadCamion",
    "PedidoInput",
    "InstanciaTurno",
    "ViajePlan",
    "PlanCamion",
    "PlanTurno",
    "RegistroActividadCamion",
    "MetricasCamion",
    "ResumenPedidoOriginal",
    "MetricasGlobalesTurno",
    "DesgloseCostoTurno",
    "ResultadoValidacionPlan",
    "ResultadoTurno",
]


class Turno(str, Enum):
    """Jornada u horario de operación del corralón."""

    MANANA = "MANANA"
    TARDE = "TARDE"


class EstadoPedido(str, Enum):
    """Ciclo de vida y estados de entrega de un pedido."""

    PENDIENTE = "PENDIENTE"
    PLANIFICADO = "PLANIFICADO"
    CARGADO = "CARGADO"
    EN_TRANSITO = "EN_TRANSITO"
    ESPERANDO = "ESPERANDO"
    ENTREGADO = "ENTREGADO"
    NO_ENTREGADO = "NO_ENTREGADO"
    PENDIENTE_DIA_SIGUIENTE = "PENDIENTE_DIA_SIGUIENTE"


class AlgoritmoPlanificacion(str, Enum):
    """Identificador del algoritmo o estrategia de resolución empleada."""

    RL = "RL"
    GA = "GA"
    GREEDY = "GREEDY"
    RANDOM = "RANDOM"
    MANUAL_TEST = "MANUAL_TEST"


class ModoEjecucion(str, Enum):
    """Contexto de ejecución y evaluación del planificador."""

    SIMULACION = "SIMULACION"
    RL_TRAINING = "RL_TRAINING"
    RL_EVALUATION = "RL_EVALUATION"
    PRODUCCION = "PRODUCCION"


class TipoActividadCamion(str, Enum):
    """Naturaleza de la actividad operativa registrada en la traza de un camión."""

    CARGA = "CARGA"
    VIAJE_CLIENTE = "VIAJE_CLIENTE"
    ESPERA_CLIENTE = "ESPERA_CLIENTE"
    DESCARGA = "DESCARGA"
    REGRESO_CORRALON = "REGRESO_CORRALON"


@dataclass
class PedidoInput:
    """Demanda individual o fracción de entrega asignable a un camión.

    Si el pedido original excede la capacidad de un camión o requiere divisiones
    logísticas, se particiona en múltiples fracciones vinculadas por `pedido_original_id`.
    Las ventanas horarias (`hora_desde_min` y `hora_hasta_min`) se expresan en minutos
    del día transcurridos desde las 00:00.
    """

    pedido_id: str
    pedido_original_id: str
    numero_parte: int
    total_partes: int
    turno: Turno
    latitud: float
    longitud: float
    unidades_capacidad: int
    requiere_volcador: bool
    tiene_ventana_especifica: bool
    hora_desde_min: int
    hora_hasta_min: int
    cliente: str = ""
    direccion: str = ""
    barrio: str = ""
    observaciones: str = ""

    @property
    def es_split(self) -> bool:
        """Indica si la entrega pertenece a un pedido particionado en varias partes."""
        return self.total_partes > 1


@dataclass
class InstanciaTurno:
    """Definición inmutable del escenario de planificación para un turno de trabajo.

    Contiene los pedidos a distribuir, los parámetros de la flota (capacidad y
    cantidad de vehículos), los límites horarios del turno y las coordenadas del corralón.
    """

    instancia_id: str
    fecha_operacion: str
    turno: Turno
    pedidos: list[PedidoInput]
    lat_corralon: float
    lon_corralon: float
    capacidad_camion: int
    cantidad_camiones: int
    hora_inicio_turno_min: int
    hora_fin_objetivo_min: int
    hora_fin_tolerancia_min: int
    seed_escenario: int
    seed_ejecucion: int


@dataclass
class ViajePlan:
    """Secuencia ordenada de entregas asignadas a una salida del camión desde el depósito."""

    numero_viaje: int
    pedido_ids: list[str] = field(default_factory=list)


@dataclass
class PlanCamion:
    """Itinerario completo y secuencial de viajes asignados a un camión durante el turno."""

    camion_id: int
    viajes: list[ViajePlan] = field(default_factory=list)


@dataclass
class PlanTurno:
    """Plan de distribución generado para la flota en un turno de trabajo."""

    instancia_id: str
    algoritmo: AlgoritmoPlanificacion
    camiones: list[PlanCamion] = field(default_factory=list)
    costo_estimado: float = 0.0
    tiempo_computo_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class RegistroActividadCamion:
    """Evento atómico registrado en la traza cronológica de operaciones de un camión.

    Los momentos `inicio_min_dia` y `fin_min_dia` se expresan en minutos desde las 00:00.
    """

    camion_id: int
    tipo: TipoActividadCamion
    inicio_min_dia: float
    fin_min_dia: float
    duracion_min: float
    numero_viaje: int = -1
    pedido_id: str = ""
    distancia_metros: float = 0.0
    personas_asignadas: int = 0
    unidades: int = 0


@dataclass
class MetricasCamion:
    """Indicadores consolidados de actividad y desempeño para un vehículo individual.

    Agrupa conteos de entregas, volúmenes transportados, distancias y tiempos acumulados
    por estado operativo (carga, viaje, espera, descarga e inactividad).
    Los campos de inicio/fin registran el minuto del día de apertura y cierre (-1.0 si no operó).
    """

    camion_id: int

    viajes_completados: int = 0

    tareas_planificadas: int = 0
    tareas_entregadas: int = 0
    tareas_no_entregadas: int = 0

    unidades_planificadas: int = 0
    unidades_entregadas: int = 0
    unidades_retornadas: int = 0

    distancia_total_metros: float = 0.0
    distancia_total_km: float = 0.0

    tiempo_carga_min: float = 0.0
    tiempo_viaje_cliente_min: float = 0.0
    tiempo_regreso_min: float = 0.0
    tiempo_viaje_total_min: float = 0.0
    tiempo_espera_cliente_min: float = 0.0
    tiempo_descarga_min: float = 0.0
    tiempo_activo_min: float = 0.0
    tiempo_inactivo_min: float = 0.0

    minuto_inicio_operacion: float = -1.0
    minuto_fin_operacion: float = -1.0
    duracion_operacion_min: float = 0.0

    ocupacion_promedio_viajes: float = 0.0
    promedio_personas_carga: float = 0.0


@dataclass
class ResumenPedidoOriginal:
    """Estado de satisfacción global de un pedido de cliente y sus entregas parciales."""

    pedido_original_id: str
    partes_planificadas: int = 0
    partes_entregadas: int = 0
    unidades_planificadas: int = 0
    unidades_entregadas: int = 0
    completo: bool = False


@dataclass
class MetricasGlobalesTurno:
    """Indicadores clave de desempeño (KPIs) consolidados para el turno completo.

    Agrupa métricas de nivel de servicio (entregas en ventana y puntualidad),
    volumen y demanda, utilización de tiempos de la flota, balance de carga
    entre vehículos y atención de camiones proveedores en el corralón.

    Nota de unidades:
        Los tiempos y duraciones se expresan en minutos, las distancias en kilómetros
        y los instantes cronológicos en minutos transcurridos desde las 00:00 del día.
    """

    tareas_planificadas: int = 0
    tareas_entregadas: int = 0
    tareas_no_entregadas: int = 0
    porcentaje_tareas_entregadas: float = 0.0

    pedidos_originales_planificados: int = 0
    pedidos_originales_completos: int = 0
    pedidos_originales_incompletos: int = 0
    porcentaje_pedidos_originales_completos: float = 0.0

    unidades_planificadas: int = 0
    unidades_entregadas: int = 0
    unidades_retornadas: int = 0
    porcentaje_unidades_entregadas: float = 0.0

    entregas_en_ventana: int = 0
    entregas_aceptadas_antes: int = 0
    entregas_tardias_aceptadas: int = 0
    llegadas_tardias_rechazadas: int = 0
    entregas_sin_tardanza: int = 0

    porcentaje_en_ventana_sobre_entregadas: float = 0.0
    porcentaje_servicio_sin_tardanza_sobre_planificadas: float = 0.0

    cantidad_llegadas_tardias: int = 0
    tardanza_total_min: float = 0.0
    tardanza_promedio_min: float = 0.0
    tardanza_max_min: float = 0.0

    viajes_totales: int = 0
    distancia_total_km: float = 0.0

    tiempo_carga_total_min: float = 0.0
    tiempo_viaje_total_min: float = 0.0
    tiempo_espera_cliente_total_min: float = 0.0
    tiempo_descarga_total_min: float = 0.0
    tiempo_activo_acumulado_camiones_min: float = 0.0

    minuto_fin_turno_real: float = -1.0
    duracion_turno_real_min: float = 0.0
    overtime_min: float = 0.0
    exceso_tolerancia_min: float = 0.0

    ocupacion_global_viajes: float = 0.0
    promedio_personas_carga_global: float = 0.0
    persona_min_carga: float = 0.0
    persona_min_proveedor: float = 0.0

    diferencia_fin_camiones_min: float = 0.0
    diferencia_distancia_camiones_km: float = 0.0
    diferencia_tiempo_activo_camiones_min: float = 0.0

    proveedores_llegados: int = 0
    proveedores_completados: int = 0
    proveedores_cancelados_fin_ejecucion: int = 0
    proveedores_que_requirieron_dos_empleados: int = 0

    espera_proveedor_total_min: float = 0.0
    espera_proveedor_promedio_min: float = 0.0
    espera_proveedor_max_min: float = 0.0
    tiempo_atencion_proveedor_total_min: float = 0.0


@dataclass
class DesgloseCostoTurno:
    """Desagregación analítica de los componentes de la función objetivo de costos."""

    costo_tareas_no_entregadas: float = 0.0
    costo_pedidos_originales_incompletos: float = 0.0
    costo_tardanza: float = 0.0
    costo_exceso_tolerancia: float = 0.0
    costo_operacion: float = 0.0
    costo_distancia: float = 0.0
    costo_viajes: float = 0.0
    costo_desbalance: float = 0.0
    costo_total: float = 0.0


@dataclass
class ResultadoValidacionPlan:
    """Diagnóstico de consistencia y factibilidad de un plan evaluado."""

    valido: bool = True
    errores: list[str] = field(default_factory=list)


@dataclass
class ResultadoTurno:
    """Contenedor integral con los resultados de simulación y evaluación del turno."""

    instancia_id: str
    algoritmo: AlgoritmoPlanificacion

    fecha_operacion: str = ""
    turno: Turno = Turno.MANANA

    seed_escenario: int = 0
    seed_ejecucion: int = 0

    plan_valido: bool = False
    ejecucion_completada: bool = False

    metricas_camiones: list[MetricasCamion] = field(default_factory=list)

    registros_actividad: list[RegistroActividadCamion] = field(default_factory=list)

    resumenes_pedidos_originales: list[ResumenPedidoOriginal] = field(default_factory=list)

    metricas_globales: MetricasGlobalesTurno = field(default_factory=MetricasGlobalesTurno)

    desglose_costo: DesgloseCostoTurno = field(default_factory=DesgloseCostoTurno)

    costo_total: float = 0.0

    errores: list[str] = field(default_factory=list)
