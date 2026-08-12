from dataclasses import dataclass, field
from enum import Enum


class Turno(str, Enum):
    MANANA = "MANANA"
    TARDE = "TARDE"


class EstadoPedido(str, Enum):
    PENDIENTE = "PENDIENTE"
    PLANIFICADO = "PLANIFICADO"
    CARGADO = "CARGADO"
    EN_TRANSITO = "EN_TRANSITO"
    ESPERANDO = "ESPERANDO"
    ENTREGADO = "ENTREGADO"
    NO_ENTREGADO = "NO_ENTREGADO"
    PENDIENTE_DIA_SIGUIENTE = "PENDIENTE_DIA_SIGUIENTE"


class AlgoritmoPlanificacion(str, Enum):
    RL = "RL"
    GA = "GA"
    GREEDY = "GREEDY"
    RANDOM = "RANDOM"
    MANUAL_TEST = "MANUAL_TEST"


class ModoEjecucion(str, Enum):
    SIMULACION = "SIMULACION"
    RL_TRAINING = "RL_TRAINING"
    RL_EVALUATION = "RL_EVALUATION"
    PRODUCCION = "PRODUCCION"


@dataclass
class PedidoInput:
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
        return self.total_partes > 1


@dataclass
class InstanciaTurno:
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
    numero_viaje: int
    pedido_ids: list[str] = field(default_factory=list)


@dataclass
class PlanCamion:
    camion_id: int
    viajes: list[ViajePlan] = field(default_factory=list)


@dataclass
class PlanTurno:
    instancia_id: str
    algoritmo: AlgoritmoPlanificacion
    camiones: list[PlanCamion] = field(default_factory=list)
    costo_estimado: float = 0.0
    tiempo_computo_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ResultadoValidacionPlan:
    valido: bool = True
    errores: list[str] = field(default_factory=list)


@dataclass
class ResultadoTurno:
    instancia_id: str
    algoritmo: AlgoritmoPlanificacion
    plan_valido: bool = False
    ejecucion_completada: bool = False
    errores: list[str] = field(default_factory=list)