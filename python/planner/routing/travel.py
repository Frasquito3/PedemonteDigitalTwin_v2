"""Cálculo de distancias, matrices de viaje y modelos de congestión vehicular."""

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, isfinite, radians, sin, sqrt
from typing import Protocol

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno

__all__ = [
    "Coordenada",
    "FuenteMatrizViaje",
    "ResultadoTramoViaje",
    "ProveedorViaje",
    "ProveedorHaversineAjustado",
    "MatrizViaje",
    "distancia_haversine_metros",
    "construir_matriz_viaje",
    "factor_trafico_esperado",
    "tiempo_viaje_esperado_min",
]

Coordenada = tuple[float, float]


class FuenteMatrizViaje(str, Enum):
    """Origen o mecanismo de estimación empleado para la matriz de viajes."""

    HAVERSINE_AJUSTADA = "HAVERSINE_AJUSTADA"
    VIAL_CACHE = "VIAL_CACHE"
    VIAL_LOCAL = "VIAL_LOCAL"


@dataclass(frozen=True)
class ResultadoTramoViaje:
    """Resultado del cálculo cinemático y geométrico entre dos puntos geográficos.

    Attributes:
        distancia_metros: Distancia total calculada en metros.
        tiempo_base_min: Tiempo de recorrido en minutos sin considerar congestión.
        fuente: Proveedor de cálculo utilizado.
        uso_fallback: Indica si se recurrió a un proveedor alternativo por falta de datos.
        advertencia: Mensaje descriptivo en caso de discrepancias o aproximaciones.
    """

    distancia_metros: float
    tiempo_base_min: float
    fuente: FuenteMatrizViaje
    uso_fallback: bool = False
    advertencia: str = ""


class ProveedorViaje(Protocol):
    """Protocolo para componentes proveedores de distancias y tiempos de traslado."""

    @property
    def fuente(self) -> FuenteMatrizViaje:
        """Tipo de fuente que identifica al proveedor."""
        ...

    @property
    def version(self) -> str:
        """Versión o firma del modelo de distancias."""
        ...

    def calcular_tramo(
        self,
        origen: Coordenada,
        destino: Coordenada,
        configuracion: ConfiguracionPlanificacion,
    ) -> ResultadoTramoViaje:
        """Calcula la distancia y el tiempo base entre un par de coordenadas."""
        ...


@dataclass(frozen=True)
class ProveedorHaversineAjustado:
    """Proveedor geométrico estándar basado en la fórmula de Haversine corregida.

    Aplica la distancia ortodrómica sobre la esfera terrestre multiplicada por un
    factor urbano de sinuosidad para estimar el recorrido real por calles, y calcula
    el tiempo de viaje dividiendo por la velocidad base de la flota.
    """

    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.HAVERSINE_AJUSTADA

    @property
    def version(self) -> str:
        return "haversine-ajustada-v1"

    def calcular_tramo(
        self,
        origen: Coordenada,
        destino: Coordenada,
        configuracion: ConfiguracionPlanificacion,
    ) -> ResultadoTramoViaje:
        distancia_geodesica = distancia_haversine_metros(
            origen[0], origen[1], destino[0], destino[1]
        )
        distancia_ajustada = distancia_geodesica * configuracion.factor_urbano_distancia
        tiempo_base_min = distancia_ajustada / 1000.0 / configuracion.velocidad_base_kmh * 60.0

        return ResultadoTramoViaje(
            distancia_metros=distancia_ajustada,
            tiempo_base_min=tiempo_base_min,
            fuente=self.fuente,
        )


@dataclass(frozen=True)
class MatrizViaje:
    """Matriz densa de distancias y tiempos base precalculados entre todos los nodos del turno."""

    nodo_ids: list[str]
    indice_por_id: dict[str, int]
    distancia_metros: list[list[float]]
    tiempo_base_min: list[list[float]]
    fuente: FuenteMatrizViaje = FuenteMatrizViaje.HAVERSINE_AJUSTADA
    version_fuente: str = "haversine-ajustada-v1"
    cantidad_fallbacks: int = 0
    advertencias: tuple[str, ...] = ()

    def distancia(self, origen_id: str, destino_id: str) -> float:
        """Retorna la distancia en metros entre dos nodos identificados por su ID."""
        i = self._indice(origen_id)
        j = self._indice(destino_id)
        return self.distancia_metros[i][j]

    def tiempo_base(self, origen_id: str, destino_id: str) -> float:
        """Retorna el tiempo base de viaje en minutos entre dos nodos (sin tráfico)."""
        i = self._indice(origen_id)
        j = self._indice(destino_id)
        return self.tiempo_base_min[i][j]

    @property
    def usa_fallback(self) -> bool:
        """Indica si al menos un tramo de la matriz requirió proveedor de respaldo."""
        return self.cantidad_fallbacks > 0

    def resumen_fuente(self) -> str:
        """Genera una cadena diagnóstica con la fuente, versión y fallbacks utilizados."""
        return (
            f"fuente={self.fuente.value}"
            f"|version={self.version_fuente}"
            f"|fallbacks={self.cantidad_fallbacks}"
        )

    def _indice(self, nodo_id: str) -> int:
        try:
            return self.indice_por_id[nodo_id]
        except KeyError as exc:
            raise ValueError(f"Nodo inexistente en MatrizViaje: {nodo_id}") from exc


def distancia_haversine_metros(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calcula la distancia geodésica ortodrómica en metros entre dos coordenadas WGS84."""
    radio_tierra_m = 6_371_000.0

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = sin(delta_lat / 2.0) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2.0) ** 2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))

    return radio_tierra_m * c


def _validar_resultado_tramo(
    resultado: ResultadoTramoViaje,
    origen_id: str,
    destino_id: str,
) -> None:
    """Verifica que la distancia y tiempo calculados sean valores numéricos finitos y no negativos."""
    if not isfinite(resultado.distancia_metros) or resultado.distancia_metros < 0.0:
        raise ValueError(
            f"El proveedor devolvió una distancia inválida para {origen_id} -> {destino_id}: "
            f"{resultado.distancia_metros}."
        )

    if not isfinite(resultado.tiempo_base_min) or resultado.tiempo_base_min < 0.0:
        raise ValueError(
            f"El proveedor devolvió un tiempo inválido para {origen_id} -> {destino_id}: "
            f"{resultado.tiempo_base_min}."
        )


def construir_matriz_viaje(
    instancia: InstanciaTurno,
    configuracion: ConfiguracionPlanificacion,
    proveedor: ProveedorViaje | None = None,
) -> MatrizViaje:
    """Construye la matriz de distancias y tiempos para todos los pedidos y el corralón.

    Args:
        instancia: Datos del turno con pedidos y coordenadas de corralón.
        configuracion: Parámetros operativos y cinemáticos.
        proveedor: Proveedor de cálculo (usa Haversine ajustada si es `None`).

    Returns:
        Instancia inmutable de `MatrizViaje` con distancias y tiempos indexados.

    Raises:
        ValueError: Si existen identificadores de nodo repetidos o cálculos inválidos.
    """
    proveedor_efectivo = proveedor if proveedor is not None else ProveedorHaversineAjustado()

    nodo_ids = [
        configuracion.id_nodo_corralon,
        *[pedido.pedido_id for pedido in instancia.pedidos],
    ]

    if len(nodo_ids) != len(set(nodo_ids)):
        raise ValueError("No se puede construir la matriz: existen IDs de nodo duplicados.")

    coordenadas: list[Coordenada] = [
        (instancia.lat_corralon, instancia.lon_corralon),
        *[(pedido.latitud, pedido.longitud) for pedido in instancia.pedidos],
    ]

    cantidad_nodos = len(nodo_ids)
    distancias = [[0.0 for _ in range(cantidad_nodos)] for _ in range(cantidad_nodos)]
    tiempos = [[0.0 for _ in range(cantidad_nodos)] for _ in range(cantidad_nodos)]

    cantidad_fallbacks = 0
    advertencias: list[str] = []

    for i in range(cantidad_nodos):
        for j in range(cantidad_nodos):
            if i == j:
                continue

            resultado = proveedor_efectivo.calcular_tramo(
                coordenadas[i],
                coordenadas[j],
                configuracion,
            )
            _validar_resultado_tramo(resultado, nodo_ids[i], nodo_ids[j])

            distancias[i][j] = resultado.distancia_metros
            tiempos[i][j] = resultado.tiempo_base_min

            if resultado.uso_fallback:
                cantidad_fallbacks += 1

            if resultado.advertencia and resultado.advertencia not in advertencias:
                advertencias.append(resultado.advertencia)

    return MatrizViaje(
        nodo_ids=nodo_ids,
        indice_por_id={nodo_id: indice for indice, nodo_id in enumerate(nodo_ids)},
        distancia_metros=distancias,
        tiempo_base_min=tiempos,
        fuente=proveedor_efectivo.fuente,
        version_fuente=proveedor_efectivo.version,
        cantidad_fallbacks=cantidad_fallbacks,
        advertencias=tuple(advertencias),
    )


def factor_trafico_esperado(
    minuto_dia: float,
    configuracion: ConfiguracionPlanificacion,
) -> float:
    """Determina el factor de congestión según el horario del día (minutos desde las 00:00)."""
    if (
        configuracion.trafico_pico_manana_inicio_min
        <= minuto_dia
        < configuracion.trafico_pico_manana_fin_min
    ):
        return configuracion.trafico_factor_pico_manana

    if (
        configuracion.trafico_pico_tarde_inicio_min
        <= minuto_dia
        <= configuracion.trafico_pico_tarde_fin_min
    ):
        return configuracion.trafico_factor_pico_tarde

    return configuracion.trafico_factor_normal


def tiempo_viaje_esperado_min(
    matriz: MatrizViaje,
    origen_id: str,
    destino_id: str,
    minuto_salida: float,
    configuracion: ConfiguracionPlanificacion,
) -> float:
    """Calcula el tiempo de viaje estimado en minutos ajustado por la congestión vial del horario."""
    return matriz.tiempo_base(origen_id, destino_id) * factor_trafico_esperado(
        minuto_salida, configuracion
    )
