from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt

from .config import ConfiguracionPlanificacion
from .schema import InstanciaTurno


@dataclass(frozen=True)
class MatrizViaje:
    nodo_ids: list[str]

    indice_por_id: dict[str, int]

    distancia_metros: list[list[float]]

    tiempo_base_min: list[list[float]]

    def distancia(
        self,
        origen_id: str,
        destino_id: str,
    ) -> float:
        i = self._indice(origen_id)
        j = self._indice(destino_id)

        return self.distancia_metros[i][j]

    def tiempo_base(
        self,
        origen_id: str,
        destino_id: str,
    ) -> float:
        i = self._indice(origen_id)
        j = self._indice(destino_id)

        return self.tiempo_base_min[i][j]

    def _indice(
        self,
        nodo_id: str,
    ) -> int:
        try:
            return self.indice_por_id[nodo_id]

        except KeyError as exc:
            raise ValueError(
                "Nodo inexistente en MatrizViaje: "
                f"{nodo_id}"
            ) from exc


def distancia_haversine_metros(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radio_tierra_m = 6_371_000.0

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = (
        sin(delta_lat / 2.0) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(delta_lon / 2.0) ** 2
    )

    c = 2.0 * atan2(
        sqrt(a),
        sqrt(1.0 - a),
    )

    return radio_tierra_m * c


def construir_matriz_viaje(
    instancia: InstanciaTurno,
    configuracion: ConfiguracionPlanificacion,
) -> MatrizViaje:
    nodo_ids = [
        configuracion.id_nodo_corralon,
        *[
            pedido.pedido_id
            for pedido in instancia.pedidos
        ],
    ]

    if len(nodo_ids) != len(set(nodo_ids)):
        raise ValueError(
            "No se puede construir la matriz: "
            "existen IDs de nodo duplicados."
        )

    coordenadas = [
        (
            instancia.lat_corralon,
            instancia.lon_corralon,
        ),
        *[
            (
                pedido.latitud,
                pedido.longitud,
            )
            for pedido in instancia.pedidos
        ],
    ]

    cantidad_nodos = len(nodo_ids)

    distancias = [
        [
            0.0
            for _ in range(cantidad_nodos)
        ]
        for _ in range(cantidad_nodos)
    ]

    tiempos = [
        [
            0.0
            for _ in range(cantidad_nodos)
        ]
        for _ in range(cantidad_nodos)
    ]

    for i in range(cantidad_nodos):
        for j in range(cantidad_nodos):
            if i == j:
                continue

            distancia_geodesica = (
                distancia_haversine_metros(
                    *coordenadas[i],
                    *coordenadas[j],
                )
            )

            distancia_ajustada = (
                distancia_geodesica
                * configuracion.factor_urbano_distancia
            )

            tiempo_min = (
                distancia_ajustada
                / 1000.0
                / configuracion.velocidad_base_kmh
                * 60.0
            )

            distancias[i][j] = (
                distancia_ajustada
            )

            tiempos[i][j] = tiempo_min

    return MatrizViaje(
        nodo_ids=nodo_ids,

        indice_por_id={
            nodo_id: indice
            for indice, nodo_id
            in enumerate(nodo_ids)
        },

        distancia_metros=distancias,

        tiempo_base_min=tiempos,
    )


def factor_trafico_esperado(
    minuto_dia: float,
    configuracion: ConfiguracionPlanificacion,
) -> float:
    if (
        configuracion
        .trafico_pico_manana_inicio_min
        <= minuto_dia
        < configuracion
        .trafico_pico_manana_fin_min
    ):
        return (
            configuracion
            .trafico_factor_pico_manana
        )

    if (
        configuracion
        .trafico_pico_tarde_inicio_min
        <= minuto_dia
        <= configuracion
        .trafico_pico_tarde_fin_min
    ):
        return (
            configuracion
            .trafico_factor_pico_tarde
        )

    return configuracion.trafico_factor_normal


def tiempo_viaje_esperado_min(
    matriz: MatrizViaje,
    origen_id: str,
    destino_id: str,
    minuto_salida: float,
    configuracion: ConfiguracionPlanificacion,
) -> float:
    return (
        matriz.tiempo_base(
            origen_id,
            destino_id,
        )
        * factor_trafico_esperado(
            minuto_salida,
            configuracion,
        )
    )