"""Proveedor persistente de distancias y tiempos viales basado en archivo CSV indexado."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from planner.core.config import ConfiguracionPlanificacion
from planner.routing.travel import (
    Coordenada,
    FuenteMatrizViaje,
    ProveedorHaversineAjustado,
    ProveedorViaje,
    ResultadoTramoViaje,
)

__all__ = [
    "COLUMNAS_CACHE_VIAL",
    "RegistroTramoVial",
    "EstadisticasCacheVial",
    "ProveedorVialCachePersistente",
]

COLUMNAS_CACHE_VIAL = (
    "version_cache",
    "lat_origen",
    "lon_origen",
    "lat_destino",
    "lon_destino",
    "distancia_metros",
    "tiempo_base_min",
    "fuente_distancia",
    "fuente_tiempo",
)


@dataclass(frozen=True)
class RegistroTramoVial:
    """Información cinemática almacenada para un tramo dirigido entre dos coordenadas."""

    origen: Coordenada
    destino: Coordenada
    distancia_metros: float
    tiempo_base_min: float | None
    fuente_distancia: str
    fuente_tiempo: str


@dataclass(frozen=True)
class EstadisticasCacheVial:
    """Metadatos e indicadores de integridad de la caché vial cargada en memoria."""

    cantidad_tramos: int
    version_cache: str
    precision_coordenadas: int
    huella_sha256: str


class ProveedorVialCachePersistente:
    """Proveedor de distancias y tiempos de viaje respaldado por una caché CSV persistente.

    La caché es dirigida (A -> B y B -> A son tramos independientes), lo que permite
    representar manos únicas, rotondas y accesos viales reales. Si un tramo no existe
    en la caché, puede utilizar un proveedor de respaldo (fallback) geométrico.
    """

    def __init__(
        self,
        ruta_cache: str | Path,
        *,
        version_cache_esperada: str,
        precision_coordenadas: int = 6,
        proveedor_fallback: ProveedorViaje | None = None,
        permitir_fallback: bool = True,
    ) -> None:
        if not version_cache_esperada.strip():
            raise ValueError("version_cache_esperada no puede estar vacía.")

        if precision_coordenadas < 0 or precision_coordenadas > 9:
            raise ValueError("precision_coordenadas debe estar entre 0 y 9.")

        self.ruta_cache = Path(ruta_cache).expanduser().resolve()
        self.version_cache_esperada = version_cache_esperada.strip()
        self.precision_coordenadas = precision_coordenadas
        self.permitir_fallback = permitir_fallback

        self.proveedor_fallback = (
            proveedor_fallback if proveedor_fallback is not None else ProveedorHaversineAjustado()
        )

        self._registros = self._cargar_registros()
        self._huella_sha256 = self._calcular_huella()

    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.VIAL_CACHE

    @property
    def version(self) -> str:
        return f"vial-cache-csv-v1:{self.version_cache_esperada}:sha256={self._huella_sha256[:12]}"

    @property
    def estadisticas(self) -> EstadisticasCacheVial:
        """Retorna las estadísticas descriptivas y la huella criptográfica de la caché."""
        return EstadisticasCacheVial(
            cantidad_tramos=len(self._registros),
            version_cache=self.version_cache_esperada,
            precision_coordenadas=self.precision_coordenadas,
            huella_sha256=self._huella_sha256,
        )

    def contiene_tramo(self, origen: Coordenada, destino: Coordenada) -> bool:
        """Determina si un par origen-destino se encuentra indexado en la caché."""
        if self._misma_coordenada(origen, destino):
            return True
        return self._clave(origen, destino) in self._registros

    def calcular_tramo(
        self,
        origen: Coordenada,
        destino: Coordenada,
        configuracion: ConfiguracionPlanificacion,
    ) -> ResultadoTramoViaje:
        """Obtiene la distancia y tiempo del tramo desde la caché o vía proveedor fallback."""
        if self._misma_coordenada(origen, destino):
            return ResultadoTramoViaje(
                distancia_metros=0.0,
                tiempo_base_min=0.0,
                fuente=self.fuente,
            )

        clave = self._clave(origen, destino)
        registro = self._registros.get(clave)

        if registro is not None:
            tiempo_base_min = registro.tiempo_base_min
            if tiempo_base_min is None:
                tiempo_base_min = (
                    registro.distancia_metros / 1000.0 / configuracion.velocidad_base_kmh * 60.0
                )

            return ResultadoTramoViaje(
                distancia_metros=registro.distancia_metros,
                tiempo_base_min=tiempo_base_min,
                fuente=self.fuente,
            )

        if not self.permitir_fallback:
            raise KeyError(
                f"La caché vial no contiene el tramo dirigido "
                f"{self._formatear_coordenada(origen)} -> "
                f"{self._formatear_coordenada(destino)}."
            )

        resultado_fallback = self.proveedor_fallback.calcular_tramo(origen, destino, configuracion)
        return ResultadoTramoViaje(
            distancia_metros=resultado_fallback.distancia_metros,
            tiempo_base_min=resultado_fallback.tiempo_base_min,
            fuente=self.fuente,
            uso_fallback=True,
            advertencia=(
                "La caché vial no contiene todos los tramos; se utilizó el proveedor fallback."
            ),
        )

    def _cargar_registros(self) -> dict[str, RegistroTramoVial]:
        if not self.ruta_cache.is_file():
            raise FileNotFoundError(f"No existe la caché vial: {self.ruta_cache}")

        registros: dict[str, RegistroTramoVial] = {}

        with self.ruta_cache.open("r", encoding="utf-8-sig", newline="") as archivo:
            lector = csv.DictReader(archivo)
            if lector.fieldnames is None:
                raise ValueError("La caché vial no contiene encabezado CSV.")

            columnas_faltantes = [
                col for col in COLUMNAS_CACHE_VIAL if col not in lector.fieldnames
            ]
            if columnas_faltantes:
                raise ValueError(
                    "La caché vial no contiene las columnas requeridas: "
                    + ", ".join(columnas_faltantes)
                )

            for numero_fila, fila in enumerate(lector, start=2):
                if self._fila_vacia(fila):
                    continue

                registro = self._convertir_fila(fila, numero_fila)
                clave = self._clave(registro.origen, registro.destino)
                registro_anterior = registros.get(clave)

                if registro_anterior is not None and registro_anterior != registro:
                    raise ValueError(
                        f"La caché vial contiene registros conflictivos para el tramo {clave}."
                    )

                registros[clave] = registro

        return registros

    def _convertir_fila(
        self,
        fila: dict[str, str],
        numero_fila: int,
    ) -> RegistroTramoVial:
        version_cache = fila["version_cache"].strip()
        if version_cache != self.version_cache_esperada:
            raise ValueError(
                f"Versión de caché inesperada en la fila {numero_fila}: {version_cache!r}. "
                f"Se esperaba {self.version_cache_esperada!r}."
            )

        origen = (
            self._leer_float(fila, "lat_origen", numero_fila),
            self._leer_float(fila, "lon_origen", numero_fila),
        )
        destino = (
            self._leer_float(fila, "lat_destino", numero_fila),
            self._leer_float(fila, "lon_destino", numero_fila),
        )

        self._validar_coordenada(origen, "origen", numero_fila)
        self._validar_coordenada(destino, "destino", numero_fila)

        distancia_metros = self._leer_float(fila, "distancia_metros", numero_fila)
        if distancia_metros < 0.0:
            raise ValueError(f"distancia_metros debe ser >= 0 en la fila {numero_fila}.")

        texto_tiempo = fila["tiempo_base_min"].strip()
        if texto_tiempo:
            try:
                tiempo_base_min = float(texto_tiempo)
            except ValueError as exc:
                raise ValueError(
                    f"tiempo_base_min inválido en la fila {numero_fila}: {texto_tiempo!r}."
                ) from exc

            if not isfinite(tiempo_base_min) or tiempo_base_min < 0.0:
                raise ValueError(
                    f"tiempo_base_min debe ser finito y >= 0 en la fila {numero_fila}."
                )
        else:
            tiempo_base_min = None

        fuente_distancia = fila["fuente_distancia"].strip()
        fuente_tiempo = fila["fuente_tiempo"].strip()

        if not fuente_distancia:
            raise ValueError(f"fuente_distancia no puede estar vacía en la fila {numero_fila}.")
        if not fuente_tiempo:
            raise ValueError(f"fuente_tiempo no puede estar vacía en la fila {numero_fila}.")

        return RegistroTramoVial(
            origen=origen,
            destino=destino,
            distancia_metros=distancia_metros,
            tiempo_base_min=tiempo_base_min,
            fuente_distancia=fuente_distancia,
            fuente_tiempo=fuente_tiempo,
        )

    def _clave(self, origen: Coordenada, destino: Coordenada) -> str:
        return f"{self._formatear_coordenada(origen)}->{self._formatear_coordenada(destino)}"

    def _formatear_coordenada(self, coordenada: Coordenada) -> str:
        precision = self.precision_coordenadas
        return f"{coordenada[0]:.{precision}f},{coordenada[1]:.{precision}f}"

    def _misma_coordenada(self, origen: Coordenada, destino: Coordenada) -> bool:
        return self._formatear_coordenada(origen) == self._formatear_coordenada(destino)

    def _calcular_huella(self) -> str:
        hash_sha256 = hashlib.sha256()
        with self.ruta_cache.open("rb") as archivo:
            for bloque in iter(lambda: archivo.read(65_536), b""):
                hash_sha256.update(bloque)
        return hash_sha256.hexdigest()

    @staticmethod
    def _fila_vacia(fila: dict[str, str]) -> bool:
        return not any((val or "").strip() for val in fila.values())

    @staticmethod
    def _leer_float(fila: dict[str, str], columna: str, numero_fila: int) -> float:
        texto = fila[columna].strip()
        try:
            valor = float(texto)
        except ValueError as exc:
            raise ValueError(f"{columna} inválido en la fila {numero_fila}: {texto!r}.") from exc

        if not isfinite(valor):
            raise ValueError(f"{columna} debe ser finito en la fila {numero_fila}.")
        return valor

    @staticmethod
    def _validar_coordenada(coordenada: Coordenada, nombre: str, numero_fila: int) -> None:
        lat, lon = coordenada
        if lat < -90.0 or lat > 90.0:
            raise ValueError(f"Latitud de {nombre} inválida en la fila {numero_fila}: {lat}.")
        if lon < -180.0 or lon > 180.0:
            raise ValueError(f"Longitud de {nombre} inválida en la fila {numero_fila}: {lon}.")
