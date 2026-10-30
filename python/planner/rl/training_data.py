"""Particionamiento y gestión de datos de demanda real para RL.

Este módulo provee estructuras de datos y utilidades para cargar, validar y
particionar registros de demanda geográfica histórica en conjuntos disjuntos
de entrenamiento, validación y prueba (TRAIN, VALIDATION, TEST), garantizando
reproducibilidad y ausencia de fuga de información por dirección física.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

__all__ = [
    "COLUMNAS_REQUERIDAS",
    "ESTADO_APTO_ENTRENAMIENTO",
    "PROPORCION_ENTRENAMIENTO",
    "PROPORCION_PRUEBA",
    "PROPORCION_VALIDACION",
    "SEED_DIVISION_DEMANDA_REAL",
    "CatalogoDemandaReal",
    "DivisionDemandaReal",
    "ParticionDemandaReal",
    "PuntoDemandaReal",
]

ESTADO_APTO_ENTRENAMIENTO: Final[str] = "APTO_ENTRENAMIENTO"
SEED_DIVISION_DEMANDA_REAL: Final[int] = 15_2026
PROPORCION_ENTRENAMIENTO: Final[float] = 0.80
PROPORCION_VALIDACION: Final[float] = 0.10
PROPORCION_PRUEBA: Final[float] = 0.10

COLUMNAS_REQUERIDAS: Final[frozenset[str]] = frozenset(
    {
        "registro_id",
        "calle",
        "altura",
        "ciudad",
        "barrio",
        "latitud",
        "longitud",
        "distancia_corralon_km",
        "direccion_osm",
        "clave_direccion_fuente",
        "frecuencia_direccion_fuente",
        "estado_calidad",
    }
)


class ParticionDemandaReal(str, Enum):
    """Identificador de subconjunto de datos particionados."""

    ENTRENAMIENTO = "TRAIN"
    VALIDACION = "VALIDATION"
    PRUEBA = "TEST"


@dataclass(frozen=True, slots=True)
class PuntoDemandaReal:
    """Ubicación histórica apta para generar demanda sintética o realista.

    Cada fila del dataset se conserva como una observación independiente.
    Si una misma dirección aparece varias veces, tendrá proporcionalmente
    más probabilidad de ser seleccionada mediante un muestreo uniforme de filas.
    """

    registro_id: str
    calle: str
    altura: str
    ciudad: str
    barrio: str
    latitud: float
    longitud: float
    distancia_corralon_km: float
    direccion_osm: str
    clave_direccion_fuente: str
    frecuencia_direccion_fuente: int

    @property
    def direccion_corta(self) -> str:
        """Retorna una representación abreviada de calle, altura y ciudad."""
        calle_altura = self.calle
        if self.altura:
            calle_altura += f" {self.altura}"
        return f"{calle_altura}, {self.ciudad}"

    @classmethod
    def desde_fila_csv(
        cls,
        fila: dict[str, str],
        numero_linea: int,
    ) -> PuntoDemandaReal:
        """Construye e higieniza una instancia a partir de una fila de un archivo CSV."""
        registro_id = _texto_obligatorio(fila, "registro_id", numero_linea)
        calle = _texto_obligatorio(fila, "calle", numero_linea)
        ciudad = _texto_obligatorio(fila, "ciudad", numero_linea)
        latitud = _float_obligatorio(fila, "latitud", numero_linea)
        longitud = _float_obligatorio(fila, "longitud", numero_linea)
        distancia_corralon_km = _float_obligatorio(fila, "distancia_corralon_km", numero_linea)
        frecuencia = _int_obligatorio(fila, "frecuencia_direccion_fuente", numero_linea)

        if not -90.0 <= latitud <= 90.0:
            raise ValueError(f"Latitud fuera de rango en la línea {numero_linea}: {latitud}.")
        if not -180.0 <= longitud <= 180.0:
            raise ValueError(f"Longitud fuera de rango en la línea {numero_linea}: {longitud}.")
        if distancia_corralon_km < 0.0:
            raise ValueError(
                f"La distancia al corralón no puede ser negativa en la línea {numero_linea}."
            )
        if frecuencia <= 0:
            raise ValueError(
                f"La frecuencia de la dirección debe ser positiva en la línea {numero_linea}."
            )

        return cls(
            registro_id=registro_id,
            calle=calle,
            altura=_texto_opcional(fila, "altura"),
            ciudad=ciudad,
            barrio=_texto_opcional(fila, "barrio", valor_por_defecto="No especificado"),
            latitud=latitud,
            longitud=longitud,
            distancia_corralon_km=distancia_corralon_km,
            direccion_osm=_texto_opcional(fila, "direccion_osm"),
            clave_direccion_fuente=_texto_obligatorio(fila, "clave_direccion_fuente", numero_linea),
            frecuencia_direccion_fuente=frecuencia,
        )


class CatalogoDemandaReal:
    """Colección inmutable de observaciones geográficas reales auditadas."""

    def __init__(
        self,
        registros: Sequence[PuntoDemandaReal],
        ruta_fuente: Path | None = None,
    ) -> None:
        if not registros:
            raise ValueError("El catálogo de demanda real no puede estar vacío.")

        ids = [registro.registro_id for registro in registros]
        if len(ids) != len(set(ids)):
            raise ValueError("El dataset contiene registro_id duplicados.")

        self._registros = tuple(registros)
        self._ruta_fuente = ruta_fuente

    @classmethod
    def desde_csv(
        cls,
        ruta: Path,
        estado_aceptado: str = ESTADO_APTO_ENTRENAMIENTO,
    ) -> CatalogoDemandaReal:
        """Carga y valida los registros aptos para entrenamiento desde un CSV."""
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el dataset: {ruta}")
        if not ruta.is_file():
            raise ValueError(f"La ruta no es un archivo: {ruta}")

        registros: list[PuntoDemandaReal] = []
        with ruta.open(mode="r", encoding="utf-8-sig", newline="") as archivo:
            lector = csv.DictReader(archivo)
            if lector.fieldnames is None:
                raise ValueError("El CSV no contiene encabezado.")

            columnas_presentes = {col.strip() for col in lector.fieldnames if col is not None}
            faltantes = COLUMNAS_REQUERIDAS - columnas_presentes
            if faltantes:
                faltantes_ordenadas = ", ".join(sorted(faltantes))
                raise ValueError(f"Faltan columnas requeridas: {faltantes_ordenadas}.")

            for numero_linea, fila in enumerate(lector, start=2):
                estado = _texto_opcional(fila, "estado_calidad")
                if estado != estado_aceptado:
                    continue
                registros.append(
                    PuntoDemandaReal.desde_fila_csv(fila=fila, numero_linea=numero_linea)
                )

        if not registros:
            raise ValueError(f"El CSV no contiene registros con estado {estado_aceptado!r}.")

        return cls(registros=registros, ruta_fuente=ruta)

    @property
    def registros(self) -> tuple[PuntoDemandaReal, ...]:
        """Tupla inmutable con todos los registros del catálogo."""
        return self._registros

    @property
    def ruta_fuente(self) -> Path | None:
        """Ruta al archivo CSV de origen, si fue cargado desde disco."""
        return self._ruta_fuente

    @property
    def claves_direccion_fuente(self) -> frozenset[str]:
        """Conjunto inmutable de claves únicas de dirección física."""
        return frozenset(registro.clave_direccion_fuente for registro in self._registros)

    def __len__(self) -> int:
        return len(self._registros)

    def muestrear(
        self,
        cantidad: int,
        rng: random.Random,
        con_reemplazo: bool = False,
    ) -> list[PuntoDemandaReal]:
        """Selecciona ubicaciones utilizando el generador aleatorio recibido.

        Al muestrear filas y no direcciones únicas se conservan las frecuencias
        históricas observadas en la operación real.
        """
        if cantidad < 0:
            raise ValueError("cantidad no puede ser negativa.")
        if cantidad == 0:
            return []
        if not con_reemplazo and cantidad > len(self._registros):
            raise ValueError(
                f"No se pueden seleccionar {cantidad} registros sin reemplazo "
                f"desde un catálogo de {len(self._registros)}."
            )
        if con_reemplazo:
            return [rng.choice(self._registros) for _ in range(cantidad)]
        return rng.sample(self._registros, k=cantidad)

    def muestrear_con_seed(
        self,
        cantidad: int,
        seed: int,
        con_reemplazo: bool = False,
    ) -> list[PuntoDemandaReal]:
        """Muestrea registros utilizando una semilla fija reproducible."""
        rng = random.Random(seed)
        return self.muestrear(cantidad=cantidad, rng=rng, con_reemplazo=con_reemplazo)

    def filtrar_por_claves_direccion_fuente(
        self,
        claves: Collection[str],
    ) -> CatalogoDemandaReal:
        """Retorna un nuevo sub-catálogo con los registros cuyas claves coincidan."""
        claves_normalizadas = frozenset(clave.strip() for clave in claves if clave.strip())
        if not claves_normalizadas:
            raise ValueError("La selección de claves no puede estar vacía.")

        registros_filtrados = [
            registro
            for registro in self._registros
            if registro.clave_direccion_fuente in claves_normalizadas
        ]
        if not registros_filtrados:
            raise ValueError("Ningún registro coincide con las claves solicitadas.")

        return CatalogoDemandaReal(registros=registros_filtrados, ruta_fuente=self._ruta_fuente)

    def dividir_por_direccion_fuente(
        self,
        seed: int = SEED_DIVISION_DEMANDA_REAL,
        proporcion_entrenamiento: float = PROPORCION_ENTRENAMIENTO,
        proporcion_validacion: float = PROPORCION_VALIDACION,
        proporcion_prueba: float = PROPORCION_PRUEBA,
    ) -> DivisionDemandaReal:
        """Divide el catálogo agrupando por dirección fuente.

        Todas las filas que comparten ``clave_direccion_fuente`` quedan en una sola
        partición. La asignación es determinista, no depende del orden de las filas
        y se basa en SHA-256 para evitar fugas entre particiones.
        """
        proporciones = (proporcion_entrenamiento, proporcion_validacion, proporcion_prueba)
        _validar_proporciones_division(proporciones)

        claves = self.claves_direccion_fuente
        if len(claves) < 3:
            raise ValueError(
                "Se requieren al menos 3 direcciones fuente únicas para crear TRAIN, VALIDATION y TEST."
            )

        claves_ordenadas = sorted(
            claves,
            key=lambda clave: (_clave_orden_division(clave=clave, seed=seed), clave),
        )
        cantidades = _calcular_cantidades_particiones(
            cantidad_total=len(claves_ordenadas),
            proporciones=proporciones,
        )

        cantidad_entrenamiento = cantidades[0]
        cantidad_validacion = cantidades[1]
        fin_validacion = cantidad_entrenamiento + cantidad_validacion

        claves_entrenamiento = frozenset(claves_ordenadas[:cantidad_entrenamiento])
        claves_validacion = frozenset(claves_ordenadas[cantidad_entrenamiento:fin_validacion])
        claves_prueba = frozenset(claves_ordenadas[fin_validacion:])

        return DivisionDemandaReal(
            entrenamiento=self.filtrar_por_claves_direccion_fuente(claves_entrenamiento),
            validacion=self.filtrar_por_claves_direccion_fuente(claves_validacion),
            prueba=self.filtrar_por_claves_direccion_fuente(claves_prueba),
            seed=int(seed),
            proporcion_entrenamiento=proporcion_entrenamiento,
            proporcion_validacion=proporcion_validacion,
            proporcion_prueba=proporcion_prueba,
        )

    def conteo_por_ciudad(self) -> dict[str, int]:
        """Calcula el número de registros agrupados por ciudad en orden descendente."""
        conteo = Counter(registro.ciudad for registro in self._registros)
        return dict(sorted(conteo.items(), key=lambda item: (-item[1], item[0])))

    def cantidad_direcciones_fuente_unicas(self) -> int:
        """Retorna la cantidad de ubicaciones físicas únicas registradas."""
        return len(self.claves_direccion_fuente)


@dataclass(frozen=True, slots=True)
class DivisionDemandaReal:
    """División reproducible y sin fuga por dirección fuente."""

    entrenamiento: CatalogoDemandaReal
    validacion: CatalogoDemandaReal
    prueba: CatalogoDemandaReal
    seed: int
    proporcion_entrenamiento: float
    proporcion_validacion: float
    proporcion_prueba: float

    def catalogo_para(self, particion: ParticionDemandaReal) -> CatalogoDemandaReal:
        """Obtiene el catálogo correspondiente a la partición solicitada."""
        if particion == ParticionDemandaReal.ENTRENAMIENTO:
            return self.entrenamiento
        if particion == ParticionDemandaReal.VALIDACION:
            return self.validacion
        if particion == ParticionDemandaReal.PRUEBA:
            return self.prueba
        raise ValueError(f"Partición no soportada: {particion!r}.")

    def resumen(self) -> dict[str, dict[str, int]]:
        """Genera un diccionario con métricas de tamaño y direcciones únicas por split."""
        return {
            particion.value: {
                "registros": len(self.catalogo_para(particion)),
                "direcciones_fuente_unicas": self.catalogo_para(
                    particion
                ).cantidad_direcciones_fuente_unicas(),
            }
            for particion in ParticionDemandaReal
        }

    def validar_sin_fuga(self) -> None:
        """Verifica formalmente la ausencia de intersección entre las particiones."""
        claves_entrenamiento = self.entrenamiento.claves_direccion_fuente
        claves_validacion = self.validacion.claves_direccion_fuente
        claves_prueba = self.prueba.claves_direccion_fuente

        if claves_entrenamiento & claves_validacion:
            raise RuntimeError("Hay direcciones compartidas entre TRAIN y VALIDATION.")
        if claves_entrenamiento & claves_prueba:
            raise RuntimeError("Hay direcciones compartidas entre TRAIN y TEST.")
        if claves_validacion & claves_prueba:
            raise RuntimeError("Hay direcciones compartidas entre VALIDATION y TEST.")


def _clave_orden_division(clave: str, seed: int) -> bytes:
    contenido = f"{int(seed)}\0{clave}".encode()
    return hashlib.sha256(contenido).digest()


def _validar_proporciones_division(proporciones: tuple[float, float, float]) -> None:
    if any(not math.isfinite(p) for p in proporciones):
        raise ValueError("Las proporciones deben ser finitas.")
    if any(p <= 0.0 for p in proporciones):
        raise ValueError("Las proporciones deben ser mayores que cero.")
    if not math.isclose(sum(proporciones), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Las proporciones de TRAIN, VALIDATION y TEST deben sumar 1.0.")


def _calcular_cantidades_particiones(
    cantidad_total: int,
    proporciones: tuple[float, float, float],
) -> tuple[int, int, int]:
    cantidades_reales = [cantidad_total * p for p in proporciones]
    cantidades = [math.floor(c) for c in cantidades_reales]
    faltantes = cantidad_total - sum(cantidades)

    orden_resto = sorted(
        range(len(proporciones)),
        key=lambda idx: (-(cantidades_reales[idx] - cantidades[idx]), idx),
    )
    for idx in orden_resto[:faltantes]:
        cantidades[idx] += 1

    for idx, cantidad in enumerate(cantidades):
        if cantidad > 0:
            continue
        donante = max(range(len(cantidades)), key=lambda pos: (cantidades[pos], -pos))
        if cantidades[donante] <= 1:
            raise ValueError("No hay suficientes direcciones para crear tres particiones.")
        cantidades[donante] -= 1
        cantidades[idx] += 1

    return (cantidades[0], cantidades[1], cantidades[2])


def _texto_obligatorio(fila: dict[str, str], columna: str, numero_linea: int) -> str:
    valor = _texto_opcional(fila=fila, columna=columna)
    if not valor:
        raise ValueError(
            f"Valor obligatorio vacío en la columna {columna!r}, línea {numero_linea}."
        )
    return valor


def _texto_opcional(fila: dict[str, str], columna: str, valor_por_defecto: str = "") -> str:
    valor = fila.get(columna)
    if valor is None:
        return valor_por_defecto
    valor_limpio = valor.strip()
    return valor_limpio if valor_limpio else valor_por_defecto


def _float_obligatorio(fila: dict[str, str], columna: str, numero_linea: int) -> float:
    valor = _texto_obligatorio(fila=fila, columna=columna, numero_linea=numero_linea)
    try:
        return float(valor)
    except ValueError as exc:
        raise ValueError(
            f"Valor numérico inválido en la columna {columna!r}, línea {numero_linea}: {valor!r}."
        ) from exc


def _int_obligatorio(fila: dict[str, str], columna: str, numero_linea: int) -> int:
    valor = _texto_obligatorio(fila=fila, columna=columna, numero_linea=numero_linea)
    try:
        return int(valor)
    except ValueError as exc:
        raise ValueError(
            f"Valor entero inválido en la columna {columna!r}, línea {numero_linea}: {valor!r}."
        ) from exc
