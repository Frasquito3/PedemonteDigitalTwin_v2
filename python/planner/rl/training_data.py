from __future__ import annotations

import csv
import hashlib
import math
import random

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Sequence


ESTADO_APTO_ENTRENAMIENTO: Final[str] = (
    "APTO_ENTRENAMIENTO"
)

SEED_DIVISION_DEMANDA_REAL: Final[int] = (
    15_2026
)

PROPORCION_ENTRENAMIENTO: Final[float] = (
    0.80
)

PROPORCION_VALIDACION: Final[float] = (
    0.10
)

PROPORCION_PRUEBA: Final[float] = (
    0.10
)

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


class ParticionDemandaReal(
    str,
    Enum,
):
    ENTRENAMIENTO = "TRAIN"
    VALIDACION = "VALIDATION"
    PRUEBA = "TEST"


@dataclass(frozen=True, slots=True)
class PuntoDemandaReal:
    """
    Ubicación histórica apta para generar demanda.

    Cada fila del dataset se conserva como una observación
    independiente. Si una misma dirección aparece varias
    veces, tendrá proporcionalmente más probabilidad de ser
    elegida mediante un muestreo uniforme de filas.
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
        registro_id = _texto_obligatorio(
            fila=fila,
            columna="registro_id",
            numero_linea=numero_linea,
        )

        calle = _texto_obligatorio(
            fila=fila,
            columna="calle",
            numero_linea=numero_linea,
        )

        ciudad = _texto_obligatorio(
            fila=fila,
            columna="ciudad",
            numero_linea=numero_linea,
        )

        latitud = _float_obligatorio(
            fila=fila,
            columna="latitud",
            numero_linea=numero_linea,
        )

        longitud = _float_obligatorio(
            fila=fila,
            columna="longitud",
            numero_linea=numero_linea,
        )

        distancia_corralon_km = _float_obligatorio(
            fila=fila,
            columna="distancia_corralon_km",
            numero_linea=numero_linea,
        )

        frecuencia = _int_obligatorio(
            fila=fila,
            columna="frecuencia_direccion_fuente",
            numero_linea=numero_linea,
        )

        if not -90.0 <= latitud <= 90.0:
            raise ValueError(
                "Latitud fuera de rango en la línea "
                f"{numero_linea}: {latitud}."
            )

        if not -180.0 <= longitud <= 180.0:
            raise ValueError(
                "Longitud fuera de rango en la línea "
                f"{numero_linea}: {longitud}."
            )

        if distancia_corralon_km < 0.0:
            raise ValueError(
                "La distancia al corralón no puede ser "
                f"negativa en la línea {numero_linea}."
            )

        if frecuencia <= 0:
            raise ValueError(
                "La frecuencia de la dirección debe ser "
                f"positiva en la línea {numero_linea}."
            )

        return cls(
            registro_id=registro_id,
            calle=calle,
            altura=_texto_opcional(
                fila,
                "altura",
            ),
            ciudad=ciudad,
            barrio=_texto_opcional(
                fila,
                "barrio",
                valor_por_defecto="No especificado",
            ),
            latitud=latitud,
            longitud=longitud,
            distancia_corralon_km=(
                distancia_corralon_km
            ),
            direccion_osm=_texto_opcional(
                fila,
                "direccion_osm",
            ),
            clave_direccion_fuente=(
                _texto_obligatorio(
                    fila=fila,
                    columna="clave_direccion_fuente",
                    numero_linea=numero_linea,
                )
            ),
            frecuencia_direccion_fuente=frecuencia,
        )


class CatalogoDemandaReal:
    """
    Catálogo inmutable de observaciones geográficas reales.
    """

    def __init__(
        self,
        registros: Sequence[PuntoDemandaReal],
        ruta_fuente: Path | None = None,
    ) -> None:
        if not registros:
            raise ValueError(
                "El catálogo de demanda real no puede "
                "estar vacío."
            )

        ids = [
            registro.registro_id
            for registro in registros
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "El dataset contiene registro_id duplicados."
            )

        self._registros = tuple(registros)
        self._ruta_fuente = ruta_fuente

    @classmethod
    def desde_csv(
        cls,
        ruta: Path,
        estado_aceptado: str = (
            ESTADO_APTO_ENTRENAMIENTO
        ),
    ) -> CatalogoDemandaReal:
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el dataset: {ruta}"
            )

        if not ruta.is_file():
            raise ValueError(
                f"La ruta no es un archivo: {ruta}"
            )

        registros: list[PuntoDemandaReal] = []

        with ruta.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as archivo:
            lector = csv.DictReader(archivo)

            if lector.fieldnames is None:
                raise ValueError(
                    "El CSV no contiene encabezado."
                )

            columnas_presentes = {
                columna.strip()
                for columna in lector.fieldnames
                if columna is not None
            }

            faltantes = (
                COLUMNAS_REQUERIDAS
                - columnas_presentes
            )

            if faltantes:
                faltantes_ordenadas = ", ".join(
                    sorted(faltantes)
                )

                raise ValueError(
                    "Faltan columnas requeridas: "
                    f"{faltantes_ordenadas}."
                )

            for numero_linea, fila in enumerate(
                lector,
                start=2,
            ):
                estado = _texto_opcional(
                    fila,
                    "estado_calidad",
                )

                if estado != estado_aceptado:
                    continue

                registros.append(
                    PuntoDemandaReal.desde_fila_csv(
                        fila=fila,
                        numero_linea=numero_linea,
                    )
                )

        if not registros:
            raise ValueError(
                "El CSV no contiene registros con estado "
                f"{estado_aceptado!r}."
            )

        return cls(
            registros=registros,
            ruta_fuente=ruta,
        )

    @property
    def registros(
        self,
    ) -> tuple[PuntoDemandaReal, ...]:
        return self._registros

    @property
    def ruta_fuente(self) -> Path | None:
        return self._ruta_fuente

    @property
    def claves_direccion_fuente(
        self,
    ) -> frozenset[str]:
        return frozenset(
            registro.clave_direccion_fuente
            for registro in self._registros
        )

    def __len__(self) -> int:
        return len(self._registros)

    def muestrear(
        self,
        cantidad: int,
        rng: random.Random,
        con_reemplazo: bool = False,
    ) -> list[PuntoDemandaReal]:
        """
        Selecciona ubicaciones utilizando el generador
        aleatorio recibido.

        Al muestrear filas y no direcciones únicas se
        conservan las frecuencias históricas observadas.
        """

        if cantidad < 0:
            raise ValueError(
                "cantidad no puede ser negativa."
            )

        if cantidad == 0:
            return []

        if (
            not con_reemplazo
            and cantidad > len(self._registros)
        ):
            raise ValueError(
                "No se pueden seleccionar "
                f"{cantidad} registros sin reemplazo "
                f"desde un catálogo de "
                f"{len(self._registros)}."
            )

        if con_reemplazo:
            return [
                rng.choice(self._registros)
                for _ in range(cantidad)
            ]

        return rng.sample(
            self._registros,
            k=cantidad,
        )

    def muestrear_con_seed(
        self,
        cantidad: int,
        seed: int,
        con_reemplazo: bool = False,
    ) -> list[PuntoDemandaReal]:
        rng = random.Random(seed)

        return self.muestrear(
            cantidad=cantidad,
            rng=rng,
            con_reemplazo=con_reemplazo,
        )

    def filtrar_por_claves_direccion_fuente(
        self,
        claves: Collection[str],
    ) -> CatalogoDemandaReal:
        claves_normalizadas = frozenset(
            clave.strip()
            for clave in claves
            if clave.strip()
        )

        if not claves_normalizadas:
            raise ValueError(
                "La selección de claves no puede "
                "estar vacía."
            )

        registros_filtrados = [
            registro
            for registro in self._registros
            if (
                registro.clave_direccion_fuente
                in claves_normalizadas
            )
        ]

        if not registros_filtrados:
            raise ValueError(
                "Ningún registro coincide con las "
                "claves solicitadas."
            )

        return CatalogoDemandaReal(
            registros=registros_filtrados,
            ruta_fuente=self._ruta_fuente,
        )

    def dividir_por_direccion_fuente(
        self,
        seed: int = SEED_DIVISION_DEMANDA_REAL,
        proporcion_entrenamiento: float = (
            PROPORCION_ENTRENAMIENTO
        ),
        proporcion_validacion: float = (
            PROPORCION_VALIDACION
        ),
        proporcion_prueba: float = (
            PROPORCION_PRUEBA
        ),
    ) -> DivisionDemandaReal:
        """
        Divide el catálogo agrupando por dirección fuente.

        Todas las filas que comparten
        ``clave_direccion_fuente`` quedan en una sola
        partición. La asignación es determinista, no depende
        del orden de las filas y se basa en SHA-256.
        """

        proporciones = (
            proporcion_entrenamiento,
            proporcion_validacion,
            proporcion_prueba,
        )

        _validar_proporciones_division(
            proporciones
        )

        claves = self.claves_direccion_fuente

        if len(claves) < 3:
            raise ValueError(
                "Se requieren al menos 3 direcciones "
                "fuente únicas para crear TRAIN, "
                "VALIDATION y TEST."
            )

        claves_ordenadas = sorted(
            claves,
            key=lambda clave: (
                _clave_orden_division(
                    clave=clave,
                    seed=seed,
                ),
                clave,
            ),
        )

        cantidades = (
            _calcular_cantidades_particiones(
                cantidad_total=len(
                    claves_ordenadas
                ),
                proporciones=proporciones,
            )
        )

        cantidad_entrenamiento = (
            cantidades[0]
        )

        cantidad_validacion = (
            cantidades[1]
        )

        fin_validacion = (
            cantidad_entrenamiento
            + cantidad_validacion
        )

        claves_entrenamiento = frozenset(
            claves_ordenadas[
                :cantidad_entrenamiento
            ]
        )

        claves_validacion = frozenset(
            claves_ordenadas[
                cantidad_entrenamiento:
                fin_validacion
            ]
        )

        claves_prueba = frozenset(
            claves_ordenadas[
                fin_validacion:
            ]
        )

        return DivisionDemandaReal(
            entrenamiento=(
                self
                .filtrar_por_claves_direccion_fuente(
                    claves_entrenamiento
                )
            ),
            validacion=(
                self
                .filtrar_por_claves_direccion_fuente(
                    claves_validacion
                )
            ),
            prueba=(
                self
                .filtrar_por_claves_direccion_fuente(
                    claves_prueba
                )
            ),
            seed=int(seed),
            proporcion_entrenamiento=(
                proporcion_entrenamiento
            ),
            proporcion_validacion=(
                proporcion_validacion
            ),
            proporcion_prueba=(
                proporcion_prueba
            ),
        )

    def conteo_por_ciudad(
        self,
    ) -> dict[str, int]:
        conteo = Counter(
            registro.ciudad
            for registro in self._registros
        )

        return dict(
            sorted(
                conteo.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        )

    def cantidad_direcciones_fuente_unicas(
        self,
    ) -> int:
        return len(
            self.claves_direccion_fuente
        )


@dataclass(frozen=True, slots=True)
class DivisionDemandaReal:
    """
    División reproducible y sin fuga por dirección fuente.
    """

    entrenamiento: CatalogoDemandaReal
    validacion: CatalogoDemandaReal
    prueba: CatalogoDemandaReal
    seed: int
    proporcion_entrenamiento: float
    proporcion_validacion: float
    proporcion_prueba: float

    def catalogo_para(
        self,
        particion: ParticionDemandaReal,
    ) -> CatalogoDemandaReal:
        if (
            particion
            == ParticionDemandaReal.ENTRENAMIENTO
        ):
            return self.entrenamiento

        if (
            particion
            == ParticionDemandaReal.VALIDACION
        ):
            return self.validacion

        if (
            particion
            == ParticionDemandaReal.PRUEBA
        ):
            return self.prueba

        raise ValueError(
            f"Partición no soportada: {particion!r}."
        )

    def resumen(
        self,
    ) -> dict[str, dict[str, int]]:
        return {
            particion.value: {
                "registros": len(
                    self.catalogo_para(
                        particion
                    )
                ),
                "direcciones_fuente_unicas": (
                    self
                    .catalogo_para(
                        particion
                    )
                    .cantidad_direcciones_fuente_unicas()
                ),
            }
            for particion in ParticionDemandaReal
        }

    def validar_sin_fuga(
        self,
    ) -> None:
        claves_entrenamiento = (
            self.entrenamiento
            .claves_direccion_fuente
        )

        claves_validacion = (
            self.validacion
            .claves_direccion_fuente
        )

        claves_prueba = (
            self.prueba
            .claves_direccion_fuente
        )

        if (
            claves_entrenamiento
            & claves_validacion
        ):
            raise RuntimeError(
                "Hay direcciones compartidas entre "
                "TRAIN y VALIDATION."
            )

        if (
            claves_entrenamiento
            & claves_prueba
        ):
            raise RuntimeError(
                "Hay direcciones compartidas entre "
                "TRAIN y TEST."
            )

        if (
            claves_validacion
            & claves_prueba
        ):
            raise RuntimeError(
                "Hay direcciones compartidas entre "
                "VALIDATION y TEST."
            )


def _clave_orden_division(
    clave: str,
    seed: int,
) -> bytes:
    contenido = (
        f"{int(seed)}\0{clave}"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        contenido
    ).digest()


def _validar_proporciones_division(
    proporciones: tuple[
        float,
        float,
        float,
    ],
) -> None:
    if any(
        not math.isfinite(
            proporcion
        )
        for proporcion in proporciones
    ):
        raise ValueError(
            "Las proporciones deben ser finitas."
        )

    if any(
        proporcion <= 0.0
        for proporcion in proporciones
    ):
        raise ValueError(
            "Las proporciones deben ser mayores "
            "que cero."
        )

    if not math.isclose(
        sum(proporciones),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Las proporciones de TRAIN, VALIDATION "
            "y TEST deben sumar 1.0."
        )


def _calcular_cantidades_particiones(
    cantidad_total: int,
    proporciones: tuple[
        float,
        float,
        float,
    ],
) -> tuple[int, int, int]:
    cantidades_reales = [
        cantidad_total
        * proporcion
        for proporcion in proporciones
    ]

    cantidades = [
        math.floor(
            cantidad
        )
        for cantidad in cantidades_reales
    ]

    faltantes = (
        cantidad_total
        - sum(cantidades)
    )

    orden_resto = sorted(
        range(
            len(proporciones)
        ),
        key=lambda indice: (
            -(
                cantidades_reales[indice]
                - cantidades[indice]
            ),
            indice,
        ),
    )

    for indice in orden_resto[
        :faltantes
    ]:
        cantidades[indice] += 1

    for indice, cantidad in enumerate(
        cantidades
    ):
        if cantidad > 0:
            continue

        donante = max(
            range(
                len(cantidades)
            ),
            key=lambda posicion: (
                cantidades[posicion],
                -posicion,
            ),
        )

        if cantidades[donante] <= 1:
            raise ValueError(
                "No hay suficientes direcciones "
                "para crear tres particiones."
            )

        cantidades[donante] -= 1
        cantidades[indice] += 1

    return (
        cantidades[0],
        cantidades[1],
        cantidades[2],
    )


def _texto_obligatorio(
    fila: dict[str, str],
    columna: str,
    numero_linea: int,
) -> str:
    valor = _texto_opcional(
        fila=fila,
        columna=columna,
    )

    if not valor:
        raise ValueError(
            f"Valor obligatorio vacío en la columna "
            f"{columna!r}, línea {numero_linea}."
        )

    return valor


def _texto_opcional(
    fila: dict[str, str],
    columna: str,
    valor_por_defecto: str = "",
) -> str:
    valor = fila.get(columna)

    if valor is None:
        return valor_por_defecto

    valor_limpio = valor.strip()

    if not valor_limpio:
        return valor_por_defecto

    return valor_limpio


def _float_obligatorio(
    fila: dict[str, str],
    columna: str,
    numero_linea: int,
) -> float:
    valor = _texto_obligatorio(
        fila=fila,
        columna=columna,
        numero_linea=numero_linea,
    )

    try:
        return float(valor)
    except ValueError as exc:
        raise ValueError(
            f"Valor numérico inválido en la columna "
            f"{columna!r}, línea {numero_linea}: "
            f"{valor!r}."
        ) from exc


def _int_obligatorio(
    fila: dict[str, str],
    columna: str,
    numero_linea: int,
) -> int:
    valor = _texto_obligatorio(
        fila=fila,
        columna=columna,
        numero_linea=numero_linea,
    )

    try:
        return int(valor)
    except ValueError as exc:
        raise ValueError(
            f"Valor entero inválido en la columna "
            f"{columna!r}, línea {numero_linea}: "
            f"{valor!r}."
        ) from exc