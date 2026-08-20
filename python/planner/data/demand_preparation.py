from __future__ import annotations

import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Final

LAT_CORRALON: Final[float] = -32.8495006
LON_CORRALON: Final[float] = -60.722653
DISTANCIA_MAX_ENTRENAMIENTO_KM: Final[float] = 30.0
VERSION_DATASET: Final[str] = "1.0.0"

COLUMNAS_ENTRADA: Final[tuple[str, ...]] = (
    "calle",
    "altura",
    "latitud",
    "longitud",
    "ciudad_detectada",
    "barrio",
    "direccion_osm",
)

COLUMNAS_SALIDA: Final[tuple[str, ...]] = (
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
    "motivo_revision",
)


def normalizar_texto(valor: str) -> str:
    texto = unicodedata.normalize(
        "NFKD",
        valor.strip().casefold(),
    )
    sin_acentos = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )
    return " ".join(sin_acentos.split())


def clave_direccion(
    calle: str,
    altura: str,
    ciudad: str,
) -> str:
    return "|".join(
        (
            normalizar_texto(calle),
            normalizar_texto(altura),
            normalizar_texto(ciudad),
        )
    )


def distancia_haversine_km(
    latitud_origen: float,
    longitud_origen: float,
    latitud_destino: float,
    longitud_destino: float,
) -> float:
    radio_tierra_km = 6371.0088

    latitud_origen_rad = math.radians(
        latitud_origen
    )
    latitud_destino_rad = math.radians(
        latitud_destino
    )
    diferencia_latitud = math.radians(
        latitud_destino - latitud_origen
    )
    diferencia_longitud = math.radians(
        longitud_destino - longitud_origen
    )

    termino = (
        math.sin(diferencia_latitud / 2.0) ** 2
        + math.cos(latitud_origen_rad)
        * math.cos(latitud_destino_rad)
        * math.sin(diferencia_longitud / 2.0) ** 2
    )

    return 2.0 * radio_tierra_km * math.asin(
        math.sqrt(termino)
    )


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()

    with ruta.open("rb") as archivo:
        while True:
            bloque = archivo.read(1024 * 1024)
            if not bloque:
                break
            digest.update(bloque)

    return digest.hexdigest()


def leer_registros(ruta_entrada: Path) -> list[dict[str, str]]:
    with ruta_entrada.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as archivo:
        lector = csv.DictReader(archivo)

        columnas = tuple(lector.fieldnames or ())
        faltantes = [
            columna
            for columna in COLUMNAS_ENTRADA
            if columna not in columnas
        ]

        if faltantes:
            raise ValueError(
                "Faltan columnas obligatorias: "
                + ", ".join(faltantes)
            )

        return [dict(fila) for fila in lector]


def percentil(
    valores_ordenados: list[float],
    proporcion: float,
) -> float:
    if not valores_ordenados:
        return 0.0

    posicion = (len(valores_ordenados) - 1) * proporcion
    inferior = math.floor(posicion)
    superior = math.ceil(posicion)

    if inferior == superior:
        return valores_ordenados[inferior]

    peso_superior = posicion - inferior
    return (
        valores_ordenados[inferior]
        * (1.0 - peso_superior)
        + valores_ordenados[superior]
        * peso_superior
    )


def preparar_dataset(
    ruta_entrada: Path,
    ruta_dataset: Path,
    ruta_revision: Path,
    ruta_metadata: Path,
    distancia_max_entrenamiento_km: float = (
        DISTANCIA_MAX_ENTRENAMIENTO_KM
    ),
) -> dict[str, object]:
    if distancia_max_entrenamiento_km <= 0.0:
        raise ValueError(
            "distancia_max_entrenamiento_km debe ser > 0."
        )

    registros_crudos = leer_registros(ruta_entrada)

    if not registros_crudos:
        raise ValueError(
            "El archivo de entrada no contiene registros."
        )

    claves = [
        clave_direccion(
            registro["calle"],
            registro["altura"],
            registro["ciudad_detectada"],
        )
        for registro in registros_crudos
    ]
    frecuencias = Counter(claves)

    registros_procesados: list[dict[str, str]] = []
    registros_revision: list[dict[str, str]] = []

    for indice, (registro, clave) in enumerate(
        zip(registros_crudos, claves, strict=True),
        start=1,
    ):
        try:
            latitud = float(registro["latitud"])
            longitud = float(registro["longitud"])
        except ValueError as exc:
            raise ValueError(
                f"Coordenadas no numéricas en fila {indice + 1}."
            ) from exc

        if not (-90.0 <= latitud <= 90.0):
            raise ValueError(
                f"Latitud inválida en fila {indice + 1}: {latitud}."
            )

        if not (-180.0 <= longitud <= 180.0):
            raise ValueError(
                f"Longitud inválida en fila {indice + 1}: {longitud}."
            )

        distancia_km = distancia_haversine_km(
            LAT_CORRALON,
            LON_CORRALON,
            latitud,
            longitud,
        )

        if distancia_km <= distancia_max_entrenamiento_km:
            estado_calidad = "APTO_ENTRENAMIENTO"
            motivo_revision = ""
        else:
            estado_calidad = "REVISAR_DISTANCIA"
            motivo_revision = (
                "Distancia al corralón superior a "
                f"{distancia_max_entrenamiento_km:.1f} km."
            )

        procesado = {
            "registro_id": f"DG-{indice:04d}",
            "calle": registro["calle"].strip(),
            "altura": registro["altura"].strip(),
            "ciudad": registro["ciudad_detectada"].strip(),
            "barrio": registro["barrio"].strip(),
            "latitud": f"{latitud:.7f}",
            "longitud": f"{longitud:.7f}",
            "distancia_corralon_km": f"{distancia_km:.6f}",
            "direccion_osm": registro["direccion_osm"].strip(),
            "clave_direccion_fuente": clave,
            "frecuencia_direccion_fuente": str(
                frecuencias[clave]
            ),
            "estado_calidad": estado_calidad,
            "motivo_revision": motivo_revision,
        }

        registros_procesados.append(procesado)

        if estado_calidad != "APTO_ENTRENAMIENTO":
            registros_revision.append(procesado)

    ruta_dataset.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ruta_revision.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ruta_metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ruta_dataset.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=COLUMNAS_SALIDA,
        )
        escritor.writeheader()
        escritor.writerows(registros_procesados)

    with ruta_revision.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=COLUMNAS_SALIDA,
        )
        escritor.writeheader()
        escritor.writerows(registros_revision)

    aptos = [
        registro
        for registro in registros_procesados
        if registro["estado_calidad"]
        == "APTO_ENTRENAMIENTO"
    ]

    ciudades_todas = Counter(
        registro["ciudad"]
        for registro in registros_procesados
    )
    ciudades_aptas = Counter(
        registro["ciudad"]
        for registro in aptos
    )

    distancias = sorted(
        float(registro["distancia_corralon_km"])
        for registro in registros_procesados
    )

    metadata: dict[str, object] = {
        "dataset": "demanda_geografica_real_pedemonte",
        "version": VERSION_DATASET,
        "archivo_fuente": ruta_entrada.name,
        "sha256_fuente": sha256_archivo(ruta_entrada),
        "corralon": {
            "latitud": LAT_CORRALON,
            "longitud": LON_CORRALON,
        },
        "criterio_calidad": {
            "distancia_max_entrenamiento_km": (
                distancia_max_entrenamiento_km
            ),
            "registros_lejanos": (
                "Se conservan en el dataset, pero quedan "
                "marcados para revisión y no deben usarse "
                "en entrenamiento hasta ser validados."
            ),
        },
        "conteos": {
            "registros_totales": len(registros_procesados),
            "registros_aptos_entrenamiento": len(aptos),
            "registros_revision": len(registros_revision),
            "direcciones_fuente_unicas": len(frecuencias),
        },
        "distancias_km": {
            "minimo": round(min(distancias), 6),
            "mediana": round(percentil(distancias, 0.50), 6),
            "p90": round(percentil(distancias, 0.90), 6),
            "p95": round(percentil(distancias, 0.95), 6),
            "p99": round(percentil(distancias, 0.99), 6),
            "maximo": round(max(distancias), 6),
        },
        "ciudades_todos_los_registros": dict(
            sorted(ciudades_todas.items())
        ),
        "ciudades_aptas_entrenamiento": dict(
            sorted(ciudades_aptas.items())
        ),
        "regla_frecuencia": (
            "Cada fila representa una aparición en la fuente. "
            "Las repeticiones se conservan para mantener la "
            "frecuencia empírica de demanda."
        ),
    }

    ruta_metadata.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return metadata