"""Validaciones de cobertura vial, costos y tiempos experimentales."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from planner.core.schema import InstanciaTurno
from planner.routing.vial_cache import ProveedorVialCachePersistente

COMPONENTES_COSTO = (
    "costo_distancia",
    "costo_operacion",
    "costo_viajes",
    "costo_tardanza",
    "costo_desbalance",
    "costo_exceso_tolerancia",
    "costo_tareas_no_entregadas",
    "costo_pedidos_originales_incompletos",
)

COLUMNAS_DESGLOSE = [
    "dia_numero",
    "repeticion",
    "seed_ejecucion",
    "metodo",
    "algoritmo_aplicado",
    "costo_total",
    *COMPONENTES_COSTO,
    "suma_componentes",
    "diferencia_costos",
    "estado_desglose",
    "tareas_entregadas",
    "tareas_no_entregadas",
]


def leer_tiempo_generacion(carpeta: Path) -> float | None:
    """Recupera la medicion real del generador cuando esta disponible."""
    ruta = carpeta / "metadata_generacion.json"
    if not ruta.is_file():
        return None

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    valor = datos.get("tiempo_generacion_s")

    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None

    numero = float(valor)
    return numero if math.isfinite(numero) and numero >= 0 else None


def verificar_cobertura_vial(
    instancias: list[InstanciaTurno],
    ruta_cache: Path,
) -> dict[str, Any]:
    """Comprueba todos los pares dirigidos antes de iniciar motores Java."""
    proveedor = ProveedorVialCachePersistente(
        ruta_cache,
        version_cache_esperada="pedemonte-vial-v1",
        permitir_fallback=False,
    )

    requeridos = 0
    faltantes = 0
    muestras: list[dict[str, Any]] = []
    escenarios: list[dict[str, Any]] = []

    for instancia in instancias:
        puntos = list(
            dict.fromkeys(
                [
                    (
                        round(instancia.lat_corralon, 6),
                        round(instancia.lon_corralon, 6),
                    ),
                    *[(round(p.latitud, 6), round(p.longitud, 6)) for p in instancia.pedidos],
                ]
            )
        )

        requeridos_escenario = len(puntos) * (len(puntos) - 1)
        faltantes_escenario = 0
        requeridos += requeridos_escenario

        for origen in puntos:
            for destino in puntos:
                if origen == destino:
                    continue

                if proveedor.contiene_tramo(origen, destino):
                    continue

                faltantes += 1
                faltantes_escenario += 1

                if len(muestras) < 30:
                    muestras.append(
                        {
                            "instancia_id": instancia.instancia_id,
                            "origen": list(origen),
                            "destino": list(destino),
                        }
                    )

        escenarios.append(
            {
                "instancia_id": instancia.instancia_id,
                "puntos": len(puntos),
                "requeridos": requeridos_escenario,
                "faltantes": faltantes_escenario,
            }
        )

    return {
        "estado": "LISTA" if faltantes == 0 else "INCOMPLETA",
        "tramos_requeridos": requeridos,
        "faltantes": faltantes,
        "muestras_faltantes": muestras,
        "escenarios": escenarios,
        "cache_sha256": proveedor.estadisticas.huella_sha256,
    }


def leer_tiempo_preparacion_gis(
    carpeta: Path,
    escenarios_esperados: int,
) -> float | None:
    """Suma las duraciones GIS, incluidos los reintentos."""
    ruta = carpeta / "tiempos_preparacion_gis.csv"

    if not ruta.is_file():
        return None

    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        requeridas = {"numero_turno", "preparacion_s", "estado"}

        if lector.fieldnames is None or not requeridas.issubset(lector.fieldnames):
            raise ValueError("El CSV GIS no contiene las columnas requeridas.")

        filas = list(lector)

    if not filas:
        raise ValueError("El CSV de tiempos GIS esta vacio.")

    total = 0.0
    completados: set[int] = set()

    for fila in filas:
        numero_turno = int(fila["numero_turno"])
        segundos = float(fila["preparacion_s"])
        estado = str(fila["estado"])

        if (
            not math.isfinite(segundos)
            or segundos < 0
            or numero_turno < 1
            or numero_turno > escenarios_esperados
            or estado not in ("OK", "ERROR")
        ):
            raise ValueError("Registro de tiempo GIS invalido.")

        total += segundos

        if estado == "OK":
            if numero_turno in completados:
                raise ValueError("Escenario GIS completado mas de una vez.")
            completados.add(numero_turno)

    esperados = set(range(1, escenarios_esperados + 1))

    if completados != esperados:
        raise ValueError("No hay mediciones GIS completas para todos los escenarios.")

    return total


def _numero(valor: Any) -> float | None:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    numero = float(valor)
    return numero if math.isfinite(numero) and numero >= 0 else None


def auditar_componentes_costos(
    carpeta_experimento: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reconcilia el costo total simulado con sus componentes originales."""
    registros: list[dict[str, Any]] = []
    inconsistencias: list[str] = []
    sin_desglose = 0

    for archivo in sorted((carpeta_experimento / "corridas").glob("*.json")):
        try:
            payload = json.loads(archivo.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            inconsistencias.append(f"{archivo.name}: {exc}")
            continue

        for item in payload.get("resultados", []):
            if item.get("estado") != "FINALIZADO":
                continue

            resultado = item.get("resultado")
            if not isinstance(resultado, dict):
                inconsistencias.append(f"{archivo.name}: resultado FINALIZADO sin datos")
                continue

            metodo = str(item.get("metodo_solicitado", ""))
            costo_total = _numero(resultado.get("costo_total"))
            valores = {nombre: _numero(resultado.get(nombre)) for nombre in COMPONENTES_COSTO}

            fila: dict[str, Any] = {
                "dia_numero": payload.get("dia_numero"),
                "repeticion": payload.get("repeticion"),
                "seed_ejecucion": payload.get("seed_ejecucion"),
                "metodo": metodo,
                "algoritmo_aplicado": resultado.get("algoritmo_aplicado"),
                "costo_total": costo_total,
                "tareas_entregadas": resultado.get("tareas_entregadas"),
                "tareas_no_entregadas": resultado.get("tareas_no_entregadas"),
            }

            fila.update(valores)

            if costo_total is None or any(v is None for v in valores.values()):
                sin_desglose += 1
                fila.update(
                    suma_componentes="",
                    diferencia_costos="",
                    estado_desglose="INCOMPLETO",
                )
            else:
                suma = sum(v for v in valores.values() if v is not None)
                diferencia = costo_total - suma
                tolerancia = max(1e-6, abs(costo_total) * 1e-7)
                coherente = abs(diferencia) <= tolerancia

                fila.update(
                    suma_componentes=suma,
                    diferencia_costos=diferencia,
                    estado_desglose="COHERENTE" if coherente else "INCONSISTENTE",
                )

                if not coherente:
                    inconsistencias.append(f"{archivo.name}/{metodo}: diferencia={diferencia:.6f}")

            registros.append(fila)

    resumen = {
        "resultados_analizados": len(registros),
        "desgloses_incompletos": sin_desglose,
        "inconsistentes": len(inconsistencias),
        "errores": inconsistencias,
    }
    return resumen, registros
