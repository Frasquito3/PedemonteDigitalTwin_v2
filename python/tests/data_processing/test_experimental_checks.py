"""Controles de preflight vial y conciliacion de costos."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.simulation.experimental_checks import (
    auditar_componentes_costos,
    leer_tiempo_generacion,
    leer_tiempo_preparacion_gis,
    verificar_cobertura_vial,
)


def test_preflight_detecta_tramo_dirigido_faltante(tmp_path: Path) -> None:
    cache = tmp_path / "cache_vial.csv"
    cache.write_text(
        "version_cache,lat_origen,lon_origen,lat_destino,lon_destino,"
        "distancia_metros,tiempo_base_min,fuente_distancia,fuente_tiempo\n"
        "pedemonte-vial-v1,0,0,0.1,0.1,1000,2,TEST,TEST\n",
        encoding="utf-8",
    )

    instancia: Any = SimpleNamespace(
        instancia_id="PRUEBA",
        lat_corralon=0.0,
        lon_corralon=0.0,
        pedidos=[SimpleNamespace(latitud=0.1, longitud=0.1)],
    )

    reporte = verificar_cobertura_vial([instancia], cache)
    assert reporte["tramos_requeridos"] == 2
    assert reporte["faltantes"] == 1

    with cache.open("a", encoding="utf-8") as archivo:
        archivo.write("pedemonte-vial-v1,0.1,0.1,0,0,1000,2,TEST,TEST\n")

    reporte = verificar_cobertura_vial([instancia], cache)
    assert reporte["faltantes"] == 0
    assert reporte["estado"] == "LISTA"


def test_desglose_detecta_penalizaciones_y_diferencias(tmp_path: Path) -> None:
    carpeta = tmp_path / "corridas"
    carpeta.mkdir()

    componentes = {
        "costo_distancia": 100.0,
        "costo_operacion": 150.0,
        "costo_viajes": 30.0,
        "costo_tardanza": 0.0,
        "costo_desbalance": 20.0,
        "costo_exceso_tolerancia": 4000.0,
        "costo_tareas_no_entregadas": 0.0,
        "costo_pedidos_originales_incompletos": 0.0,
    }

    payload = {
        "dia_numero": 1,
        "repeticion": 1,
        "seed_ejecucion": 123,
        "resultados": [
            {
                "metodo_solicitado": "GREEDY",
                "estado": "FINALIZADO",
                "resultado": {
                    "algoritmo_aplicado": "GREEDY",
                    "costo_total": 4300.0,
                    **componentes,
                },
            }
        ],
    }

    archivo = carpeta / "turno_01_rep_01_seed_123.json"
    archivo.write_text(json.dumps(payload), encoding="utf-8")

    auditoria, filas = auditar_componentes_costos(tmp_path)
    assert auditoria["inconsistentes"] == 0
    assert filas[0]["costo_exceso_tolerancia"] == 4000.0

    payload["resultados"][0]["resultado"]["costo_total"] = 4301.0
    archivo.write_text(json.dumps(payload), encoding="utf-8")

    auditoria, _ = auditar_componentes_costos(tmp_path)
    assert auditoria["inconsistentes"] == 1


def test_generacion_no_medida_no_se_confunde_con_cero(tmp_path: Path) -> None:
    assert leer_tiempo_generacion(tmp_path) is None

    (tmp_path / "metadata_generacion.json").write_text(
        json.dumps({"tiempo_generacion_s": 0.08}),
        encoding="utf-8",
    )

    assert leer_tiempo_generacion(tmp_path) == 0.08


def test_tiempos_gis_con_reintentos(tmp_path: Path) -> None:
    assert leer_tiempo_preparacion_gis(tmp_path, 2) is None

    archivo = tmp_path / "tiempos_preparacion_gis.csv"
    archivo.write_text(
        "numero_turno,preparacion_s,estado\n1,1.25,OK\n2,0.50,ERROR\n2,2.00,OK\n",
        encoding="utf-8",
    )

    assert leer_tiempo_preparacion_gis(tmp_path, 2) == 3.75
