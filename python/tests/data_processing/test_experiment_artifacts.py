"""Pruebas de persistencia y estadísticas independientes de AnyLogic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from scripts.simulation.experiment_artifacts import (
    METRICAS_SIMULADAS,
    escribir_csv_atomico,
    escribir_json_atomico,
    generar_analisis,
    sha256_archivo,
)


def _fila(dia: int, rep: int, metodo: str, costo: float) -> dict[str, str]:
    fila = {
        "dia_numero": str(dia),
        "seed_escenario": str(100 + dia),
        "repeticion": str(rep),
        "seed_ejecucion": str(10000 + 10 * dia + rep),
        "metodo": metodo,
        "estado_simulado": "FINALIZADO",
        "algoritmo_simulado": metodo,
    }
    for metrica in METRICAS_SIMULADAS:
        fila[metrica] = str(costo if metrica == "costo_simulado" else 1.0)
    return fila


def test_escritura_atomica_preserva_json_csv_y_hash(tmp_path: Path) -> None:
    archivo_json = tmp_path / "datos.json"
    archivo_csv = tmp_path / "datos.csv"
    escribir_json_atomico(archivo_json, {"dato": 1})
    escribir_json_atomico(archivo_json, {"dato": 2})
    escribir_csv_atomico(archivo_csv, [{"x": 3}], ["x"])
    escribir_csv_atomico(archivo_csv, [{"x": 4}], ["x"])
    assert json.loads(archivo_json.read_text(encoding="utf-8")) == {"dato": 2}
    with archivo_csv.open(encoding="utf-8-sig", newline="") as archivo:
        assert list(csv.DictReader(archivo)) == [{"x": "4"}]
    assert len(sha256_archivo(archivo_json) or "") == 64
    assert sha256_archivo(tmp_path / "ausente") is None
    assert not list(tmp_path.glob("*.tmp"))


def test_analisis_pareado_por_escenario_y_auditoria(tmp_path: Path) -> None:
    manifest = {
        "entradas": [{"dia_numero": d, "seed_escenario": 100 + d} for d in (1, 2)],
        "corridas": [
            {"dia_numero": d, "repeticion": rep, "seed_ejecucion": 10000 + 10 * d + rep}
            for d in (1, 2)
            for rep in (1, 2)
        ],
    }
    escribir_json_atomico(tmp_path / "manifest_corridas.json", manifest)
    filas = []
    for dia in (1, 2):
        for rep in (1, 2):
            for metodo in ("RL", "GA"):
                costo = 100.0 + 20.0 * dia if metodo == "RL" else 50.0 + 10.0 * dia
                filas.append(_fila(dia, rep, metodo, costo))
            nombre = f"turno_{dia:02d}_rep_{rep:02d}_seed_{10000 + 10 * dia + rep}.json"
            for carpeta, nombre_archivo in (
                ("corridas", nombre),
                ("instancias", nombre),
                ("planes", f"turno_{dia:02d}_rep_{rep:02d}_rl.json"),
                ("planes", f"turno_{dia:02d}_rep_{rep:02d}_ga.json"),
            ):
                escribir_json_atomico(tmp_path / carpeta / nombre_archivo, {})
    escribir_csv_atomico(tmp_path / "resumen_resultados.csv", filas, list(filas[0]))
    auditoria = generar_analisis(tmp_path, metodos=("RL", "GA"))
    assert auditoria["corridas_esperadas"] == 8
    assert auditoria["corridas_validas"] == 8
    assert auditoria["faltantes"] == 0
    assert auditoria["duplicados"] == []
    assert auditoria["archivos_faltantes"] == []

    with (tmp_path / "analisis" / "comparaciones_rl.csv").open(
        encoding="utf-8-sig", newline=""
    ) as archivo:
        diferencias = list(csv.DictReader(archivo))
    costo = next(fila for fila in diferencias if fila["metrica"] == "costo_simulado")
    assert int(costo["n_replicas_emparejadas"]) == 4
    assert int(costo["n_escenarios_independientes"]) == 2
    assert float(costo["diferencia_media"]) == pytest.approx(65.0)
    assert float(costo["margen_ic95"]) > 0


def test_auditoria_detecta_fallo_y_registro_duplicado(tmp_path: Path) -> None:
    filas = [_fila(1, 1, "RL", 100.0), _fila(1, 1, "RL", 100.0)]
    filas.append(_fila(1, 1, "GA", 50.0))
    filas[-1]["estado_simulado"] = "ERROR"
    escribir_csv_atomico(tmp_path / "resumen_resultados.csv", filas, list(filas[0]))
    auditoria = generar_analisis(tmp_path, metodos=("RL", "GA"))
    assert len(auditoria["duplicados"]) == 1
    assert auditoria["corridas_validas"] == 1
    assert auditoria["corridas_invalidas"] == 1
