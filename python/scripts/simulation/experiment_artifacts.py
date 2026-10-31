"""Persistencia atómica, integridad y estadísticas del estudio experimental.

Los resultados completos son locales; los CSV consolidados mantienen el esquema
utilizado por el dashboard AnyLogic. Una réplica es la unidad de recuperación.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

from scripts.simulation.experimental_checks import (
    COLUMNAS_DESGLOSE,
    auditar_componentes_costos,
)

METRICAS_SIMULADAS: dict[str, str] = {
    "costo_simulado": "Costo total",
    "distancia_simulada_km": "Distancia (km)",
    "duracion_simulada_min": "Duración (min)",
    "viajes_simulados": "Viajes",
    "tardanza_simulada_min": "Tardanza (min)",
    "desbalance_simulado_min": "Desbalance (min)",
    "tiempo_plan_ms": "Cómputo de planificación (ms)",
    "tiempo_motor_s": "Cómputo del motor AnyLogic (s)",
}


def escribir_json_atomico(ruta: Path, contenido: Any) -> None:
    """Guarda JSON mediante escritura temporal y reemplazo atómico."""
    _escribir_atomico(
        ruta, json.dumps(contenido, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def escribir_csv_atomico(ruta: Path, filas: list[dict[str, Any]], columnas: list[str]) -> None:
    """Evita exponer un resumen CSV truncado tras una interrupción."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=ruta.parent,
            prefix=f".{ruta.name}.",
            suffix=".tmp",
            delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            escritor = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
            escritor.writeheader()
            escritor.writerows(filas)
            archivo.flush()
            os.fsync(archivo.fileno())
        os.replace(temporal, ruta)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def _escribir_atomico(ruta: Path, texto: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=ruta.parent,
            prefix=f".{ruta.name}.",
            suffix=".tmp",
            delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            archivo.write(texto)
            archivo.flush()
            os.fsync(archivo.fileno())
        os.replace(temporal, ruta)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def sha256_archivo(ruta: Path) -> str | None:
    """Obtiene huella del archivo, sin incorporarlo a los metadatos."""
    if not ruta.is_file():
        return None
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _estado_fuentes_git(repo: Path) -> dict[str, Any]:
    """Registra cambios locales de codigo, sin inventariar resultados masivos."""
    try:
        proceso = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
                "--",
                "python/planner",
                "python/scripts",
                "python/tests",
                "anylogic/PedemonteDigitalTwin_v2/PedemonteDigitalTwin_v2.alp",
                "pyproject.toml",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        cambios = proceso.stdout.splitlines()
        return {
            "modificado": bool(cambios),
            "cantidad": len(cambios),
            "archivos": cambios[:100],
        }
    except (OSError, subprocess.SubprocessError):
        return {"modificado": None, "cantidad": None, "archivos": []}


def contexto_versiones(raiz_python: Path, modelo_exportado: Path) -> dict[str, Any]:
    """Registra revisión de Git y hashes de los modelos de referencia."""
    repo = raiz_python.parent
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        revision = git.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    rutas = {
        "codigo_ejecutor": raiz_python / "scripts" / "simulation" / "run_simulated_comparison.py",
        "codigo_generador": raiz_python / "scripts" / "data_processing" / "generar_pedidos_dia.py",
        "codigo_validaciones": raiz_python / "scripts" / "simulation" / "experimental_checks.py",
        "codigo_comparador_simulado": raiz_python
        / "planner"
        / "integration"
        / "simulated_comparison.py",
        "codigo_ejecucion_simulada": raiz_python
        / "planner"
        / "integration"
        / "simulated_execution.py",
        "codigo_selector_bridge": raiz_python / "planner" / "integration" / "selector_bridge.py",
        "codigo_comparacion_estimada": raiz_python
        / "planner"
        / "integration"
        / "estimated_comparison.py",
        "codigo_captura_entorno": raiz_python
        / "scripts"
        / "simulation"
        / "environment_snapshot.py",
        "modelo_alp": repo / "anylogic" / "PedemonteDigitalTwin_v2" / "PedemonteDigitalTwin_v2.alp",
        "datos_historicos": raiz_python / "data" / "processed" / "demanda_geografica.csv",
        "probabilidades_demanda": raiz_python
        / "data"
        / "processed"
        / "estadistica"
        / "probabilidades_demanda.csv",
        "cant_pedidos_historicos": raiz_python / "data" / "raw" / "cant_pedidos_turnos.csv",
        "modelo_anylogic_exportado": modelo_exportado,
        "modelo_rl": raiz_python / "models" / "rl" / "rl_policy.zip",
        "configuracion_rl": raiz_python / "models" / "rl" / "rl_policies.json",
        "cache_vial": raiz_python / "data" / "routing" / "cache_vial.csv",
    }
    return {
        "git_commit": revision,
        "estado_fuentes_git": _estado_fuentes_git(repo),
        "python_version": sys.version.split()[0],
        "dependencias": {
            nombre: _version_paquete(nombre)
            for nombre in ("numpy", "pandas", "stable-baselines3", "anylogic-alpyne")
        },
        "archivos": {
            nombre: {"ruta": str(ruta), "sha256": sha256_archivo(ruta)}
            for nombre, ruta in rutas.items()
        },
    }


def _version_paquete(nombre: str) -> str | None:
    try:
        return package_metadata.version(nombre)
    except package_metadata.PackageNotFoundError:
        return None


def cargar_filas_checkpoint(ruta: Path) -> list[dict[str, Any]]:
    """Obtiene las observaciones consolidadas anteriormente persistidas."""
    if not ruta.is_file():
        return []
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, list) or not all(isinstance(fila, dict) for fila in datos):
        raise ValueError("El checkpoint de observaciones tiene un formato inválido.")
    return datos


def _numero_valido(valor: Any) -> float | None:
    if valor is None or isinstance(valor, bool) or str(valor).strip() == "":
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) and numero >= 0.0 else None


def _t_critico_95(n: int) -> float:
    """Cuantil bilateral 95 % de Student sin incorporar dependencias nuevas."""
    tabla = (
        0.0,
        12.706204736,
        4.30265273,
        3.182446305,
        2.776445105,
        2.570581836,
        2.446911851,
        2.364624252,
        2.306004135,
        2.262157163,
        2.228138852,
        2.20098516,
        2.17881283,
        2.160368656,
        2.144786688,
        2.131449546,
        2.119905299,
        2.109815578,
        2.10092204,
        2.093024054,
        2.085963447,
        2.079613845,
        2.073873068,
        2.06865761,
        2.063898562,
        2.059538553,
        2.055529439,
        2.051830516,
        2.048407142,
        2.045229642,
        2.042272456,
    )
    if n < 2:
        return math.nan
    gl = n - 1
    if gl <= 30:
        return tabla[gl]
    # Expansión asintótica para df > 30, con error pequeño en esa región.
    z = statistics.NormalDist().inv_cdf(0.975)
    return (
        z
        + (z**3 + z) / (4 * gl)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * gl**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * gl**3)
    )


def generar_analisis(
    carpeta_experimento: Path,
    *,
    metodos: tuple[str, ...] = ("RL", "HIBRIDO", "GREEDY", "RANDOM", "GA"),
) -> dict[str, Any]:
    """Audita corridas y exporta estadísticas sin ejecutar el simulador.

    Los intervalos se calculan sobre medias por escenario, no sobre réplicas
    independientes, para evitar pseudorreplicación. Se excluyen fallos de
    simulación de los promedios, pero permanecen contabilizados en auditoría.
    """
    ruta_csv = carpeta_experimento / "resumen_resultados.csv"
    with ruta_csv.open(encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    carpeta_analisis = carpeta_experimento / "analisis"
    carpeta_analisis.mkdir(parents=True, exist_ok=True)
    manifiesto_ruta = carpeta_experimento / "manifest_corridas.json"
    manifiesto = (
        json.loads(manifiesto_ruta.read_text(encoding="utf-8")) if manifiesto_ruta.is_file() else {}
    )

    llaves: set[tuple[str, str, str, str, str]] = set()
    duplicados: list[str] = []
    invalidos: list[dict[str, str]] = []
    validos: list[dict[str, Any]] = []
    estados: dict[str, int] = defaultdict(int)
    for fila in filas:
        llave = (
            str(fila.get("dia_numero", "")),
            str(fila.get("seed_escenario", "")),
            str(fila.get("repeticion", "")),
            str(fila.get("seed_ejecucion", "")),
            str(fila.get("metodo", "")),
        )
        if llave in llaves:
            duplicados.append("|".join(llave))
            continue
        llaves.add(llave)
        estados[fila.get("estado_simulado", "DESCONOCIDO")] += 1
        if fila.get("metodo") not in metodos:
            invalidos.append({"id": "|".join(llave), "motivo": "Método desconocido"})
            continue
        if fila.get("estado_simulado") != "FINALIZADO" or not fila.get("algoritmo_simulado"):
            invalidos.append({"id": "|".join(llave), "motivo": "Simulación no finalizada"})
            continue
        metricas = {k: _numero_valido(fila.get(k)) for k in METRICAS_SIMULADAS}
        if any(valor is None for valor in metricas.values()):
            invalidos.append({"id": "|".join(llave), "motivo": "Métricas incompletas"})
            continue
        validos.append(fila)

    registros_metricas: list[dict[str, Any]] = []
    for metodo in metodos:
        datos = [fila for fila in validos if fila["metodo"] == metodo]
        for metrica, etiqueta in METRICAS_SIMULADAS.items():
            valores = [float(fila[metrica]) for fila in datos]
            registros_metricas.append(
                {
                    "metodo": metodo,
                    "metrica": metrica,
                    "descripcion": etiqueta,
                    "n_corridas_validas": len(valores),
                    "media": statistics.mean(valores) if valores else "",
                    "mediana": statistics.median(valores) if valores else "",
                    "desviacion_estandar": statistics.stdev(valores) if len(valores) > 1 else "",
                    "minimo": min(valores) if valores else "",
                    "maximo": max(valores) if valores else "",
                }
            )

    emparejados: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for fila in validos:
        llave_pareja = tuple(
            fila.get(k, "")
            for k in ("dia_numero", "seed_escenario", "repeticion", "seed_ejecucion")
        )
        emparejados[llave_pareja][fila["metodo"]] = fila

    comparaciones: list[dict[str, Any]] = []
    for metodo in metodos:
        if metodo == "RL":
            continue
        for metrica in METRICAS_SIMULADAS:
            por_escenario: dict[tuple[str, str], list[float]] = defaultdict(list)
            n_replicas = 0
            for llave, pareja in emparejados.items():
                if "RL" in pareja and metodo in pareja:
                    por_escenario[llave[:2]].append(
                        float(pareja["RL"][metrica]) - float(pareja[metodo][metrica])
                    )
                    n_replicas += 1
            diferencias = [statistics.mean(valores) for valores in por_escenario.values()]
            n_escenarios = len(diferencias)
            media = statistics.mean(diferencias) if diferencias else None
            margen = (
                _t_critico_95(n_escenarios)
                * statistics.stdev(diferencias)
                / math.sqrt(n_escenarios)
                if n_escenarios >= 2
                else None
            )
            comparaciones.append(
                {
                    "referencia": "RL",
                    "metodo": metodo,
                    "metrica": metrica,
                    "direccion": "RL-método (negativo favorece RL)",
                    "n_replicas_emparejadas": n_replicas,
                    "n_escenarios_independientes": n_escenarios,
                    "diferencia_media": media if media is not None else "",
                    "margen_ic95": margen if margen is not None else "",
                    "limite_inferior_ic95": (media - margen)
                    if media is not None and margen is not None
                    else "",
                    "limite_superior_ic95": (media + margen)
                    if media is not None and margen is not None
                    else "",
                    "ic95_disponible": margen is not None,
                }
            )

    esperados = len(manifiesto.get("corridas", [])) * len(metodos) if manifiesto else None
    llaves_esperadas = {
        (
            str(item["dia_numero"]),
            str(entrada["seed_escenario"]),
            str(item["repeticion"]),
            str(item["seed_ejecucion"]),
            metodo,
        )
        for entrada in manifiesto.get("entradas", [])
        for item in manifiesto.get("corridas", [])
        if item["dia_numero"] == entrada["dia_numero"]
        for metodo in metodos
    }
    llaves_faltantes = sorted(llaves_esperadas - llaves)
    llaves_inesperadas = sorted(llaves - llaves_esperadas) if manifiesto else []
    archivos_faltantes: list[str] = []
    if manifiesto:
        for item in manifiesto["corridas"]:
            dia, rep, semilla = (
                int(item["dia_numero"]),
                int(item["repeticion"]),
                int(item["seed_ejecucion"]),
            )
            grupo = {llave for llave in llaves if llave[0] == str(dia) and llave[2] == str(rep)}
            if not grupo:
                continue
            nombre_corrida = f"turno_{dia:02d}_rep_{rep:02d}_seed_{semilla}.json"
            archivos = [
                carpeta_experimento / "corridas" / nombre_corrida,
                carpeta_experimento / "instancias" / nombre_corrida,
            ] + [
                carpeta_experimento
                / "planes"
                / f"turno_{dia:02d}_rep_{rep:02d}_{metodo.lower()}.json"
                for metodo in metodos
            ]
            archivos_faltantes.extend(
                ruta.relative_to(carpeta_experimento).as_posix()
                for ruta in archivos
                if not ruta.is_file()
            )
    auditoria = {
        "fecha_auditoria": datetime.now().isoformat(timespec="seconds"),
        "corridas_esperadas": esperados,
        "filas_presentes": len(filas),
        "corridas_validas": len(validos),
        "corridas_invalidas": len(invalidos),
        "duplicados": duplicados,
        "identificadores_inesperados": ["|".join(llave) for llave in llaves_inesperadas],
        "archivos_faltantes": archivos_faltantes,
        "estado_simulado": dict(estados),
        "errores": invalidos,
        "faltantes": len(llaves_faltantes) if manifiesto else None,
        "identificadores_faltantes": ["|".join(llave) for llave in llaves_faltantes],
        "advertencia_inferencia": (
            "Los IC95 se calculan sobre escenarios independientes; las réplicas "
            "de un mismo escenario se promedian antes de estimar la varianza. "
            "No son intervalos ajustados por comparaciones múltiples."
        ),
    }
    auditoria_costos, filas_costos = auditar_componentes_costos(carpeta_experimento)
    auditoria["auditoria_costos"] = auditoria_costos
    escribir_csv_atomico(
        carpeta_analisis / "desglose_costos.csv",
        filas_costos,
        COLUMNAS_DESGLOSE,
    )

    columnas_resumen = list(registros_metricas[0]) if registros_metricas else []
    columnas_comparacion = list(comparaciones[0]) if comparaciones else []
    escribir_csv_atomico(
        carpeta_analisis / "estadisticas_metodos.csv", registros_metricas, columnas_resumen
    )
    escribir_csv_atomico(
        carpeta_analisis / "comparaciones_rl.csv", comparaciones, columnas_comparacion
    )
    escribir_json_atomico(carpeta_analisis / "auditoria.json", auditoria)
    return auditoria
