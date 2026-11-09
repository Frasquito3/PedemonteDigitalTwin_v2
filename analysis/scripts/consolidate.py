"""Consolidación de resultados experimentales en datasets maestros."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def consolidar_datos(
    ruta_experimentos: Path = Path("python/data/experiments"),
    ruta_resultados: Path = Path("python/data/results/estudios"),
    dir_salida: Path = Path("analysis/data"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee, une, valida y exporta los datasets consolidados."""
    dir_salida.mkdir(parents=True, exist_ok=True)

    filas_resultados: list[dict[str, object]] = []
    filas_reproducibilidad: list[dict[str, object]] = []

    carpetas_estudios = sorted([p for p in ruta_resultados.iterdir() if p.is_dir()])

    for exp_dir in carpetas_estudios:
        exp_name = exp_dir.name
        if "47827" in exp_name:
            continue

        exp_input_dir = ruta_experimentos / exp_name
        meta_gen_file = exp_input_dir / "metadata_generacion.json"
        pedidos_info_map: dict[int, dict[str, int]] = {}

        if meta_gen_file.exists():
            with open(meta_gen_file, encoding="utf-8") as f:
                mg = json.load(f)
            for t in mg.get("turnos", []):
                dia = int(t.get("dia_numero", 0))
                pedidos_info_map[dia] = {
                    "cantidad_pedidos": int(t.get("cantidad_pedidos", 0)),
                    "unidades_totales": int(t.get("unidades_totales", 0)),
                }

        ejecuciones = [p for p in exp_dir.iterdir() if p.is_dir()]
        for ejec in ejecuciones:
            inners = [p for p in ejec.iterdir() if p.is_dir()]
            for inner in inners:
                meta_file = inner / "metadata_experimento.json"
                csv_file = inner / "resumen_resultados.csv"
                desglose_file = inner / "analisis" / "desglose_costos.csv"
                entorno_file = inner / "entorno_ejecucion.json"

                meta_data: dict[str, object] = {}
                if meta_file.exists():
                    with open(meta_file, encoding="utf-8") as f:
                        meta_data = json.load(f)

                entorno_data: dict[str, object] = {}
                if entorno_file.exists():
                    with open(entorno_file, encoding="utf-8") as f:
                        entorno_data = json.load(f)

                df_resumen = pd.read_csv(csv_file)
                df_desglose = (
                    pd.read_csv(desglose_file) if desglose_file.exists() else pd.DataFrame()
                )

                desglose_cols_merge = [
                    "dia_numero",
                    "repeticion",
                    "metodo",
                    "costo_distancia",
                    "costo_operacion",
                    "costo_viajes",
                    "costo_tardanza",
                    "costo_desbalance",
                    "costo_exceso_tolerancia",
                    "costo_tareas_no_entregadas",
                    "costo_pedidos_originales_incompletos",
                    "tareas_entregadas",
                    "tareas_no_entregadas",
                ]

                if not df_desglose.empty:
                    cols_presentes = [c for c in desglose_cols_merge if c in df_desglose.columns]
                    df_merged = pd.merge(
                        df_resumen,
                        df_desglose[cols_presentes],
                        on=["dia_numero", "repeticion", "metodo"],
                        how="left",
                        suffixes=("", "_desglose"),
                    )
                else:
                    df_merged = df_resumen.copy()

                for _, row in df_merged.iterrows():
                    dia = int(row["dia_numero"])
                    info_ped = pedidos_info_map.get(
                        dia, {"cantidad_pedidos": 0, "unidades_totales": 0}
                    )

                    sim_id = f"{exp_name}_D{dia:02d}_R{int(row['repeticion']):02d}_{row['metodo']}"

                    filas_resultados.append(
                        {
                            "id_experimento": exp_name,
                            "id_simulacion": sim_id,
                            "seed_inicial": int(meta_data.get("seed_inicial", 0)),
                            "dia_numero": dia,
                            "instancia_id": str(row.get("instancia_id", "")),
                            "archivo_excel": str(row.get("archivo_excel", "")),
                            "turno": str(row.get("turno", "MANANA")),
                            "cantidad_pedidos": info_ped["cantidad_pedidos"],
                            "cantidad_tareas": int(row.get("cantidad_tareas", 0)),
                            "unidades_totales": info_ped["unidades_totales"],
                            "repeticion": int(row.get("repeticion", 1)),
                            "seed_escenario": int(row.get("seed_escenario", 0)),
                            "seed_ejecucion": int(row.get("seed_ejecucion", 0)),
                            "metodo": str(row.get("metodo", "")),
                            "seed_algoritmo": int(row.get("seed_algoritmo", 0)),
                            "algoritmo_estimado": str(row.get("algoritmo_estimado", "")),
                            "factible_estimado": bool(row.get("factible_estimado", True)),
                            "costo_estimado": float(row.get("costo_estimado", 0.0)),
                            "distancia_estimada_km": float(row.get("distancia_estimada_km", 0.0)),
                            "duracion_estimada_min": float(row.get("duracion_estimada_min", 0.0)),
                            "viajes_estimados": int(row.get("viajes_estimados", 0)),
                            "pedidos_tardios_estimados": int(
                                row.get("pedidos_tardios_estimados", 0)
                            ),
                            "tardanza_estimada_min": float(row.get("tardanza_estimada_min", 0.0)),
                            "desbalance_estimado_min": float(
                                row.get("desbalance_estimado_min", 0.0)
                            ),
                            "tiempo_plan_ms": float(row.get("tiempo_plan_ms", 0.0)),
                            "estado_simulado": str(row.get("estado_simulado", "")),
                            "algoritmo_simulado": str(row.get("algoritmo_simulado", "")),
                            "costo_simulado": float(row.get("costo_simulado", 0.0)),
                            "distancia_simulada_km": float(row.get("distancia_simulada_km", 0.0)),
                            "duracion_simulada_min": float(row.get("duracion_simulada_min", 0.0)),
                            "viajes_simulados": int(row.get("viajes_simulados", 0)),
                            "tardanza_simulada_min": float(row.get("tardanza_simulada_min", 0.0)),
                            "desbalance_simulado_min": float(
                                row.get("desbalance_simulado_min", 0.0)
                            ),
                            "tiempo_motor_s": float(row.get("tiempo_motor_s", 0.0)),
                            "costo_distancia": float(row.get("costo_distancia", 0.0)),
                            "costo_operacion": float(row.get("costo_operacion", 0.0)),
                            "costo_viajes": float(row.get("costo_viajes", 0.0)),
                            "costo_tardanza": float(row.get("costo_tardanza", 0.0)),
                            "costo_desbalance": float(row.get("costo_desbalance", 0.0)),
                            "costo_exceso_tolerancia": float(
                                row.get("costo_exceso_tolerancia", 0.0)
                            ),
                            "costo_tareas_no_entregadas": float(
                                row.get("costo_tareas_no_entregadas", 0.0)
                            ),
                            "costo_pedidos_incompletos": float(
                                row.get("costo_pedidos_originales_incompletos", 0.0)
                            ),
                            "tareas_entregadas": int(row.get("tareas_entregadas", 0)),
                            "tareas_no_entregadas": int(row.get("tareas_no_entregadas", 0)),
                            "detalle_estimado": str(row.get("detalle_estimado", "")),
                        }
                    )

                filas_reproducibilidad.append(
                    {
                        "id_experimento": exp_name,
                        "algoritmo": str(meta_data.get("algoritmo", "")),
                        "commit_hash": str(
                            meta_data.get("commit_hash", "DESCONOCIDO_NO_REGISTRADO")
                        ),
                        "fecha_inicio": str(meta_data.get("fecha_inicio", "")),
                        "fecha_fin": str(meta_data.get("fecha_fin", "")),
                        "fecha_captura_entorno": str(entorno_data.get("fecha_captura", "")),
                        "version_python": str(entorno_data.get("python", {}).get("version", "")),
                        "implementacion_python": str(
                            entorno_data.get("python", {}).get("implementation", "")
                        ),
                        "version_java": str(entorno_data.get("java", {}).get("version", "")),
                        "version_alpyne": str(entorno_data.get("packages", {}).get("alpyne", "")),
                        "version_gymnasium": str(
                            entorno_data.get("packages", {}).get("gymnasium", "")
                        ),
                        "version_numpy": str(entorno_data.get("packages", {}).get("numpy", "")),
                        "version_pandas": str(entorno_data.get("packages", {}).get("pandas", "")),
                        "version_stable_baselines3": str(
                            entorno_data.get("packages", {}).get("stable_baselines3", "")
                        ),
                        "version_torch": str(entorno_data.get("packages", {}).get("torch", "")),
                        "sistema_operativo": str(entorno_data.get("system", {}).get("system", "")),
                        "version_so": str(entorno_data.get("system", {}).get("version", "")),
                        "arquitectura_so": str(
                            entorno_data.get("system", {}).get("architecture", "")
                        ),
                        "cpu_modelo": str(entorno_data.get("cpu", {}).get("model", "")),
                        "nucleos_fisicos": int(
                            entorno_data.get("cpu", {}).get("physical_cores", 0)
                        ),
                        "nucleos_logicos": int(entorno_data.get("cpu", {}).get("logical_cores", 0)),
                        "ram_total_gib": float(entorno_data.get("ram", {}).get("total_gib", 0.0)),
                        "gpu_modelos": str([g.get("name") for g in entorno_data.get("gpus", [])]),
                        "cantidad_turnos": int(
                            meta_data.get("configuracion", {}).get("cantidad_turnos", 100)
                        ),
                        "repeticiones_por_turno": int(
                            meta_data.get("configuracion", {}).get("repeticiones_por_turno", 3)
                        ),
                        "horizonte_simulacion_min": float(
                            meta_data.get("configuracion", {}).get(
                                "horizonte_simulacion_min", 600.0
                            )
                        ),
                        "timeout_segundos": float(
                            meta_data.get("configuracion", {}).get("timeout_segundos", 240.0)
                        ),
                        "proveedores_habilitados": bool(
                            meta_data.get("configuracion", {}).get("proveedores_habilitados", True)
                        ),
                        "solo_estimado": bool(
                            meta_data.get("configuracion", {}).get("solo_estimado", False)
                        ),
                    }
                )

    df_morning = pd.DataFrame(filas_resultados)
    df_reprod = pd.DataFrame(filas_reproducibilidad).drop_duplicates(
        subset=["id_experimento", "algoritmo"]
    )

    df_morning.to_csv(dir_salida / "morning_results.csv", index=False)
    df_reprod.to_csv(dir_salida / "reproducibility.csv", index=False)

    return df_morning, df_reprod


if __name__ == "__main__":
    df_m, df_r = consolidar_datos()
    print(f"morning_results.csv: {len(df_m)} filas")
    print(f"reproducibility.csv: {len(df_r)} filas")
