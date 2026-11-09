"""Cálculo de estadísticas descriptivas, intervalos de confianza y pruebas pareadas."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA_REPRODUCIBILIDAD = 42


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Calcula la prueba de rangos con signo de Wilcoxon (aproximación asintótica)."""
    diff = x - y
    diff = diff[diff != 0.0]
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 1.0

    ranks = pd.Series(np.abs(diff)).rank().values
    w_pos = np.sum(ranks[diff > 0.0])
    w_neg = np.sum(ranks[diff < 0.0])
    w = min(w_pos, w_neg)

    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mean_w) / std_w

    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(w), float(z), float(p_val)


def friedman_test(matrix: np.ndarray) -> tuple[float, float]:
    """Calcula la prueba de Friedman para k algoritmos sobre n instancias."""
    n, k = matrix.shape
    ranks = np.zeros_like(matrix)
    for i in range(n):
        ranks[i, :] = pd.Series(matrix[i, :]).rank().values

    r_mean = np.mean(ranks, axis=0)
    chi2 = (12.0 * n / (k * (k + 1))) * np.sum((r_mean - (k + 1) / 2.0) ** 2)

    df = k - 1
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    p_val = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(chi2), float(max(0.0, min(1.0, p_val)))


def bootstrap_ci_median(
    data: np.ndarray,
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = SEMILLA_REPRODUCIBILIDAD,
) -> tuple[float, float]:
    """Calcula intervalo de confianza bootstrap para la mediana."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_medians = np.median(data[boot_indices], axis=1)
    low = np.percentile(boot_medians, 100 * (alpha / 2))
    high = np.percentile(boot_medians, 100 * (1 - alpha / 2))
    return float(low), float(high)


def ejecutar_analisis_estadistico(
    ruta_morning_csv: Path = Path("analysis/data/morning_results.csv"),
    dir_salida: Path = Path("analysis/statistics"),
) -> None:
    """Ejecuta los análisis descriptivos e inferenciales y exporta CSVs."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ruta_morning_csv)

    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    # 1. Estadísticas descriptivas
    filas_desc = []
    for algo in algos:
        sub = df[df["metodo"] == algo]
        c = sub["costo_simulado"].values
        d = sub["distancia_simulada_km"].values
        t = sub["duracion_simulada_min"].values
        v = sub["viajes_simulados"].values
        tard = sub["tardanza_simulada_min"].values
        desb = sub["desbalance_simulado_min"].values
        rt = sub["tiempo_plan_ms"].values

        q25, q50, q75 = np.percentile(c, [25, 50, 75])
        media_c = np.mean(c)
        std_c = np.std(c, ddof=1)
        cv_c = (std_c / media_c) * 100 if media_c > 0 else 0.0

        filas_desc.append(
            {
                "algoritmo": algo,
                "muestra_n": len(sub),
                "costo_mediana": float(q50),
                "costo_q25": float(q25),
                "costo_q75": float(q75),
                "costo_iqr": float(q75 - q25),
                "costo_media": float(media_c),
                "costo_std": float(std_c),
                "costo_cv_porc": float(cv_c),
                "costo_min": float(np.min(c)),
                "costo_max": float(np.max(c)),
                "distancia_km_mediana": float(np.median(d)),
                "distancia_km_media": float(np.mean(d)),
                "duracion_min_mediana": float(np.median(t)),
                "duracion_min_media": float(np.mean(t)),
                "viajes_mediana": float(np.median(v)),
                "viajes_media": float(np.mean(v)),
                "tardanza_min_mediana": float(np.median(tard)),
                "tardanza_min_media": float(np.mean(tard)),
                "desbalance_min_mediana": float(np.median(desb)),
                "desbalance_min_media": float(np.mean(desb)),
                "tiempo_plan_ms_mediana": float(np.median(rt)),
                "tiempo_plan_ms_media": float(np.mean(rt)),
            }
        )

    df_desc = pd.DataFrame(filas_desc)
    df_desc.to_csv(dir_salida / "descriptive_statistics.csv", index=False)

    # 2. Intervalos de confianza
    filas_ci = []
    for algo in algos:
        sub = df[df["metodo"] == algo]
        c = sub["costo_simulado"].values
        n = len(c)
        med = float(np.median(c))
        mean = float(np.mean(c))
        std = float(np.std(c, ddof=1))
        se_mean = std / math.sqrt(n)

        # IC 95% para la media
        ci_mean_low = mean - 1.96 * se_mean
        ci_mean_high = mean + 1.96 * se_mean

        # IC 95% bootstrap para la mediana
        ci_med_low, ci_med_high = bootstrap_ci_median(c)

        filas_ci.append(
            {
                "algoritmo": algo,
                "metrica": "mediana_costo",
                "estimador_puntual": med,
                "ci_95_inferior": ci_med_low,
                "ci_95_superior": ci_med_high,
                "metodo_ci": "bootstrap_percentil_2000",
            }
        )
        filas_ci.append(
            {
                "algoritmo": algo,
                "metrica": "media_costo",
                "estimador_puntual": mean,
                "ci_95_inferior": ci_mean_low,
                "ci_95_superior": ci_mean_high,
                "metodo_ci": "asintotico_normal",
            }
        )

    df_ci = pd.DataFrame(filas_ci)
    df_ci.to_csv(dir_salida / "confidence_intervals.csv", index=False)

    # 3. Comparaciones pareadas sobre las mismas instancias
    instancias_comunes = df["id_simulacion"].apply(lambda s: "_".join(s.split("_")[:-1]))
    df["instancia_clave"] = instancias_comunes

    pivote = df.pivot(index="instancia_clave", columns="metodo", values="costo_simulado").dropna()

    filas_comp = []
    pares = [
        ("GA", "HIBRIDO"),
        ("GA", "RL"),
        ("GA", "GREEDY"),
        ("GA", "RANDOM"),
        ("HIBRIDO", "RL"),
        ("HIBRIDO", "GREEDY"),
        ("HIBRIDO", "RANDOM"),
        ("RL", "GREEDY"),
        ("RL", "RANDOM"),
        ("GREEDY", "RANDOM"),
    ]

    for a1, a2 in pares:
        s1 = pivote[a1].values
        s2 = pivote[a2].values
        n_total = len(s1)
        gana_1 = int(np.sum(s1 < s2))
        gana_2 = int(np.sum(s2 < s1))
        empata = int(np.sum(s1 == s2))

        diff = s1 - s2
        med_diff = float(np.median(diff))
        media_diff = float(np.mean(diff))

        filas_comp.append(
            {
                "algoritmo_1": a1,
                "algoritmo_2": a2,
                "instancias_evaluadas": n_total,
                "victorias_1": gana_1,
                "porc_victorias_1": float(gana_1 / n_total * 100),
                "victorias_2": gana_2,
                "porc_victorias_2": float(gana_2 / n_total * 100),
                "empates": empata,
                "porc_empates": float(empata / n_total * 100),
                "mediana_diferencia_1_menos_2": med_diff,
                "media_diferencia_1_menos_2": media_diff,
            }
        )

    df_comp = pd.DataFrame(filas_comp)
    df_comp.to_csv(dir_salida / "pairwise_comparisons.csv", index=False)

    # 4. Pruebas de hipótesis pareadas
    matrix_friedman = pivote[algos].values
    chi2_f, p_friedman = friedman_test(matrix_friedman)
    n_inst = len(matrix_friedman)
    k_algos = len(algos)
    kendall_w = chi2_f / (n_inst * (k_algos - 1))

    filas_tests = []
    raw_p_values = []
    test_records = []

    for a1, a2 in pares:
        s1 = pivote[a1].values
        s2 = pivote[a2].values
        w, z, p = wilcoxon_signed_rank(s1, s2)
        r_effect = abs(z) / math.sqrt(len(s1))
        raw_p_values.append(p)
        test_records.append(
            {
                "comparacion": f"{a1} vs {a2}",
                "estadistico_w": w,
                "z_score": z,
                "p_valor_crudo": p,
                "tamano_efecto_r": r_effect,
            }
        )

    # Corrección de Holm-Bonferroni
    indices_ordenados = np.argsort(raw_p_values)
    m = len(raw_p_values)
    p_adj = [0.0] * m
    for rank, idx in enumerate(indices_ordenados):
        raw_p = raw_p_values[idx]
        adj = raw_p * (m - rank)
        p_adj[idx] = min(1.0, adj)

    for i in range(1, m):
        idx_curr = indices_ordenados[i]
        idx_prev = indices_ordenados[i - 1]
        if p_adj[idx_curr] < p_adj[idx_prev]:
            p_adj[idx_curr] = p_adj[idx_prev]

    for i, rec in enumerate(test_records):
        rec["p_valor_ajustado_holm"] = p_adj[i]
        rec["significativo_al_5porc"] = p_adj[i] < 0.05
        rec["friedman_chi2"] = chi2_f
        rec["friedman_p"] = p_friedman
        rec["kendall_w"] = kendall_w
        filas_tests.append(rec)

    df_tests = pd.DataFrame(filas_tests)
    df_tests.to_csv(dir_salida / "pairwise_tests.csv", index=False)

    # 5. Análisis estratificado por cantidad de tareas
    bins = [6, 10, 15, 20, 25, 35]
    labels_bins = ["06-10", "11-15", "16-20", "21-25", "26-35"]
    df["estrato_tareas"] = pd.cut(df["cantidad_tareas"], bins=bins, labels=labels_bins)

    filas_estratos = []
    for estrato in labels_bins:
        sub_e = df[df["estrato_tareas"] == estrato]
        if sub_e.empty:
            continue
        n_turnos = len(sub_e) // len(algos)
        for algo in algos:
            sub_a = sub_e[sub_e["metodo"] == algo]
            c = sub_a["costo_simulado"].values
            tard = sub_a["tardanza_simulada_min"].values
            filas_estratos.append(
                {
                    "estrato_tareas": estrato,
                    "turnos_en_estrato": n_turnos,
                    "algoritmo": algo,
                    "costo_mediana": float(np.median(c)),
                    "costo_media": float(np.mean(c)),
                    "costo_std": float(np.std(c, ddof=1)),
                    "tardanza_min_mediana": float(np.median(tard)),
                    "tardanza_min_media": float(np.mean(tard)),
                }
            )

    df_estratos = pd.DataFrame(filas_estratos)
    df_estratos.to_csv(dir_salida / "results_by_order_count.csv", index=False)


if __name__ == "__main__":
    ejecutar_analisis_estadistico()
    print("Análisis estadístico completado exitosamente.")
