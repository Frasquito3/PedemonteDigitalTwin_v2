"""Generación de figuras para visualización de resultados experimentales."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORES = {
    "GA": "#1f77b4",
    "HIBRIDO": "#2ca02c",
    "RL": "#ff7f0e",
    "GREEDY": "#d62728",
    "RANDOM": "#7f7f7f",
}

ESTILOS = {
    "GA": "-",
    "HIBRIDO": "--",
    "RL": "-.",
    "GREEDY": ":",
    "RANDOM": "-",
}


def configurar_estilo_matplotlib() -> None:
    """Configura parámetros generales de renderizado para publicaciones."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
            "grid.alpha": 0.4,
            "grid.linestyle": ":",
        }
    )


def generar_boxplots_globales(
    df: pd.DataFrame,
    dir_salida: Path = Path("analysis/figures/global"),
) -> None:
    """Genera boxplots globales individuales y multifiguras para todas las métricas."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    # 1. Boxplot global de costo simulado (principal)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    datos_costo = [df[df["metodo"] == a]["costo_simulado"].values for a in algos]

    bp = ax.boxplot(
        datos_costo,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={
            "marker": "D",
            "markeredgecolor": "#b30000",
            "markerfacecolor": "#b30000",
            "markersize": 5,
        },
    )
    ax.set_xticks(range(1, len(algos) + 1))
    ax.set_xticklabels(algos, fontweight="bold")

    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(COLORES[algo])
        patch.set_alpha(0.75)

    ax.set_yscale("log")
    ax.set_ylabel("Costo Simulado Total [Escala Logarítmica]")
    ax.set_title("Distribución Global de Costo Simulado por Algoritmo", pad=12)
    ax.grid(True, which="both", axis="y")

    line_med = plt.Line2D([0], [0], color="black", linewidth=1.4, label="Mediana")
    line_mean = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="#b30000",
        markersize=6,
        label="Media aritmética",
    )
    ax.legend(handles=[line_med, line_mean], loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(dir_salida / "global_quality_boxplot.png", dpi=300)
    fig.savefig(dir_salida / "global_cost_boxplot.png", dpi=300)
    plt.close(fig)

    # 2. Boxplots para cada una de las demás métricas operativas
    metricas = [
        (
            "distancia_simulada_km",
            "Distancia Total Recorrida [km]",
            "global_distance_boxplot.png",
            False,
        ),
        ("duracion_simulada_min", "Duración del Turno [min]", "global_duration_boxplot.png", False),
        (
            "tardanza_simulada_min",
            "Tardanza Acumulada en Clientes [min]",
            "global_lateness_boxplot.png",
            True,
        ),
        ("viajes_simulados", "Cantidad Total de Viajes", "global_trips_boxplot.png", False),
        (
            "desbalance_simulado_min",
            "Desbalance entre Camiones [min]",
            "global_imbalance_boxplot.png",
            False,
        ),
        (
            "tiempo_plan_ms",
            "Tiempo de Cálculo del Planificador [ms]",
            "global_runtime_boxplot.png",
            True,
        ),
    ]

    for col, ylabel, nombre_archivo, usar_log in metricas:
        fig_m, ax_m = plt.subplots(figsize=(7.5, 4.2), dpi=300)
        datos_m = [df[df["metodo"] == a][col].values for a in algos]

        bp_m = ax_m.boxplot(
            datos_m,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            meanprops={
                "marker": "D",
                "markeredgecolor": "#b30000",
                "markerfacecolor": "#b30000",
                "markersize": 5,
            },
        )
        ax_m.set_xticks(range(1, len(algos) + 1))
        ax_m.set_xticklabels(algos, fontweight="bold")

        for patch, algo in zip(bp_m["boxes"], algos):
            patch.set_facecolor(COLORES[algo])
            patch.set_alpha(0.75)

        if usar_log:
            # Para log con valores >= 0
            min_val = min(np.min(d) for d in datos_m)
            if min_val <= 0:
                ax_m.set_yscale("symlog", linthresh=1.0)
            else:
                ax_m.set_yscale("log")

        ax_m.set_ylabel(ylabel)
        ax_m.set_title(f"Distribución Global de {ylabel.split(' [')[0]} por Algoritmo", pad=12)
        ax_m.grid(True, axis="y")

        ax_m.legend(handles=[line_med, line_mean], loc="upper right", framealpha=0.9)
        plt.tight_layout()
        fig_m.savefig(dir_salida / nombre_archivo, dpi=300)
        plt.close(fig_m)

    # 3. Multifigura comparativa 2x3 con todas las métricas
    fig_multi, axes_multi = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=300)
    axes_flat = axes_multi.flatten()

    for idx, (col, ylabel, _, usar_log) in enumerate(metricas):
        ax_curr = axes_flat[idx]
        datos_m = [df[df["metodo"] == a][col].values for a in algos]

        bp_curr = ax_curr.boxplot(
            datos_m,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            meanprops={
                "marker": "D",
                "markeredgecolor": "#b30000",
                "markerfacecolor": "#b30000",
                "markersize": 4,
            },
        )
        ax_curr.set_xticks(range(1, len(algos) + 1))
        ax_curr.set_xticklabels(algos, fontsize=8, fontweight="bold")

        for patch, algo in zip(bp_curr["boxes"], algos):
            patch.set_facecolor(COLORES[algo])
            patch.set_alpha(0.75)

        if usar_log:
            min_val = min(np.min(d) for d in datos_m)
            if min_val <= 0:
                ax_curr.set_yscale("symlog", linthresh=1.0)
            else:
                ax_curr.set_yscale("log")

        ax_curr.set_ylabel(ylabel, fontsize=8.5)
        ax_curr.set_title(ylabel.split(" [")[0], fontsize=9.5, fontweight="bold")
        ax_curr.grid(True, axis="y")

    plt.tight_layout()
    fig_multi.savefig(dir_salida / "global_all_metrics_boxplots.png", dpi=300)
    plt.close(fig_multi)


def generar_figura_escalabilidad(
    df: pd.DataFrame,
    dir_salida: Path = Path("analysis/figures/order_count"),
) -> None:
    """Genera la curva de escalabilidad de costo mediano según carga de tareas."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    bins = [6, 10, 15, 20, 25, 35]
    labels_bins = ["06-10", "11-15", "16-20", "21-25", "26-35"]
    df["estrato_tareas"] = pd.cut(df["cantidad_tareas"], bins=bins, labels=labels_bins)

    pivot_escalabilidad = df.pivot_table(
        index="estrato_tareas",
        columns="metodo",
        values="costo_simulado",
        aggfunc="median",
        observed=False,
    )[algos]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    x = range(len(labels_bins))

    for algo in algos:
        valores = pivot_escalabilidad[algo].values
        ax.plot(
            x,
            valores,
            marker="o",
            linewidth=1.8,
            linestyle=ESTILOS[algo],
            color=COLORES[algo],
            label=algo,
            markersize=5,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bins)
    ax.set_xlabel("Rango de Tareas por Turno")
    ax.set_ylabel("Mediana del Costo Simulado [Escala Logarítmica]")
    ax.set_title("Curva de Escalabilidad Operativa según Carga de Tareas", pad=12)
    ax.grid(True, which="both", axis="y")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    fig.savefig(dir_salida / "scalability_order_count.png", dpi=300)
    plt.close(fig)


def generar_figura_heatmap_enfrentamientos(
    ruta_pairwise_csv: Path = Path("analysis/statistics/pairwise_comparisons.csv"),
    dir_salida: Path = Path("analysis/figures/pairwise"),
) -> None:
    """Genera mapa de calor de victorias relativas en comparaciones pareadas."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    df_pw = pd.read_csv(ruta_pairwise_csv)
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    matriz_win_rate = np.zeros((len(algos), len(algos)))

    for _, row in df_pw.iterrows():
        a1, a2 = row["algoritmo_1"], row["algoritmo_2"]
        i, j = algos.index(a1), algos.index(a2)
        wr1 = row["porc_victorias_1"]
        wr2 = row["porc_victorias_2"]
        matriz_win_rate[i, j] = wr1
        matriz_win_rate[j, i] = wr2

    for k in range(len(algos)):
        matriz_win_rate[k, k] = 50.0

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
    im = ax.imshow(matriz_win_rate, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(algos)))
    ax.set_yticks(range(len(algos)))
    ax.set_xticklabels(algos, fontweight="bold")
    ax.set_yticklabels(algos, fontweight="bold")

    for i in range(len(algos)):
        for j in range(len(algos)):
            val = matriz_win_rate[i, j]
            color_txt = "white" if val > 65 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color_txt, fontsize=9.5)

    ax.set_title(
        "Tasa de Victorias en Enfrentamientos Directos Pareados (Fila vs. Columna)", pad=12
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% Victorias")

    plt.tight_layout()
    fig.savefig(dir_salida / "pairwise_win_rate_heatmap.png", dpi=300)
    plt.close(fig)


def generar_figura_compromiso_runtime(
    df: pd.DataFrame,
    dir_salida: Path = Path("analysis/figures/runtime"),
) -> None:
    """Genera diagrama de Pareto (calidad vs. tiempo de cálculo) con cruces IQR."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    for algo in algos:
        sub = df[df["metodo"] == algo]
        x_med = np.median(sub["tiempo_plan_ms"])
        x_q25 = np.percentile(sub["tiempo_plan_ms"], 25)
        x_q75 = np.percentile(sub["tiempo_plan_ms"], 75)

        y_med = np.median(sub["costo_simulado"])
        y_q25 = np.percentile(sub["costo_simulado"], 25)
        y_q75 = np.percentile(sub["costo_simulado"], 75)

        ax.errorbar(
            x_med,
            y_med,
            xerr=[[x_med - x_q25], [x_q75 - x_med]],
            yerr=[[y_med - y_q25], [y_q75 - y_med]],
            fmt="o",
            color=COLORES[algo],
            ecolor=COLORES[algo],
            elinewidth=1.5,
            capsize=4,
            markersize=9,
            label=f"{algo} ({x_med:.1f} ms, ${y_med:.1f})",
            alpha=0.9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tiempo de Cómputo del Planificador [ms, Escala Logarítmica]")
    ax.set_ylabel("Mediana del Costo Simulado [Escala Logarítmica]")
    ax.set_title("Compromiso Calidad vs. Tiempo de Cómputo (Mediana e IQR Bivariado)", pad=12)
    ax.grid(True, which="both", alpha=0.5)

    # Leyenda ubicada arriba a la derecha sin solapar puntos
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(dir_salida / "runtime_vs_quality_scatter.png", dpi=300)
    plt.close(fig)


def generar_figura_desglose_costos(
    df: pd.DataFrame,
    dir_salida: Path = Path("analysis/figures/cost_breakdown"),
) -> None:
    """Genera gráfico de barras apiladas de los componentes promedio del costo."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    cols_fisicas = ["costo_distancia", "costo_operacion", "costo_viajes", "costo_desbalance"]

    medias_fisicas = []
    medias_tardanza = []
    medias_exceso = []
    medias_no_entregadas = []

    for a in algos:
        sub = df[df["metodo"] == a]
        f_val = sub[cols_fisicas].sum(axis=1).mean()
        t_val = sub["costo_tardanza"].mean()
        e_val = sub["costo_exceso_tolerancia"].mean()
        ne_val = (sub["costo_tareas_no_entregadas"] + sub["costo_pedidos_incompletos"]).mean()

        medias_fisicas.append(f_val)
        medias_tardanza.append(t_val)
        medias_exceso.append(e_val)
        medias_no_entregadas.append(ne_val)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = np.arange(len(algos))
    w = 0.55

    ax.bar(
        x,
        medias_fisicas,
        w,
        label="Costos Físicos Directos (Km, Horas, Viajes, Desbalance)",
        color="#2b5c8f",
    )
    ax.bar(
        x,
        medias_tardanza,
        w,
        bottom=medias_fisicas,
        label="Penalización por Tardanza en Ventana",
        color="#d95f02",
    )
    bottom_3 = np.array(medias_fisicas) + np.array(medias_tardanza)
    ax.bar(
        x,
        medias_exceso,
        w,
        bottom=bottom_3,
        label="Penalización por Exceso de Turno (>270 min)",
        color="#7570b3",
    )
    bottom_4 = bottom_3 + np.array(medias_exceso)
    ax.bar(
        x,
        medias_no_entregadas,
        w,
        bottom=bottom_4,
        label="Penalización por Tareas Omitidas",
        color="#e7298a",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Costo Promedio [$ por Turno, Escala Logarítmica]")
    ax.set_title("Desglose Estructural de Componentes del Costo Total Simulado", pad=12)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(dir_salida / "cost_components_stacked.png", dpi=300)
    plt.close(fig)


def generar_figura_ecdf_tardanza(
    df: pd.DataFrame,
    dir_salida: Path = Path("analysis/figures/service_level"),
) -> None:
    """Genera función de distribución acumulada empírica (ECDF) de tardanza."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    for algo in algos:
        vals = np.sort(df[df["metodo"] == algo]["tardanza_simulada_min"].values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(
            vals,
            y,
            label=algo,
            color=COLORES[algo],
            linestyle=ESTILOS[algo],
            linewidth=1.8,
            where="post",
        )

    ax.set_xlim(-1, 60)
    ax.set_xlabel("Tardanza Acumulada en Clientes [minutos]")
    ax.set_ylabel("Probabilidad Acumulada P(Tardanza ≤ x)")
    ax.set_title("Nivel de Servicio al Cliente: Distribución Acumulada (ECDF) de Tardanza", pad=12)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, label="Puntualidad Perfecta (0 min)")
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(dir_salida / "lateness_ecdf.png", dpi=300)
    plt.close(fig)


def generar_figura_zoom_11_15(
    df: pd.DataFrame,
    dir_salida: Path = Path("analysis/figures/order_count"),
) -> None:
    """Genera comparativa del régimen típico (11 a 15 tareas, N=858 turnos)."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    sub_11_15 = df[(df["cantidad_tareas"] >= 11) & (df["cantidad_tareas"] <= 15)]
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=300)
    metricas = [
        ("costo_simulado", "Costo Simulado [$]", True),
        ("tardanza_simulada_min", "Tardanza en Clientes [min]", True),
        ("duracion_simulada_min", "Duración Turno [min]", False),
    ]

    for ax, (col, label, log_scale) in zip(axes, metricas):
        datos = [sub_11_15[sub_11_15["metodo"] == a][col].values for a in algos]
        bp = ax.boxplot(
            datos,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            meanprops={
                "marker": "D",
                "markeredgecolor": "#b30000",
                "markerfacecolor": "#b30000",
                "markersize": 4,
            },
        )
        ax.set_xticks(range(1, len(algos) + 1))
        ax.set_xticklabels(algos, fontweight="bold", fontsize=8.5)

        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(COLORES[algo])
            patch.set_alpha(0.75)

        if log_scale:
            min_val = min(np.min(d) for d in datos)
            if min_val <= 0:
                ax.set_yscale("symlog", linthresh=1.0)
            else:
                ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, axis="y")

    fig.suptitle(
        "Desempeño Operativo en el Régimen Típico del Negocio (11 a 15 Tareas, N = 858 Turnos)",
        fontsize=11,
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(dir_salida / "typical_load_11_15_comparison.png", dpi=300)
    plt.close(fig)


def generar_figura_brecha_simulacion(
    df: pd.DataFrame,
    dir_salida: Path = Path("analysis/figures/simulation_gap"),
) -> None:
    """Genera gráfico de brecha entre costo simulado y costo estimado."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    algos = ["GA", "HIBRIDO", "RL", "GREEDY", "RANDOM"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    df_temp = df.copy()
    df_temp["brecha_costo"] = df_temp["costo_simulado"] - df_temp["costo_estimado"]

    datos = [df_temp[df_temp["metodo"] == a]["brecha_costo"].values for a in algos]

    bp = ax.boxplot(
        datos,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={"color": "#b30000", "linewidth": 1.2, "linestyle": "--"},
    )
    ax.set_xticks(range(1, len(algos) + 1))
    ax.set_xticklabels(algos, fontweight="bold")

    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(COLORES[algo])
        patch.set_alpha(0.75)

    ax.set_ylabel("Brecha de Desvío: Costo Simulado - Costo Estimado [$]")
    ax.set_title("Brecha entre Costo Simulado (AnyLogic) y Costo Estimado (Python)", pad=12)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.grid(True, axis="y")

    line_med = plt.Line2D([0], [0], color="black", linewidth=1.4, label="Mediana de desvío")
    line_mean = plt.Line2D(
        [0], [0], color="#b30000", linestyle="--", linewidth=1.2, label="Media de desvío"
    )
    ax.legend(handles=[line_med, line_mean], loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(dir_salida / "estimated_vs_simulated_gap.png", dpi=300)
    plt.close(fig)


def generar_todas_las_figuras(
    ruta_morning_csv: Path = Path("analysis/data/morning_results.csv"),
    ruta_pairwise_csv: Path = Path("analysis/statistics/pairwise_comparisons.csv"),
) -> None:
    """Ejecuta el pipeline completo de generación de gráficos."""
    configurar_estilo_matplotlib()
    df = pd.read_csv(ruta_morning_csv)
    print("Generando figuras en PNG (300 DPI)...")
    generar_boxplots_globales(df)
    generar_figura_escalabilidad(df)
    generar_figura_heatmap_enfrentamientos(ruta_pairwise_csv)
    generar_figura_compromiso_runtime(df)
    generar_figura_desglose_costos(df)
    generar_figura_ecdf_tardanza(df)
    generar_figura_zoom_11_15(df)
    generar_figura_brecha_simulacion(df)
    print("Figuras generadas exitosamente.")


if __name__ == "__main__":
    generar_todas_las_figuras()
