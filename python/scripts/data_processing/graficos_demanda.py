from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

PYTHON_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = PYTHON_ROOT / "data" / "processed" / "geocoding" / "direcciones_geolocalizadas.csv"
DEFAULT_OUTPUT_DIR = PYTHON_ROOT / "data" / "processed" / "estadistica"
DEFAULT_BAR_FILE = DEFAULT_OUTPUT_DIR / "barrios_top15.png"
DEFAULT_DONUT_FILE = DEFAULT_OUTPUT_DIR / "pedidos_por_ciudad.png"

DPI = 150
PALETA_CIUDADES: dict[str, str] = {
    "Granadero Baigorria": "#4361EE",
    "Rosario": "#F72585",
    "Funes": "#F8961E",
    "Ibarlucea": "#4CC9F0",
    "Roldan": "#43AA8B",
    "Roldán": "#43AA8B",
    "San Lorenzo": "#577590",
    "Capitan Bermudez": "#7209B7",
    "Capitán Bermúdez": "#7209B7",
    "Fray Luis Beltran": "#3F37C9",
    "Fray Luis Beltrán": "#3F37C9",
    "Otros": "#9CA3AF",
}


def _configurar_estilo_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def generar_grafico_top_barrios(
    df: pd.DataFrame,
    ruta_salida: Path = DEFAULT_BAR_FILE,
    top_n: int = 15,
) -> Path:
    """Genera y guarda el gráfico de barras horizontales para el Top N de barrios."""
    _configurar_estilo_matplotlib()

    col_ciudad = "ciudad_detectada" if "ciudad_detectada" in df.columns else "ciudad"
    df_clean = df.copy()
    df_clean["barrio"] = df_clean["barrio"].astype(str).str.strip()
    df_clean[col_ciudad] = df_clean[col_ciudad].astype(str).str.strip()

    stats = (
        df_clean.groupby([col_ciudad, "barrio"])
        .size()
        .reset_index(name="cantidad_pedidos")
        .sort_values("cantidad_pedidos", ascending=False)
    )

    top = stats.head(top_n).copy()
    top["etiqueta"] = top["barrio"] + " (" + top[col_ciudad].str[:3] + ")"
    colores_barra = [PALETA_CIUDADES.get(c, "#6B7280") for c in top[col_ciudad]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["etiqueta"], top["cantidad_pedidos"], color=colores_barra, height=0.65)

    for bar, val in zip(bars, top["cantidad_pedidos"]):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=10,
            color="#374151",
        )

    ax.set_xlabel("Cantidad de pedidos", fontsize=11, color="#374151")
    ax.set_title(
        f"Top {top_n} barrios por volumen de pedidos",
        fontsize=14,
        fontweight="bold",
        pad=14,
        color="#111827",
    )
    max_val = top["cantidad_pedidos"].max() if not top.empty else 10
    ax.set_xlim(0, max_val * 1.15)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)

    leyenda = [
        mpatches.Patch(color=v, label=k)
        for k, v in PALETA_CIUDADES.items()
        if k in top[col_ciudad].values
    ]
    ax.legend(handles=leyenda, fontsize=9, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Barras -> {ruta_salida}")
    return ruta_salida


def generar_grafico_donut_ciudades(
    df: pd.DataFrame,
    ruta_salida: Path = DEFAULT_DONUT_FILE,
    umbral_pct: float = 1.0,
) -> Path:
    """Genera y guarda el gráfico donut agrupando ciudades menores al umbral en 'Otros'."""
    _configurar_estilo_matplotlib()

    col_ciudad = "ciudad_detectada" if "ciudad_detectada" in df.columns else "ciudad"
    df_clean = df.copy()
    df_clean[col_ciudad] = df_clean[col_ciudad].astype(str).str.strip()

    por_ciudad = df_clean.groupby(col_ciudad).size().reset_index(name="n")
    total_pedidos = int(por_ciudad["n"].sum())
    por_ciudad["pct"] = (por_ciudad["n"] / total_pedidos * 100) if total_pedidos > 0 else 0

    principales = por_ciudad[por_ciudad["pct"] >= umbral_pct].copy()
    otros_n = int(por_ciudad[por_ciudad["pct"] < umbral_pct]["n"].sum())

    if otros_n > 0:
        otros_df = pd.DataFrame([{
            col_ciudad: "Otros",
            "n": otros_n,
            "pct": (otros_n / total_pedidos * 100) if total_pedidos > 0 else 0,
        }])
        por_ciudad_graf = pd.concat([principales, otros_df], ignore_index=True)
    else:
        por_ciudad_graf = principales

    por_ciudad_graf = por_ciudad_graf.sort_values("n", ascending=False)
    colores_donut = [PALETA_CIUDADES.get(c, "#9CA3AF") for c in por_ciudad_graf[col_ciudad]]

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, *_ = ax.pie(
        por_ciudad_graf["n"],
        colors=colores_donut,
        startangle=90,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
    )

    leyenda_donut = []
    for wedge, ciudad, n, pct in zip(
        wedges,
        por_ciudad_graf[col_ciudad],
        por_ciudad_graf["n"],
        por_ciudad_graf["pct"],
    ):
        leyenda_donut.append(
            mpatches.Patch(color=wedge.get_facecolor(), label=f"{ciudad}: {n} ({pct:.1f}%)")
        )

    ax.legend(
        handles=leyenda_donut,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=False,
    )

    ax.text(
        0,
        0,
        f"{total_pedidos}\npedidos",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )

    ax.set_title("Pedidos por ciudad", fontsize=14, fontweight="bold", pad=16, color="#111827")
    ax.axis("equal")

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Donut  -> {ruta_salida}")
    return ruta_salida


def generar_graficos_demanda(
    archivo_entrada: Path = DEFAULT_INPUT_FILE,
    ruta_barras: Path = DEFAULT_BAR_FILE,
    ruta_donut: Path = DEFAULT_DONUT_FILE,
) -> tuple[Path, Path]:
    """Genera el conjunto completo de gráficos a partir del CSV geolocalizado."""
    if not archivo_entrada.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {archivo_entrada}")

    df = pd.read_csv(archivo_entrada)
    p_bar = generar_grafico_top_barrios(df, ruta_salida=ruta_barras)
    p_donut = generar_grafico_donut_ciudades(df, ruta_salida=ruta_donut)
    return p_bar, p_donut


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generación de gráficos de demanda histórica")
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="CSV geolocalizado")
    p.add_argument("--bar-file", type=Path, default=DEFAULT_BAR_FILE, help="PNG de salida gráfico de barras")
    p.add_argument("--donut-file", type=Path, default=DEFAULT_DONUT_FILE, help="PNG de salida gráfico donut")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        generar_graficos_demanda(
            archivo_entrada=args.input_file.resolve(),
            ruta_barras=args.bar_file.resolve(),
            ruta_donut=args.donut_file.resolve(),
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
