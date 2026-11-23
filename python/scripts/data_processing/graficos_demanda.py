"""Módulo de generación de visualizaciones analíticas de demanda histórica y geolocalización.

Genera gráficos de barras horizontales para el Top N de barrios de mayor demanda,
diagramas donut de distribución porcentual por localidad y mapas de calor (KDE)
sobre base cartográfica abierta en escala de grises para la zona núcleo operativa,
empleando los estándares de publicación formal del proyecto (300 DPI, tipografía serif).
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageEnhance, ImageOps

PYTHON_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = (
    PYTHON_ROOT / "data" / "processed" / "geocoding" / "direcciones_geolocalizadas.csv"
)
DEFAULT_OUTPUT_DIR = PYTHON_ROOT / "data" / "processed" / "estadistica"
DEFAULT_BAR_FILE = DEFAULT_OUTPUT_DIR / "barrios_top15.png"
DEFAULT_DONUT_FILE = DEFAULT_OUTPUT_DIR / "pedidos_por_ciudad.png"
DEFAULT_DISPERSION_FILE = DEFAULT_OUTPUT_DIR / "dispersion_geografica.png"

DPI = 300
LAT_CORRALON = -32.8495006
LON_CORRALON = -60.722653

PALETA_CIUDADES: dict[str, str] = {
    "Granadero Baigorria": "#2b5c8f",
    "Rosario": "#d95f02",
    "Funes": "#bc6c25",
    "Ibarlucea": "#2ca02c",
    "Roldan": "#4a6984",
    "Roldán": "#4a6984",
    "San Lorenzo": "#7570b3",
    "Capitan Bermudez": "#008080",
    "Capitán Bermúdez": "#008080",
    "Fray Luis Beltran": "#8c564b",
    "Fray Luis Beltrán": "#8c564b",
    "Otros": "#7f7f7f",
}

__all__ = [
    "DEFAULT_BAR_FILE",
    "DEFAULT_DISPERSION_FILE",
    "DEFAULT_DONUT_FILE",
    "DEFAULT_INPUT_FILE",
    "DEFAULT_OUTPUT_DIR",
    "DPI",
    "LAT_CORRALON",
    "LON_CORRALON",
    "PALETA_CIUDADES",
    "generar_grafico_dispersion_geografica",
    "generar_grafico_donut_ciudades",
    "generar_grafico_top_barrios",
    "generar_graficos_demanda",
    "main",
]


def _configurar_estilo_matplotlib() -> None:
    """Configura parámetros tipográficos y estéticos formales para publicaciones."""
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _abreviar_ciudad(ciudad: str) -> str:
    """Abrevia nombres de municipios para optimizar espacio en etiquetas."""
    abreviaturas = {
        "Granadero Baigorria": "G. Baigorria",
        "Capitán Bermúdez": "C. Bermúdez",
        "Capitan Bermudez": "C. Bermúdez",
        "Fray Luis Beltrán": "F. L. Beltrán",
        "Fray Luis Beltran": "F. L. Beltrán",
    }
    return abreviaturas.get(ciudad, ciudad)


def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Convierte coordenadas geográficas a índices de tesela en proyección Mercator."""
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Convierte índices de tesela Mercator a coordenadas geográficas (lat, lon)."""
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def _obtener_mapa_base_escala_grises(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int = 13,
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    """Descarga y ensambla teselas abiertas de OpenStreetMap en escala de grises.

    Utiliza el servidor público de OpenStreetMap sin requerir credenciales ni API keys.
    """
    try:
        x0, y1 = _deg2num(lat_min, lon_min, zoom)
        x1, y0 = _deg2num(lat_max, lon_max, zoom)

        headers = {"User-Agent": "PedemonteDigitalTwin/2.0 (OpenStreetMap academic use)"}
        width = (x1 - x0 + 1) * 256
        height = (y1 - y0 + 1) * 256
        stitched = Image.new("RGB", (width, height), (245, 245, 245))

        for xi, x in enumerate(range(x0, x1 + 1)):
            for yi, y in enumerate(range(y0, y1 + 1)):
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    tile = Image.open(io.BytesIO(resp.content))
                    stitched.paste(tile, (xi * 256, yi * 256))

        stitched_bw = ImageOps.grayscale(stitched)
        enhancer = ImageEnhance.Brightness(stitched_bw)
        stitched_bw_clarificado = np.array(enhancer.enhance(1.15))

        nw_lat, nw_lon = _num2deg(x0, y0, zoom)
        se_lat, se_lon = _num2deg(x1 + 1, y1 + 1, zoom)
        extent = (nw_lon, se_lon, se_lat, nw_lat)
        return stitched_bw_clarificado, extent
    except Exception:
        return None, None


def generar_grafico_top_barrios(
    df: pd.DataFrame,
    ruta_salida: Path = DEFAULT_BAR_FILE,
    top_n: int = 15,
) -> Path:
    """Genera y guarda el gráfico de barras horizontales para el Top N de barrios.

    Args:
        df: DataFrame con las columnas de ciudad y barrio.
        ruta_salida: Ruta del archivo PNG resultante.
        top_n: Cantidad de barrios principales a graficar.

    Returns:
        Ruta resuelta del archivo PNG generado.
    """
    _configurar_estilo_matplotlib()

    col_ciudad = "ciudad_detectada" if "ciudad_detectada" in df.columns else "ciudad"
    df_clean = df.copy()
    df_clean["barrio"] = df_clean["barrio"].astype(str).str.strip()
    df_clean[col_ciudad] = df_clean[col_ciudad].astype(str).str.strip()

    total_pedidos = len(df_clean)

    stats = (
        df_clean.groupby([col_ciudad, "barrio"])
        .size()
        .reset_index(name="cantidad_pedidos")
        .sort_values("cantidad_pedidos", ascending=False)
    )

    top = stats.head(top_n).copy()
    top["pct"] = (top["cantidad_pedidos"] / total_pedidos * 100) if total_pedidos > 0 else 0.0
    top["etiqueta"] = [
        f"{b} ({_abreviar_ciudad(c)})" for b, c in zip(top["barrio"], top[col_ciudad])
    ]
    colores_barra = [PALETA_CIUDADES.get(c, "#7f7f7f") for c in top[col_ciudad]]

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=DPI)

    y_pos = list(range(len(top) - 1, -1, -1))
    bars = ax.barh(
        y_pos,
        top["cantidad_pedidos"],
        color=colores_barra,
        height=0.68,
        edgecolor="#1f2937",
        linewidth=0.6,
        alpha=0.85,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["etiqueta"])

    max_val = top["cantidad_pedidos"].max() if not top.empty else 10
    for bar, val, pct in zip(bars, top["cantidad_pedidos"], top["pct"]):
        ax.text(
            bar.get_width() + (max_val * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#1f2937",
        )

    ax.set_xlabel("Cantidad de Pedidos Históricos [Unidades]")
    ax.set_ylabel("Barrio y Localidad de Destino")
    ax.set_title(f"Top {top_n} Barrios con Mayor Concentración de Demanda Histórica", pad=12)
    ax.set_xlim(0, max_val * 1.25)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)

    ciudades_presentes = list(dict.fromkeys(top[col_ciudad]))
    leyenda = [
        mpatches.Patch(
            facecolor=PALETA_CIUDADES.get(c, "#7f7f7f"),
            edgecolor="#1f2937",
            linewidth=0.6,
            alpha=0.85,
            label=c,
        )
        for c in ciudades_presentes
    ]
    ax.legend(
        handles=leyenda,
        title="Municipio",
        loc="lower right",
        framealpha=0.9,
        edgecolor="#d1d5db",
    )

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
    """Genera y guarda el gráfico donut formal agrupando localidades menores al umbral.

    Args:
        df: DataFrame con las columnas de ciudad detectada.
        ruta_salida: Ruta del archivo PNG resultante.
        umbral_pct: Porcentaje mínimo para figurar con porción independiente.

    Returns:
        Ruta resuelta del archivo PNG guardado.
    """
    _configurar_estilo_matplotlib()

    col_ciudad = "ciudad_detectada" if "ciudad_detectada" in df.columns else "ciudad"
    df_clean = df.copy()
    df_clean[col_ciudad] = df_clean[col_ciudad].astype(str).str.strip()

    por_ciudad = df_clean.groupby(col_ciudad).size().reset_index(name="n")
    total_pedidos = int(por_ciudad["n"].sum())
    por_ciudad["pct"] = (por_ciudad["n"] / total_pedidos * 100) if total_pedidos > 0 else 0.0

    principales = por_ciudad[por_ciudad["pct"] >= umbral_pct].copy()
    menores = por_ciudad[por_ciudad["pct"] < umbral_pct]
    otros_n = int(menores["n"].sum())
    cantidad_menores = len(menores)

    if otros_n > 0:
        etiqueta_otros = f"Otros ({cantidad_menores} localidades)"
        otros_df = pd.DataFrame(
            [
                {
                    col_ciudad: etiqueta_otros,
                    "n": otros_n,
                    "pct": (otros_n / total_pedidos * 100) if total_pedidos > 0 else 0.0,
                }
            ]
        )
        por_ciudad_graf = pd.concat([principales, otros_df], ignore_index=True)
    else:
        por_ciudad_graf = principales

    por_ciudad_graf = por_ciudad_graf.sort_values("n", ascending=False)
    colores_donut = [
        PALETA_CIUDADES.get(c, PALETA_CIUDADES["Otros"]) for c in por_ciudad_graf[col_ciudad]
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=DPI)
    res_pie = ax.pie(
        por_ciudad_graf["n"],
        colors=colores_donut,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5, antialiased=True),
    )
    wedges = res_pie[0]

    leyenda_donut = [
        mpatches.Patch(
            facecolor=wedge.get_facecolor(),
            edgecolor="#d1d5db",
            linewidth=0.6,
            label=f"{ciudad}: {n} ({pct:.1f}%)",
        )
        for wedge, ciudad, n, pct in zip(
            wedges,
            por_ciudad_graf[col_ciudad],
            por_ciudad_graf["n"],
            por_ciudad_graf["pct"],
        )
    ]

    ax.legend(
        handles=leyenda_donut,
        title="Municipio / Localidad",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.9,
        edgecolor="#d1d5db",
    )

    ax.text(
        0,
        0.06,
        f"N = {total_pedidos}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0,
        -0.08,
        "pedidos\n(100.0%)",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#4b5563",
    )

    ax.set_title(
        "Distribución de Demanda por Municipio en el Área Metropolitana",
        pad=14,
    )
    ax.axis("equal")

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Donut  -> {ruta_salida}")
    return ruta_salida


def generar_grafico_dispersion_geografica(
    df: pd.DataFrame,
    ruta_salida: Path = DEFAULT_DISPERSION_FILE,
    lat_depot: float = LAT_CORRALON,
    lon_depot: float = LON_CORRALON,
    distancia_max_km: float = 6.0,
) -> Path:
    """Genera y guarda el mapa de calor sobre base cartográfica en escala de grises.

    Filtra los pedidos lejanos para enfocar la visualización en la zona núcleo operativa,
    estimando zonas de densidad mediante KDE e identificando al depósito con un punto rojo.

    Args:
        df: DataFrame con latitud, longitud y ciudad_detectada.
        ruta_salida: Ruta del archivo PNG resultante.
        lat_depot: Latitud del depósito central (Corralón Pedemonte).
        lon_depot: Longitud del depósito central.
        distancia_max_km: Radio máximo en km para delimitar la zona núcleo.

    Returns:
        Ruta resuelta del archivo PNG guardado.
    """
    _configurar_estilo_matplotlib()

    col_ciudad = "ciudad_detectada" if "ciudad_detectada" in df.columns else "ciudad"
    df_clean = df.dropna(subset=["latitud", "longitud"]).copy()
    df_clean[col_ciudad] = df_clean[col_ciudad].astype(str).str.strip()

    km_por_grado_lat = 111.0
    km_por_grado_lon = 111.0 * float(np.cos(np.radians(lat_depot)))

    dlat_km = (df_clean["latitud"] - lat_depot) * km_por_grado_lat
    dlon_km = (df_clean["longitud"] - lon_depot) * km_por_grado_lon
    distancias_km = np.sqrt(dlat_km**2 + dlon_km**2)

    mascara_nucleo = distancias_km <= distancia_max_km
    df_core = df_clean[mascara_nucleo].copy()

    lat_min, lat_max = -32.890, -32.812
    lon_min, lon_max = -60.755, -60.680

    fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=DPI)

    mapa_bw, extent = _obtener_mapa_base_escala_grises(lat_min, lat_max, lon_min, lon_max, zoom=13)
    if mapa_bw is not None and extent is not None:
        ax.imshow(mapa_bw, extent=extent, cmap="gray", alpha=0.88, zorder=1)
    else:
        ax.set_facecolor("#f9fafb")
        ax.grid(True, linestyle=":", alpha=0.5, color="#d1d5db")

    grid_x = np.linspace(lon_min, lon_max, 180)
    grid_y = np.linspace(lat_min, lat_max, 180)
    x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)

    bw_x = float(np.std(df_core["longitud"]) * (len(df_core) ** (-1 / 6)) * 0.70)
    bw_y = float(np.std(df_core["latitud"]) * (len(df_core) ** (-1 / 6)) * 0.70)

    z_densidad = np.zeros_like(x_mesh)
    for xi, yi in zip(df_core["longitud"], df_core["latitud"]):
        dx = (x_mesh - xi) / bw_x
        dy = (y_mesh - yi) / bw_y
        z_densidad += np.exp(-0.5 * (dx**2 + dy**2))

    z_densidad /= 2.0 * np.pi * bw_x * bw_y * len(df_core)
    z_norm = (z_densidad - z_densidad.min()) / (z_densidad.max() - z_densidad.min())
    z_enmascarado = np.ma.masked_where(z_norm < 0.04, z_norm)

    cf = ax.contourf(
        x_mesh,
        y_mesh,
        z_enmascarado,
        levels=np.linspace(0.04, 1.0, 18),
        cmap="YlOrRd",
        alpha=0.62,
        zorder=2,
    )
    ax.contour(
        x_mesh,
        y_mesh,
        z_enmascarado,
        levels=np.linspace(0.08, 1.0, 6),
        colors="#991b1b",
        linewidths=0.5,
        alpha=0.40,
        zorder=3,
    )

    angulos = np.linspace(0, 2 * np.pi, 200)
    for r in [1, 2, 3, 5]:
        dlat_r = r / km_por_grado_lat
        dlon_r = r / km_por_grado_lon
        ax.plot(
            lon_depot + dlon_r * np.cos(angulos),
            lat_depot + dlat_r * np.sin(angulos),
            linestyle="--",
            color="#374151",
            linewidth=0.75,
            alpha=0.50,
            zorder=4,
        )
        ax.text(
            lon_depot - (dlon_r * 0.72),
            lat_depot - (dlat_r * 0.70),
            f"{r} km",
            fontsize=7.5,
            color="#374151",
            fontstyle="italic",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.80),
            zorder=5,
        )

    ax.scatter(
        lon_depot,
        lat_depot,
        color="#d90429",
        s=140,
        edgecolor="white",
        linewidth=1.8,
        zorder=10,
        label="Corralón Pedemonte (Depósito Central)",
    )
    ax.annotate(
        "Corralón Pedemonte\n(Depósito Central)",
        xy=(lon_depot, lat_depot),
        xytext=(lon_depot + 0.010, lat_depot + 0.008),
        arrowprops=dict(arrowstyle="->", color="#1f2937", lw=1.0),
        fontsize=8.5,
        fontweight="bold",
        color="#111827",
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d90429", lw=1.2, alpha=0.92
        ),
        zorder=12,
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_xlabel("Longitud Geográfica [°O]")
    ax.set_ylabel("Latitud Geográfica [°S]")
    ax.set_title("Mapa de Calor: Concentración Espacial de Demanda en Zona Núcleo", pad=12)

    cbar = fig.colorbar(cf, ax=ax, fraction=0.032, pad=0.03)
    cbar.set_label("Densidad Relativa de Pedidos [KDE]", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    elementos_leyenda = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d90429",
            markeredgecolor="white",
            markersize=8.5,
            label="Corralón Pedemonte",
        ),
        mpatches.Patch(
            facecolor="#f97316",
            edgecolor="#b91c1c",
            alpha=0.75,
            label="Zona de calor (demanda)",
        ),
    ]
    ax.legend(
        handles=elementos_leyenda,
        loc="lower left",
        framealpha=0.92,
        edgecolor="#d1d5db",
        fontsize=8.5,
    )

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Dispersión (Mapa de Calor) -> {ruta_salida}")
    return ruta_salida


def generar_graficos_demanda(
    archivo_entrada: Path = DEFAULT_INPUT_FILE,
    ruta_barras: Path = DEFAULT_BAR_FILE,
    ruta_donut: Path = DEFAULT_DONUT_FILE,
    ruta_dispersion: Path = DEFAULT_DISPERSION_FILE,
) -> tuple[Path, Path, Path]:
    """Genera el conjunto completo de gráficos a partir del CSV geolocalizado.

    Args:
        archivo_entrada: Ruta al CSV de direcciones geolocalizadas.
        ruta_barras: Ruta del PNG para barras horizontales.
        ruta_donut: Ruta del PNG para diagrama donut.
        ruta_dispersion: Ruta del PNG para mapa de dispersión espacial.

    Returns:
        Tupla `(ruta_barras, ruta_donut, ruta_dispersion)` con las rutas generadas.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de entrada.
    """
    if not archivo_entrada.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {archivo_entrada}")

    df = pd.read_csv(archivo_entrada)
    p_bar = generar_grafico_top_barrios(df, ruta_salida=ruta_barras)
    p_donut = generar_grafico_donut_ciudades(df, ruta_salida=ruta_donut)
    p_disp = generar_grafico_dispersion_geografica(df, ruta_salida=ruta_dispersion)
    return p_bar, p_donut, p_disp


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generación de gráficos de demanda histórica")
    p.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="CSV geolocalizado",
    )
    p.add_argument(
        "--bar-file",
        type=Path,
        default=DEFAULT_BAR_FILE,
        help="PNG de salida gráfico de barras",
    )
    p.add_argument(
        "--donut-file",
        type=Path,
        default=DEFAULT_DONUT_FILE,
        help="PNG de salida gráfico donut",
    )
    p.add_argument(
        "--dispersion-file",
        type=Path,
        default=DEFAULT_DISPERSION_FILE,
        help="PNG de salida mapa de dispersión espacial",
    )
    return p


def main() -> int:
    """Punto de entrada de línea de comandos para la generación de gráficos."""
    args = _parser().parse_args()
    try:
        generar_graficos_demanda(
            archivo_entrada=args.input_file.resolve(),
            ruta_barras=args.bar_file.resolve(),
            ruta_donut=args.donut_file.resolve(),
            ruta_dispersion=args.dispersion_file.resolve(),
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
