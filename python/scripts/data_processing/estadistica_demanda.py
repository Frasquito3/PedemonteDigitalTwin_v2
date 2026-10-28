from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PYTHON_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = PYTHON_ROOT / "data" / "processed" / "geocoding" / "direcciones_geolocalizadas.csv"
DEFAULT_OUTPUT_DIR = PYTHON_ROOT / "data" / "processed" / "estadistica"
DEFAULT_PROBABILIDADES_FILE = DEFAULT_OUTPUT_DIR / "probabilidades_demanda.csv"
DEFAULT_RESUMEN_FILE = DEFAULT_OUTPUT_DIR / "resumen_ciudades.csv"


def calcular_probabilidades_demanda(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las frecuencias y pesos relativos de demanda agrupados por ciudad y barrio."""
    df_clean = df.copy()
    if "ciudad_detectada" in df_clean.columns and "ciudad" not in df_clean.columns:
        df_clean["ciudad"] = df_clean["ciudad_detectada"]

    df_clean["barrio"] = df_clean["barrio"].astype(str).str.strip()
    df_clean["ciudad"] = df_clean["ciudad"].astype(str).str.strip()

    estadisticas = (
        df_clean.groupby(["ciudad", "barrio"])
        .size()
        .reset_index(name="cantidad_pedidos")
    )

    total_pedidos = int(estadisticas["cantidad_pedidos"].sum())
    if total_pedidos > 0:
        estadisticas["probabilidad_demanda"] = (
            estadisticas["cantidad_pedidos"] / total_pedidos
        ).round(4)
    else:
        estadisticas["probabilidad_demanda"] = 0.0

    return estadisticas.sort_values(by="cantidad_pedidos", ascending=False)


def calcular_resumen_ciudades(df: pd.DataFrame) -> pd.DataFrame:
    """Genera el resumen porcentual de pedidos agrupados únicamente por ciudad."""
    df_clean = df.copy()
    if "ciudad_detectada" in df_clean.columns and "ciudad" not in df_clean.columns:
        df_clean["ciudad"] = df_clean["ciudad_detectada"]

    df_clean["ciudad"] = df_clean["ciudad"].astype(str).str.strip()

    resumen = (
        df_clean.groupby("ciudad")
        .size()
        .reset_index(name="cantidad_pedidos")
        .sort_values(by="cantidad_pedidos", ascending=False)
    )

    total_pedidos = int(resumen["cantidad_pedidos"].sum())
    if total_pedidos > 0:
        resumen["porcentaje"] = (resumen["cantidad_pedidos"] / total_pedidos * 100).round(2)
    else:
        resumen["porcentaje"] = 0.0

    return resumen


def procesar_estadisticas_demanda(
    archivo_entrada: Path = DEFAULT_INPUT_FILE,
    archivo_probabilidades: Path = DEFAULT_PROBABILIDADES_FILE,
    archivo_resumen: Path = DEFAULT_RESUMEN_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee el dataset geolocalizado, genera las estadísticas y las guarda en CSV."""
    if not archivo_entrada.exists():
        raise FileNotFoundError(f"No se encontró el archivo de direcciones: {archivo_entrada}")

    df = pd.read_csv(archivo_entrada)

    estadisticas = calcular_probabilidades_demanda(df)
    resumen = calcular_resumen_ciudades(df)

    print("--- Estadísticas de Pedidos por Ciudad y Barrio (Top 15) ---")
    print(estadisticas.head(15).to_string(index=False))

    print("\n--- Resumen por Ciudad ---")
    print(resumen.to_string(index=False))

    archivo_probabilidades.parent.mkdir(parents=True, exist_ok=True)
    archivo_resumen.parent.mkdir(parents=True, exist_ok=True)

    estadisticas.to_csv(archivo_probabilidades, index=False, encoding="utf-8-sig")
    resumen.to_csv(archivo_resumen, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Estadísticas exportadas:")
    print(f"     Probabilidades: {archivo_probabilidades}")
    print(f"     Resumen:        {archivo_resumen}")

    return estadisticas, resumen


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cálculo de estadísticas de demanda por ciudad y barrio")
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="CSV de direcciones geolocalizadas")
    p.add_argument("--prob-output", type=Path, default=DEFAULT_PROBABILIDADES_FILE, help="CSV de salida de probabilidades")
    p.add_argument("--resumen-output", type=Path, default=DEFAULT_RESUMEN_FILE, help="CSV de salida de resumen de ciudades")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        procesar_estadisticas_demanda(
            archivo_entrada=args.input_file.resolve(),
            archivo_probabilidades=args.prob_output.resolve(),
            archivo_resumen=args.resumen_output.resolve(),
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
