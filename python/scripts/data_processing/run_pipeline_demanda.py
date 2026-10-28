from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from scripts.data_processing.estadistica_demanda import procesar_estadisticas_demanda
from scripts.data_processing.extraccion_ocr import ejecutar_pipeline_extraccion
from scripts.data_processing.geocoding import ejecutar_geocodificacion
from scripts.data_processing.graficos_demanda import generar_graficos_demanda


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline completo de demanda histórica de pedidos")
    parser.add_argument(
        "--etapa",
        choices=["all", "ocr", "geocoding", "estadistica", "graficos"],
        default="all",
        help="Etapa a ejecutar (por defecto: all)",
    )
    args = parser.parse_args()

    print(f"=== Iniciando Pipeline de Demanda Histórica [Etapa: {args.etapa}] ===")

    try:
        if args.etapa in ("all", "ocr"):
            print("\n--- 1. Extracción OCR (Google AI Studio) ---")
            ejecutar_pipeline_extraccion()

        if args.etapa in ("all", "geocoding"):
            print("\n--- 2. Geocodificación y normalización (Nominatim / OSM) ---")
            ejecutar_geocodificacion()

        if args.etapa in ("all", "estadistica"):
            print("\n--- 3. Cálculo de distribución estadística de demanda ---")
            procesar_estadisticas_demanda()

        if args.etapa in ("all", "graficos"):
            print("\n--- 4. Generación de gráficos y visualizaciones EDA ---")
            generar_graficos_demanda()

        print("\n=== Pipeline ejecutado exitosamente ===")
        return 0

    except Exception as exc:
        print(f"\n[ERROR EN PIPELINE] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
