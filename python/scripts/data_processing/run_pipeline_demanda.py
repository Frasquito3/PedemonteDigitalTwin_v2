"""Script orquestador del pipeline completo de demanda histórica de pedidos.

Permite ejecutar de forma secuencial o modular las cinco etapas del pipeline de datos:
1. `ocr`: Extracción multimodal de direcciones manuscritas mediante Google GenAI.
2. `geocoding`: Geolocalización y normalización con OpenStreetMap (Nominatim).
3. `estadistica`: Estimación de frecuencias empíricas y pesos por barrio y ciudad.
4. `graficos`: Generación de gráficos analíticos (barras Top N y donut por municipio).
5. `generar`: Muestreo estocástico de turnos y pedidos sintéticos para benchmarking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

__all__ = ["main"]


def main() -> int:
    """Ejecuta el pipeline completo o una etapa específica según argumentos de CLI.

    Returns:
        0 si la ejecución fue exitosa, 1 en caso de error no controlado.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline completo de demanda histórica de pedidos"
    )
    parser.add_argument(
        "--etapa",
        choices=["all", "ocr", "geocoding", "estadistica", "graficos", "generar"],
        default="all",
        help="Etapa a ejecutar (por defecto: all)",
    )
    parser.add_argument(
        "-n",
        "--dias",
        type=int,
        default=1,
        help="Cantidad de días a generar cuando se ejecuta la etapa 'generar' (por defecto: 1)",
    )
    parser.add_argument(
        "-p",
        "--pedidos-por-dia",
        type=str,
        default="8-12",
        help="Rango o cantidad fija de pedidos por día (por defecto: 8-12)",
    )
    args = parser.parse_args()

    print(f"=== Iniciando Pipeline de Demanda Histórica [Etapa: {args.etapa}] ===")

    try:
        if args.etapa in ("all", "ocr"):
            print("\n--- 1. Extracción OCR (Google AI Studio) ---")
            from scripts.data_processing.extraccion_ocr import (
                ejecutar_pipeline_extraccion,
            )

            ejecutar_pipeline_extraccion()

        if args.etapa in ("all", "geocoding"):
            print("\n--- 2. Geocodificación y normalización (Nominatim / OSM) ---")
            from scripts.data_processing.geocoding import ejecutar_geocodificacion

            ejecutar_geocodificacion()

        if args.etapa in ("all", "estadistica"):
            print("\n--- 3. Cálculo de distribución estadística de demanda ---")
            from scripts.data_processing.estadistica_demanda import (
                procesar_estadisticas_demanda,
            )

            procesar_estadisticas_demanda()

        if args.etapa in ("all", "graficos"):
            print("\n--- 4. Generación de gráficos y visualizaciones EDA ---")
            from scripts.data_processing.graficos_demanda import (
                generar_graficos_demanda,
            )

            generar_graficos_demanda()

        if args.etapa in ("all", "generar"):
            print(f"\n--- 5. Generación estocástica de nuevos pedidos ({args.dias} días) ---")
            from scripts.data_processing.generar_pedidos_dia import (
                ejecutar_generacion,
                parsear_rango_pedidos,
            )

            rango = parsear_rango_pedidos(args.pedidos_por_dia)
            ejecutar_generacion(dias_n=args.dias, rango_pedidos=rango)

        print("\n=== Pipeline ejecutado exitosamente ===")
        return 0

    except Exception as exc:
        print(f"\n[ERROR EN PIPELINE] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
