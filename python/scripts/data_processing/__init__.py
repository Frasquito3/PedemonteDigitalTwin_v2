from __future__ import annotations

from pathlib import Path

__doc__ = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

from scripts.data_processing import (
    estadistica_demanda,
    extraccion_ocr,
    generar_pedidos_dia,
    geocoding,
    graficos_demanda,
    run_pipeline_demanda,
)

__all__ = [
    "estadistica_demanda",
    "extraccion_ocr",
    "generar_pedidos_dia",
    "geocoding",
    "graficos_demanda",
    "run_pipeline_demanda",
]
