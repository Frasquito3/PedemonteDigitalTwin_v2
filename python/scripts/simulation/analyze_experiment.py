"""CLI para auditar y regenerar tablas de experimentos guardados.

Uso: python -m scripts.simulation.analyze_experiment --directorio <carpeta_seed>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.simulation.experiment_artifacts import generar_analisis


def main() -> int:
    """Analiza una carpeta existente sin volver a ejecutar AnyLogic."""
    parser = argparse.ArgumentParser(description="Audita un experimento Pedemonte ya guardado.")
    parser.add_argument("--directorio", type=Path, required=True)
    args = parser.parse_args()
    auditoria = generar_analisis(args.directorio.expanduser().resolve())
    print(json.dumps(auditoria, ensure_ascii=False, indent=2))
    print(f"ANALISIS | {args.directorio / 'analisis'}")
    if (
        auditoria["duplicados"]
        or auditoria["faltantes"]
        or auditoria["archivos_faltantes"]
        or auditoria["identificadores_inesperados"]
    ):
        return 1
    # Si hay errores registrados, no ocultarlos detrás de un código de salida 0.
    return 1 if auditoria["corridas_invalidas"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
