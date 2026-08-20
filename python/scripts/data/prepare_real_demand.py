from __future__ import annotations

import argparse
from pathlib import Path

from planner.data.demand_preparation import (
    DISTANCIA_MAX_ENTRENAMIENTO_KM,
    preparar_dataset,
)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara y versiona el dataset geográfico real "
            "de Pedemonte."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV geolocalizado original.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV procesado de salida.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        required=True,
        help="CSV con registros que requieren revisión.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        required=True,
        help="JSON de metadatos del dataset.",
    )
    parser.add_argument(
        "--max-training-distance-km",
        type=float,
        default=DISTANCIA_MAX_ENTRENAMIENTO_KM,
        help=(
            "Distancia máxima al corralón para marcar un "
            "registro como apto para entrenamiento."
        ),
    )

    return parser


def main() -> None:
    argumentos = crear_parser().parse_args()

    metadata = preparar_dataset(
        ruta_entrada=argumentos.input,
        ruta_dataset=argumentos.output,
        ruta_revision=argumentos.review_output,
        ruta_metadata=argumentos.metadata_output,
        distancia_max_entrenamiento_km=(
            argumentos.max_training_distance_km
        ),
    )

    conteos = metadata["conteos"]
    if not isinstance(conteos, dict):
        raise TypeError(
            "metadata['conteos'] debe ser un diccionario."
        )

    print("DATASET DEMANDA REAL PREPARADO")
    print(
        "Registros totales: "
        f"{conteos['registros_totales']}"
    )
    print(
        "Aptos para entrenamiento: "
        f"{conteos['registros_aptos_entrenamiento']}"
    )
    print(
        "En revisión: "
        f"{conteos['registros_revision']}"
    )


if __name__ == "__main__":
    main()