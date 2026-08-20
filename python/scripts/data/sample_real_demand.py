from __future__ import annotations

import argparse
from pathlib import Path

from planner.data.real_demand import (
    CatalogoDemandaReal,
)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Carga el dataset geográfico real y obtiene "
            "una muestra reproducible de ubicaciones."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset procesado de demanda real.",
    )

    parser.add_argument(
        "--cantidad",
        type=int,
        required=True,
        help="Cantidad de ubicaciones a seleccionar.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Semilla utilizada para el muestreo.",
    )

    parser.add_argument(
        "--con-reemplazo",
        action="store_true",
        help=(
            "Permite seleccionar varias veces la misma "
            "fila del dataset."
        ),
    )

    return parser


def main() -> None:
    argumentos = crear_parser().parse_args()

    catalogo = CatalogoDemandaReal.desde_csv(
        argumentos.dataset
    )

    muestra = catalogo.muestrear_con_seed(
        cantidad=argumentos.cantidad,
        seed=argumentos.seed,
        con_reemplazo=argumentos.con_reemplazo,
    )

    print("CATÁLOGO DE DEMANDA REAL")
    print(f"Dataset: {argumentos.dataset}")
    print(f"Registros aptos: {len(catalogo)}")
    print(
        "Direcciones fuente únicas: "
        f"{catalogo.cantidad_direcciones_fuente_unicas()}"
    )
    print()

    print("MUESTRA DETERMINISTA")
    print(f"Seed: {argumentos.seed}")
    print(f"Cantidad: {len(muestra)}")
    print(
        "Con reemplazo: "
        f"{argumentos.con_reemplazo}"
    )
    print()

    for indice, punto in enumerate(
        muestra,
        start=1,
    ):
        print(
            f"{indice:02d}. "
            f"{punto.registro_id} | "
            f"{punto.direccion_corta} | "
            f"barrio={punto.barrio} | "
            f"lat={punto.latitud:.7f} | "
            f"lon={punto.longitud:.7f} | "
            f"distancia={punto.distancia_corralon_km:.3f} km"
        )

    print()
    print("DISTRIBUCIÓN COMPLETA POR CIUDAD")

    for ciudad, cantidad in (
        catalogo.conteo_por_ciudad().items()
    ):
        porcentaje = (
            100.0
            * cantidad
            / len(catalogo)
        )

        print(
            f"{ciudad}: "
            f"{cantidad} "
            f"({porcentaje:.2f} %)"
        )


if __name__ == "__main__":
    main()