from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from planner.evaluation.classic_instances import (  # noqa: E402
    crear_casos_benchmark_clasico,
)
from planner.evaluation.comparison_contract import (  # noqa: E402
    escribir_contrato_comparacion,
    preparar_contrato_comparacion,
)
from planner.integration.planner_selector import (  # noqa: E402
    SelectorPlanificadores,
)
from planner.routing.vial_cache import (  # noqa: E402
    ProveedorVialCachePersistente,
)


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara el contrato reproducible de cinco alternativas "
            "para la comparación Python-AnyLogic de la Fase 16A."
        )
    )
    parser.add_argument(
        "--case",
        dest="caso_id",
        default="B05_VOLCADOR",
        help=(
            "Caso controlado. Acepta B01..B06 o el nombre canónico. "
            "Por defecto: B05_VOLCADOR."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PYTHON_ROOT
            / "models"
            / "rl"
            / "pedemonte_maskable_ppo.zip"
        ),
        help="Ruta del modelo RL histórico.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=(
            PYTHON_ROOT
            / "data"
            / "routing"
            / "cache_vial_v1.csv"
        ),
        help="Ruta de la caché vial compartida.",
    )
    parser.add_argument(
        "--cache-version",
        default="pedemonte-vial-v1",
    )
    parser.add_argument(
        "--seed-execution",
        type=int,
        default=None,
        help=(
            "Sobrescribe seed_ejecucion manteniendo intacta "
            "seed_escenario."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PYTHON_ROOT
            / "results"
            / "comparison_contract"
        ),
    )
    return parser


def main() -> int:
    args = _crear_parser().parse_args()

    casos_lista = crear_casos_benchmark_clasico()
    casos = {
        caso.caso_id.upper(): caso
        for caso in casos_lista
    }
    alias = {
        caso.caso_id.split("_", 1)[0].upper(): caso.caso_id.upper()
        for caso in casos_lista
    }
    caso_solicitado = args.caso_id.strip().upper()
    caso_id = alias.get(caso_solicitado, caso_solicitado)
    if caso_id not in casos:
        disponibles = ", ".join(sorted(casos))
        raise ValueError(
            f"Caso inexistente: {caso_solicitado}. "
            f"Disponibles: {disponibles}."
        )

    instancia = casos[caso_id].instancia
    if args.seed_execution is not None:
        instancia = replace(
            instancia,
            seed_ejecucion=args.seed_execution,
        )

    proveedor = ProveedorVialCachePersistente(
        args.cache,
        version_cache_esperada=args.cache_version,
        permitir_fallback=False,
    )
    selector = SelectorPlanificadores(
        model_path_rl=args.model,
        proveedor_viaje=proveedor,
        deterministic=True,
    )
    selector.precargar_rl()

    contrato = preparar_contrato_comparacion(
        instancia,
        selector=selector,
        proveedor_viaje=proveedor,
    )
    rutas = escribir_contrato_comparacion(
        contrato,
        args.output,
    )

    print("=== CONTRATO COMPARACIÓN 16A ===")
    print(f"Instancia: {contrato.instancia_id}")
    print(f"Seed escenario: {contrato.seed_escenario}")
    print(f"Seed ejecución: {contrato.seed_ejecucion}")
    print(f"Proveedor: {contrato.version_viaje}")
    print(f"Planes OK: {contrato.planes_ok}/5")

    for registro in contrato.planes:
        costo = (
            "—"
            if registro.costo_estimado is None
            else f"{registro.costo_estimado:.6f}"
        )
        print(
            f"{registro.orden}. {registro.modo_solicitado} | "
            f"estado={registro.estado} | "
            f"resultado={registro.algoritmo_resultante or '—'} | "
            f"costo={costo}"
        )

    for nombre, ruta in rutas.items():
        print(f"{nombre}: {ruta}")

    return 0 if contrato.planes_error == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
