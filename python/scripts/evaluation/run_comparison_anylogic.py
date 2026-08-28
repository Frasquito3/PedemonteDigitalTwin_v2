from __future__ import annotations

import argparse
import sys

from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[2]

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


from planner.evaluation.comparison_execution import (
    ConfiguracionEjecucionComparacion,
    cargar_contrato_comparacion,
    escribir_resultado_ejecucion_comparacion,
    ejecutar_contrato_comparacion,
)
from planner.integration.anylogic_vector_client import (
    AnyLogicVectorClient,
)


DEFAULT_CONTRACT = (
    PYTHON_ROOT
    / "results"
    / "comparison_contract"
    / "comparison_contract.json"
)

DEFAULT_MODEL = (
    PYTHON_ROOT
    / "anylogic_export"
    / "comparison"
    / "PedemonteDigitalTwin_v2.zip"
)

DEFAULT_OUTPUT = (
    PYTHON_ROOT
    / "results"
    / "comparison_anylogic"
    / "16B"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta en AnyLogic los cinco planes del contrato 16A "
            "usando la misma seed de ejecución."
        )
    )

    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT),
        help="Ruta a comparison_contract.json.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="ZIP exportado desde AlpyneExperiment.",
    )
    parser.add_argument(
        "--java",
        default="",
        help="Ruta opcional a java.exe.",
    )
    parser.add_argument(
        "--python-root",
        default=str(PYTHON_ROOT),
        help=(
            "Carpeta python del proyecto. Se envía al modelo exportado "
            "para localizar la caché vial durante la ejecución."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directorio de salida para JSON y CSV.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout por corrida en segundos.",
    )
    parser.add_argument(
        "--max-server-await",
        type=float,
        default=45.0,
        help="Espera máxima de arranque del servidor Alpyne.",
    )
    parser.add_argument(
        "--ple-limit",
        type=float,
        default=300.0,
        help=(
            "Límite preventivo de tiempo simulado por corrida, "
            "en minutos."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Detiene el lote ante el primer error.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Desactiva logs detallados de Alpyne y Java.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    contrato = cargar_contrato_comparacion(args.contract)

    cliente = AnyLogicVectorClient(
        model_path=args.model,
        java_exe=(args.java if args.java.strip() else None),
        python_root=args.python_root,
        timeout_segundos=args.timeout,
        max_server_await_time=args.max_server_await,
        limite_ple_min=args.ple_limit,
        log_id_base="comparison",
        habilitar_logs=not args.no_logs,
    )

    resultado = ejecutar_contrato_comparacion(
        contrato,
        ejecutor=cliente,
        configuracion=ConfiguracionEjecucionComparacion(
            continuar_ante_error=not args.fail_fast,
            exigir_cinco_planes_ok=True,
            exigir_orden_rl_primero=True,
        ),
    )

    rutas = escribir_resultado_ejecucion_comparacion(
        resultado,
        args.output,
    )

    print("=== EJECUCIÓN COMPARABLE ANYLOGIC 16B ===")
    print(f"Instancia: {resultado.instancia_id}")
    print(f"Seed escenario: {resultado.seed_escenario}")
    print(f"Seed ejecución común: {resultado.seed_ejecucion}")
    print("Proceso nuevo por plan: SI")
    print(
        f"Ejecuciones OK: {resultado.ejecuciones_ok}/"
        f"{len(resultado.registros)}"
    )

    for registro in resultado.registros:
        costo_estimado = (
            "—"
            if registro.costo_estimado is None
            else f"{registro.costo_estimado:.6f}"
        )
        costo_real = (
            "—"
            if registro.costo_real is None
            else f"{registro.costo_real:.6f}"
        )
        duracion = (
            "—"
            if registro.tiempo_simulado_min is None
            else f"{registro.tiempo_simulado_min:.6f}"
        )

        print(
            f"{registro.orden}. {registro.modo_solicitado} | "
            f"estado={registro.estado_ejecucion} | "
            f"resultado={registro.algoritmo_resultante} | "
            f"estimado={costo_estimado} | "
            f"real={costo_real} | "
            f"tiempo_sim={duracion}"
        )

        if registro.error_ejecucion:
            print(f"   ERROR: {registro.error_ejecucion}")

    for nombre, ruta in rutas.items():
        print(f"{nombre}: {ruta}")

    return 0 if resultado.ejecuciones_error == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
