from __future__ import annotations

import argparse
import sys

from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from planner.evaluation.comparison_service_audit import (  # noqa: E402
    auditar_suite_servicio,
    cargar_suite_cruda,
    escribir_auditoria_servicio,
)
from planner.routing.vial_cache import (  # noqa: E402
    ProveedorVialCachePersistente,
)


DEFAULT_16C = (
    PYTHON_ROOT
    / "results"
    / "comparison_anylogic"
    / "16C"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reprocesa la suite 16C sin repetir AnyLogic, separa estado "
            "técnico y servicio, y audita las ventanas horarias."
        )
    )
    parser.add_argument(
        "--suite",
        default=str(DEFAULT_16C / "comparison_suite.json"),
        help="JSON crudo generado por la suite 16C.",
    )
    parser.add_argument(
        "--contracts",
        default=str(DEFAULT_16C / "contracts"),
        help="Directorio que contiene los contratos por caso.",
    )
    parser.add_argument(
        "--cache",
        default=str(PYTHON_ROOT / "data" / "routing" / "cache_vial_v1.csv"),
        help="Caché vial estricta usada por la auditoría temporal.",
    )
    parser.add_argument(
        "--cache-version",
        default="pedemonte-vial-v1",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_16C / "service_audit"),
        help="Directorio de salida de la auditoría 16C.2.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ruta_suite = Path(args.suite).expanduser().resolve()
    suite = cargar_suite_cruda(ruta_suite)
    proveedor = ProveedorVialCachePersistente(
        args.cache,
        version_cache_esperada=args.cache_version,
        permitir_fallback=False,
    )

    resultado = auditar_suite_servicio(
        suite,
        suite_origen=str(ruta_suite),
        contratos_dir=args.contracts,
        proveedor_viaje=proveedor,
    )
    rutas = escribir_auditoria_servicio(resultado, args.output)

    print("=== AUDITORÍA DE SERVICIO 16C.2 ===")
    print(
        "Ejecuciones técnicas OK: "
        f"{resultado.ejecuciones_tecnicas_ok}/"
        f"{resultado.corridas_esperadas}"
    )
    print(f"Servicios completos: {resultado.servicios_completos}")
    print(f"Servicios incompletos: {resultado.servicios_incompletos}")
    print(f"Servicios con error: {resultado.servicios_error}")
    print(
        "Tasa de completitud global: "
        f"{resultado.tasa_completitud_global_pct:.2f}%"
    )

    print("\nCasos:")
    for caso in resultado.casos:
        ranking_rl = caso.ranking_rl if caso.ranking_rl is not None else "—"
        nivel_rl = (
            "—"
            if caso.nivel_servicio_rl_pct is None
            else f"{caso.nivel_servicio_rl_pct:.2f}%"
        )
        print(
            f"{caso.caso_id} | completos={caso.servicios_completos}/5 | "
            f"RL={caso.estado_servicio_rl} | servicio_RL={nivel_rl} | "
            f"ranking_RL={ranking_rl}"
        )

    print("\nAuditoría de ventanas:")
    for resumen in resultado.resumen_planes_ventanas:
        riesgos = (
            ",".join(resumen.pedidos_riesgo_rechazo)
            if resumen.pedidos_riesgo_rechazo
            else "NINGUNO"
        )
        print(
            f"{resumen.modo_solicitado} | "
            f"servicio={resumen.estado_servicio_real} | "
            f"tardías={resumen.llegadas_tardias_estimadas} | "
            f"riesgo={riesgos}"
        )

    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"{nombre}: {ruta}")

    return 0 if resultado.servicios_error == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
