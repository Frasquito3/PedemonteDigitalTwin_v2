from __future__ import annotations

import argparse
import sys
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from planner.evaluation.robust_hybrid_benchmark import (
    ejecutar_benchmark_hibrido_robusto,
    escribir_resultados_benchmark_hibrido_robusto,
)
from planner.evaluation.rl_stress_instances import crear_casos_stress_rl
from planner.rl.planner import RLPlanner
from planner.routing.vial_cache import ProveedorVialCachePersistente


def _parsear_modelo(texto: str) -> tuple[str, Path]:
    if "=" not in texto:
        raise ValueError("Cada --modelo debe usar ALIAS=RUTA.")

    alias, ruta_texto = texto.split("=", 1)
    alias = alias.strip().upper()
    ruta_texto = ruta_texto.strip()

    if not alias or not ruta_texto:
        raise ValueError("Alias y ruta del modelo no pueden estar vacíos.")

    return alias, Path(ruta_texto).expanduser().resolve()


def _fmt(valor: float | None) -> str:
    return "-" if valor is None else f"{valor:+.3f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa el modo HIBRIDO robusto que selecciona el menor costo "
            "entre GREEDY, GA y RL sobre la batería stress estratificada."
        )
    )
    parser.add_argument(
        "--modelo",
        action="append",
        default=None,
        help="Modelo ALIAS=RUTA. Puede repetirse.",
    )
    parser.add_argument(
        "--cantidad-por-estrato",
        type=int,
        default=12,
    )
    parser.add_argument("--seed-base", type=int, default=16_600)
    parser.add_argument(
        "--cache-vial",
        default="data/routing/cache_vial_v1.csv",
    )
    parser.add_argument(
        "--version-cache",
        default="pedemonte-vial-v1",
    )
    parser.add_argument(
        "--salida",
        default="results/benchmark_rl/15R_6C_hibrido_robusto",
    )
    parser.add_argument("--max-pedidos", type=int, default=30)
    args = parser.parse_args()

    if args.cantidad_por_estrato <= 0:
        parser.error("--cantidad-por-estrato debe ser > 0.")
    if args.max_pedidos <= 0:
        parser.error("--max-pedidos debe ser > 0.")

    especificaciones = args.modelo or [
        "HISTORICO=models/rl/pedemonte_maskable_ppo.zip",
        "REAL_V2=models/rl/pedemonte_maskable_ppo_real_v2.zip",
    ]

    modelos: dict[str, Path] = {}
    for texto in especificaciones:
        alias, ruta = _parsear_modelo(texto)
        if alias in modelos:
            raise ValueError(f"Alias duplicado: {alias}")
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo {alias}: {ruta}")
        modelos[alias] = ruta

    proveedor = ProveedorVialCachePersistente(
        args.cache_vial,
        version_cache_esperada=args.version_cache,
        permitir_fallback=False,
    )
    casos = crear_casos_stress_rl(
        cantidad_por_estrato=args.cantidad_por_estrato,
        seed_base=args.seed_base,
    )
    planners_rl = {
        alias: RLPlanner(
            model_path=ruta,
            proveedor_viaje=proveedor,
            max_pedidos=args.max_pedidos,
            deterministic=True,
        )
        for alias, ruta in modelos.items()
    }

    resultado = ejecutar_benchmark_hibrido_robusto(
        casos,
        proveedor_viaje=proveedor,
        planners_rl=planners_rl,
    )
    rutas = escribir_resultados_benchmark_hibrido_robusto(
        resultado,
        args.salida,
    )

    print("\n=== BENCHMARK HÍBRIDO ROBUSTO COMPLETADO ===")
    print(f"Versión: {resultado.version_benchmark}")
    print(f"Objetivo: {resultado.version_objetivo}")
    print(
        "Proveedor: "
        f"{resultado.fuente_viaje} | {resultado.version_viaje}"
    )
    print(f"Casos: {resultado.cantidad_casos}")
    print(f"Modelos: {resultado.cantidad_modelos}")
    print(f"Filas generadas: {resultado.cantidad_filas}")
    print(
        "MODELO | FUENTE G/GA/RL | MEJORA MEDIA G | "
        "MEJORA MEDIA GA | TIEMPO P90 MS | GARANTÍAS"
    )
    for resumen in resultado.resumenes:
        garantias = (
            resumen.violaciones_garantia_greedy == 0
            and resumen.violaciones_garantia_ga == 0
        )
        print(
            f"{resumen.modelo_alias} | "
            f"{resumen.fuente_greedy}/"
            f"{resumen.fuente_ga}/"
            f"{resumen.fuente_rl} | "
            f"{_fmt(resumen.mejora_media_vs_greedy_pct)} | "
            f"{_fmt(resumen.mejora_media_vs_ga_pct)} | "
            f"{resumen.tiempo_p90_ms:.3f} | "
            f"{garantias}"
        )

    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"  {nombre}: {ruta}")


if __name__ == "__main__":
    main()
