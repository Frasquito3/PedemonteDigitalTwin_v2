from __future__ import annotations

import argparse
import sys
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from planner.algorithms.ga import ConfiguracionGA
from planner.evaluation.rl_controlled_benchmark import (
    ConfiguracionBenchmarkRLControlado,
    MetadatosModeloRL,
    calcular_sha256_archivo,
)
from planner.evaluation.rl_stress_benchmark import (
    ejecutar_benchmark_rl_stress,
    escribir_resultados_benchmark_rl_stress,
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


def _imprimir_resumen(resultado) -> None:
    print("\n=== BENCHMARK RL STRESS COMPLETADO ===")
    print(f"Versión: {resultado.version_benchmark}")
    print(f"Objetivo: {resultado.controlado.version_objetivo}")
    print(
        "Proveedor: "
        f"{resultado.controlado.fuente_viaje} | "
        f"{resultado.controlado.version_viaje}"
    )
    print(f"Casos: {resultado.cantidad_casos}")
    print(f"Modelos: {resultado.cantidad_modelos}")
    print(f"Filas generadas: {resultado.cantidad_filas}")
    print(
        "MODELO | RL G/E/P vs GREEDY | RL G/E/P vs GA | "
        "HIBRIDO RL/GREEDY | GAP MEDIO G | GAP P90 G | PEOR G | GARANTIA"
    )
    for resumen in resultado.resumen_modelos:
        print(
            f"{resumen.modelo_alias} | "
            f"{resumen.rl_gana_greedy}/"
            f"{resumen.rl_empata_greedy}/"
            f"{resumen.rl_pierde_greedy} | "
            f"{resumen.rl_gana_ga}/"
            f"{resumen.rl_empata_ga}/"
            f"{resumen.rl_pierde_ga} | "
            f"{resumen.hibrido_fuente_rl}/"
            f"{resumen.hibrido_fuente_greedy} | "
            f"{_fmt(resumen.gap_rl_vs_greedy_promedio_pct)} | "
            f"{_fmt(resumen.gap_rl_vs_greedy_p90_pct)} | "
            f"{_fmt(resumen.gap_rl_vs_greedy_peor_pct)} | "
            f"{resumen.violaciones_garantia_hibrida == 0}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa modelos RL sobre una batería stress estratificada "
            "que usa exclusivamente los nodos de la caché vial validada."
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
        default="results/benchmark_rl/15R_6B_stress",
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
    metadatos = {
        alias: MetadatosModeloRL(
            alias=alias,
            ruta_modelo=str(ruta),
            sha256=calcular_sha256_archivo(ruta),
        )
        for alias, ruta in modelos.items()
    }

    resultado = ejecutar_benchmark_rl_stress(
        casos,
        proveedor_viaje=proveedor,
        planners_rl=planners_rl,
        metadatos_modelos=metadatos,
        configuracion_benchmark=ConfiguracionBenchmarkRLControlado(
            configuracion_ga=ConfiguracionGA(),
            seed_ga=101,
        ),
    )
    rutas = escribir_resultados_benchmark_rl_stress(
        resultado,
        args.salida,
    )

    _imprimir_resumen(resultado)
    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"  {nombre}: {ruta}")


if __name__ == "__main__":
    main()
