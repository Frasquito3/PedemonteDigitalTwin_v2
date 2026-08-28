from __future__ import annotations

import argparse
import sys
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from planner.algorithms.ga import ConfiguracionGA
from planner.evaluation.classic_benchmark import (
    ConfiguracionBenchmarkClasico,
    ejecutar_benchmark_clasico,
    escribir_resultados_benchmark,
)
from planner.evaluation.classic_instances import (
    CasoBenchmarkClasico,
    crear_casos_benchmark_clasico,
)
from planner.routing.vial_cache import ProveedorVialCachePersistente


def _crear_configuracion_perfil(
    perfil: str,
) -> ConfiguracionBenchmarkClasico:
    if perfil == "rapido":
        return ConfiguracionBenchmarkClasico(
            seeds_estocasticas=(101, 211),
            configuracion_ga=ConfiguracionGA(
                tamano_poblacion=16,
                generaciones=20,
                tamano_elite=2,
                tamano_torneo=3,
                probabilidad_crossover=0.90,
                probabilidad_mutacion_swap=0.20,
                probabilidad_mutacion_inversion=0.10,
                generaciones_sin_mejora_max=8,
            ),
        )

    if perfil == "formal":
        return ConfiguracionBenchmarkClasico(
            seeds_estocasticas=(101, 211, 307, 401, 503),
            configuracion_ga=ConfiguracionGA(),
        )

    raise ValueError(f"Perfil desconocido: {perfil}")


def _seleccionar_casos(
    casos: tuple[CasoBenchmarkClasico, ...],
    seleccion: str,
) -> tuple[CasoBenchmarkClasico, ...]:
    texto = seleccion.strip()

    if not texto or texto.upper() == "TODAS":
        return casos

    ids_solicitados = {
        parte.strip().upper()
        for parte in texto.split(",")
        if parte.strip()
    }
    por_id = {
        caso.caso_id.upper(): caso
        for caso in casos
    }
    faltantes = sorted(ids_solicitados - set(por_id))

    if faltantes:
        raise ValueError(
            "Casos inexistentes: " + ", ".join(faltantes)
        )

    return tuple(
        caso
        for caso in casos
        if caso.caso_id.upper() in ids_solicitados
    )


def _imprimir_resumen(resultado) -> None:
    print("\n=== BENCHMARK CLÁSICO COMPLETADO ===")
    print(f"Versión: {resultado.version_benchmark}")
    print(f"Objetivo: {resultado.version_objetivo}")
    print(
        "Proveedor: "
        f"{resultado.fuente_viaje} | {resultado.version_viaje}"
    )
    print(f"Corridas aceptadas: {len(resultado.corridas)}")

    encabezado = (
        "CASO",
        "ALGORITMO",
        "N",
        "COSTO MEDIO",
        "MEJOR",
        "Δ% GREEDY",
        "FIRMAS",
        "MS MEDIO",
    )
    print(" | ".join(encabezado))

    for resumen in resultado.resumenes:
        print(
            f"{resumen.caso_id} | "
            f"{resumen.algoritmo} | "
            f"{resumen.corridas} | "
            f"{resumen.costo_promedio:.6f} | "
            f"{resumen.costo_minimo:.6f} | "
            f"{resumen.diferencia_promedio_vs_greedy_pct:+.3f}% | "
            f"{resumen.firmas_distintas} | "
            f"{resumen.tiempo_computo_promedio_ms:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reproducible de GREEDY, RANDOM y GA usando "
            "caché vial estricta y estimacion-costo-v3."
        )
    )
    parser.add_argument(
        "--perfil",
        choices=("rapido", "formal"),
        default="rapido",
    )
    parser.add_argument(
        "--cache-vial",
        default="data/routing/cache_vial.csv",
    )
    parser.add_argument(
        "--version-cache",
        default="pedemonte-vial-v1",
    )
    parser.add_argument(
        "--salida",
        default=None,
    )
    parser.add_argument(
        "--instancias",
        default="TODAS",
        help=(
            "TODAS o lista separada por comas de caso_id, por ejemplo "
            "B01_SIMPLE,B03_MULTIVIAJE."
        ),
    )
    args = parser.parse_args()

    salida = (
        Path(args.salida)
        if args.salida is not None
        else Path("results/benchmark_classic") / f"15R_5A_{args.perfil}"
    )

    proveedor = ProveedorVialCachePersistente(
        args.cache_vial,
        version_cache_esperada=args.version_cache,
        permitir_fallback=False,
    )
    casos = _seleccionar_casos(
        crear_casos_benchmark_clasico(),
        args.instancias,
    )
    configuracion = _crear_configuracion_perfil(args.perfil)

    resultado = ejecutar_benchmark_clasico(
        casos,
        proveedor_viaje=proveedor,
        configuracion_benchmark=configuracion,
    )
    rutas = escribir_resultados_benchmark(
        resultado,
        salida,
    )

    _imprimir_resumen(resultado)
    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"  {nombre}: {ruta}")


if __name__ == "__main__":
    main()
