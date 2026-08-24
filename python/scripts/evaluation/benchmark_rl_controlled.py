from __future__ import annotations

import argparse
import sys
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from planner.algorithms.ga import ConfiguracionGA
from planner.evaluation.classic_instances import (
    CasoBenchmarkClasico,
    crear_casos_benchmark_clasico,
)
from planner.evaluation.rl_controlled_benchmark import (
    ConfiguracionBenchmarkRLControlado,
    MetadatosModeloRL,
    calcular_sha256_archivo,
    ejecutar_benchmark_rl_controlado,
    escribir_resultados_benchmark_rl_controlado,
)
from planner.rl.planner import RLPlanner
from planner.routing.vial_cache import ProveedorVialCachePersistente


def _parsear_modelo(texto: str) -> tuple[str, Path]:
    if "=" not in texto:
        raise ValueError(
            "Cada --modelo debe usar el formato ALIAS=RUTA."
        )
    alias, ruta_texto = texto.split("=", 1)
    alias = alias.strip().upper()
    ruta_texto = ruta_texto.strip()
    if not alias or not ruta_texto:
        raise ValueError(
            "Cada --modelo debe usar un alias y una ruta no vacíos."
        )
    return alias, Path(ruta_texto).expanduser().resolve()


def _seleccionar_casos(
    casos: tuple[CasoBenchmarkClasico, ...],
    seleccion: str,
) -> tuple[CasoBenchmarkClasico, ...]:
    texto = seleccion.strip()
    if not texto or texto.upper() == "TODAS":
        return casos

    ids = {
        parte.strip().upper()
        for parte in texto.split(",")
        if parte.strip()
    }
    por_id = {caso.caso_id.upper(): caso for caso in casos}
    faltantes = sorted(ids - set(por_id))
    if faltantes:
        raise ValueError(
            "Casos inexistentes: " + ", ".join(faltantes)
        )
    return tuple(
        caso for caso in casos if caso.caso_id.upper() in ids
    )


def _formatear_numero(valor: float | None) -> str:
    return "ERROR" if valor is None else f"{valor:.6f}"


def _formatear_pct(valor: float | None) -> str:
    return "-" if valor is None else f"{valor:+.3f}%"


def _imprimir_resumen(resultado) -> None:
    print("\n=== BENCHMARK RL CONTROLADO COMPLETADO ===")
    print(f"Versión: {resultado.version_benchmark}")
    print(f"Objetivo: {resultado.version_objetivo}")
    print(
        "Proveedor: "
        f"{resultado.fuente_viaje} | {resultado.version_viaje}"
    )
    print(f"Casos: {len(resultado.casos)}")
    print(f"Modelos: {len(resultado.modelos)}")
    print(f"Filas generadas: {len(resultado.corridas)}")
    print(
        "CASO | MODELO | RL ESTADO | RL COSTO | "
        "HIBRIDO COSTO | FUENTE | MEJORA H/GREEDY | GARANTIA"
    )

    for resumen in resultado.resumenes:
        print(
            f"{resumen.caso_id} | "
            f"{resumen.modelo_alias} | "
            f"{resumen.rl_estado} | "
            f"{_formatear_numero(resumen.costo_rl)} | "
            f"{_formatear_numero(resumen.costo_hibrido)} | "
            f"{resumen.fuente_hibrida or '-'} | "
            f"{_formatear_pct(resumen.mejora_hibrido_vs_greedy_pct)} | "
            f"{resumen.hibrido_cumple_garantia}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnóstico controlado de modelos RL y sus híbridos "
            "contra GREEDY y GA, usando los seis casos clásicos."
        )
    )
    parser.add_argument(
        "--modelo",
        action="append",
        default=None,
        help=(
            "Modelo en formato ALIAS=RUTA. Puede repetirse. "
            "Por defecto evalúa HISTORICO y REAL_V2."
        ),
    )
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
        default="results/benchmark_rl/15R_6A_controlado",
    )
    parser.add_argument(
        "--instancias",
        default="TODAS",
    )
    parser.add_argument(
        "--max-pedidos",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    especificaciones = args.modelo or [
        "HISTORICO=models/rl/pedemonte_maskable_ppo.zip",
        "REAL_V2=models/rl/pedemonte_maskable_ppo_real_v2.zip",
    ]

    modelos: dict[str, Path] = {}
    for texto in especificaciones:
        alias, ruta = _parsear_modelo(texto)
        if alias in modelos:
            raise ValueError(f"Alias de modelo duplicado: {alias}")
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No existe el modelo {alias}: {ruta}"
            )
        modelos[alias] = ruta

    proveedor = ProveedorVialCachePersistente(
        args.cache_vial,
        version_cache_esperada=args.version_cache,
        permitir_fallback=False,
    )
    casos = _seleccionar_casos(
        crear_casos_benchmark_clasico(),
        args.instancias,
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

    resultado = ejecutar_benchmark_rl_controlado(
        casos,
        proveedor_viaje=proveedor,
        planners_rl=planners_rl,
        metadatos_modelos=metadatos,
        configuracion_benchmark=ConfiguracionBenchmarkRLControlado(
            configuracion_ga=ConfiguracionGA(),
            seed_ga=101,
        ),
    )
    rutas = escribir_resultados_benchmark_rl_controlado(
        resultado,
        args.salida,
    )

    _imprimir_resumen(resultado)
    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"  {nombre}: {ruta}")


if __name__ == "__main__":
    main()
