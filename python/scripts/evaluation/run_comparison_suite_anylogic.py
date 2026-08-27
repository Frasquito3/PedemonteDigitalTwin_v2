from __future__ import annotations

import argparse
import sys

from dataclasses import asdict
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[2]

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


from planner.evaluation.classic_instances import (  # noqa: E402
    CasoBenchmarkClasico,
    crear_casos_benchmark_clasico,
)
from planner.evaluation.comparison_contract import (  # noqa: E402
    escribir_contrato_comparacion,
    preparar_contrato_comparacion,
)
from planner.evaluation.comparison_suite import (  # noqa: E402
    CasoContratoComparacion,
    ConfiguracionSuiteComparacion,
    escribir_resultado_suite_comparacion,
    ejecutar_suite_comparacion,
)
from planner.integration.anylogic_vector_client import (  # noqa: E402
    AnyLogicVectorClient,
)
from planner.integration.planner_selector import (  # noqa: E402
    SelectorPlanificadores,
)
from planner.routing.vial_cache import (  # noqa: E402
    ProveedorVialCachePersistente,
)


DEFAULT_RL_MODEL = (
    PYTHON_ROOT
    / "models"
    / "rl"
    / "pedemonte_maskable_ppo.zip"
)

DEFAULT_CACHE = (
    PYTHON_ROOT
    / "data"
    / "routing"
    / "cache_vial_v1.csv"
)

DEFAULT_ANYLOGIC_MODEL = (
    PYTHON_ROOT
    / "anylogic_export"
    / "phase16b_comparison"
    / "PedemonteDigitalTwin_v2.zip"
)

DEFAULT_OUTPUT = (
    PYTHON_ROOT
    / "results"
    / "comparison_anylogic"
    / "16C"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara y ejecuta en AnyLogic los cinco modos para los "
            "seis casos controlados de la Fase 16C."
        )
    )
    parser.add_argument(
        "--cases",
        default="TODAS",
        help=(
            "TODAS o lista de casos. Acepta B01..B06 o los nombres "
            "canónicos, por ejemplo B01_SIMPLE,B04_VENTANAS."
        ),
    )
    parser.add_argument(
        "--rl-model",
        default=str(DEFAULT_RL_MODEL),
        help="Modelo RL histórico utilizado por el selector.",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Caché vial compartida y estricta.",
    )
    parser.add_argument(
        "--cache-version",
        default="pedemonte-vial-v1",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_ANYLOGIC_MODEL),
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
        help="Carpeta python del proyecto.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directorio de resultados de la suite.",
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
        help="Límite preventivo de tiempo simulado por corrida.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Detiene la suite ante el primer error.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Reduce los logs de Alpyne y Java.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    salida = Path(args.output).expanduser().resolve()

    casos = _seleccionar_casos(
        crear_casos_benchmark_clasico(),
        args.cases,
    )

    proveedor = ProveedorVialCachePersistente(
        args.cache,
        version_cache_esperada=args.cache_version,
        permitir_fallback=False,
    )
    selector = SelectorPlanificadores(
        model_path_rl=args.rl_model,
        proveedor_viaje=proveedor,
        deterministic=True,
    )
    selector.precargar_rl()

    casos_contrato: list[CasoContratoComparacion] = []

    print("=== PREPARACIÓN SUITE ANYLOGIC 16C ===")
    for indice, caso in enumerate(casos, start=1):
        contrato = preparar_contrato_comparacion(
            caso.instancia,
            selector=selector,
            proveedor_viaje=proveedor,
        )

        rutas_contrato = escribir_contrato_comparacion(
            contrato,
            salida / "contracts" / caso.caso_id,
        )

        casos_contrato.append(
            CasoContratoComparacion(
                caso_id=caso.caso_id,
                categoria=caso.categoria,
                descripcion=caso.descripcion,
                contrato=asdict(contrato),
            )
        )

        print(
            f"{indice}/{len(casos)} {caso.caso_id} | "
            f"planes={contrato.planes_ok}/5 | "
            f"seed_ejecucion={contrato.seed_ejecucion} | "
            f"contrato={rutas_contrato['contrato_json']}"
        )

    cliente = AnyLogicVectorClient(
        model_path=args.model,
        java_exe=(args.java if args.java.strip() else None),
        python_root=args.python_root,
        timeout_segundos=args.timeout,
        max_server_await_time=args.max_server_await,
        limite_ple_min=args.ple_limit,
        log_id_base="phase16c",
        habilitar_logs=not args.no_logs,
    )

    resultado = ejecutar_suite_comparacion(
        casos_contrato,
        ejecutor=cliente,
        configuracion=ConfiguracionSuiteComparacion(
            continuar_ante_error=not args.fail_fast,
            exigir_seis_casos=(args.cases.strip().upper() == "TODAS"),
            exigir_orden_rl_primero=True,
            exigir_cinco_planes_ok=True,
        ),
    )
    rutas = escribir_resultado_suite_comparacion(resultado, salida)

    print("\n=== SUITE ANYLOGIC 16C COMPLETADA ===")
    print(f"Casos: {resultado.cantidad_casos}")
    print(
        f"Corridas OK: {resultado.corridas_ok}/"
        f"{resultado.corridas_esperadas}"
    )
    print(f"Nivel de evidencia: {resultado.nivel_evidencia}")
    print(f"Proveedor vial: {resultado.version_viaje}")

    for caso in resultado.casos:
        mejores = (
            ",".join(caso.modos_mejor_costo)
            if caso.modos_mejor_costo
            else "—"
        )
        mejor = (
            "—"
            if caso.mejor_costo_real is None
            else f"{caso.mejor_costo_real:.6f}"
        )
        rl = (
            "—"
            if caso.costo_real_rl is None
            else f"{caso.costo_real_rl:.6f}"
        )
        print(
            f"{caso.caso_id} | "
            f"ok={caso.ejecuciones_ok}/5 | "
            f"mejor={mejores} ({mejor}) | "
            f"RL={rl} | ranking_RL={caso.ranking_rl or '—'}"
        )
        if caso.error_caso:
            print(f"   ERROR CASO: {caso.error_caso}")

    print("\nResumen por modo solicitado:")
    for resumen in resultado.resumen_algoritmos:
        mejora_rl = (
            "—"
            if resumen.mejora_media_vs_rl_pct is None
            else f"{resumen.mejora_media_vs_rl_pct:+.3f}%"
        )
        mejora_greedy = (
            "—"
            if resumen.mejora_media_vs_greedy_pct is None
            else f"{resumen.mejora_media_vs_greedy_pct:+.3f}%"
        )
        print(
            f"{resumen.modo_solicitado} | "
            f"ok={resumen.casos_ok}/{resumen.casos_totales} | "
            f"primeros={resumen.primeros_puestos} | "
            f"G/E/P vs RL="
            f"{resumen.victorias_vs_rl}/"
            f"{resumen.empates_vs_rl}/"
            f"{resumen.derrotas_vs_rl} | "
            f"media vs RL={mejora_rl} | "
            f"media vs Greedy={mejora_greedy}"
        )

    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"{nombre}: {ruta}")

    return 0 if resultado.corridas_error == 0 else 2


def _seleccionar_casos(
    casos: tuple[CasoBenchmarkClasico, ...],
    seleccion: str,
) -> tuple[CasoBenchmarkClasico, ...]:
    texto = seleccion.strip().upper()
    if not texto or texto == "TODAS":
        return casos

    por_id = {caso.caso_id.upper(): caso for caso in casos}
    alias = {
        caso.caso_id.split("_", 1)[0].upper(): caso.caso_id.upper()
        for caso in casos
    }

    solicitados: list[str] = []
    for parte in texto.split(","):
        token = parte.strip().upper()
        if not token:
            continue
        solicitados.append(alias.get(token, token))

    faltantes = sorted(set(solicitados) - set(por_id))
    if faltantes:
        raise ValueError(
            "Casos inexistentes: "
            + ", ".join(faltantes)
            + ". Disponibles: "
            + ", ".join(sorted(por_id))
        )

    seleccionados = tuple(
        caso
        for caso in casos
        if caso.caso_id.upper() in set(solicitados)
    )
    if not seleccionados:
        raise ValueError("No se seleccionó ningún caso.")
    return seleccionados


if __name__ == "__main__":
    raise SystemExit(main())
