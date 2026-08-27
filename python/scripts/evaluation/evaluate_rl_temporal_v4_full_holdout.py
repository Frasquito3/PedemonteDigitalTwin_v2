from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from planner.algorithms.greedy import GreedyFeasiblePlanner  # noqa: E402
from planner.evaluation.rl_temporal_v4_full_holdout import (  # noqa: E402
    ESTRATOS_FASE_16D7,
    MODO_GREEDY,
    MODO_RL_HISTORICO,
    MODO_RL_TEMPORAL_V4_EXTENSION,
    MODO_RL_TEMPORAL_V4_FULL,
    MODO_RL_TEMPORAL_V4_QUICK,
    SEED_HOLDOUT_FINAL_MINIMO,
    analizar_clasicos,
    construir_veredicto,
    crear_casos_clasicos,
    crear_casos_sinteticos_finales,
    escribir_resultados,
    evaluar_casos_holdout,
    resumir_casos,
    resumir_registros,
    validar_metadatos_fase16d9,
)
from planner.routing.vial_cache import ProveedorVialCachePersistente  # noqa: E402


def parse_args() -> argparse.Namespace:
    quick_dir = PYTHON_ROOT / "rl_artifacts" / "phase16d_temporal_v4_quick"
    extension_dir = PYTHON_ROOT / "rl_artifacts" / "phase16d_temporal_v4_extension_9_12"
    full_dir = PYTHON_ROOT / "rl_artifacts" / "phase16d_temporal_v4_full_11_12"
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el holdout final independiente de la Fase 16D.9: "
            "histórico, quick v4, extensión 9-12, full 11-12 y Greedy."
        )
    )
    parser.add_argument(
        "--historical-model", type=Path,
        default=PYTHON_ROOT / "models" / "rl" / "pedemonte_maskable_ppo.zip",
    )
    parser.add_argument("--quick-model", type=Path, default=quick_dir / "final_model.zip")
    parser.add_argument("--quick-config", type=Path, default=quick_dir / "temporal_v4_config.json")
    parser.add_argument("--quick-selection", type=Path, default=quick_dir / "final_model_selection.json")
    parser.add_argument("--extension-model", type=Path, default=extension_dir / "final_model.zip")
    parser.add_argument("--extension-config", type=Path, default=extension_dir / "temporal_v4_extension_config.json")
    parser.add_argument("--extension-selection", type=Path, default=extension_dir / "final_model_selection.json")
    parser.add_argument("--extension-summary", type=Path, default=extension_dir / "final_external_summary.json")
    parser.add_argument("--full-model", type=Path, default=full_dir / "final_model.zip")
    parser.add_argument("--full-config", type=Path, default=full_dir / "temporal_v4_full_11_12_config.json")
    parser.add_argument("--full-selection", type=Path, default=full_dir / "final_model_selection.json")
    parser.add_argument("--full-summary", type=Path, default=full_dir / "final_external_summary.json")
    parser.add_argument(
        "--full-audit", type=Path,
        default=PYTHON_ROOT / "results" / "rl_temporal" / "16D_8_auditoria" / "auditoria_16d8.json",
    )
    parser.add_argument(
        "--holdout-16d7-result", type=Path,
        default=(
            PYTHON_ROOT / "results" / "rl_temporal"
            / "16D_7_holdout_extension_9_12_formal_272000"
            / "rl_temporal_v4_extension_holdout.json"
        ),
    )
    parser.add_argument(
        "--quick-holdout-result", type=Path,
        default=(
            PYTHON_ROOT / "results" / "rl_temporal" / "16D_5_holdout_v4"
            / "rl_temporal_v4_holdout.json"
        ),
    )
    parser.add_argument(
        "--cases-per-stratum", type=int, default=30,
        help="Casos por cada uno de los ocho estratos. Protocolo final: >=20; recomendado: 30.",
    )
    parser.add_argument("--seed-start", type=int, default=SEED_HOLDOUT_FINAL_MINIMO)
    parser.add_argument(
        "--output-dir", type=Path,
        default=(
            PYTHON_ROOT / "results" / "rl_temporal"
            / "16D_9_holdout_full_11_12_formal_274000"
        ),
    )
    parser.add_argument(
        "--only-classic", action="store_true",
        help="Ejecuta sólo B01-B06. No reemplaza el holdout final.",
    )
    args = parser.parse_args()
    if args.cases_per_stratum <= 0:
        parser.error("--cases-per-stratum debe ser > 0.")
    if not args.only_classic and args.cases_per_stratum < 20:
        parser.error("El protocolo final exige --cases-per-stratum >= 20.")
    if args.seed_start < SEED_HOLDOUT_FINAL_MINIMO:
        parser.error(
            f"La Fase 16D.9 exige --seed-start >= {SEED_HOLDOUT_FINAL_MINIMO}."
        )
    return args


def _crear_planificadores(
    args: argparse.Namespace,
    proveedor: Any | None,
) -> dict[str, Any]:
    from planner.rl.rl_planner import RLPlanner
    from planner.rl.rl_temporal_v4_planner import RLTemporalV4Planner

    return {
        MODO_RL_HISTORICO: RLPlanner(args.historical_model, proveedor_viaje=proveedor),
        MODO_RL_TEMPORAL_V4_QUICK: RLTemporalV4Planner(args.quick_model, proveedor_viaje=proveedor),
        MODO_RL_TEMPORAL_V4_EXTENSION: RLTemporalV4Planner(args.extension_model, proveedor_viaje=proveedor),
        MODO_RL_TEMPORAL_V4_FULL: RLTemporalV4Planner(args.full_model, proveedor_viaje=proveedor),
        MODO_GREEDY: GreedyFeasiblePlanner(proveedor_viaje=proveedor),
    }


def _fmt(valor: Any, decimales: int = 3) -> str:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return "—"
    return f"{float(valor):.{decimales}f}"


def _imprimir_clasicos(clasicos: dict[str, Any]) -> None:
    print("\nCasos clásicos clave:")
    for caso_id in ("B04_VENTANAS", "B05_VOLCADOR", "B06_SPLIT"):
        print(f"\n{caso_id}")
        detalle = clasicos.get(caso_id, {})
        modos = detalle.get("modos", {})
        for modo in (
            MODO_RL_HISTORICO,
            MODO_RL_TEMPORAL_V4_QUICK,
            MODO_RL_TEMPORAL_V4_EXTENSION,
            MODO_RL_TEMPORAL_V4_FULL,
            MODO_GREEDY,
        ):
            item = modos.get(modo, {})
            print(
                f"{modo} | estado={item.get('estado', 'FALTANTE')} | "
                f"tardíos={item.get('pedidos_tardios', '—')} | "
                f"tardanza={_fmt(item.get('tardanza_total_min'))} min | "
                f"costo={_fmt(item.get('costo_recalculado'))} | "
                f"ruta={item.get('firma_plan') or '—'}"
            )
        print(
            f"Full vs extensión={detalle.get('full_vs_extension', 'NO_DISPONIBLE')} | "
            f"vs quick={detalle.get('full_vs_quick', 'NO_DISPONIBLE')} | "
            f"vs histórico={detalle.get('full_vs_historico', 'NO_DISPONIBLE')}"
        )


def _imprimir_global(resumenes: list[Any]) -> None:
    print("\nResumen global del holdout sintético:")
    for r in resumenes:
        if r.grupo != "HOLDOUT_SINTETICO" or r.alcance != "TODOS":
            continue
        print(
            f"{r.modo} | OK={r.casos_ok}/{r.casos_totales} | errores={r.casos_error} | "
            f"sin riesgo={_fmt(r.tasa_sin_riesgo_pct, 2)}% | "
            f"tardanza media/mediana/p95={_fmt(r.tardanza_media_min)}/"
            f"{_fmt(r.tardanza_mediana_min)}/{_fmt(r.tardanza_p95_min)} min | "
            f"G/E/P vs extensión={r.victorias_vs_extension}/"
            f"{r.empates_vs_extension}/{r.derrotas_vs_extension} | "
            f"G/E/P vs Greedy={r.victorias_vs_greedy}/"
            f"{r.empates_vs_greedy}/{r.derrotas_vs_greedy} | "
            f"gap costo mediano/p95 vs Greedy="
            f"{_fmt(r.gap_costo_mediano_vs_greedy_pct)}/"
            f"{_fmt(r.gap_costo_p95_vs_greedy_pct)}% | "
            f"extremos={r.costos_extremos_vs_greedy}"
        )


def _imprimir_full_por_alcance(titulo: str, resumenes: list[Any]) -> None:
    print(f"\n{titulo}:")
    for r in resumenes:
        if r.modo != MODO_RL_TEMPORAL_V4_FULL:
            continue
        print(
            f"{r.alcance} | n={r.casos_totales} | "
            f"sin riesgo={_fmt(r.tasa_sin_riesgo_pct, 2)}% | "
            f"G/E/P vs extensión={r.victorias_vs_extension}/"
            f"{r.empates_vs_extension}/{r.derrotas_vs_extension} | "
            f"tardanza p95={_fmt(r.tardanza_p95_min)} min | "
            f"extremos={r.costos_extremos_vs_greedy}"
        )


def main() -> None:
    args = parse_args()
    casos_clasicos = crear_casos_clasicos()
    casos_sinteticos = (
        [] if args.only_classic
        else crear_casos_sinteticos_finales(
            casos_por_estrato=args.cases_per_stratum,
            seed_inicio=args.seed_start,
        )
    )
    semillas_holdout = [c.instancia.seed_escenario for c in casos_sinteticos]

    auditoria = validar_metadatos_fase16d9(
        historical_model=args.historical_model,
        quick_model=args.quick_model,
        quick_config=args.quick_config,
        quick_selection=args.quick_selection,
        extension_model=args.extension_model,
        extension_config=args.extension_config,
        extension_selection=args.extension_selection,
        extension_summary=args.extension_summary,
        quick_holdout_result=args.quick_holdout_result,
        full_model=args.full_model,
        full_config=args.full_config,
        full_selection=args.full_selection,
        full_summary=args.full_summary,
        full_audit=args.full_audit,
        holdout_16d7_result=args.holdout_16d7_result,
        seed_inicio=args.seed_start,
        semillas_holdout=semillas_holdout,
    )

    proveedor_vial = ProveedorVialCachePersistente(
        PYTHON_ROOT / "data" / "routing" / "cache_vial_v1.csv",
        version_cache_esperada="pedemonte-vial-v1",
        permitir_fallback=False,
    )
    planificadores_clasicos = _crear_planificadores(args, proveedor_vial)
    planificadores_sinteticos = _crear_planificadores(args, None)

    print("=== FASE 16D.9 — HOLDOUT FINAL INDEPENDIENTE V4 FULL 11-12 ===")
    print(f"RL histórico: {args.historical_model}")
    print(f"RL temporal v4 quick: {args.quick_model}")
    print(f"RL temporal v4 extensión: {args.extension_model}")
    print(f"RL temporal v4 full: {args.full_model}")
    print("Modelo promovido: NO")
    print("Comparación: pedidos tardíos -> tardanza total -> costo estimado")

    registros = evaluar_casos_holdout(
        casos_clasicos,
        planificadores_clasicos,
        proveedor_viaje=proveedor_vial,
    )
    if casos_sinteticos:
        print(
            f"\nEvaluando {len(casos_sinteticos)} casos nuevos, "
            f"{args.cases_per_stratum} por cada uno de "
            f"{len(ESTRATOS_FASE_16D7)} estratos..."
        )
        registros.extend(
            evaluar_casos_holdout(casos_sinteticos, planificadores_sinteticos)
        )

    globales, estratos, segmentos = resumir_registros(registros)
    casos = resumir_casos(registros)
    clasicos = analizar_clasicos(registros)
    veredicto = construir_veredicto(registros, globales, segmentos)

    _imprimir_clasicos(clasicos)
    _imprimir_global(globales)
    _imprimir_full_por_alcance("Full por estrato", estratos)
    _imprimir_full_por_alcance("Full por segmento", segmentos)

    print("\nVeredicto automático:")
    print(veredicto["estado"])
    for criterio, cumple in veredicto.get("criterios", {}).items():
        print(f"- {criterio}: {'SI' if cumple else 'NO'}")

    semillas_csv = [
        {
            "caso_id": c.caso_id,
            "estrato": c.estrato,
            "banda_pedidos": c.banda_pedidos,
            "cantidad_objetivo": c.cantidad_objetivo,
            "cantidad_pedidos": len(c.instancia.pedidos),
            "patron_conflictivo": c.patron_conflictivo,
            "seed_escenario": c.instancia.seed_escenario,
            "seed_ejecucion": c.instancia.seed_ejecucion,
        }
        for c in casos_sinteticos
    ]
    full_cfg = auditoria["full_config"]
    full_sel = auditoria["full_selection"]
    metadatos = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "fase": "16D.9",
        "version_run_full": full_cfg.get("version_run"),
        "timestep_full_seleccionado": full_sel.get("timestep_final_seleccionado"),
        "modelo_historico": str(args.historical_model.resolve()),
        "modelo_quick": str(args.quick_model.resolve()),
        "modelo_extension": str(args.extension_model.resolve()),
        "modelo_full": str(args.full_model.resolve()),
        "sha256_modelos": auditoria["hashes_modelos"],
        "casos_clasicos": len(casos_clasicos),
        "casos_sinteticos": len(casos_sinteticos),
        "casos_por_estrato": args.cases_per_stratum,
        "estratos": [e.nombre for e in ESTRATOS_FASE_16D7],
        "seed_inicio": args.seed_start,
        "seed_min_usada": min(semillas_holdout) if semillas_holdout else None,
        "seed_max_usada": max(semillas_holdout) if semillas_holdout else None,
        "semillas_holdout": semillas_holdout,
        "semillas_prohibidas_declaradas": auditoria["semillas_prohibidas_declaradas"],
        "proveedor_clasicos": proveedor_vial.version,
        "proveedor_sinteticos": "haversine-ajustada-v1",
        "criterio_lexicografico": [
            "pedidos_tardios", "tardanza_total_min", "costo_recalculado"
        ],
        "modelo_promovido": False,
        "modo_solo_clasicos": args.only_classic,
    }
    rutas = escribir_resultados(
        args.output_dir,
        metadatos=metadatos,
        registros=registros,
        resumen_global=globales,
        resumen_estratos=estratos,
        resumen_segmentos=segmentos,
        casos=casos,
        clasicos=clasicos,
        veredicto=veredicto,
        semillas=semillas_csv,
    )
    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"{nombre}: {ruta.resolve()}")


if __name__ == "__main__":
    main()
