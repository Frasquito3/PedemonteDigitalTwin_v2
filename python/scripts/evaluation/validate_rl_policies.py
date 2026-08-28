from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from planner.algorithms.greedy import GreedyFeasiblePlanner  # noqa: E402
from planner.evaluation.high_demand_policy_holdout import (  # noqa: E402
    crear_casos_clasicos,
    crear_casos_sinteticos_finales,
)
from planner.evaluation.policy_runtime_holdout import (  # noqa: E402
    MODO_EXTENSION,
    MODO_FULL,
    MODO_GREEDY,
    MODO_OPERACIONAL,
    construir_veredicto,
    escribir_resultados,
    evaluar_casos_operacionales,
    resumir,
)
from planner.rl.policy_runtime import (  # noqa: E402
    RLTemporalV4OperationalPlanner,
    cargar_configuracion_operacional,
)
from planner.rl.policy_planner import RLTemporalV4Planner  # noqa: E402
from planner.routing.vial_cache import ProveedorVialCachePersistente  # noqa: E402


SEED_MINIMO = 278_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida una única vez la política operacional temporal v4 "
            "antes de conectarla a AnyLogic."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PYTHON_ROOT / "models" / "rl"
            / "rl_policies.json"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=SEED_MINIMO)
    parser.add_argument("--cases-per-stratum", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PYTHON_ROOT / "results" / "rl_temporal"
            / "policy_runtime_holdout"
        ),
    )
    args = parser.parse_args()
    if args.seed_start < SEED_MINIMO:
        parser.error(f"--seed-start debe ser >= {SEED_MINIMO}.")
    if args.cases_per_stratum < 20:
        parser.error("--cases-per-stratum debe ser >= 20.")
    return args


def crear_planificadores(manifest: Path, proveedor):
    cfg = cargar_configuracion_operacional(manifest)
    return {
        MODO_OPERACIONAL: RLTemporalV4OperationalPlanner(
            manifest_path=manifest,
            proveedor_viaje=proveedor,
        ),
        MODO_EXTENSION: RLTemporalV4Planner(
            cfg.modelo_extension,
            proveedor_viaje=proveedor,
        ),
        MODO_FULL: RLTemporalV4Planner(
            cfg.modelo_full,
            proveedor_viaje=proveedor,
        ),
        MODO_GREEDY: GreedyFeasiblePlanner(
            proveedor_viaje=proveedor,
        ),
    }


def main() -> None:
    args = parse_args()
    casos_clasicos = crear_casos_clasicos()
    casos_sinteticos = crear_casos_sinteticos_finales(
        casos_por_estrato=args.cases_per_stratum,
        seed_inicio=args.seed_start,
    )

    proveedor_vial = ProveedorVialCachePersistente(
        PYTHON_ROOT / "data" / "routing" / "cache_vial.csv",
        version_cache_esperada="pedemonte-vial-v1",
        permitir_fallback=False,
    )

    print("=== FASE 16D.11 — VALIDACIÓN OPERACIONAL RL TEMPORAL V4 ===")
    print(f"Manifiesto: {args.manifest.resolve()}")
    print(f"Casos clásicos: {len(casos_clasicos)}")
    print(f"Casos sintéticos: {len(casos_sinteticos)}")
    print(f"Seed inicial reservada: {args.seed_start}")
    print("Criterio: pedidos tardíos -> tardanza total -> costo")
    print("Entrenamiento adicional: NO")
    print("Promoción automática: NO")

    registros = evaluar_casos_operacionales(
        casos_clasicos,
        crear_planificadores(args.manifest, proveedor_vial),
        proveedor_viaje=proveedor_vial,
    )
    registros.extend(
        evaluar_casos_operacionales(
            casos_sinteticos,
            crear_planificadores(args.manifest, None),
        )
    )

    resumenes = resumir(registros)
    veredicto = construir_veredicto(registros, resumenes)
    cfg = cargar_configuracion_operacional(args.manifest)
    fuentes = {}
    for registro in registros:
        if registro.modo == MODO_OPERACIONAL and registro.estado == "OK":
            fuentes[registro.fuente_operacional] = (
                fuentes.get(registro.fuente_operacional, 0) + 1
            )

    print("\nResumen operacional:")
    for resumen in resumenes:
        if resumen.modo != MODO_OPERACIONAL:
            continue
        print(
            f"{resumen.alcance} | OK={resumen.ok}/{resumen.casos} | "
            f"sin riesgo={resumen.tasa_sin_riesgo_pct} | "
            f"tardíos={resumen.pedidos_tardios_total} | "
            f"tardanza media={resumen.tardanza_media_min}"
        )
    print(f"Fuentes seleccionadas: {fuentes}")
    print("\nVeredicto:")
    print(veredicto["estado"])
    for nombre, valor in veredicto["criterios"].items():
        print(f"- {nombre}: {'SI' if valor else 'NO'}")

    metadatos = {
        "fase": "16D.11",
        "version": "rl-temporal-v4-operational-holdout-v1",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "modelo_extension": str(cfg.modelo_extension),
        "modelo_full": str(cfg.modelo_full),
        "sha256_extension": cfg.sha256_extension,
        "sha256_full": cfg.sha256_full,
        "casos_clasicos": len(casos_clasicos),
        "casos_sinteticos": len(casos_sinteticos),
        "casos_por_estrato": args.cases_per_stratum,
        "seed_start": args.seed_start,
        "fuentes_seleccionadas": fuentes,
        "entrenamiento_adicional": False,
        "modelo_promovido": False,
    }
    rutas = escribir_resultados(
        args.output_dir,
        metadatos=metadatos,
        registros=registros,
        resumenes=resumenes,
        veredicto=veredicto,
    )
    print("\nArchivos:")
    for nombre, ruta in rutas.items():
        print(f"{nombre}: {ruta}")
    if veredicto["estado"] == "APTO_PARA_INTEGRACION_ANYLOGIC":
        print("RESULTADO: VALIDACION_OPERACIONAL_16D_11_OK")
    else:
        print("RESULTADO: VALIDACION_OPERACIONAL_16D_11_NO_APTA")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
