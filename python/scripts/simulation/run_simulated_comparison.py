from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[2]

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from planner.core.schema import (  # noqa: E402
    InstanciaTurno,
    PedidoInput,
    Turno,
)
from planner.domain.preprocess import preprocesar_pedidos  # noqa: E402
from planner.integration import selector_bridge  # noqa: E402
from planner.integration.alpyne_codec import (  # noqa: E402
    codificar_instancia_alpyne,
)
from planner.integration.order_excel_import import (  # noqa: E402
    importar_pedidos_excel,
)
from planner.integration.simulated_comparison import (  # noqa: E402
    METODOS_COMPARACION_SIMULADA,
    ejecutar_comparacion_simulada,
)
from planner.integration.simulated_execution import (  # noqa: E402
    codificar_identificadores_pedidos,
)


def main() -> int:
    args = _parser().parse_args()

    modelo_exportado = (
        Path(args.modelo_exportado)
        .expanduser()
        .resolve()
    )
    raiz_python = (
        Path(args.raiz_python)
        .expanduser()
        .resolve()
    )
    archivo_excel = (
        Path(args.archivo_excel)
        .expanduser()
        .resolve()
    )
    salida = (
        Path(args.salida)
        .expanduser()
        .resolve()
    )

    instancia = _crear_instancia_ejemplo(
        archivo_excel=archivo_excel,
        seed_escenario=args.seed_escenario,
    )
    instancia_vector = codificar_instancia_alpyne(
        instancia
    )

    estado_selector = selector_bridge.inicializar(
        model_path=str(
            raiz_python
            / "models"
            / "rl"
            / "rl_policies.json"
        ),
        max_pedidos=30,
        deterministic=True,
        cache_vial_path=str(
            raiz_python
            / "data"
            / "routing"
            / "cache_vial.csv"
        ),
        version_cache_vial="pedemonte-vial-v1",
        permitir_fallback_vial=False,
    )

    print(f"SELECTOR | {estado_selector}")

    selector_bridge.comparar_estimado_vector(
        instancia_vector,
        instancia.seed_escenario,
        instancia.seed_ejecucion,
    )

    planes = {
        metodo: (
            selector_bridge
            .obtener_plan_comparacion_vector(
                metodo
            )
        )
        for metodo in METODOS_COMPARACION_SIMULADA
    }

    for metodo in METODOS_COMPARACION_SIMULADA:
        print(
            "PLAN | "
            f"metodo={metodo} | "
            f"plan_vector={len(planes[metodo])}"
        )

    comparacion = ejecutar_comparacion_simulada(
        modelo_exportado=modelo_exportado,
        raiz_python=raiz_python,
        instancia_vector=instancia_vector,
        planes_por_metodo=planes,
        identificadores_pedidos=(
            codificar_identificadores_pedidos(
                instancia.pedidos
            )
        ),
        instancia_id=instancia.instancia_id,
        fecha_operacion=instancia.fecha_operacion,
        seed_escenario=instancia.seed_escenario,
        seed_ejecucion=instancia.seed_ejecucion,
        proveedores_habilitados=(
            not args.sin_proveedores
        ),
        timeout_segundos_por_metodo=args.timeout,
        horizonte_simulacion_min=args.horizonte_min,
        continuar_ante_error=True,
    )

    for resultado_metodo in comparacion.resultados:
        if resultado_metodo.resultado is None:
            print(
                "METODO | "
                f"solicitado={resultado_metodo.metodo_solicitado} | "
                f"estado={resultado_metodo.estado} | "
                f"error={resultado_metodo.error}"
            )
            continue

        resultado = resultado_metodo.resultado
        print(
            "METODO | "
            f"solicitado={resultado_metodo.metodo_solicitado} | "
            f"estado={resultado_metodo.estado} | "
            f"aplicado={resultado.algoritmo_aplicado} | "
            f"costo={resultado.costo_total:.6f} | "
            f"distancia_km={resultado.distancia_total_km:.6f} | "
            f"duracion_min={resultado.duracion_simulada_min:.6f} | "
            f"viajes={resultado.viajes_totales} | "
            f"tardanza_min={resultado.tardanza_total_min:.6f} | "
            f"desbalance_min={resultado.diferencia_fin_camiones_min:.6f}"
        )

    salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    salida.write_text(
        json.dumps(
            comparacion.como_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(comparacion.resumen())
    print(f"COMPARACION_JSON | {salida}")

    return 0 if comparacion.completa else 1


def _crear_instancia_ejemplo(
    *,
    archivo_excel: Path,
    seed_escenario: int,
) -> InstanciaTurno:
    importacion = importar_pedidos_excel(
        archivo_excel,
        turno="MANANA",
        capacidad_camion=8,
        max_tareas=30,
    )

    originales = [
        PedidoInput(
            pedido_id=pedido.pedido_id,
            pedido_original_id=pedido.pedido_id,
            numero_parte=1,
            total_partes=1,
            turno=Turno.MANANA,
            latitud=pedido.latitud,
            longitud=pedido.longitud,
            unidades_capacidad=pedido.unidades,
            requiere_volcador=pedido.requiere_volcador,
            tiene_ventana_especifica=pedido.tiene_ventana,
            hora_desde_min=pedido.hora_desde_min,
            hora_hasta_min=pedido.hora_hasta_min,
            cliente=pedido.cliente,
            direccion=pedido.direccion,
            barrio=pedido.barrio,
            observaciones=pedido.observaciones,
        )
        for pedido in importacion.pedidos
    ]

    tareas = preprocesar_pedidos(
        originales,
        capacidad_camion=8,
    )

    return InstanciaTurno(
        instancia_id=(
            "UI-2026-08-30-"
            f"{seed_escenario}"
        ),
        fecha_operacion="2026-08-30",
        turno=Turno.MANANA,
        pedidos=tareas,
        lat_corralon=-32.8495006,
        lon_corralon=-60.722653,
        capacidad_camion=8,
        cantidad_camiones=2,
        hora_inicio_turno_min=450,
        hora_fin_objetivo_min=720,
        hora_fin_tolerancia_min=735,
        seed_escenario=seed_escenario,
        seed_ejecucion=1_000_000 + seed_escenario,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta RL, HÍBRIDO, GREEDY, RANDOM y GA en "
            "cinco motores AnyLogic independientes."
        )
    )
    parser.add_argument(
        "--modelo-exportado",
        default=str(
            PYTHON_ROOT
            / "anylogic_export"
            / "simulated_comparison"
            / "PedemonteDigitalTwin_v2.zip"
        ),
    )
    parser.add_argument(
        "--raiz-python",
        default=str(PYTHON_ROOT),
    )
    parser.add_argument(
        "--archivo-excel",
        default=str(
            PYTHON_ROOT
            / "templates"
            / "ejemplo_pedidos_importacion.xlsx"
        ),
    )
    parser.add_argument(
        "--seed-escenario",
        type=int,
        default=6001,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
    )
    parser.add_argument(
        "--horizonte-min",
        type=float,
        default=600.0,
        help=(
            "Horizonte máximo de tiempo simulado, en minutos, "
            "para cada motor AnyLogic."
        ),
    )
    parser.add_argument(
        "--sin-proveedores",
        action="store_true",
    )
    parser.add_argument(
        "--salida",
        default=str(
            PYTHON_ROOT
            / "data"
            / "results"
            / "simulated_comparison.json"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
