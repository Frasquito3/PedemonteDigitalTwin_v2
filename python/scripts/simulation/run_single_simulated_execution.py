from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from planner.integration.alpyne_codec import (  # noqa: E402
    codificar_instancia_alpyne,
)
from planner.integration.order_excel_import import (  # noqa: E402
    importar_pedidos_excel,
)
from planner.integration import selector_bridge  # noqa: E402
from planner.integration.simulated_execution import (  # noqa: E402
    codificar_identificadores_pedidos,
    ejecutar_plan_en_modelo_exportado,
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

    plan_vector = (
        selector_bridge
        .obtener_plan_comparacion_vector(
            args.metodo
        )
    )

    print(
        "PLAN | "
        f"metodo={args.metodo} | "
        f"instancia_vector={len(instancia_vector)} | "
        f"plan_vector={len(plan_vector)}"
    )

    resultado = ejecutar_plan_en_modelo_exportado(
        modelo_exportado=modelo_exportado,
        raiz_python=raiz_python,
        instancia_vector=instancia_vector,
        plan_vector=plan_vector,
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
        timeout_segundos=args.timeout,
        log_id="single-simulated-execution",
    )

    salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    salida.write_text(
        json.dumps(
            resultado.como_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(resultado.resumen())
    print(f"RESULTADO_JSON | {salida}")

    return 0


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
            "Ejecuta un plan estimado en una corrida limpia "
            "del modelo AnyLogic exportado."
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
        "--metodo",
        default="HIBRIDO",
        choices=[
            "RL",
            "HIBRIDO",
            "GREEDY",
            "RANDOM",
            "GA",
        ],
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
        "--sin-proveedores",
        action="store_true",
    )
    parser.add_argument(
        "--salida",
        default=str(
            PYTHON_ROOT
            / "data"
            / "results"
            / "simulated_execution.json"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
