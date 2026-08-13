from __future__ import annotations

import argparse
import json
import sys

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


PYTHON_ROOT = Path(
    __file__
).resolve().parents[2]

if str(
    PYTHON_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PYTHON_ROOT
        ),
    )


from planner.algorithms.hybrid_rl_greedy import (  # noqa: E402
    HybridRLGreedyPlanner,
)

from planner.domain.validator import (  # noqa: E402
    validar_instancia,
    validar_plan,
)

from planner.integration.anylogic_client import (  # noqa: E402
    AnyLogicDynamicClient,
)

from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)

from planner.rl.planner import (  # noqa: E402
    RLPlanner,
)


DEFAULT_ANYLOGIC_MODEL = (
    PYTHON_ROOT
    / "anylogic_export"
    / "phase10c_dynamic"
    / "PedemonteDigitalTwin_v2.zip"
)

DEFAULT_RL_MODEL = (
    PYTHON_ROOT
    / "rl_artifacts"
    / "phase9c_curriculum"
    / "stage_03_4_12"
    / "best"
    / "best_model.zip"
)

DEFAULT_OUTPUT = (
    PYTHON_ROOT
    / "rl_artifacts"
    / "phase10d_real_planner"
    / "hybrid_anylogic_result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta una instancia generada en "
            "Python, planificada por el híbrido "
            "RL-Greedy y simulada en AnyLogic."
        )
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_ANYLOGIC_MODEL
        ),
        help=(
            "ZIP exportado desde "
            "AlpyneExperiment."
        ),
    )

    parser.add_argument(
        "--rl-model",
        default=str(
            DEFAULT_RL_MODEL
        ),
        help=(
            "Modelo MaskablePPO utilizado "
            "por RLPlanner."
        ),
    )

    parser.add_argument(
        "--java",
        default="",
        help=(
            "Ruta opcional al java.exe "
            "incluido con AnyLogic."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=4_003,
        help=(
            "Seed de la instancia de "
            "aceptación."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
        help=(
            "Ruta del JSON de resultados."
        ),
    )

    return parser.parse_args()


def resolver_ruta(
    texto: str,
) -> Path:
    ruta = Path(
        texto
    ).expanduser()

    if not ruta.is_absolute():
        ruta = (
            Path.cwd()
            / ruta
        )

    return ruta.resolve()


def convertir_json(
    valor: Any,
) -> Any:
    if isinstance(
        valor,
        Enum,
    ):
        return valor.value

    if is_dataclass(
        valor
    ) and not isinstance(
        valor,
        type,
    ):
        return convertir_json(
            asdict(
                valor
            )
        )

    if isinstance(
        valor,
        dict,
    ):
        return {
            str(
                clave
            ): convertir_json(
                contenido
            )
            for clave, contenido
            in valor.items()
        }

    if isinstance(
        valor,
        (
            list,
            tuple,
        ),
    ):
        return [
            convertir_json(
                contenido
            )
            for contenido in valor
        ]

    return valor


def main() -> None:
    args = parse_args()

    model_path = resolver_ruta(
        args.model
    )

    rl_model_path = resolver_ruta(
        args.rl_model
    )

    output_path = resolver_ruta(
        args.output
    )

    java_path: str | None = None

    if args.java.strip():
        java_path = str(
            resolver_ruta(
                args.java
            )
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            "No existe el ZIP de AnyLogic: "
            f"{model_path}"
        )

    if not rl_model_path.is_file():
        raise FileNotFoundError(
            "No existe el modelo RL: "
            f"{rl_model_path}"
        )

    print()
    print(
        "=== FASE 10D: HÍBRIDO → ANYLOGIC ==="
    )
    print(
        f"ZIP AnyLogic: {model_path}"
    )
    print(
        f"Modelo RL: {rl_model_path}"
    )
    print(
        f"Seed: {args.seed}"
    )

    configuracion_generador = (
        ConfiguracionGeneradorInstancias(
            min_pedidos_finales=2,
            max_pedidos_finales=2,
            capacidad_camion=8,
            cantidad_camiones=2,
            probabilidad_volcador=0.0,
            probabilidad_ventana_especifica=0.0,
            probabilidad_pedido_mayor_capacidad=0.0,
            max_unidades_pedido_grande=16,
            desplazamiento_max_grados=0.003,
            max_intentos_generacion=100,
        )
    )

    generador = GeneradorInstanciasRL(
        configuracion_generador
    )

    instancia = generador.generar(
        args.seed
    )

    errores_instancia = validar_instancia(
        instancia
    )

    if errores_instancia:
        raise RuntimeError(
            "El generador produjo una "
            "instancia inválida: "
            + " | ".join(
                errores_instancia
            )
        )

    print()
    print(
        "Instancia generada:"
    )
    print(
        f"  ID: {instancia.instancia_id}"
    )
    print(
        f"  Turno: {instancia.turno.value}"
    )
    print(
        f"  Pedidos: {len(instancia.pedidos)}"
    )
    print(
        "  Unidades: "
        + str(
            [
                pedido
                .unidades_capacidad
                for pedido
                in instancia.pedidos
            ]
        )
    )

    planner_rl = RLPlanner(
        model_path=rl_model_path,
        max_pedidos=30,
        deterministic=True,
    )

    planner_hibrido = (
        HybridRLGreedyPlanner(
            planner_rl=planner_rl
        )
    )

    plan = planner_hibrido.generar_plan(
        instancia
    )

    validacion_plan = validar_plan(
        instancia,
        plan,
    )

    if not validacion_plan.valido:
        raise RuntimeError(
            "El híbrido produjo un "
            "plan inválido: "
            + " | ".join(
                validacion_plan.errores
            )
        )

    decision = (
        planner_hibrido
        .ultima_decision
    )

    if decision is None:
        raise RuntimeError(
            "El planificador híbrido no "
            "registró su decisión."
        )

    cantidad_viajes = sum(
        len(
            camion.viajes
        )
        for camion in plan.camiones
    )

    print()
    print(
        "Plan híbrido generado:"
    )
    print(
        "  Fuente seleccionada: "
        f"{decision.fuente_seleccionada.value}"
    )
    print(
        f"  Motivo: {decision.motivo.value}"
    )
    print(
        f"  Algoritmo del plan: "
        f"{plan.algoritmo.value}"
    )
    print(
        f"  Viajes: {cantidad_viajes}"
    )
    print(
        "  Costo estimado Python: "
        f"{plan.costo_estimado:.6f}"
    )
    print(
        "  Tiempo de planificación: "
        f"{plan.tiempo_computo_ms:.3f} ms"
    )

    cliente = AnyLogicDynamicClient(
        model_path=model_path,
        java_exe=java_path,
        limite_ple_min=59.0,
        log_id="phase10d",
        habilitar_logs=True,
    )

    resultado_anylogic = cliente.ejecutar(
        instancia=instancia,
        plan=plan,
    )

    observacion = (
        resultado_anylogic
        .observacion_final
    )

    entregadas = int(
        observacion[
            "tareasEntregadas"
        ]
    )

    no_entregadas = int(
        observacion[
            "tareasNoEntregadas"
        ]
    )

    if entregadas != len(
        instancia.pedidos
    ):
        raise RuntimeError(
            "La instancia controlada de "
            "aceptación no entregó todos "
            "los pedidos. "
            f"Entregados={entregadas}, "
            f"esperados={len(instancia.pedidos)}."
        )

    if no_entregadas != 0:
        raise RuntimeError(
            "La instancia controlada registró "
            "pedidos no entregados: "
            f"{no_entregadas}."
        )

    salida = {
        "fase": "10D",
        "seed_generacion": args.seed,
        "instancia": convertir_json(
            instancia
        ),
        "plan": convertir_json(
            plan
        ),
        "decision_hibrida": (
            convertir_json(
                decision
            )
        ),
        "anylogic": (
            resultado_anylogic
            .a_dict()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            salida,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=== RESULTADO OPERATIVO ANYLOGIC ==="
    )
    print(
        f"Mensaje: {observacion['mensaje']}"
    )
    print(
        f"Entregados: {entregadas}"
    )
    print(
        f"No entregados: {no_entregadas}"
    )
    print(
        f"Viajes: {observacion['viajesTotales']}"
    )
    print(
        "Duración: "
        f"{float(observacion['tiempoSimuladoMin']):.3f} min"
    )
    print(
        "Costo AnyLogic: "
        f"{float(observacion['costoTotal']):.6f}"
    )

    print()
    print(
        "FASE 10D HÍBRIDO–ANYLOGIC: OK"
    )
    print(
        f"JSON: {output_path}"
    )


if __name__ == "__main__":
    main()