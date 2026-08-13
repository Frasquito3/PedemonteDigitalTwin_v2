from __future__ import annotations

import argparse
import json
import math
import os

from pathlib import Path
from typing import Any

from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim


PYTHON_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_MODEL = (
    PYTHON_ROOT
    / "anylogic_export"
    / "phase10c_dynamic"
    / "PedemonteDigitalTwin_v2.zip"
)

DEFAULT_OUTPUT = (
    PYTHON_ROOT
    / "rl_artifacts"
    / "phase10c_dynamic"
    / "dynamic_contract_result.json"
)

LAT_CORRALON = -32.8495006
LON_CORRALON = -60.722653

PROTOCOL_VERSION = 1


EXPECTED_CONFIGURATION_FIELDS = {
    "seedEjecucion",
    "instanciaVector",
}

EXPECTED_ACTION_FIELDS = {
    "accionCodigo",
    "planVector",
}

EXPECTED_OBSERVATION_FIELDS = {
    "protocoloVersion",
    "configurado",
    "accionRecibida",
    "instanciaAceptada",
    "planAceptado",
    "cantidadPedidos",
    "ejecucionEnCurso",
    "ejecucionFinalizada",
    "error",
    "mensaje",
    "costoTotal",
    "tareasEntregadas",
    "tareasNoEntregadas",
    "viajesTotales",
    "tiempoSimuladoMin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida el contrato dinámico "
            "Python-Alpyne-AnyLogic."
        )
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
    )

    parser.add_argument(
        "--java",
        default="",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3_001,
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    return parser.parse_args()


def resolver_ruta(
    texto: str,
) -> Path:
    ruta = Path(
        texto
    ).expanduser()

    if not ruta.is_absolute():
        ruta = Path.cwd() / ruta

    return ruta.resolve()


def resolver_java(
    recibido: str,
) -> Path:
    if recibido.strip():
        ruta = resolver_ruta(
            recibido
        )

        if not ruta.is_file():
            raise FileNotFoundError(
                f"No existe java.exe: {ruta}"
            )

        return ruta

    raices = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ]

    for raiz_texto in raices:
        if not raiz_texto:
            continue

        raiz = Path(
            raiz_texto
        )

        for anylogic_dir in raiz.glob(
            "AnyLogic*"
        ):
            candidatos = (
                anylogic_dir
                / "jre"
                / "bin"
                / "java.exe",

                anylogic_dir
                / "jdk"
                / "bin"
                / "java.exe",

                anylogic_dir
                / "runtime"
                / "bin"
                / "java.exe",
            )

            for candidato in candidatos:
                if candidato.is_file():
                    return candidato.resolve()

    raise FileNotFoundError(
        "No se encontró el Java de AnyLogic. "
        "Usá --java con la ruta a java.exe."
    )


def construir_instancia_vector(
) -> list[float]:
    pedidos = [
        [
            0.0,  # índice
            0.0,  # original
            1.0,  # parte
            1.0,  # total partes
            1.0,  # unidades
            0.0,  # volcador
            LAT_CORRALON + 0.0008,
            LON_CORRALON + 0.0005,
            -1.0,
            -1.0,
        ],
        [
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            0.0,
            LAT_CORRALON - 0.0007,
            LON_CORRALON - 0.0006,
            -1.0,
            -1.0,
        ],
    ]

    vector = [
        float(PROTOCOL_VERSION),
        0.0,  # turno mañana
        float(len(pedidos)),
        8.0,
        2.0,
        LAT_CORRALON,
        LON_CORRALON,
        0.0,
    ]

    for pedido in pedidos:
        vector.extend(
            pedido
        )

    return vector


def construir_plan_vector(
) -> list[float]:
    asignaciones = [
        [
            0.0,  # camión
            1.0,  # viaje
            1.0,  # orden
            0.0,  # pedido
        ],
        [
            0.0,
            1.0,
            2.0,
            1.0,
        ],
    ]

    vector = [
        float(PROTOCOL_VERSION),
        float(len(asignaciones)),
        2.0,  # GREEDY
        0.0,  # costo estimado
        0.1,  # tiempo de cómputo
    ]

    for asignacion in asignaciones:
        vector.extend(
            asignacion
        )

    return vector


def exigir_status(
    status: Any,
    operacion: str,
) -> Any:
    if status is None:
        raise RuntimeError(
            f"{operacion} no devolvió status."
        )

    return status


def contiene_estado(
    status: Any,
    estado: EngineState,
) -> bool:
    return bool(
        status.state & estado
    )


def validar_schema(
    sim: AnyLogicSim,
) -> dict[str, list[str]]:
    schema = sim.schema

    if schema is None:
        raise RuntimeError(
            "Alpyne no devolvió el schema."
        )

    configuration = set(
        schema.configuration
    )

    action = set(
        schema.action
    )

    observation = set(
        schema.observation
    )

    if configuration != (
        EXPECTED_CONFIGURATION_FIELDS
    ):
        raise RuntimeError(
            "Configuration inesperada: "
            f"{sorted(configuration)}"
        )

    if action != (
        EXPECTED_ACTION_FIELDS
    ):
        raise RuntimeError(
            "Action inesperada: "
            f"{sorted(action)}"
        )

    if observation != (
        EXPECTED_OBSERVATION_FIELDS
    ):
        raise RuntimeError(
            "Observation inesperada: "
            f"{sorted(observation)}"
        )

    return {
        "configuration": sorted(
            configuration
        ),
        "action": sorted(
            action
        ),
        "observation": sorted(
            observation
        ),
    }


def validar_final(
    status: Any,
) -> dict[str, Any]:
    observacion = dict(
        status.observation
    )

    if contiene_estado(
        status,
        EngineState.ERROR,
    ):
        raise RuntimeError(
            f"AnyLogic terminó con ERROR: "
            f"{status.message}"
        )

    if not contiene_estado(
        status,
        EngineState.FINISHED,
    ):
        raise RuntimeError(
            "Estado final inesperado: "
            f"{status.state}"
        )

    if observacion["error"]:
        raise RuntimeError(
            "El modelo rechazó los datos: "
            f"{observacion['mensaje']}"
        )

    if not observacion[
        "instanciaAceptada"
    ]:
        raise RuntimeError(
            "La instancia no fue aceptada."
        )

    if not observacion[
        "planAceptado"
    ]:
        raise RuntimeError(
            "El plan no fue aceptado."
        )

    if observacion[
        "cantidadPedidos"
    ] != 2:
        raise RuntimeError(
            "Cantidad de pedidos incorrecta: "
            f"{observacion['cantidadPedidos']}"
        )

    if observacion[
        "tareasEntregadas"
    ] != 2:
        raise RuntimeError(
            "No se entregaron las dos tareas."
        )

    if observacion[
        "tareasNoEntregadas"
    ] != 0:
        raise RuntimeError(
            "Aparecieron tareas no entregadas."
        )

    if observacion[
        "viajesTotales"
    ] != 1:
        raise RuntimeError(
            "Se esperaba exactamente un viaje."
        )

    costo = float(
        observacion[
            "costoTotal"
        ]
    )

    if (
        not math.isfinite(costo)
        or costo < 0.0
    ):
        raise RuntimeError(
            f"Costo inválido: {costo}"
        )

    duracion = float(
        observacion[
            "tiempoSimuladoMin"
        ]
    )

    if (
        duracion <= 0.0
        or duracion >= 59.0
    ):
        raise RuntimeError(
            "Duración operativa inválida: "
            f"{duracion}"
        )

    return observacion


def main() -> None:
    args = parse_args()

    model_path = resolver_ruta(
        args.model
    )

    java_path = resolver_java(
        args.java
    )

    output_path = resolver_ruta(
        args.output
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"No existe el modelo: {model_path}"
        )

    instancia_vector = (
        construir_instancia_vector()
    )

    plan_vector = (
        construir_plan_vector()
    )

    print()
    print(
        "=== CONTRATO DINÁMICO ALPYNE ==="
    )
    print(
        f"Modelo: {model_path}"
    )
    print(
        f"Java: {java_path}"
    )
    print(
        f"Seed: {args.seed}"
    )

    sim = AnyLogicSim(
        model_path=str(
            model_path
        ),
        java_exe=str(
            java_path
        ),
        auto_lock=True,
        auto_finish=True,
        py_log_level=True,
        java_log_level=True,
        log_id="phase10c",
        lock_defaults={
            "flag": (
                EngineState.PAUSED
                | EngineState.FINISHED
                | EngineState.ERROR
            ),
            "timeout": 120,
        },
        max_server_await_time=30.0,
    )

    schema = validar_schema(
        sim
    )

    print()
    print(
        "Schema dinámico validado."
    )

    status_inicial = exigir_status(
        sim.reset(
            seedEjecucion=args.seed,
            instanciaVector=(
                instancia_vector
            ),
        ),
        "reset",
    )

    if not contiene_estado(
        status_inicial,
        EngineState.PAUSED,
    ):
        raise RuntimeError(
            "El modelo no se detuvo en "
            "el punto de decisión."
        )

    observacion_inicial = dict(
        status_inicial.observation
    )

    if observacion_inicial[
        "error"
    ]:
        raise RuntimeError(
            observacion_inicial[
                "mensaje"
            ]
        )

    print(
        "Configuration recibida."
    )
    print(
        "Enviando instancia de 2 pedidos "
        "y plan de 1 viaje..."
    )

    status_final = exigir_status(
        sim.take_action(
            accionCodigo=2,
            planVector=plan_vector,
        ),
        "take_action",
    )

    observacion_final = validar_final(
        status_final
    )

    resultado = {
        "modelo": str(
            model_path
        ),
        "java": str(
            java_path
        ),
        "seed": args.seed,
        "schema": schema,
        "instancia_vector": (
            instancia_vector
        ),
        "plan_vector": (
            plan_vector
        ),
        "estado_inicial": str(
            status_inicial.state
        ),
        "observacion_inicial": (
            observacion_inicial
        ),
        "estado_final": str(
            status_final.state
        ),
        "observacion_final": (
            observacion_final
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
            resultado,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=== RESULTADO DINÁMICO ==="
    )
    print(
        f"Mensaje: "
        f"{observacion_final['mensaje']}"
    )
    print(
        f"Pedidos: "
        f"{observacion_final['cantidadPedidos']}"
    )
    print(
        f"Entregados: "
        f"{observacion_final['tareasEntregadas']}"
    )
    print(
        f"No entregados: "
        f"{observacion_final['tareasNoEntregadas']}"
    )
    print(
        f"Viajes: "
        f"{observacion_final['viajesTotales']}"
    )
    print(
        f"Duración: "
        f"{observacion_final['tiempoSimuladoMin']:.3f} min"
    )
    print(
        f"Costo: "
        f"{observacion_final['costoTotal']:.6f}"
    )
    print()
    print(
        "CONTRATO DINÁMICO "
        "PYTHON–ANYLOGIC: OK"
    )
    print(
        f"JSON: {output_path}"
    )


if __name__ == "__main__":
    main()