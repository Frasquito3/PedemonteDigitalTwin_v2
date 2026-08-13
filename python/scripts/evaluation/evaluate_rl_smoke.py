from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from statistics import mean
from typing import Any


# ============================================================
# RAÍZ DEL PAQUETE PYTHON
# ============================================================
#
# Archivo actual:
#
# python/scripts/evaluation/evaluate_rl_smoke.py
#
# parents[0] = evaluation
# parents[1] = scripts
# parents[2] = python

PYTHON_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_ROOT),
    )


from planner.algorithms.ga import (  # noqa: E402
    ConfiguracionGA,
    generar_plan_ga,
)

from planner.algorithms.greedy import (  # noqa: E402
    generar_plan_greedy,
)

from planner.algorithms.random_feasible import (  # noqa: E402
    generar_plan_random,
)

from planner.domain.validator import (  # noqa: E402
    validar_plan,
)

from planner.rl.instance_generator import (  # noqa: E402
    GeneradorInstanciasRL,
)

from planner.rl.planner import (  # noqa: E402
    RLPlanner,
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación de humo del modelo RL "
            "contra Greedy, Random y GA."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="phase9b_smoke",
        help=(
            "Nombre del directorio ubicado dentro "
            "de rl_artifacts."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help=(
            "Cantidad de instancias utilizadas "
            "en la evaluación."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "best",
            "final",
        ],
        default="best",
        help=(
            "Modelo que se cargará desde el "
            "directorio de la corrida."
        ),
    )

    args = parser.parse_args()

    if not args.run_name.strip():
        parser.error(
            "--run-name no puede estar vacío."
        )

    if args.episodes <= 0:
        parser.error(
            "--episodes debe ser > 0."
        )

    return args


def resolver_model_path(
    run_dir: Path,
    tipo_modelo: str,
) -> Path:
    if tipo_modelo == "best":
        model_path = (
            run_dir
            / "best"
            / "best_model.zip"
        )

    else:
        model_path = (
            run_dir
            / "final_model.zip"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "No existe el modelo solicitado: "
            f"{model_path}"
        )

    return model_path


def main(
) -> None:
    args = parse_args()

    run_dir = (
        PYTHON_ROOT
        / "rl_artifacts"
        / args.run_name
    )

    if not run_dir.exists():
        raise FileNotFoundError(
            "No existe el directorio de corrida: "
            f"{run_dir}"
        )

    model_path = resolver_model_path(
        run_dir=run_dir,
        tipo_modelo=args.model,
    )

    planner_rl = RLPlanner(
        model_path=model_path,
        deterministic=True,
    )

    generador = GeneradorInstanciasRL()

    configuracion_ga = ConfiguracionGA(
        tamano_poblacion=30,

        generaciones=40,

        tamano_elite=3,

        tamano_torneo=3,

        probabilidad_crossover=0.90,

        probabilidad_mutacion_swap=0.20,

        probabilidad_mutacion_inversion=0.10,

        generaciones_sin_mejora_max=15,
    )

    costos: dict[
        str,
        list[float],
    ] = {
        "RL": [],
        "GREEDY": [],
        "RANDOM": [],
        "GA": [],
    }

    resultados: list[
        dict[str, Any]
    ] = []

    print("")
    print(
        "=== EVALUACIÓN ESTIMADA SMOKE ==="
    )

    print(
        f"Modelo: {model_path}"
    )

    print(
        f"Episodios: {args.episodes}"
    )

    for indice in range(
        args.episodes
    ):
        seed = (
            160_000
            + indice
        )

        instancia = generador.generar(
            seed
        )

        planes = {
            "RL": (
                planner_rl.generar_plan(
                    instancia
                )
            ),

            "GREEDY": (
                generar_plan_greedy(
                    instancia
                )
            ),

            "RANDOM": (
                generar_plan_random(
                    instancia,

                    seed=(
                        seed
                        + 7_001
                    ),
                )
            ),

            "GA": (
                generar_plan_ga(
                    instancia,

                    seed=(
                        seed
                        + 8_001
                    ),

                    configuracion_ga=(
                        configuracion_ga
                    ),
                )
            ),
        }

        costos_instancia: dict[
            str,
            float,
        ] = {}

        for nombre, plan in planes.items():
            validacion = validar_plan(
                instancia,
                plan,
            )

            if not validacion.valido:
                raise RuntimeError(
                    f"{nombre} produjo un plan "
                    f"inválido en seed={seed}: "
                    + " | ".join(
                        validacion.errores
                    )
                )

            costo = plan.costo_estimado

            costos[
                nombre
            ].append(
                costo
            )

            costos_instancia[
                nombre
            ] = costo

        # No se utiliza costos_instancia.get porque
        # dict.get() puede devolver None según su
        # contrato de tipos. La lambda garantiza que
        # el resultado sea siempre float.
        ganador = min(
            costos_instancia,

            key=lambda nombre: (
                costos_instancia[
                    nombre
                ]
            ),
        )

        resultados.append(
            {
                "seed": seed,

                "cantidad_pedidos": len(
                    instancia.pedidos
                ),

                "costos": (
                    costos_instancia
                ),

                "ganador": ganador,
            }
        )

        print(
            f"Seed={seed} "
            f"| tareas={len(instancia.pedidos)} "
            f"| RL={costos_instancia['RL']:.3f} "
            f"| Greedy="
            f"{costos_instancia['GREEDY']:.3f} "
            f"| Random="
            f"{costos_instancia['RANDOM']:.3f} "
            f"| GA={costos_instancia['GA']:.3f} "
            f"| ganador={ganador}"
        )

    resumen: dict[
        str,
        dict[str, float],
    ] = {
        nombre: {
            "promedio": mean(
                valores
            ),

            "minimo": min(
                valores
            ),

            "maximo": max(
                valores
            ),
        }

        for nombre, valores
        in costos.items()
    }

    print("")
    print(
        "=== PROMEDIOS ==="
    )

    for nombre, datos in resumen.items():
        print(
            f"{nombre}: "
            f"{datos['promedio']:.6f}"
        )

    salida: dict[
        str,
        Any,
    ] = {
        "modelo": str(
            model_path
        ),

        "episodios": (
            args.episodes
        ),

        "resumen": resumen,

        "resultados": resultados,
    }

    salida_path = (
        run_dir
        / "smoke_evaluation.json"
    )

    with salida_path.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            salida,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    print("")
    print(
        "Resultado guardado en: "
        f"{salida_path}"
    )


if __name__ == "__main__":
    main()