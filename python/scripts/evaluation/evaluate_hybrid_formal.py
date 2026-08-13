from __future__ import annotations

import argparse
import csv
import json
import sys

from collections import Counter
from math import isfinite
from pathlib import Path
from statistics import (
    mean,
    median,
    pstdev,
)
from typing import Any


PYTHON_ROOT = Path(
    __file__
).resolve().parents[2]

PLANNER_DIR = (
    PYTHON_ROOT
    / "planner"
)

if not PLANNER_DIR.is_dir():
    raise RuntimeError(
        "No se encontró el paquete planner. "
        f"Raíz calculada: {PYTHON_ROOT}"
    )

python_root_texto = str(
    PYTHON_ROOT
)

if python_root_texto not in sys.path:
    sys.path.insert(
        0,
        python_root_texto,
    )


from planner.algorithms.ga import (  # noqa: E402
    ConfiguracionGA,
    generar_plan_ga,
)

from planner.algorithms.greedy import (  # noqa: E402
    generar_plan_greedy,
)

from planner.algorithms.hybrid_rl_greedy import (  # noqa: E402
    HybridRLGreedyPlanner,
)

from planner.algorithms.random_feasible import (  # noqa: E402
    generar_plan_random,
)

from planner.domain.validator import (  # noqa: E402
    validar_plan,
)

from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)

from planner.rl.planner import (  # noqa: E402
    RLPlanner,
)


NOMBRES_ALGORITMOS = (
    "HYBRID",
    "RL",
    "GREEDY",
    "RANDOM",
    "GA",
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación formal del planificador "
            "híbrido RL-Greedy."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error(
            "--episodes debe ser > 0."
        )

    if not args.model.strip():
        parser.error(
            "--model no puede estar vacío."
        )

    if not args.output_dir.strip():
        parser.error(
            "--output-dir no puede estar vacío."
        )

    return args


def resolver_ruta(
    ruta_texto: str,
) -> Path:
    ruta = Path(
        ruta_texto
    )

    if not ruta.is_absolute():
        ruta = (
            Path.cwd()
            / ruta
        )

    return ruta.resolve()


def percentil(
    valores: list[float],
    proporcion: float,
) -> float:
    ordenados = sorted(
        valores
    )

    posicion = round(
        proporcion
        * (
            len(ordenados)
            - 1
        )
    )

    return ordenados[
        posicion
    ]


def resumir(
    valores: list[float],
) -> dict[str, float]:
    return {
        "promedio": mean(
            valores
        ),

        "mediana": median(
            valores
        ),

        "desvio": (
            pstdev(
                valores
            )
            if len(valores) > 1
            else 0.0
        ),

        "minimo": min(
            valores
        ),

        "maximo": max(
            valores
        ),

        "p90": percentil(
            valores,
            0.90,
        ),
    }


def validar_plan_y_metricas(
    nombre: str,
    instancia,
    plan,
    seed: int,
) -> None:
    validacion = validar_plan(
        instancia,
        plan,
    )

    if not validacion.valido:
        raise RuntimeError(
            f"{nombre} produjo un plan inválido "
            f"en seed={seed}: "
            + " | ".join(
                validacion.errores
            )
        )

    if not isfinite(
        plan.costo_estimado
    ):
        raise RuntimeError(
            f"{nombre} produjo costo no finito "
            f"en seed={seed}."
        )

    if plan.costo_estimado < 0.0:
        raise RuntimeError(
            f"{nombre} produjo costo negativo "
            f"en seed={seed}."
        )

    if not isfinite(
        plan.tiempo_computo_ms
    ):
        raise RuntimeError(
            f"{nombre} produjo tiempo no finito "
            f"en seed={seed}."
        )


def main(
) -> None:
    args = parse_args()

    model_path = resolver_ruta(
        args.model
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "No existe el modelo RL: "
            f"{model_path}"
        )

    output_dir = resolver_ruta(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generador = GeneradorInstanciasRL(
        ConfiguracionGeneradorInstancias(
            min_pedidos_finales=4,
            max_pedidos_finales=12,
        )
    )

    planner_rl = RLPlanner(
        model_path=model_path,
        deterministic=True,
    )

    planner_hibrido = (
        HybridRLGreedyPlanner(
            planner_rl=planner_rl
        )
    )

    configuracion_ga = ConfiguracionGA(
        tamano_poblacion=40,
        generaciones=60,
        tamano_elite=4,
        tamano_torneo=3,
        probabilidad_crossover=0.90,
        probabilidad_mutacion_swap=0.20,
        probabilidad_mutacion_inversion=0.10,
        generaciones_sin_mejora_max=20,
    )

    costos = {
        nombre: []
        for nombre
        in NOMBRES_ALGORITMOS
    }

    tiempos = {
        nombre: []
        for nombre
        in NOMBRES_ALGORITMOS
    }

    gaps_greedy = {
        nombre: []
        for nombre
        in NOMBRES_ALGORITMOS
    }

    victorias = {
        nombre: 0
        for nombre
        in NOMBRES_ALGORITMOS
    }

    decisiones = Counter()

    registros: list[
        dict[str, Any]
    ] = []

    garantia_cumplida = True

    print("")
    print(
        "=== EVALUACIÓN HÍBRIDA FORMAL ==="
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
            args.seed_start
            + indice
        )

        instancia = generador.generar(
            seed
        )

        plan_hibrido = (
            planner_hibrido
            .generar_plan(
                instancia
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

        plan_rl = (
            planner_rl
            .generar_plan(
                instancia
            )
        )

        plan_greedy = (
            generar_plan_greedy(
                instancia
            )
        )

        plan_random = (
            generar_plan_random(
                instancia,

                seed=(
                    seed
                    + 7_001
                ),
            )
        )

        plan_ga = (
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
        )

        planes = {
            "HYBRID": plan_hibrido,
            "RL": plan_rl,
            "GREEDY": plan_greedy,
            "RANDOM": plan_random,
            "GA": plan_ga,
        }

        costos_instancia: dict[
            str,
            float,
        ] = {}

        tiempos_instancia: dict[
            str,
            float,
        ] = {}

        for nombre, plan in planes.items():
            validar_plan_y_metricas(
                nombre=nombre,
                instancia=instancia,
                plan=plan,
                seed=seed,
            )

            costos_instancia[
                nombre
            ] = (
                plan.costo_estimado
            )

            tiempos_instancia[
                nombre
            ] = (
                plan.tiempo_computo_ms
            )

            costos[
                nombre
            ].append(
                plan.costo_estimado
            )

            tiempos[
                nombre
            ].append(
                plan.tiempo_computo_ms
            )

        costo_greedy = (
            costos_instancia[
                "GREEDY"
            ]
        )

        gaps_instancia = {}

        for nombre in NOMBRES_ALGORITMOS:
            gap = (
                costos_instancia[nombre]
                - costo_greedy
            ) / max(
                abs(
                    costo_greedy
                ),
                1.0,
            )

            gaps_instancia[
                nombre
            ] = gap

            gaps_greedy[
                nombre
            ].append(
                gap
            )

        if (
            costos_instancia["HYBRID"]
            >
            costo_greedy
            + 1e-9
        ):
            garantia_cumplida = False

            raise RuntimeError(
                "El híbrido empeoró Greedy "
                f"en seed={seed}."
            )

        mejor_costo = min(
            costos_instancia.values()
        )

        ganadores = [
            nombre

            for nombre, costo
            in costos_instancia.items()

            if abs(
                costo
                - mejor_costo
            ) <= 1e-9
        ]

        for ganador in ganadores:
            victorias[
                ganador
            ] += 1

        fuente = (
            decision
            .fuente_seleccionada
            .value
        )

        decisiones[
            fuente
        ] += 1

        registros.append(
            {
                "seed": seed,

                "cantidad_pedidos": len(
                    instancia.pedidos
                ),

                "fuente_hibrida": fuente,

                "motivo_hibrido": (
                    decision
                    .motivo
                    .value
                ),

                "costos": (
                    costos_instancia
                ),

                "tiempos_ms": (
                    tiempos_instancia
                ),

                "gaps_greedy": (
                    gaps_instancia
                ),

                "ganadores": (
                    ganadores
                ),
            }
        )

        print(
            f"[{indice + 1}/{args.episodes}] "
            f"seed={seed} "
            f"| tareas={len(instancia.pedidos)} "
            f"| híbrido={costos_instancia['HYBRID']:.3f} "
            f"| fuente={fuente} "
            f"| RL={costos_instancia['RL']:.3f} "
            f"| G={costo_greedy:.3f} "
            f"| GA={costos_instancia['GA']:.3f}"
        )

    resumen = {
        nombre: {
            "costo": resumir(
                costos[nombre]
            ),

            "tiempo_ms": resumir(
                tiempos[nombre]
            ),

            "gap_greedy": resumir(
                gaps_greedy[nombre]
            ),

            "victorias": (
                victorias[nombre]
            ),
        }

        for nombre
        in NOMBRES_ALGORITMOS
    }

    resumen_por_tamano = {}

    tamanos = sorted(
        {
            registro[
                "cantidad_pedidos"
            ]

            for registro in registros
        }
    )

    for tamano in tamanos:
        registros_tamano = [
            registro

            for registro in registros

            if (
                registro[
                    "cantidad_pedidos"
                ]
                == tamano
            )
        ]

        resumen_por_tamano[
            str(tamano)
        ] = {
            "cantidad_instancias": len(
                registros_tamano
            ),

            "costo_promedio_hibrido": mean(
                registro[
                    "costos"
                ]["HYBRID"]

                for registro
                in registros_tamano
            ),

            "costo_promedio_rl": mean(
                registro[
                    "costos"
                ]["RL"]

                for registro
                in registros_tamano
            ),

            "costo_promedio_greedy": mean(
                registro[
                    "costos"
                ]["GREEDY"]

                for registro
                in registros_tamano
            ),

            "gap_promedio_hibrido": mean(
                registro[
                    "gaps_greedy"
                ]["HYBRID"]

                for registro
                in registros_tamano
            ),
        }

    resultado = {
        "modelo": str(
            model_path
        ),

        "episodios": (
            args.episodes
        ),

        "seed_start": (
            args.seed_start
        ),

        "garantias": {
            "hibrido_nunca_empeora_greedy": (
                garantia_cumplida
            ),
        },

        "decisiones_hibridas": dict(
            decisiones
        ),

        "resumen": resumen,

        "resumen_por_tamano": (
            resumen_por_tamano
        ),

        "registros": registros,
    }

    json_path = (
        output_dir
        / "hybrid_evaluation.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            resultado,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = (
        output_dir
        / "hybrid_evaluation.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as archivo:
        writer = csv.writer(
            archivo
        )

        writer.writerow(
            [
                "seed",
                "cantidad_pedidos",
                "fuente_hibrida",
                "costo_hibrido",
                "costo_rl",
                "costo_greedy",
                "costo_random",
                "costo_ga",
                "gap_hibrido_greedy",
                "ganadores",
            ]
        )

        for registro in registros:
            writer.writerow(
                [
                    registro["seed"],

                    registro[
                        "cantidad_pedidos"
                    ],

                    registro[
                        "fuente_hibrida"
                    ],

                    registro[
                        "costos"
                    ]["HYBRID"],

                    registro[
                        "costos"
                    ]["RL"],

                    registro[
                        "costos"
                    ]["GREEDY"],

                    registro[
                        "costos"
                    ]["RANDOM"],

                    registro[
                        "costos"
                    ]["GA"],

                    registro[
                        "gaps_greedy"
                    ]["HYBRID"],

                    "|".join(
                        registro[
                            "ganadores"
                        ]
                    ),
                ]
            )

    print("")
    print(
        "=== RESUMEN HÍBRIDO ==="
    )

    for nombre in NOMBRES_ALGORITMOS:
        datos = resumen[
            nombre
        ]

        print(
            f"{nombre}: "
            f"promedio="
            f"{datos['costo']['promedio']:.6f} "
            f"| mediana="
            f"{datos['costo']['mediana']:.6f} "
            f"| p90="
            f"{datos['costo']['p90']:.6f} "
            f"| victorias="
            f"{datos['victorias']}"
        )

    print("")
    print(
        "Decisiones híbridas: "
        f"{dict(decisiones)}"
    )

    print(
        "Garantía híbrido <= Greedy: "
        f"{garantia_cumplida}"
    )

    print("")
    print(
        f"JSON: {json_path}"
    )

    print(
        f"CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()