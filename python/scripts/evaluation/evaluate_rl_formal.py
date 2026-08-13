from __future__ import annotations

import argparse
import csv
import json
import sys

from math import isfinite
from pathlib import Path
from statistics import (
    mean,
    median,
    pstdev,
)
from typing import Any


# ============================================================
# CONFIGURACIÓN DE LA RAÍZ DEL PROYECTO
# ============================================================
#
# Ubicación de este archivo:
#
# python/
# └── scripts/
#     └── evaluation/
#         └── evaluate_rl_formal.py
#
# parents[0] = evaluation
# parents[1] = scripts
# parents[2] = python
#
# Debemos agregar python/ al sys.path antes de importar planner.

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


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================
#
# Estos imports deben aparecer después de modificar sys.path.

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
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)

from planner.rl.planner import (  # noqa: E402
    RLPlanner,
)


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación formal estimada de "
            "RL, Greedy, Random y GA."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Ruta al modelo principal de RL."
        ),
    )

    parser.add_argument(
        "--comparison-model",
        type=str,
        default="",
        help=(
            "Ruta opcional a otro modelo RL "
            "para utilizar como comparación."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help=(
            "Cantidad de instancias que se "
            "evaluarán."
        ),
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=200_000,
        help=(
            "Primera seed del conjunto de "
            "evaluación."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Directorio donde se guardarán "
            "los resultados."
        ),
    )

    args = parser.parse_args()

    if not args.model.strip():
        parser.error(
            "--model no puede estar vacío."
        )

    if args.episodes <= 0:
        parser.error(
            "--episodes debe ser > 0."
        )

    if not args.output_dir.strip():
        parser.error(
            "--output-dir no puede estar vacío."
        )

    return args


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def resolver_ruta(
    ruta_recibida: str,
) -> Path:
    ruta = Path(
        ruta_recibida
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
    if not valores:
        raise ValueError(
            "No se puede calcular un percentil "
            "sobre una lista vacía."
        )

    if not (
        0.0
        <= proporcion
        <= 1.0
    ):
        raise ValueError(
            "La proporción debe estar entre "
            "0 y 1."
        )

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
    if not valores:
        raise ValueError(
            "No se puede resumir una lista vacía."
        )

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


def validar_numero_finito_no_negativo(
    nombre: str,
    valor: float,
    seed: int,
) -> None:
    if not isfinite(
        valor
    ):
        raise RuntimeError(
            f"{nombre} produjo un valor no finito "
            f"en seed={seed}: {valor}"
        )

    if valor < 0.0:
        raise RuntimeError(
            f"{nombre} produjo un valor negativo "
            f"en seed={seed}: {valor}"
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main(
) -> None:
    args = parse_args()

    model_path = resolver_ruta(
        args.model
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "No existe el modelo principal: "
            f"{model_path}"
        )

    comparison_model_path: (
        Path | None
    ) = None

    if args.comparison_model.strip():
        comparison_model_path = resolver_ruta(
            args.comparison_model
        )

        if not comparison_model_path.is_file():
            raise FileNotFoundError(
                "No existe el modelo de comparación: "
                f"{comparison_model_path}"
            )

    output_dir = resolver_ruta(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuracion_generador = (
        ConfiguracionGeneradorInstancias(
            min_pedidos_finales=4,
            max_pedidos_finales=12,
        )
    )

    generador = GeneradorInstanciasRL(
        configuracion_generador
    )

    planner_rl = RLPlanner(
        model_path=model_path,
        deterministic=True,
    )

    planner_comparacion: (
        RLPlanner | None
    ) = None

    if comparison_model_path is not None:
        planner_comparacion = RLPlanner(
            model_path=(
                comparison_model_path
            ),
            deterministic=True,
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

    nombres = [
        "RL_CURRICULUM",
        "GREEDY",
        "RANDOM",
        "GA",
    ]

    if planner_comparacion is not None:
        nombres.append(
            "RL_SMOKE"
        )

    costos: dict[
        str,
        list[float],
    ] = {
        nombre: []
        for nombre in nombres
    }

    tiempos: dict[
        str,
        list[float],
    ] = {
        nombre: []
        for nombre in nombres
    }

    gaps_greedy: dict[
        str,
        list[float],
    ] = {
        nombre: []
        for nombre in nombres
    }

    victorias: dict[
        str,
        int,
    ] = {
        nombre: 0
        for nombre in nombres
    }

    registros: list[
        dict[str, Any]
    ] = []

    print("")
    print(
        "=== EVALUACIÓN FORMAL ESTIMADA ==="
    )

    print(
        f"Raíz Python: {PYTHON_ROOT}"
    )

    print(
        f"Modelo principal: {model_path}"
    )

    if comparison_model_path is not None:
        print(
            "Modelo de comparación: "
            f"{comparison_model_path}"
        )

    print(
        f"Episodios: {args.episodes}"
    )

    print(
        f"Seed inicial: {args.seed_start}"
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

        planes = {
            "RL_CURRICULUM": (
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

        if planner_comparacion is not None:
            planes[
                "RL_SMOKE"
            ] = (
                planner_comparacion
                .generar_plan(
                    instancia
                )
            )

        costos_instancia: dict[
            str,
            float,
        ] = {}

        tiempos_instancia: dict[
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

            costo = (
                plan.costo_estimado
            )

            tiempo_ms = (
                plan.tiempo_computo_ms
            )

            validar_numero_finito_no_negativo(
                nombre=(
                    f"Costo {nombre}"
                ),
                valor=costo,
                seed=seed,
            )

            validar_numero_finito_no_negativo(
                nombre=(
                    f"Tiempo {nombre}"
                ),
                valor=tiempo_ms,
                seed=seed,
            )

            costos_instancia[
                nombre
            ] = costo

            tiempos_instancia[
                nombre
            ] = tiempo_ms

            costos[
                nombre
            ].append(
                costo
            )

            tiempos[
                nombre
            ].append(
                tiempo_ms
            )

        costo_greedy = (
            costos_instancia[
                "GREEDY"
            ]
        )

        gaps_instancia: dict[
            str,
            float,
        ] = {}

        for nombre in nombres:
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

        registro: dict[
            str,
            Any,
        ] = {
            "seed": seed,

            "cantidad_pedidos": len(
                instancia.pedidos
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

        registros.append(
            registro
        )

        texto_comparacion = ""

        if (
            "RL_SMOKE"
            in costos_instancia
        ):
            texto_comparacion = (
                " | RL_SMOKE="
                f"{costos_instancia['RL_SMOKE']:.3f}"
            )

        print(
            f"[{indice + 1}/{args.episodes}] "
            f"seed={seed} "
            f"| tareas={len(instancia.pedidos)} "
            f"| RL={costos_instancia['RL_CURRICULUM']:.3f} "
            f"| Greedy={costos_instancia['GREEDY']:.3f} "
            f"| Random={costos_instancia['RANDOM']:.3f} "
            f"| GA={costos_instancia['GA']:.3f}"
            f"{texto_comparacion}"
        )

    resumen: dict[
        str,
        dict[str, Any],
    ] = {}

    for nombre in nombres:
        resumen[
            nombre
        ] = {
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

    resultado: dict[
        str,
        Any,
    ] = {
        "modelo": str(
            model_path
        ),

        "modelo_comparacion": (
            str(
                comparison_model_path
            )
            if comparison_model_path
            is not None
            else ""
        ),

        "episodios": (
            args.episodes
        ),

        "seed_start": (
            args.seed_start
        ),

        "configuracion_generador": {
            "min_pedidos_finales": 4,
            "max_pedidos_finales": 12,
        },

        "resumen": resumen,

        "registros": registros,
    }

    json_path = (
        output_dir
        / "formal_evaluation.json"
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
        / "formal_evaluation.csv"
    )

    encabezado = [
        "seed",
        "cantidad_pedidos",
    ]

    encabezado.extend(
        f"costo_{nombre}"
        for nombre in nombres
    )

    encabezado.extend(
        f"tiempo_ms_{nombre}"
        for nombre in nombres
    )

    encabezado.extend(
        f"gap_greedy_{nombre}"
        for nombre in nombres
    )

    encabezado.append(
        "ganadores"
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
            encabezado
        )

        for registro in registros:
            fila: list[Any] = [
                registro["seed"],
                registro["cantidad_pedidos"],
            ]

            fila.extend(
                registro[
                    "costos"
                ][nombre]
                for nombre in nombres
            )

            fila.extend(
                registro[
                    "tiempos_ms"
                ][nombre]
                for nombre in nombres
            )

            fila.extend(
                registro[
                    "gaps_greedy"
                ][nombre]
                for nombre in nombres
            )

            fila.append(
                "|".join(
                    registro[
                        "ganadores"
                    ]
                )
            )

            writer.writerow(
                fila
            )

    print("")
    print(
        "=== RESUMEN FORMAL ==="
    )

    for nombre in nombres:
        datos = resumen[
            nombre
        ]

        print(
            f"{nombre}: "
            "promedio="
            f"{datos['costo']['promedio']:.6f} "
            "| mediana="
            f"{datos['costo']['mediana']:.6f} "
            "| p90="
            f"{datos['costo']['p90']:.6f} "
            "| gap promedio="
            f"{datos['gap_greedy']['promedio']:.6f} "
            "| victorias="
            f"{datos['victorias']}"
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