from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any


PYTHON_ROOT = Path(__file__).resolve().parents[2]

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


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
    ModoDemandaGeografica,
)
from planner.rl.planner import RLPlanner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación de humo del modelo RL contra "
            "Greedy, Random y GA."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="phase9b_smoke",
        help=(
            "Nombre del directorio ubicado dentro de "
            "python/rl_artifacts."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=160_000,
    )

    parser.add_argument(
        "--model",
        choices=[
            "best",
            "final",
        ],
        default="best",
    )

    parser.add_argument(
        "--demand-mode",
        choices=[
            ModoDemandaGeografica.SINTETICA.value,
            ModoDemandaGeografica.REAL.value,
        ],
        default=ModoDemandaGeografica.SINTETICA.value,
    )

    parser.add_argument(
        "--demand-dataset",
        type=str,
        default="",
    )

    parser.add_argument(
        "--large-order-probability",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    if not args.run_name.strip():
        parser.error("--run-name no puede estar vacío.")

    if args.episodes <= 0:
        parser.error("--episodes debe ser > 0.")

    if (
        args.large_order_probability is not None
        and not 0.0 <= args.large_order_probability <= 1.0
    ):
        parser.error(
            "--large-order-probability debe estar entre 0 y 1."
        )

    return args


def resolver_model_path(
    run_dir: Path,
    tipo_modelo: str,
) -> Path:
    if tipo_modelo == "best":
        model_path = run_dir / "best" / "best_model.zip"
    else:
        model_path = run_dir / "final_model.zip"

    if not model_path.is_file():
        raise FileNotFoundError(
            "No existe el modelo solicitado: "
            f"{model_path}"
        )

    return model_path


def resolver_probabilidad_pedido_grande(
    args: argparse.Namespace,
    modo: ModoDemandaGeografica,
) -> float:
    if args.large_order_probability is not None:
        return float(args.large_order_probability)

    if modo == ModoDemandaGeografica.REAL:
        return 0.0

    return 0.10


def crear_configuracion_generador(
    args: argparse.Namespace,
) -> ConfiguracionGeneradorInstancias:
    modo = ModoDemandaGeografica(args.demand_mode)

    return ConfiguracionGeneradorInstancias(
        min_pedidos_finales=4,
        max_pedidos_finales=8,
        probabilidad_pedido_mayor_capacidad=(
            resolver_probabilidad_pedido_grande(
                args=args,
                modo=modo,
            )
        ),
        modo_demanda_geografica=modo,
        ruta_demanda_real=args.demand_dataset.strip(),
    )


def calcular_sha256(ruta: Path | None) -> str:
    if ruta is None:
        return ""

    digest = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b"",
        ):
            digest.update(bloque)

    return digest.hexdigest()


def serializar_configuracion_generador(
    configuracion: ConfiguracionGeneradorInstancias,
) -> dict[str, Any]:
    datos: dict[str, Any] = {}

    for clave, valor in asdict(configuracion).items():
        datos[str(clave)] = valor

    datos["modo_demanda_geografica"] = (
        configuracion.modo_demanda_geografica.value
    )

    return datos


def validar_plan_evaluado(
    nombre: str,
    instancia: Any,
    plan: Any,
    seed: int,
) -> None:
    validacion = validar_plan(
        instancia,
        plan,
    )

    if not validacion.valido:
        raise RuntimeError(
            f"{nombre} produjo un plan inválido en seed={seed}: "
            + " | ".join(validacion.errores)
        )

    if not math.isfinite(plan.costo_estimado):
        raise RuntimeError(
            f"{nombre} produjo costo no finito en seed={seed}."
        )

    if plan.costo_estimado < 0.0:
        raise RuntimeError(
            f"{nombre} produjo costo negativo en seed={seed}."
        )


def main() -> None:
    args = parse_args()

    run_dir = (
        PYTHON_ROOT
        / "rl_artifacts"
        / args.run_name
    )

    if not run_dir.is_dir():
        raise FileNotFoundError(
            "No existe el directorio de corrida: "
            f"{run_dir}"
        )

    model_path = resolver_model_path(
        run_dir=run_dir,
        tipo_modelo=args.model,
    )

    configuracion_generador = (
        crear_configuracion_generador(args)
    )

    generador = GeneradorInstanciasRL(
        configuracion=configuracion_generador
    )

    planner_rl = RLPlanner(
        model_path=model_path,
        deterministic=True,
    )

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

    costos: dict[str, list[float]] = {
        "RL": [],
        "GREEDY": [],
        "RANDOM": [],
        "GA": [],
    }

    resultados: list[dict[str, Any]] = []

    ruta_demanda_real = (
        generador.ruta_demanda_real_resuelta
    )
    catalogo_demanda_real = (
        generador.catalogo_demanda_real
    )

    print("")
    print("=== EVALUACIÓN ESTIMADA SMOKE ===")
    print(f"Modelo: {model_path}")
    print(f"Episodios: {args.episodes}")
    print(f"Seed inicial: {args.seed_start}")
    print(
        "Demanda geográfica: "
        f"{configuracion_generador.modo_demanda_geografica.value}"
    )

    if ruta_demanda_real is not None:
        print(f"Dataset: {ruta_demanda_real}")
        print(
            "Registros aptos: "
            f"{len(catalogo_demanda_real or ())}"
        )

    for indice in range(args.episodes):
        seed = args.seed_start + indice
        instancia = generador.generar(seed)

        planes = {
            "RL": planner_rl.generar_plan(instancia),
            "GREEDY": generar_plan_greedy(instancia),
            "RANDOM": generar_plan_random(
                instancia,
                seed=seed + 7_001,
            ),
            "GA": generar_plan_ga(
                instancia,
                seed=seed + 8_001,
                configuracion_ga=configuracion_ga,
            ),
        }

        costos_instancia: dict[str, float] = {}

        for nombre, plan in planes.items():
            validar_plan_evaluado(
                nombre=nombre,
                instancia=instancia,
                plan=plan,
                seed=seed,
            )

            costo = float(plan.costo_estimado)
            costos[nombre].append(costo)
            costos_instancia[nombre] = costo

        ganador = min(
            costos_instancia,
            key=lambda nombre: costos_instancia[nombre],
        )

        resultados.append(
            {
                "seed": seed,
                "instancia_id": instancia.instancia_id,
                "turno": instancia.turno.value,
                "cantidad_pedidos": len(instancia.pedidos),
                "direcciones": [
                    pedido.direccion
                    for pedido in instancia.pedidos
                ],
                "costos": costos_instancia,
                "ganador": ganador,
            }
        )

        print(
            f"Seed={seed} "
            f"| tareas={len(instancia.pedidos)} "
            f"| RL={costos_instancia['RL']:.3f} "
            f"| Greedy={costos_instancia['GREEDY']:.3f} "
            f"| Random={costos_instancia['RANDOM']:.3f} "
            f"| GA={costos_instancia['GA']:.3f} "
            f"| ganador={ganador}"
        )

    resumen: dict[str, dict[str, float]] = {
        nombre: {
            "promedio": mean(valores),
            "minimo": min(valores),
            "maximo": max(valores),
        }
        for nombre, valores in costos.items()
    }

    print("")
    print("=== PROMEDIOS ===")

    for nombre, datos in resumen.items():
        print(
            f"{nombre}: "
            f"{datos['promedio']:.6f}"
        )

    salida: dict[str, Any] = {
        "modelo": str(model_path),
        "episodios": args.episodes,
        "seed_start": args.seed_start,
        "configuracion_generador": (
            serializar_configuracion_generador(
                configuracion_generador
            )
        ),
        "demanda_geografica": {
            "modo": (
                configuracion_generador
                .modo_demanda_geografica
                .value
            ),
            "ruta_dataset_resuelta": (
                str(ruta_demanda_real)
                if ruta_demanda_real is not None
                else ""
            ),
            "sha256_dataset": calcular_sha256(
                ruta_demanda_real
            ),
            "registros_aptos": (
                len(catalogo_demanda_real)
                if catalogo_demanda_real is not None
                else 0
            ),
            "direcciones_fuente_unicas": (
                catalogo_demanda_real
                .cantidad_direcciones_fuente_unicas()
                if catalogo_demanda_real is not None
                else 0
            ),
        },
        "resumen": resumen,
        "resultados": resultados,
    }

    sufijo_modo = (
        configuracion_generador
        .modo_demanda_geografica
        .value
        .lower()
    )

    salida_path = (
        run_dir
        / f"smoke_evaluation_{sufijo_modo}.json"
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