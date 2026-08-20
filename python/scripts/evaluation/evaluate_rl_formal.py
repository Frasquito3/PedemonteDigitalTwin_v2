from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys

from dataclasses import asdict
from math import isfinite
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


PYTHON_ROOT = Path(__file__).resolve().parents[2]
PLANNER_DIR = PYTHON_ROOT / "planner"

if not PLANNER_DIR.is_dir():
    raise RuntimeError(
        "No se encontró el paquete planner. "
        f"Raíz calculada: {PYTHON_ROOT}"
    )

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
from planner.data.real_demand import (  # noqa: E402
    ParticionDemandaReal,
    SEED_DIVISION_DEMANDA_REAL_V1,
)
from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
    ModoDemandaGeografica,
)
from planner.rl.planner import RLPlanner  # noqa: E402


FACTOR_RESULTADO_EXTREMO = 5.0
TOLERANCIA_COSTO = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación formal estimada de uno o dos "
            "modelos RL contra Greedy, Random y GA."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al modelo RL principal.",
    )

    parser.add_argument(
        "--comparison-model",
        type=str,
        default="",
        help="Ruta opcional a un segundo modelo RL.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=350_000,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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
        "--demand-partition",
        choices=[
            ParticionDemandaReal.ENTRENAMIENTO.value,
            ParticionDemandaReal.VALIDACION.value,
            ParticionDemandaReal.PRUEBA.value,
        ],
        default=ParticionDemandaReal.PRUEBA.value,
        help=(
            "Partición geográfica utilizada en modo REAL. "
            "Por defecto TEST."
        ),
    )

    parser.add_argument(
        "--large-order-probability",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    if not args.model.strip():
        parser.error("--model no puede estar vacío.")

    if args.episodes <= 0:
        parser.error("--episodes debe ser > 0.")

    if not args.output_dir.strip():
        parser.error("--output-dir no puede estar vacío.")

    if (
        args.large_order_probability is not None
        and not 0.0 <= args.large_order_probability <= 1.0
    ):
        parser.error(
            "--large-order-probability debe estar entre 0 y 1."
        )

    return args


def resolver_ruta(ruta_texto: str) -> Path:
    ruta = Path(ruta_texto).expanduser()

    if not ruta.is_absolute():
        ruta = Path.cwd() / ruta

    return ruta.resolve()


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

    particion = (
        ParticionDemandaReal(args.demand_partition)
        if modo == ModoDemandaGeografica.REAL
        else None
    )

    return ConfiguracionGeneradorInstancias(
        min_pedidos_finales=4,
        max_pedidos_finales=12,
        probabilidad_pedido_mayor_capacidad=(
            resolver_probabilidad_pedido_grande(
                args=args,
                modo=modo,
            )
        ),
        modo_demanda_geografica=modo,
        ruta_demanda_real=args.demand_dataset.strip(),
        particion_demanda_real=particion,
        seed_division_demanda_real=(
            SEED_DIVISION_DEMANDA_REAL_V1
        ),
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
    datos["particion_demanda_real"] = (
        configuracion.particion_demanda_real.value
        if configuracion.particion_demanda_real
        is not None
        else ""
    )

    return datos


def percentil(
    valores: list[float],
    proporcion: float,
) -> float:
    if not valores:
        raise ValueError(
            "No se puede calcular un percentil sobre "
            "una lista vacía."
        )

    if not 0.0 <= proporcion <= 1.0:
        raise ValueError(
            "La proporción debe estar entre 0 y 1."
        )

    ordenados = sorted(valores)
    posicion = round(
        proporcion
        * (len(ordenados) - 1)
    )

    return ordenados[posicion]


def resumir(
    valores: list[float],
) -> dict[str, float]:
    if not valores:
        raise ValueError(
            "No se puede resumir una lista vacía."
        )

    return {
        "promedio": mean(valores),
        "mediana": median(valores),
        "desvio": (
            pstdev(valores)
            if len(valores) > 1
            else 0.0
        ),
        "minimo": min(valores),
        "maximo": max(valores),
        "p90": percentil(valores, 0.90),
    }


def validar_plan_y_metricas(
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
            f"{nombre} produjo un plan inválido "
            f"en seed={seed}: "
            + " | ".join(validacion.errores)
        )

    if not isfinite(plan.costo_estimado):
        raise RuntimeError(
            f"{nombre} produjo costo no finito "
            f"en seed={seed}."
        )

    if plan.costo_estimado < 0.0:
        raise RuntimeError(
            f"{nombre} produjo costo negativo "
            f"en seed={seed}."
        )

    if not isfinite(plan.tiempo_computo_ms):
        raise RuntimeError(
            f"{nombre} produjo tiempo no finito "
            f"en seed={seed}."
        )

    if plan.tiempo_computo_ms < 0.0:
        raise RuntimeError(
            f"{nombre} produjo tiempo negativo "
            f"en seed={seed}."
        )


def main() -> None:
    args = parse_args()

    model_path = resolver_ruta(args.model)

    if not model_path.is_file():
        raise FileNotFoundError(
            "No existe el modelo principal: "
            f"{model_path}"
        )

    comparison_model_path: Path | None = None

    if args.comparison_model.strip():
        comparison_model_path = resolver_ruta(
            args.comparison_model
        )

        if not comparison_model_path.is_file():
            raise FileNotFoundError(
                "No existe el modelo de comparación: "
                f"{comparison_model_path}"
            )

    output_dir = resolver_ruta(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuracion_generador = crear_configuracion_generador(
        args
    )
    generador = GeneradorInstanciasRL(
        configuracion=configuracion_generador
    )

    planner_principal = RLPlanner(
        model_path=model_path,
        deterministic=True,
    )

    planner_comparacion: RLPlanner | None = None

    if comparison_model_path is not None:
        planner_comparacion = RLPlanner(
            model_path=comparison_model_path,
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
        "RL_PRINCIPAL",
        "GREEDY",
        "RANDOM",
        "GA",
    ]

    if planner_comparacion is not None:
        nombres.insert(1, "RL_COMPARACION")

    costos: dict[str, list[float]] = {
        nombre: []
        for nombre in nombres
    }
    tiempos: dict[str, list[float]] = {
        nombre: []
        for nombre in nombres
    }
    gaps_greedy: dict[str, list[float]] = {
        nombre: []
        for nombre in nombres
    }
    victorias: dict[str, int] = {
        nombre: 0
        for nombre in nombres
    }
    mejoras_greedy: dict[str, int] = {
        nombre: 0
        for nombre in nombres
    }
    empates_greedy: dict[str, int] = {
        nombre: 0
        for nombre in nombres
    }
    empeora_greedy: dict[str, int] = {
        nombre: 0
        for nombre in nombres
    }
    resultados_extremos: dict[str, int] = {
        nombre: 0
        for nombre in nombres
    }

    registros: list[dict[str, Any]] = []

    ruta_demanda_real = (
        generador.ruta_demanda_real_resuelta
    )
    catalogo_demanda_real = (
        generador.catalogo_demanda_real
    )
    catalogo_demanda_real_completo = (
        generador.catalogo_demanda_real_completo
    )
    division_demanda_real = (
        generador.division_demanda_real
    )

    print("")
    print("=== EVALUACIÓN FORMAL ESTIMADA ===")
    print(f"Modelo principal: {model_path}")

    if comparison_model_path is not None:
        print(
            "Modelo de comparación: "
            f"{comparison_model_path}"
        )

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
        print(
            "Partición: "
            f"{configuracion_generador.particion_demanda_real.value if configuracion_generador.particion_demanda_real is not None else 'CATALOGO_COMPLETO'}"
        )
        print(
            "Direcciones de la partición: "
            f"{catalogo_demanda_real.cantidad_direcciones_fuente_unicas() if catalogo_demanda_real is not None else 0}"
        )

    for indice in range(args.episodes):
        seed = args.seed_start + indice
        instancia = generador.generar(seed)

        planes: dict[str, Any] = {
            "RL_PRINCIPAL": (
                planner_principal.generar_plan(instancia)
            ),
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

        if planner_comparacion is not None:
            planes["RL_COMPARACION"] = (
                planner_comparacion.generar_plan(
                    instancia
                )
            )

        costos_instancia: dict[str, float] = {}
        tiempos_instancia: dict[str, float] = {}

        for nombre in nombres:
            plan = planes[nombre]

            validar_plan_y_metricas(
                nombre=nombre,
                instancia=instancia,
                plan=plan,
                seed=seed,
            )

            costo = float(plan.costo_estimado)
            tiempo_ms = float(plan.tiempo_computo_ms)

            costos_instancia[nombre] = costo
            tiempos_instancia[nombre] = tiempo_ms
            costos[nombre].append(costo)
            tiempos[nombre].append(tiempo_ms)

        costo_greedy = costos_instancia["GREEDY"]
        gaps_instancia: dict[str, float] = {}
        extremos_instancia: list[str] = []

        for nombre in nombres:
            costo = costos_instancia[nombre]
            gap = (
                costo - costo_greedy
            ) / max(abs(costo_greedy), 1.0)

            gaps_instancia[nombre] = gap
            gaps_greedy[nombre].append(gap)

            diferencia = costo - costo_greedy

            if diferencia < -TOLERANCIA_COSTO:
                mejoras_greedy[nombre] += 1
            elif abs(diferencia) <= TOLERANCIA_COSTO:
                empates_greedy[nombre] += 1
            else:
                empeora_greedy[nombre] += 1

            if (
                nombre != "GREEDY"
                and costo
                > FACTOR_RESULTADO_EXTREMO
                * max(costo_greedy, 1.0)
            ):
                resultados_extremos[nombre] += 1
                extremos_instancia.append(nombre)

        mejor_costo = min(costos_instancia.values())
        ganadores = [
            nombre
            for nombre, costo
            in costos_instancia.items()
            if abs(costo - mejor_costo)
            <= TOLERANCIA_COSTO
        ]

        for ganador in ganadores:
            victorias[ganador] += 1

        registros.append(
            {
                "seed": seed,
                "instancia_id": instancia.instancia_id,
                "turno": instancia.turno.value,
                "cantidad_pedidos": len(
                    instancia.pedidos
                ),
                "direcciones": [
                    pedido.direccion
                    for pedido in instancia.pedidos
                ],
                "costos": costos_instancia,
                "tiempos_ms": tiempos_instancia,
                "gaps_greedy": gaps_instancia,
                "ganadores": ganadores,
                "resultados_extremos": (
                    extremos_instancia
                ),
            }
        )

        texto_comparacion = ""

        if "RL_COMPARACION" in costos_instancia:
            texto_comparacion = (
                " | RL_COMP="
                f"{costos_instancia['RL_COMPARACION']:.3f}"
            )

        print(
            f"[{indice + 1}/{args.episodes}] "
            f"seed={seed} "
            f"| tareas={len(instancia.pedidos)} "
            f"| RL={costos_instancia['RL_PRINCIPAL']:.3f}"
            f"{texto_comparacion} "
            f"| G={costo_greedy:.3f} "
            f"| R={costos_instancia['RANDOM']:.3f} "
            f"| GA={costos_instancia['GA']:.3f}"
        )

    resumen: dict[str, dict[str, Any]] = {}

    for nombre in nombres:
        resumen[nombre] = {
            "costo": resumir(costos[nombre]),
            "tiempo_ms": resumir(tiempos[nombre]),
            "gap_greedy": resumir(
                gaps_greedy[nombre]
            ),
            "victorias": victorias[nombre],
            "mejora_greedy": mejoras_greedy[nombre],
            "empata_greedy": empates_greedy[nombre],
            "empeora_greedy": empeora_greedy[nombre],
            "resultados_extremos": (
                resultados_extremos[nombre]
            ),
        }

    resultado: dict[str, Any] = {
        "modelo_principal": str(model_path),
        "modelo_comparacion": (
            str(comparison_model_path)
            if comparison_model_path is not None
            else ""
        ),
        "episodios": args.episodes,
        "seed_start": args.seed_start,
        "criterio_resultado_extremo": {
            "factor_sobre_greedy": (
                FACTOR_RESULTADO_EXTREMO
            ),
        },
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
            "particion": (
                configuracion_generador
                .particion_demanda_real
                .value
                if configuracion_generador
                .particion_demanda_real
                is not None
                else "CATALOGO_COMPLETO"
            ),
            "seed_division": (
                configuracion_generador
                .seed_division_demanda_real
                if configuracion_generador
                .particion_demanda_real
                is not None
                else None
            ),
            "ruta_dataset_resuelta": (
                str(ruta_demanda_real)
                if ruta_demanda_real is not None
                else ""
            ),
            "sha256_dataset": calcular_sha256(
                ruta_demanda_real
            ),
            "catalogo_completo": {
                "registros": (
                    len(catalogo_demanda_real_completo)
                    if catalogo_demanda_real_completo
                    is not None
                    else 0
                ),
                "direcciones_fuente_unicas": (
                    catalogo_demanda_real_completo
                    .cantidad_direcciones_fuente_unicas()
                    if catalogo_demanda_real_completo
                    is not None
                    else 0
                ),
            },
            "catalogo_evaluado": {
                "registros": (
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
            "particiones": (
                division_demanda_real.resumen()
                if division_demanda_real is not None
                else {}
            ),
        },
        "resumen": resumen,
        "registros": registros,
    }

    sufijo_modo = (
        configuracion_generador
        .modo_demanda_geografica
        .value
        .lower()
    )

    if (
        configuracion_generador
        .particion_demanda_real
        is not None
    ):
        sufijo_modo += (
            "_"
            + configuracion_generador
            .particion_demanda_real
            .value
            .lower()
        )

    json_path = (
        output_dir
        / f"formal_evaluation_{sufijo_modo}.json"
    )
    csv_path = (
        output_dir
        / f"formal_evaluation_{sufijo_modo}.csv"
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

    encabezado = [
        "seed",
        "instancia_id",
        "turno",
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
    encabezado.extend(
        [
            "ganadores",
            "resultados_extremos",
            "direcciones",
        ]
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as archivo:
        writer = csv.writer(archivo)
        writer.writerow(encabezado)

        for registro in registros:
            fila: list[Any] = [
                registro["seed"],
                registro["instancia_id"],
                registro["turno"],
                registro["cantidad_pedidos"],
            ]
            fila.extend(
                registro["costos"][nombre]
                for nombre in nombres
            )
            fila.extend(
                registro["tiempos_ms"][nombre]
                for nombre in nombres
            )
            fila.extend(
                registro["gaps_greedy"][nombre]
                for nombre in nombres
            )
            fila.extend(
                [
                    "|".join(registro["ganadores"]),
                    "|".join(
                        registro[
                            "resultados_extremos"
                        ]
                    ),
                    "|".join(registro["direcciones"]),
                ]
            )
            writer.writerow(fila)

    print("")
    print("=== RESUMEN FORMAL ===")

    for nombre in nombres:
        datos = resumen[nombre]
        print(
            f"{nombre}: "
            "promedio="
            f"{datos['costo']['promedio']:.6f} "
            "| mediana="
            f"{datos['costo']['mediana']:.6f} "
            "| p90="
            f"{datos['costo']['p90']:.6f} "
            "| victorias="
            f"{datos['victorias']} "
            "| extremos="
            f"{datos['resultados_extremos']}"
        )

    print("")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()