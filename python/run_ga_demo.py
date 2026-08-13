from planner.ga import (
    ConfiguracionGA,
    GeneticAlgorithmPlanner,
)

from planner.greedy import (
    generar_plan_greedy,
)

from planner.validator import (
    validar_plan,
)

from tests.fixtures import (
    crear_instancia_demo,
)


def imprimir_plan(
    titulo: str,
    plan,
) -> None:
    print("")
    print(titulo)

    for camion in plan.camiones:
        print(
            f"Camión {camion.camion_id}"
        )

        if not camion.viajes:
            print("  Sin viajes")
            continue

        for viaje in camion.viajes:
            print(
                f"  Viaje "
                f"{viaje.numero_viaje}: "
                f"{viaje.pedido_ids}"
            )

    print(
        "Costo estimado: "
        f"{plan.costo_estimado:.6f}"
    )

    print(
        "Tiempo de cómputo: "
        f"{plan.tiempo_computo_ms:.3f} ms"
    )


def main() -> None:
    instancia = crear_instancia_demo()

    greedy = generar_plan_greedy(
        instancia
    )

    configuracion_ga = ConfiguracionGA(
        tamano_poblacion=60,

        generaciones=100,

        tamano_elite=4,

        tamano_torneo=3,

        probabilidad_crossover=0.90,

        probabilidad_mutacion_swap=0.20,

        probabilidad_mutacion_inversion=0.10,

        generaciones_sin_mejora_max=30,
    )

    planificador = (
        GeneticAlgorithmPlanner(
            seed=8001,

            configuracion_ga=(
                configuracion_ga
            ),
        )
    )

    ga = planificador.generar_plan(
        instancia
    )

    validacion = validar_plan(
        instancia,
        ga,
    )

    imprimir_plan(
        "=== GREEDY DE REFERENCIA ===",
        greedy,
    )

    imprimir_plan(
        "=== GENETIC ALGORITHM ===",
        ga,
    )

    print("")
    print(
        "Plan GA válido: "
        f"{validacion.valido}"
    )

    print(
        "Seed utilizada: "
        f"{planificador.ultima_seed_utilizada}"
    )

    print(
        "Generaciones ejecutadas: "
        f"{planificador.generaciones_ejecutadas}"
    )

    print(
        "Costo inicial/mejor final: "
        f"{planificador.mejor_costo_por_generacion[0]:.6f}"
        " / "
        f"{planificador.mejor_costo_por_generacion[-1]:.6f}"
    )

    print(
        "GA no es peor que Greedy: "
        f"{ga.costo_estimado <= greedy.costo_estimado + 1e-9}"
    )


if __name__ == "__main__":
    main()