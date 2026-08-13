from planner.algorithms.random_feasible import (
    RandomFeasiblePlanner,
)

from planner.domain.validator import (
    validar_plan,
)

from tests.fixtures import (
    crear_instancia_demo,
)


def firma_plan(plan) -> tuple:
    return tuple(
        tuple(
            tuple(
                viaje.pedido_ids
            )
            for viaje in camion.viajes
        )
        for camion in plan.camiones
    )


def imprimir_plan(
    seed: int,
) -> tuple:
    instancia = crear_instancia_demo()

    planificador = (
        RandomFeasiblePlanner(
            seed=seed
        )
    )

    plan = planificador.generar_plan(
        instancia
    )

    validacion = validar_plan(
        instancia,
        plan,
    )

    print("")
    print(
        f"=== RANDOM FEASIBLE "
        f"| seed={seed} ==="
    )

    print(
        f"Plan válido: "
        f"{validacion.valido}"
    )

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

    return firma_plan(
        plan
    )


def main() -> None:
    firma_7001_a = imprimir_plan(
        7001
    )

    firma_7001_b = imprimir_plan(
        7001
    )

    firma_7002 = imprimir_plan(
        7002
    )

    print("")
    print(
        "Misma seed reproduce plan: "
        f"{firma_7001_a == firma_7001_b}"
    )

    print(
        "Seed diferente cambió el plan: "
        f"{firma_7001_a != firma_7002}"
    )


if __name__ == "__main__":
    main()