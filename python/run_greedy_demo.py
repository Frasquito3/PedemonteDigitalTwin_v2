from planner.greedy import (
    generar_plan_greedy,
)

from planner.validator import (
    validar_plan,
)

from tests.fixtures import (
    crear_instancia_demo,
)


def main() -> None:
    instancia = crear_instancia_demo()

    plan = generar_plan_greedy(
        instancia
    )

    validacion = validar_plan(
        instancia,
        plan,
    )

    print("=== GREEDY FEASIBLE ===")

    print(
        f"Plan válido: {validacion.valido}"
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

    if not validacion.valido:
        for error in validacion.errores:
            print(
                f"ERROR: {error}"
            )


if __name__ == "__main__":
    main()