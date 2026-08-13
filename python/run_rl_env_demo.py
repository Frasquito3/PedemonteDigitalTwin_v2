from random import Random

import numpy as np

from planner.rl_env import (
    PedemontePlanEnv,
)

from planner.validator import (
    validar_plan,
)

from tests.fixtures import (
    crear_instancia_demo,
)


def imprimir_plan(
    env: PedemontePlanEnv,
) -> None:
    plan = env.ultimo_plan

    if plan is None:
        print("No existe plan terminal.")
        return

    validacion = validar_plan(
        env.instancia,
        plan,
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


def ejecutar_secuencia_manual() -> None:
    env = PedemontePlanEnv(
        crear_instancia_demo()
    )

    observacion, info = env.reset(
        seed=9001
    )

    print(
        "=== RL ENV | SECUENCIA MANUAL ==="
    )

    print(
        f"Dimensión observación: "
        f"{observacion.shape}"
    )

    print(
        f"Acciones válidas iniciales: "
        f"{info['acciones_validas']}"
    )

    secuencia = [
        "P004",
        "P003",
        "P001",
        "P002",
    ]

    for pedido_id in secuencia:
        accion = (
            env.accion_de_pedido_id(
                pedido_id
            )
        )

        (
            _,
            recompensa,
            terminado,
            truncado,
            info_step,
        ) = env.step(
            accion
        )

        print(
            f"Acción={accion} "
            f"pedido={pedido_id} "
            f"reward={recompensa:.6f} "
            f"terminated={terminado} "
            f"truncated={truncado}"
        )

        if terminado:
            print(
                "Costo terminal: "
                f"{info_step['costo_estimado']:.6f}"
            )

    imprimir_plan(
        env
    )


def ejecutar_politica_aleatoria_enmascarada() -> None:
    env = PedemontePlanEnv(
        crear_instancia_demo()
    )

    env.reset(
        seed=9002
    )

    rng = Random(
        9002
    )

    terminado = False

    print("")
    print(
        "=== RL ENV | POLÍTICA ALEATORIA "
        "ENMASCARADA ==="
    )

    while not terminado:
        acciones_validas = (
            np.flatnonzero(
                env.action_masks()
            )
            .tolist()
        )

        accion = int(
            rng.choice(
                acciones_validas
            )
        )

        pedido_id = (
            env.pedido_id_de_accion(
                accion
            )
        )

        (
            _,
            recompensa,
            terminado,
            truncado,
            _,
        ) = env.step(
            accion
        )

        print(
            f"Seleccionado {pedido_id} "
            f"| reward={recompensa:.6f}"
        )

        if truncado:
            raise RuntimeError(
                "El episodio aleatorio "
                "fue truncado."
            )

    imprimir_plan(
        env
    )


def main() -> None:
    ejecutar_secuencia_manual()

    ejecutar_politica_aleatoria_enmascarada()


if __name__ == "__main__":
    main()