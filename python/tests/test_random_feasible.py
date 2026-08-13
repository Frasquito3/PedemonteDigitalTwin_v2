import unittest
from dataclasses import replace

from planner.preprocess import (
    preprocesar_instancia,
)

from planner.random_feasible import (
    RandomFeasiblePlanner,
    generar_plan_random,
)

from planner.schema import (
    AlgoritmoPlanificacion,
    PedidoInput,
    Turno,
)

from planner.validator import (
    validar_plan,
)

from tests.fixtures import (
    crear_instancia_demo,
)


def firma_plan(plan) -> tuple:
    return tuple(
        (
            camion.camion_id,

            tuple(
                (
                    viaje.numero_viaje,

                    tuple(
                        viaje.pedido_ids
                    ),
                )
                for viaje in camion.viajes
            ),
        )
        for camion in plan.camiones
    )


class RandomFeasibleTest(unittest.TestCase):
    def test_genera_planes_validos_en_muchas_seeds(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        for seed in range(50):
            plan = generar_plan_random(
                instancia,
                seed=seed,
            )

            validacion = validar_plan(
                instancia,
                plan,
            )

            self.assertTrue(
                validacion.valido,

                msg=(
                    f"Seed={seed}: "
                    + " | ".join(
                        validacion.errores
                    )
                ),
            )

    def test_misma_seed_genera_mismo_plan(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        plan_a = generar_plan_random(
            instancia,
            seed=7001,
        )

        plan_b = generar_plan_random(
            instancia,
            seed=7001,
        )

        self.assertEqual(
            firma_plan(plan_a),
            firma_plan(plan_b),
        )

    def test_distintas_seeds_generan_diversidad(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        firmas = {
            firma_plan(
                generar_plan_random(
                    instancia,
                    seed=seed,
                )
            )
            for seed in range(20)
        }

        self.assertGreater(
            len(firmas),
            1,
        )

    def test_volcador_siempre_es_ultimo(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        pedidos_por_id = {
            pedido.pedido_id: pedido
            for pedido in instancia.pedidos
        }

        for seed in range(30):
            plan = generar_plan_random(
                instancia,
                seed=seed,
            )

            for camion in plan.camiones:
                for viaje in camion.viajes:
                    posiciones_volcador = [
                        indice

                        for indice, pedido_id
                        in enumerate(
                            viaje.pedido_ids
                        )

                        if pedidos_por_id[
                            pedido_id
                        ].requiere_volcador
                    ]

                    self.assertLessEqual(
                        len(posiciones_volcador),
                        1,
                    )

                    if posiciones_volcador:
                        self.assertEqual(
                            posiciones_volcador[0],

                            len(
                                viaje.pedido_ids
                            ) - 1,
                        )

    def test_registra_algoritmo_costo_y_tiempo(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        planificador = (
            RandomFeasiblePlanner(
                seed=7001
            )
        )

        plan = planificador.generar_plan(
            instancia
        )

        self.assertEqual(
            plan.algoritmo,
            AlgoritmoPlanificacion.RANDOM,
        )

        self.assertGreater(
            plan.costo_estimado,
            0.0,
        )

        self.assertGreaterEqual(
            plan.tiempo_computo_ms,
            0.0,
        )

        self.assertEqual(
            planificador
            .ultima_seed_utilizada,
            7001,
        )

    def test_split_volcador_genera_viajes_factibles(
        self,
    ) -> None:
        instancia_base = (
            crear_instancia_demo()
        )

        pedido_grande = PedidoInput(
            pedido_id="PV17",

            pedido_original_id="PV17",

            numero_parte=1,

            total_partes=1,

            turno=Turno.MANANA,

            latitud=(
                instancia_base.lat_corralon
                + 0.02
            ),

            longitud=(
                instancia_base.lon_corralon
                + 0.02
            ),

            unidades_capacidad=17,

            requiere_volcador=True,

            tiene_ventana_especifica=False,

            hora_desde_min=450,

            hora_hasta_min=720,
        )

        instancia_raw = replace(
            instancia_base,

            instancia_id=(
                "RANDOM-SPLIT-VOLCADOR"
            ),

            pedidos=[
                pedido_grande
            ],
        )

        instancia = preprocesar_instancia(
            instancia_raw
        )

        plan = generar_plan_random(
            instancia,
            seed=7001,
        )

        validacion = validar_plan(
            instancia,
            plan,
        )

        self.assertTrue(
            validacion.valido,

            msg=" | ".join(
                validacion.errores
            ),
        )

        viajes = [
            viaje

            for camion in plan.camiones
            for viaje in camion.viajes
        ]

        # 17 unidades -> 8 + 8 + 1.
        # Todas las partes heredan volcador.
        # Máximo un volcador por viaje.
        self.assertEqual(
            len(viajes),
            3,
        )

        self.assertTrue(
            all(
                len(viaje.pedido_ids) == 1
                for viaje in viajes
            )
        )


if __name__ == "__main__":
    unittest.main()