import unittest
from dataclasses import replace

from planner.ga import (
    ConfiguracionGA,
    GeneticAlgorithmPlanner,
    generar_plan_ga,
)

from planner.greedy import (
    generar_plan_greedy,
)

from planner.preprocess import (
    preprocesar_instancia,
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


def configuracion_ga_test() -> ConfiguracionGA:
    return ConfiguracionGA(
        tamano_poblacion=24,

        generaciones=40,

        tamano_elite=3,

        tamano_torneo=3,

        probabilidad_crossover=0.90,

        probabilidad_mutacion_swap=0.25,

        probabilidad_mutacion_inversion=0.15,

        generaciones_sin_mejora_max=15,
    )


class GeneticAlgorithmTest(unittest.TestCase):
    def test_genera_plan_valido(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        plan = generar_plan_ga(
            instancia,

            seed=8001,

            configuracion_ga=(
                configuracion_ga_test()
            ),
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

    def test_misma_seed_reproduce_plan(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        plan_a = generar_plan_ga(
            instancia,

            seed=8001,

            configuracion_ga=(
                configuracion_ga_test()
            ),
        )

        plan_b = generar_plan_ga(
            instancia,

            seed=8001,

            configuracion_ga=(
                configuracion_ga_test()
            ),
        )

        self.assertEqual(
            firma_plan(plan_a),
            firma_plan(plan_b),
        )

        self.assertAlmostEqual(
            plan_a.costo_estimado,
            plan_b.costo_estimado,
            places=9,
        )

    def test_no_es_peor_que_greedy_en_demo(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        greedy = generar_plan_greedy(
            instancia
        )

        ga = generar_plan_ga(
            instancia,

            seed=8001,

            configuracion_ga=(
                configuracion_ga_test()
            ),
        )

        self.assertLessEqual(
            ga.costo_estimado,
            greedy.costo_estimado
            + 1e-9,
        )

    def test_mejor_costo_no_empeora(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        planificador = (
            GeneticAlgorithmPlanner(
                seed=8001,

                configuracion_ga=(
                    configuracion_ga_test()
                ),
            )
        )

        planificador.generar_plan(
            instancia
        )

        historial = (
            planificador
            .mejor_costo_por_generacion
        )

        self.assertGreaterEqual(
            len(historial),
            1,
        )

        for anterior, siguiente in zip(
            historial,
            historial[1:],
        ):
            self.assertLessEqual(
                siguiente,
                anterior + 1e-9,
            )

    def test_volcador_siempre_ultimo(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        pedidos_por_id = {
            pedido.pedido_id: pedido
            for pedido in instancia.pedidos
        }

        for seed in range(
            8001,
            8011,
        ):
            plan = generar_plan_ga(
                instancia,

                seed=seed,

                configuracion_ga=(
                    configuracion_ga_test()
                ),
            )

            for camion in plan.camiones:
                for viaje in camion.viajes:
                    posiciones = [
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
                        len(posiciones),
                        1,
                    )

                    if posiciones:
                        self.assertEqual(
                            posiciones[0],

                            len(
                                viaje.pedido_ids
                            ) - 1,
                        )

    def test_split_volcador_es_factible(
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

            instancia_id="GA-SPLIT-VOLCADOR",

            pedidos=[
                pedido_grande
            ],
        )

        instancia = preprocesar_instancia(
            instancia_raw
        )

        plan = generar_plan_ga(
            instancia,

            seed=8001,

            configuracion_ga=(
                ConfiguracionGA(
                    tamano_poblacion=12,

                    generaciones=20,

                    tamano_elite=2,

                    tamano_torneo=3,

                    generaciones_sin_mejora_max=8,
                )
            ),
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

    def test_registra_diagnostico(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        planificador = (
            GeneticAlgorithmPlanner(
                seed=8001,

                configuracion_ga=(
                    configuracion_ga_test()
                ),
            )
        )

        plan = planificador.generar_plan(
            instancia
        )

        self.assertEqual(
            plan.algoritmo,
            AlgoritmoPlanificacion.GA,
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
            8001,
        )

        self.assertGreater(
            planificador
            .generaciones_ejecutadas,
            0,
        )

        self.assertGreaterEqual(
            len(
                planificador
                .mejor_costo_por_generacion
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()