import unittest

from planner.algorithms.ga import ConfiguracionGA, GeneticAlgorithmPlanner, generar_plan_ga
from planner.domain.validator import validar_plan

from tests.fixtures import crear_instancia_demo


def firma_plan(plan) -> tuple:
    return tuple(
        (
            camion.camion_id,
            tuple((viaje.numero_viaje, tuple(viaje.pedido_ids)) for viaje in camion.viajes),
        )
        for camion in plan.camiones
    )


def configuracion_ga_test() -> ConfiguracionGA:
    return ConfiguracionGA(
        tamano_poblacion=24,
        generaciones=40,
        tamano_elite=3,
        tamano_torneo=3,
        probabilidad_crossover=0.9,
        probabilidad_mutacion_swap=0.25,
        probabilidad_mutacion_inversion=0.15,
        generaciones_sin_mejora_max=15,
    )


class GeneticAlgorithmTest(unittest.TestCase):
    def test_genera_plan_valido(self) -> None:
        instancia = crear_instancia_demo()
        plan = generar_plan_ga(instancia, seed=8001, configuracion_ga=configuracion_ga_test())
        validacion = validar_plan(instancia, plan)
        self.assertTrue(validacion.valido, msg=" | ".join(validacion.errores))

    def test_misma_seed_reproduce_plan(self) -> None:
        instancia = crear_instancia_demo()
        plan_a = generar_plan_ga(instancia, seed=8001, configuracion_ga=configuracion_ga_test())
        plan_b = generar_plan_ga(instancia, seed=8001, configuracion_ga=configuracion_ga_test())
        self.assertEqual(firma_plan(plan_a), firma_plan(plan_b))
        self.assertAlmostEqual(plan_a.costo_estimado, plan_b.costo_estimado, places=9)

    def test_mejor_costo_no_empeora(self) -> None:
        instancia = crear_instancia_demo()
        planificador = GeneticAlgorithmPlanner(seed=8001, configuracion_ga=configuracion_ga_test())
        planificador.generar_plan(instancia)
        historial = planificador.mejor_costo_por_generacion
        self.assertGreaterEqual(len(historial), 1)
        for anterior, siguiente in zip(historial, historial[1:]):
            self.assertLessEqual(siguiente, anterior + 1e-09)

    def test_volcador_siempre_ultimo(self) -> None:
        instancia = crear_instancia_demo()
        pedidos_por_id = {pedido.pedido_id: pedido for pedido in instancia.pedidos}
        for seed in range(8001, 8011):
            plan = generar_plan_ga(instancia, seed=seed, configuracion_ga=configuracion_ga_test())
            for camion in plan.camiones:
                for viaje in camion.viajes:
                    posiciones = [
                        indice
                        for indice, pedido_id in enumerate(viaje.pedido_ids)
                        if pedidos_por_id[pedido_id].requiere_volcador
                    ]
                    self.assertLessEqual(len(posiciones), 1)
                    if posiciones:
                        self.assertEqual(posiciones[0], len(viaje.pedido_ids) - 1)

    def test_tasas_adaptativas_bajan_crossover_y_suben_mutacion(self) -> None:
        cfg = ConfiguracionGA(
            generaciones=50,
            generaciones_sin_mejora_max=20,
            probabilidad_crossover=0.90,
            probabilidad_crossover_min=0.65,
            probabilidad_mutacion_swap=0.20,
            probabilidad_mutacion_swap_max=0.45,
            probabilidad_mutacion_inversion=0.10,
            probabilidad_mutacion_inversion_max=0.30,
            usar_tasas_adaptativas=True,
        )
        planner = GeneticAlgorithmPlanner(configuracion_ga=cfg)
        c_ini, s_ini, i_ini = planner._calcular_tasas_generacion(
            numero_generacion=1, generaciones_sin_mejora=0
        )
        c_fin, s_fin, i_fin = planner._calcular_tasas_generacion(
            numero_generacion=50, generaciones_sin_mejora=20
        )
        self.assertAlmostEqual(c_ini, 0.90)
        self.assertAlmostEqual(s_ini, 0.20)
        self.assertAlmostEqual(i_ini, 0.10)
        self.assertLess(c_fin, c_ini)
        self.assertAlmostEqual(c_fin, 0.65)
        self.assertGreater(s_fin, s_ini)
        self.assertAlmostEqual(s_fin, 0.45)
        self.assertGreater(i_fin, i_ini)
        self.assertAlmostEqual(i_fin, 0.30)


if __name__ == "__main__":
    unittest.main()
