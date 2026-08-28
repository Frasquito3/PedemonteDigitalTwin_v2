import unittest
from dataclasses import replace
from planner.domain.preprocess import preprocesar_instancia
from planner.algorithms.random_feasible import RandomFeasiblePlanner, generar_plan_random
from planner.core.schema import AlgoritmoPlanificacion, PedidoInput, Turno
from planner.domain.validator import validar_plan
from tests.fixtures import crear_instancia_demo

def firma_plan(plan) -> tuple:
    return tuple(((camion.camion_id, tuple(((viaje.numero_viaje, tuple(viaje.pedido_ids)) for viaje in camion.viajes))) for camion in plan.camiones))

class RandomFeasibleTest(unittest.TestCase):

    def test_genera_planes_validos_en_muchas_seeds(self) -> None:
        instancia = crear_instancia_demo()
        for seed in range(50):
            plan = generar_plan_random(instancia, seed=seed)
            validacion = validar_plan(instancia, plan)
            self.assertTrue(validacion.valido, msg=f'Seed={seed}: ' + ' | '.join(validacion.errores))

    def test_volcador_siempre_es_ultimo(self) -> None:
        instancia = crear_instancia_demo()
        pedidos_por_id = {pedido.pedido_id: pedido for pedido in instancia.pedidos}
        for seed in range(30):
            plan = generar_plan_random(instancia, seed=seed)
            for camion in plan.camiones:
                for viaje in camion.viajes:
                    posiciones_volcador = [indice for indice, pedido_id in enumerate(viaje.pedido_ids) if pedidos_por_id[pedido_id].requiere_volcador]
                    self.assertLessEqual(len(posiciones_volcador), 1)
                    if posiciones_volcador:
                        self.assertEqual(posiciones_volcador[0], len(viaje.pedido_ids) - 1)
if __name__ == '__main__':
    unittest.main()
