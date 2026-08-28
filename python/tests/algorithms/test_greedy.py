import unittest
from planner.algorithms.greedy import generar_plan_greedy
from planner.domain.validator import validar_plan
from tests.fixtures import crear_instancia_demo

class GreedyFeasibleTest(unittest.TestCase):

    def test_genera_plan_valido(self) -> None:
        instancia = crear_instancia_demo()
        plan = generar_plan_greedy(instancia)
        validacion = validar_plan(instancia, plan)
        self.assertTrue(validacion.valido, msg=' | '.join(validacion.errores))

    def test_volcador_siempre_ultimo(self) -> None:
        instancia = crear_instancia_demo()
        pedidos_por_id = {pedido.pedido_id: pedido for pedido in instancia.pedidos}
        plan = generar_plan_greedy(instancia)
        for camion in plan.camiones:
            for viaje in camion.viajes:
                posiciones = [indice for indice, pedido_id in enumerate(viaje.pedido_ids) if pedidos_por_id[pedido_id].requiere_volcador]
                self.assertLessEqual(len(posiciones), 1)
                if posiciones:
                    self.assertEqual(posiciones[0], len(viaje.pedido_ids) - 1)
if __name__ == '__main__':
    unittest.main()
