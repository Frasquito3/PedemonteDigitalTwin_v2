import unittest
from planner.core.schema import AlgoritmoPlanificacion, PlanCamion, PlanTurno, ViajePlan
from planner.domain.validator import validar_plan
from tests.fixtures import crear_instancia_demo

class ValidatorTest(unittest.TestCase):

    def test_rechaza_capacidad_excedida(self) -> None:
        instancia = crear_instancia_demo()
        plan = PlanTurno(instancia_id=instancia.instancia_id, algoritmo=AlgoritmoPlanificacion.MANUAL_TEST, camiones=[PlanCamion(camion_id=0, viajes=[ViajePlan(numero_viaje=1, pedido_ids=['P001', 'P004']), ViajePlan(numero_viaje=2, pedido_ids=['P003'])]), PlanCamion(camion_id=1, viajes=[ViajePlan(numero_viaje=1, pedido_ids=['P002'])])])
        resultado = validar_plan(instancia, plan)
        self.assertFalse(resultado.valido)
        self.assertTrue(any(('supera capacidad' in error for error in resultado.errores)))

    def test_rechaza_volcador_no_ultimo(self) -> None:
        instancia = crear_instancia_demo()
        plan = PlanTurno(instancia_id=instancia.instancia_id, algoritmo=AlgoritmoPlanificacion.MANUAL_TEST, camiones=[PlanCamion(camion_id=0, viajes=[ViajePlan(numero_viaje=1, pedido_ids=['P003', 'P001']), ViajePlan(numero_viaje=2, pedido_ids=['P002'])]), PlanCamion(camion_id=1, viajes=[ViajePlan(numero_viaje=1, pedido_ids=['P004'])])])
        resultado = validar_plan(instancia, plan)
        self.assertFalse(resultado.valido)
        self.assertTrue(any(('volcador no es el último' in error for error in resultado.errores)))
if __name__ == '__main__':
    unittest.main()
