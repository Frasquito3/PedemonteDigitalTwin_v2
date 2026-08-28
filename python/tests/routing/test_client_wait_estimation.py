from __future__ import annotations
import unittest
from planner.algorithms.greedy import GreedyFeasiblePlanner
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import PedidoInput, Turno
from planner.routing.decoder import estimar_fin_viaje
from planner.routing.objective import estimar_espera_cliente, tiempo_espera_respuesta_cliente_esperado_min
from planner.routing.travel import FuenteMatrizViaje, MatrizViaje

class ClientWaitEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.configuracion = ConfiguracionPlanificacion()
        self.pedido = PedidoInput(pedido_id='P1', pedido_original_id='P1', numero_parte=1, total_partes=1, turno=Turno.MANANA, latitud=-32.831, longitud=-60.719, unidades_capacidad=1, requiere_volcador=False, tiene_ventana_especifica=True, hora_desde_min=480, hora_hasta_min=540)

    def test_media_triangular_coincide_con_anylogic(self) -> None:
        self.assertAlmostEqual(tiempo_espera_respuesta_cliente_esperado_min(self.configuracion), 10.0 / 3.0)

    def test_llegada_temprana_suma_ventana_y_respuesta(self) -> None:
        estimacion = estimar_espera_cliente(self.pedido, minuto_llegada=470.0, configuracion=self.configuracion)
        self.assertAlmostEqual(estimacion.tiempo_espera_ventana_min, 10.0)
        self.assertAlmostEqual(estimacion.tiempo_espera_respuesta_min, 10.0 / 3.0)
        self.assertAlmostEqual(estimacion.minuto_inicio_descarga, 480.0 + 10.0 / 3.0)

    def test_llegada_tardia_no_agrega_respuesta(self) -> None:
        estimacion = estimar_espera_cliente(self.pedido, minuto_llegada=541.0, configuracion=self.configuracion)
        self.assertEqual(estimacion.tiempo_espera_ventana_min, 0.0)
        self.assertEqual(estimacion.tiempo_espera_respuesta_min, 0.0)
        self.assertEqual(estimacion.minuto_inicio_descarga, 541.0)
if __name__ == '__main__':
    unittest.main()
