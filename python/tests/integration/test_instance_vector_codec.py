from __future__ import annotations
import unittest
from planner.core.schema import Turno
from planner.integration.instance_vector_codec import decodificar_instancia_vector

class InstanceVectorCodecTest(unittest.TestCase):

    def vector_base(self) -> list[float]:
        return [1.0, 0.0, 2.0, 8.0, 2.0, -32.8495006, -60.722653, 0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 0.0, -32.8487, -60.7221, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 0.0, -32.8502, -60.7232, 500.0, 650.0]

    def test_decodifica_instancia_completa(self) -> None:
        instancia = decodificar_instancia_vector(self.vector_base(), seed_escenario=4003, seed_ejecucion=1004003)
        self.assertEqual(instancia.turno, Turno.MANANA)
        self.assertEqual(len(instancia.pedidos), 2)
        self.assertEqual(instancia.pedidos[0].unidades_capacidad, 5)
        self.assertEqual(instancia.pedidos[1].unidades_capacidad, 3)

    def test_conserva_ventana_especifica(self) -> None:
        instancia = decodificar_instancia_vector(self.vector_base(), seed_escenario=1, seed_ejecucion=2)
        pedido = instancia.pedidos[1]
        self.assertTrue(pedido.tiene_ventana_especifica)
        self.assertEqual(pedido.hora_desde_min, 500)
        self.assertEqual(pedido.hora_hasta_min, 650)

    def test_rechaza_longitud_incorrecta(self) -> None:
        with self.assertRaises(ValueError):
            decodificar_instancia_vector(self.vector_base()[:-1], seed_escenario=1, seed_ejecucion=2)
if __name__ == '__main__':
    unittest.main()
