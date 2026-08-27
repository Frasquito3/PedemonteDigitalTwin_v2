import unittest

from planner.rl.instance_generator import (
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)
from planner.rl.temporal_v4_instance_generator import (
    ConfiguracionGeneradorTemporalV4,
    GeneradorInstanciasTemporalV4RL,
)


class TemporalV4InstanceGeneratorTest(unittest.TestCase):
    def test_genera_patron_v4_temprano_medio_tardio(self) -> None:
        base = GeneradorInstanciasRL(
            ConfiguracionGeneradorInstancias(
                min_pedidos_finales=4,
                max_pedidos_finales=4,
                probabilidad_volcador=0.0,
                probabilidad_pedido_mayor_capacidad=0.0,
            )
        )
        generador = GeneradorInstanciasTemporalV4RL(
            base,
            ConfiguracionGeneradorTemporalV4(
                probabilidad_patron_ventanas_conflictivas=1.0
            ),
        )
        instancia = generador.generar(164_101)
        marcados = [
            pedido
            for pedido in instancia.pedidos
            if GeneradorInstanciasTemporalV4RL.MARCA_PATRON
            in pedido.observaciones
        ]

        self.assertEqual(len(marcados), 3)
        self.assertTrue(instancia.instancia_id.endswith("-TEMPORAL-V4"))
        ordenados = sorted(marcados, key=lambda p: p.hora_desde_min)
        self.assertLess(
            ordenados[0].hora_hasta_min,
            ordenados[1].hora_desde_min,
        )
        self.assertLess(
            ordenados[1].hora_hasta_min,
            ordenados[2].hora_desde_min,
        )


if __name__ == "__main__":
    unittest.main()
