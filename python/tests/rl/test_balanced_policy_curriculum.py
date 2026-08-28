import unittest

from planner.rl.balanced_policy_curriculum import (
    EtapaExtensionTemporalV4RL,
    crear_curriculum_extension_temporal_v4_diagnostico,
)


class TemporalV4ExtensionCurriculumTest(unittest.TestCase):
    def test_curriculum_enfoca_9_10_y_11_12(self) -> None:
        etapas = crear_curriculum_extension_temporal_v4_diagnostico()
        self.assertEqual(len(etapas), 2)
        self.assertEqual(
            (etapas[0].min_pedidos_finales, etapas[0].max_pedidos_finales),
            (9, 10),
        )
        self.assertEqual(
            (etapas[1].min_pedidos_finales, etapas[1].max_pedidos_finales),
            (11, 12),
        )
        self.assertEqual(sum(item.timesteps for item in etapas), 48_000)

    def test_segunda_etapa_conserva_replay_3_8_y_9_10(self) -> None:
        etapa = crear_curriculum_extension_temporal_v4_diagnostico()[1]
        self.assertGreater(etapa.probabilidad_replay_3_8, 0.0)
        self.assertGreater(etapa.probabilidad_replay_9_10, 0.0)
        self.assertGreater(etapa.probabilidad_banda_actual, 0.0)

    def test_rechaza_suma_replay_mayor_a_uno(self) -> None:
        with self.assertRaises(ValueError):
            EtapaExtensionTemporalV4RL(
                nombre="invalida",
                min_pedidos_finales=9,
                max_pedidos_finales=10,
                timesteps=1,
                eval_freq=1,
                checkpoint_freq=1,
                probabilidad_ventana_especifica=1.0,
                probabilidad_patron_conflictivo=1.0,
                probabilidad_replay_3_8=0.7,
                probabilidad_replay_9_10=0.4,
                probabilidad_volcador=0.0,
                probabilidad_pedido_grande=0.0,
                ancho_ventana_min=45,
                ancho_ventana_max=90,
            )


if __name__ == "__main__":
    unittest.main()
