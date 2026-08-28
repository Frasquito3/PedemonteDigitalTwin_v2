import unittest

from planner.rl.high_demand_policy_curriculum import (
    TIMESTEPS_ACUMULADOS_MAXIMOS_V4,
    TIMESTEPS_BASE_EXTENSION_V4,
    TIMESTEPS_COMPLETOS_ADICIONALES_V4,
    EtapaEntrenamientoCompletoTemporalV4RL,
    crear_curriculum_entrenamiento_completo_temporal_v4,
)


class TemporalV4FullCurriculumTest(unittest.TestCase):
    def test_curriculum_suma_280000_y_llega_a_348288(self) -> None:
        etapas = crear_curriculum_entrenamiento_completo_temporal_v4()
        self.assertEqual(len(etapas), 3)
        self.assertEqual(
            sum(etapa.timesteps for etapa in etapas),
            TIMESTEPS_COMPLETOS_ADICIONALES_V4,
        )
        self.assertEqual(TIMESTEPS_BASE_EXTENSION_V4, 68_288)
        self.assertEqual(TIMESTEPS_ACUMULADOS_MAXIMOS_V4, 348_288)

    def test_etapas_enfocan_general_exactos_y_consolidacion(self) -> None:
        etapas = crear_curriculum_entrenamiento_completo_temporal_v4()
        self.assertEqual(
            (etapas[0].min_pedidos_finales, etapas[0].max_pedidos_finales),
            (11, 12),
        )
        self.assertEqual(
            (etapas[1].min_pedidos_finales, etapas[1].max_pedidos_finales),
            (12, 12),
        )
        self.assertEqual(
            (etapas[2].min_pedidos_finales, etapas[2].max_pedidos_finales),
            (9, 12),
        )
        self.assertLess(etapas[0].probabilidad_patron_conflictivo, 0.5)
        self.assertGreater(etapas[1].probabilidad_banda_actual, 0.0)
        self.assertGreater(etapas[2].probabilidad_replay_exactos_12, 0.0)

    def test_todas_las_etapas_preservan_replay_3_8_y_9_10(self) -> None:
        for etapa in crear_curriculum_entrenamiento_completo_temporal_v4():
            self.assertGreater(etapa.probabilidad_replay_3_8, 0.0)
            self.assertGreater(etapa.probabilidad_replay_9_10, 0.0)
            self.assertGreater(etapa.probabilidad_banda_actual, 0.0)

    def test_rechaza_replay_sin_espacio_para_banda_actual(self) -> None:
        with self.assertRaises(ValueError):
            EtapaEntrenamientoCompletoTemporalV4RL(
                nombre="invalida",
                min_pedidos_finales=12,
                max_pedidos_finales=12,
                timesteps=1,
                eval_freq=1,
                checkpoint_freq=1,
                probabilidad_ventana_especifica=1.0,
                probabilidad_patron_conflictivo=0.0,
                probabilidad_replay_3_8=0.25,
                probabilidad_replay_9_10=0.25,
                probabilidad_replay_general_11_12=0.25,
                probabilidad_replay_exactos_12=0.25,
                probabilidad_volcador=0.0,
                probabilidad_pedido_grande=0.0,
                ancho_ventana_min=45,
                ancho_ventana_max=90,
            )


if __name__ == "__main__":
    unittest.main()
