import unittest

from planner.rl.policy_curriculum import (
    crear_curriculum_temporal_v4,
    crear_curriculum_temporal_v4_rapido,
)


class TemporalV4CurriculumTest(unittest.TestCase):
    def test_curriculum_completo_conserva_350000_pasos(self) -> None:
        etapas = crear_curriculum_temporal_v4()
        self.assertEqual(sum(etapa.timesteps for etapa in etapas), 350_000)
        self.assertEqual(len(etapas), 3)

    def test_curriculum_rapido_tiene_replay_en_etapas_posteriores(self) -> None:
        etapas = crear_curriculum_temporal_v4_rapido()
        self.assertEqual(sum(etapa.timesteps for etapa in etapas), 21_000)
        self.assertGreater(etapas[1].probabilidad_replay_core, 0.0)
        self.assertGreater(etapas[2].probabilidad_replay_core, 0.0)


if __name__ == "__main__":
    unittest.main()
