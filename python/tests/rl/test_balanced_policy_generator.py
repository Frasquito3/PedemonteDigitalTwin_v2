import unittest

from planner.rl.balanced_policy_generator import (
    GeneradorReplayMultibandaTemporalV4RL,
)


class _GeneradorFalso:
    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.seeds: list[int] = []

    def generar(self, seed: int):
        self.seeds.append(seed)
        return self.nombre, seed


class TemporalV4ExtensionGeneratorTest(unittest.TestCase):
    def test_fuente_es_determinista_por_seed(self) -> None:
        generador = GeneradorReplayMultibandaTemporalV4RL(
            generador_actual=_GeneradorFalso("actual"),
            generador_replay_3_8=_GeneradorFalso("3_8"),
            generador_replay_9_10=_GeneradorFalso("9_10"),
            probabilidad_replay_3_8=0.2,
            probabilidad_replay_9_10=0.3,
        )
        self.assertEqual(
            generador.seleccionar_fuente(12345),
            generador.seleccionar_fuente(12345),
        )

    def test_probabilidad_cero_usa_banda_actual(self) -> None:
        actual = _GeneradorFalso("actual")
        generador = GeneradorReplayMultibandaTemporalV4RL(
            generador_actual=actual,
            generador_replay_3_8=_GeneradorFalso("3_8"),
            generador_replay_9_10=_GeneradorFalso("9_10"),
            probabilidad_replay_3_8=0.0,
            probabilidad_replay_9_10=0.0,
        )
        resultado = generador.generar(77)
        self.assertEqual(resultado[0], "actual")
        self.assertEqual(actual.seeds, [77])


if __name__ == "__main__":
    unittest.main()
