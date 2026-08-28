import unittest

from planner.rl.high_demand_policy_generator import (
    FuenteReplayTemporalV4,
    GeneradorMezclaCompletaTemporalV4RL,
)


class _GeneradorFalso:
    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.seeds: list[int] = []

    def generar(self, seed: int):
        self.seeds.append(seed)
        return self.nombre, seed


class TemporalV4FullGeneratorTest(unittest.TestCase):
    def test_fuente_es_determinista_por_seed(self) -> None:
        generador = GeneradorMezclaCompletaTemporalV4RL(
            generador_actual=_GeneradorFalso("actual"),
            fuentes_replay=(
                FuenteReplayTemporalV4(
                    "3_8",
                    _GeneradorFalso("3_8"),
                    0.2,
                    0x38,
                ),
                FuenteReplayTemporalV4(
                    "12",
                    _GeneradorFalso("12"),
                    0.3,
                    0x12,
                ),
            ),
        )
        self.assertEqual(
            generador.seleccionar_fuente(12345),
            generador.seleccionar_fuente(12345),
        )

    def test_sin_replay_usa_banda_actual(self) -> None:
        actual = _GeneradorFalso("actual")
        generador = GeneradorMezclaCompletaTemporalV4RL(
            generador_actual=actual,
            fuentes_replay=(),
        )
        resultado = generador.generar(77)
        self.assertEqual(resultado[0], "actual")
        self.assertEqual(actual.seeds, [77])
        self.assertEqual(generador.probabilidad_banda_actual, 1.0)

    def test_rechaza_nombres_duplicados(self) -> None:
        with self.assertRaises(ValueError):
            GeneradorMezclaCompletaTemporalV4RL(
                generador_actual=_GeneradorFalso("actual"),
                fuentes_replay=(
                    FuenteReplayTemporalV4(
                        "replay",
                        _GeneradorFalso("a"),
                        0.1,
                        1,
                    ),
                    FuenteReplayTemporalV4(
                        "replay",
                        _GeneradorFalso("b"),
                        0.1,
                        2,
                    ),
                ),
            )

    def test_rechaza_suma_igual_a_uno(self) -> None:
        with self.assertRaises(ValueError):
            GeneradorMezclaCompletaTemporalV4RL(
                generador_actual=_GeneradorFalso("actual"),
                fuentes_replay=(
                    FuenteReplayTemporalV4(
                        "todo",
                        _GeneradorFalso("replay"),
                        1.0,
                        1,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
