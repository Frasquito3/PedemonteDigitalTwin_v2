import unittest

from planner.rl.rl_temporal_v4_validation import (
    ResumenValidacionExternaV4,
    es_mejor_validacion_externa_v4,
)


class TemporalV4ValidationTest(unittest.TestCase):
    def _resumen(
        self,
        *,
        tardios_b04: int,
        tardanza_b04: float,
        sin_riesgo: int,
        tardanza_total: float,
        gap: float,
    ) -> ResumenValidacionExternaV4:
        return ResumenValidacionExternaV4(
            timestep=1000,
            b04_pedidos_tardios=tardios_b04,
            b04_tardanza_min=tardanza_b04,
            b04_costo_estimado=100.0,
            sinteticos_totales=12,
            sinteticos_sin_riesgo=sin_riesgo,
            tasa_sintetica_sin_riesgo_pct=100.0 * sin_riesgo / 12,
            tardanza_sintetica_total_min=tardanza_total,
            tardanza_sintetica_mediana_min=0.0,
            gap_costo_mediano_vs_greedy_pct=gap,
            casos=(),
        )

    def test_b04_factible_domina_mejor_costo_sintetico(self) -> None:
        actual = self._resumen(
            tardios_b04=1,
            tardanza_b04=2.0,
            sin_riesgo=12,
            tardanza_total=0.0,
            gap=-50.0,
        )
        candidata = self._resumen(
            tardios_b04=0,
            tardanza_b04=0.0,
            sin_riesgo=9,
            tardanza_total=10.0,
            gap=5.0,
        )
        self.assertTrue(
            es_mejor_validacion_externa_v4(candidata, actual)
        )


if __name__ == "__main__":
    unittest.main()
