import unittest
from pathlib import Path

from planner.core.config import ConfiguracionPlanificacion
from planner.evaluation.classic_instances import (
    crear_casos_benchmark_clasico,
)
from planner.rl.rl_temporal_v4_config import ConfiguracionTemporalV4RL
from planner.rl.temporal_estimator import analizar_prefijo_temporal
from planner.rl.temporal_v4_estimator import (
    calcular_arrepentimiento_local_v4,
    calcular_reward_terminal_v4,
    proyectar_consecuencias_segundo_orden_v4,
)
from planner.routing.travel import construir_matriz_viaje
from planner.routing.vial_cache import ProveedorVialCachePersistente


PYTHON_ROOT = Path(__file__).resolve().parents[2]


class TemporalV4EstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.instancia = next(
            caso.instancia
            for caso in crear_casos_benchmark_clasico()
            if caso.caso_id == "B04_VENTANAS"
        )
        self.configuracion = ConfiguracionPlanificacion()
        self.proveedor = ProveedorVialCachePersistente(
            PYTHON_ROOT / "data" / "routing" / "cache_vial_v1.csv",
            version_cache_esperada="pedemonte-vial-v1",
            permitir_fallback=False,
        )
        self.matriz = construir_matriz_viaje(
            self.instancia,
            self.configuracion,
            proveedor=self.proveedor,
        )
        self.temporal = ConfiguracionTemporalV4RL()

    def test_segundo_orden_expone_riesgo_de_cercana(self) -> None:
        consecuencias = proyectar_consecuencias_segundo_orden_v4(
            self.instancia,
            self.matriz,
            self.configuracion,
            ("B04-NORTE-TEMPRANO",),
        )

        este = consecuencias["B04-ESTE-MEDIO"]
        cercana = consecuencias["B04-CERCANA-TARDE"]

        self.assertEqual(este.pedidos_tardios_finales, 0)
        self.assertEqual(
            este.secuencia_completada,
            (
                "B04-NORTE-TEMPRANO",
                "B04-ESTE-MEDIO",
                "B04-CERCANA-TARDE",
            ),
        )
        self.assertGreaterEqual(cercana.pedidos_tardios_finales, 1)
        self.assertGreater(cercana.tardanza_total_final_min, 0.0)
        self.assertGreaterEqual(cercana.pedidos_nuevos_en_riesgo, 1)

    def test_arrepentimiento_prefiere_este(self) -> None:
        consecuencias = proyectar_consecuencias_segundo_orden_v4(
            self.instancia,
            self.matriz,
            self.configuracion,
            ("B04-NORTE-TEMPRANO",),
        )
        resultado_este = calcular_arrepentimiento_local_v4(
            consecuencias,
            "B04-ESTE-MEDIO",
            self.temporal,
        )
        resultado_cercana = calcular_arrepentimiento_local_v4(
            consecuencias,
            "B04-CERCANA-TARDE",
            self.temporal,
        )

        self.assertTrue(resultado_este.es_mejor_accion)
        self.assertEqual(resultado_este.mejor_pedido_id, "B04-ESTE-MEDIO")
        self.assertGreater(
            resultado_este.reward_local,
            resultado_cercana.reward_local,
        )
        self.assertGreater(
            resultado_cercana.arrepentimiento_normalizado,
            0.0,
        )

    def test_banda_terminal_factible_domina_costo_extremo(self) -> None:
        resumen_factible = analizar_prefijo_temporal(
            self.instancia,
            self.matriz,
            self.configuracion,
            (
                "B04-NORTE-TEMPRANO",
                "B04-ESTE-MEDIO",
                "B04-CERCANA-TARDE",
            ),
        )
        resumen_riesgoso = analizar_prefijo_temporal(
            self.instancia,
            self.matriz,
            self.configuracion,
            (
                "B04-NORTE-TEMPRANO",
                "B04-CERCANA-TARDE",
                "B04-ESTE-MEDIO",
            ),
        )
        factible = calcular_reward_terminal_v4(
            resumen_factible,
            reward_costo_base=-100.0,
            configuracion=self.temporal,
        )
        riesgoso = calcular_reward_terminal_v4(
            resumen_riesgoso,
            reward_costo_base=100.0,
            configuracion=self.temporal,
        )

        self.assertTrue(factible.factible_temporalmente)
        self.assertFalse(riesgoso.factible_temporalmente)
        self.assertGreater(
            factible.reward_terminal_total,
            riesgoso.reward_terminal_total,
        )


if __name__ == "__main__":
    unittest.main()
