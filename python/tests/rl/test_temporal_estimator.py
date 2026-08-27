import unittest
from pathlib import Path

from planner.core.config import ConfiguracionPlanificacion
from planner.evaluation.classic_instances import (
    crear_casos_benchmark_clasico,
)
from planner.rl.rl_temporal_config import ConfiguracionTemporalRL
from planner.rl.temporal_estimator import (
    analizar_prefijo_temporal,
    calcular_potencial_temporal,
    proyectar_acciones_pendientes,
)
from planner.routing.travel import construir_matriz_viaje
from planner.routing.vial_cache import ProveedorVialCachePersistente


PYTHON_ROOT = Path(__file__).resolve().parents[2]


class TemporalEstimatorTest(unittest.TestCase):
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

    def test_detecta_secuencia_b04_riesgosa(self) -> None:
        resumen = analizar_prefijo_temporal(
            self.instancia,
            self.matriz,
            self.configuracion,
            (
                "B04-NORTE-TEMPRANO",
                "B04-CERCANA-TARDE",
                "B04-ESTE-MEDIO",
            ),
        )

        registro = resumen.registro_de(
            "B04-ESTE-MEDIO"
        )

        self.assertIsNotNone(registro)
        assert registro is not None

        self.assertTrue(registro.llegada_tardia)
        self.assertGreater(
            registro.tardanza_llegada_min,
            0.0,
        )
        self.assertEqual(resumen.pedidos_tardios, 1)

    def test_secuencia_factible_no_tiene_tardanza(self) -> None:
        resumen = analizar_prefijo_temporal(
            self.instancia,
            self.matriz,
            self.configuracion,
            (
                "B04-NORTE-TEMPRANO",
                "B04-ESTE-MEDIO",
                "B04-CERCANA-TARDE",
            ),
        )

        self.assertEqual(resumen.pedidos_tardios, 0)
        self.assertAlmostEqual(
            resumen.tardanza_total_min,
            0.0,
            places=9,
        )

    def test_potencial_prefiere_este_antes_que_cercana(self) -> None:
        configuracion_temporal = ConfiguracionTemporalRL()

        def potencial(prefijo: tuple[str, ...]) -> float:
            resumen = analizar_prefijo_temporal(
                self.instancia,
                self.matriz,
                self.configuracion,
                prefijo,
            )
            proyecciones = proyectar_acciones_pendientes(
                self.instancia,
                self.matriz,
                self.configuracion,
                prefijo,
            )
            return calcular_potencial_temporal(
                resumen,
                proyecciones,
                configuracion_temporal,
            ).potencial

        potencial_este = potencial(
            (
                "B04-NORTE-TEMPRANO",
                "B04-ESTE-MEDIO",
            )
        )
        potencial_cercana = potencial(
            (
                "B04-NORTE-TEMPRANO",
                "B04-CERCANA-TARDE",
            )
        )

        self.assertGreater(
            potencial_este,
            potencial_cercana,
        )


if __name__ == "__main__":
    unittest.main()
