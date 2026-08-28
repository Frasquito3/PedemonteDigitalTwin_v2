import unittest
from pathlib import Path
from planner.core.config import ConfiguracionPlanificacion
from planner.rl.training_cases import crear_casos_benchmark_clasico
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.schedule_estimator import analizar_prefijo_temporal
from planner.rl.policy_estimator import calcular_arrepentimiento_local, calcular_reward_terminal, proyectar_consecuencias_segundo_orden
from planner.routing.travel import construir_matriz_viaje
from planner.routing.vial_cache import ProveedorVialCachePersistente
PYTHON_ROOT = Path(__file__).resolve().parents[2]

class EstimadorPoliticaRLTest(unittest.TestCase):

    def setUp(self) -> None:
        self.instancia = next((caso.instancia for caso in crear_casos_benchmark_clasico() if caso.caso_id == 'B04_VENTANAS'))
        self.configuracion = ConfiguracionPlanificacion()
        self.proveedor = ProveedorVialCachePersistente(PYTHON_ROOT / 'data' / 'routing' / 'cache_vial.csv', version_cache_esperada='pedemonte-vial-v1', permitir_fallback=False)
        self.matriz = construir_matriz_viaje(self.instancia, self.configuracion, proveedor=self.proveedor)
        self.temporal = ConfiguracionPoliticaRL()

    def test_arrepentimiento_prefiere_este(self) -> None:
        consecuencias = proyectar_consecuencias_segundo_orden(self.instancia, self.matriz, self.configuracion, ('B04-NORTE-TEMPRANO',))
        resultado_este = calcular_arrepentimiento_local(consecuencias, 'B04-ESTE-MEDIO', self.temporal)
        resultado_cercana = calcular_arrepentimiento_local(consecuencias, 'B04-CERCANA-TARDE', self.temporal)
        self.assertTrue(resultado_este.es_mejor_accion)
        self.assertEqual(resultado_este.mejor_pedido_id, 'B04-ESTE-MEDIO')
        self.assertGreater(resultado_este.reward_local, resultado_cercana.reward_local)
        self.assertGreater(resultado_cercana.arrepentimiento_normalizado, 0.0)

    def test_banda_terminal_factible_domina_costo_extremo(self) -> None:
        resumen_factible = analizar_prefijo_temporal(self.instancia, self.matriz, self.configuracion, ('B04-NORTE-TEMPRANO', 'B04-ESTE-MEDIO', 'B04-CERCANA-TARDE'))
        resumen_riesgoso = analizar_prefijo_temporal(self.instancia, self.matriz, self.configuracion, ('B04-NORTE-TEMPRANO', 'B04-CERCANA-TARDE', 'B04-ESTE-MEDIO'))
        factible = calcular_reward_terminal(resumen_factible, reward_costo_base=-100.0, configuracion=self.temporal)
        riesgoso = calcular_reward_terminal(resumen_riesgoso, reward_costo_base=100.0, configuracion=self.temporal)
        self.assertTrue(factible.factible_temporalmente)
        self.assertFalse(riesgoso.factible_temporalmente)
        self.assertGreater(factible.reward_terminal_total, riesgoso.reward_terminal_total)
if __name__ == '__main__':
    unittest.main()
