import unittest
from pathlib import Path
from typing import Any
from planner.rl.training_cases import crear_casos_benchmark_clasico
from planner.rl.base_env import PedemontePlanEnv
from planner.rl.reward import ConfiguracionRewardRL, ModoRewardRL
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.policy_env import EntornoPlanificacionRL
from planner.routing.vial_cache import ProveedorVialCachePersistente
PYTHON_ROOT = Path(__file__).resolve().parents[2]

def ejecutar(env: EntornoPlanificacionRL, secuencia: tuple[str, ...]) -> tuple[float, dict[str, Any]]:
    env.reset(seed=15104)
    reward_total = 0.0
    info_final: dict[str, Any] = {}
    terminado = False
    for pedido_id in secuencia:
        _, reward, terminado, truncado, info = env.step(env.accion_de_pedido_id(pedido_id))
        if truncado:
            raise AssertionError('El episodio de la política fue truncado.')
        reward_total += reward
        info_final = dict(info)
    if not terminado:
        raise AssertionError('La secuencia no terminó el episodio.')
    return (reward_total, info_final)

class EntornoPoliticaRLTest(unittest.TestCase):

    def setUp(self) -> None:
        self.instancia = next((caso.instancia for caso in crear_casos_benchmark_clasico() if caso.caso_id == 'B04_VENTANAS'))
        self.proveedor = ProveedorVialCachePersistente(PYTHON_ROOT / 'data' / 'routing' / 'cache_vial.csv', version_cache_esperada='pedemonte-vial-v1', permitir_fallback=False)
        self.reward = ConfiguracionRewardRL(modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA)

    def crear_env(
        self,
        *,
        usar_mascara_temporal_dura: bool = True,
    ) -> EntornoPlanificacionRL:
        return EntornoPlanificacionRL(
            self.instancia,
            proveedor_viaje=self.proveedor,
            configuracion_reward=self.reward,
            configuracion_temporal=ConfiguracionPoliticaRL(
                usar_mascara_temporal_dura=usar_mascara_temporal_dura,
            ),
        )

    def test_entorno_politica_mantiene_dimension_esperada(self) -> None:
        entorno_base = PedemontePlanEnv(self.instancia, proveedor_viaje=self.proveedor)
        entorno_politica = self.crear_env()
        try:
            obs_h, _ = entorno_base.reset()
            obs_politica, _ = entorno_politica.reset()
            self.assertEqual(obs_h.shape, (276,))
            self.assertEqual(obs_politica.shape, (702,))
            self.assertTrue(entorno_politica.observation_space.contains(obs_politica))
        finally:
            entorno_base.close()
            entorno_politica.close()

    def test_reward_acumulado_prefiere_secuencia_factible(self) -> None:
        # Esta prueba compara la recompensa de dos secuencias completas.
        # La máscara dura se desactiva solo aquí para permitir ejecutar
        # deliberadamente la secuencia temporalmente riesgosa.
        riesgoso = self.crear_env(usar_mascara_temporal_dura=False)
        factible = self.crear_env(usar_mascara_temporal_dura=False)
        try:
            reward_r, info_r = ejecutar(riesgoso, ('B04-NORTE-TEMPRANO', 'B04-CERCANA-TARDE', 'B04-ESTE-MEDIO'))
            reward_f, info_f = ejecutar(factible, ('B04-NORTE-TEMPRANO', 'B04-ESTE-MEDIO', 'B04-CERCANA-TARDE'))
            self.assertGreater(reward_f, reward_r)
            self.assertEqual(info_f['pedidos_tardios_prefijo'], 0)
            self.assertGreaterEqual(info_r['pedidos_tardios_prefijo'], 1)
            self.assertTrue(info_f['factible_temporal_terminal'])
            self.assertFalse(info_r['factible_temporal_terminal'])
        finally:
            riesgoso.close()
            factible.close()
if __name__ == '__main__':
    unittest.main()
