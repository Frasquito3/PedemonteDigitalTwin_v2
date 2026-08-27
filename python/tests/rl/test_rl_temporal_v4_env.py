import unittest
from pathlib import Path
from typing import Any

from planner.evaluation.classic_instances import (
    crear_casos_benchmark_clasico,
)
from planner.rl.rl_env import PedemontePlanEnv
from planner.rl.rl_reward import ConfiguracionRewardRL, ModoRewardRL
from planner.rl.rl_temporal_v4_env import PedemonteTemporalV4PlanEnv
from planner.routing.vial_cache import ProveedorVialCachePersistente


PYTHON_ROOT = Path(__file__).resolve().parents[2]


def ejecutar(
    env: PedemonteTemporalV4PlanEnv,
    secuencia: tuple[str, ...],
) -> tuple[float, dict[str, Any]]:
    env.reset(seed=15_104)
    reward_total = 0.0
    info_final: dict[str, Any] = {}
    terminado = False

    for pedido_id in secuencia:
        _, reward, terminado, truncado, info = env.step(
            env.accion_de_pedido_id(pedido_id)
        )
        if truncado:
            raise AssertionError("El episodio v4 fue truncado.")
        reward_total += reward
        info_final = dict(info)

    if not terminado:
        raise AssertionError("La secuencia no terminó el episodio.")

    return reward_total, info_final


class RLTemporalV4EnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.instancia = next(
            caso.instancia
            for caso in crear_casos_benchmark_clasico()
            if caso.caso_id == "B04_VENTANAS"
        )
        self.proveedor = ProveedorVialCachePersistente(
            PYTHON_ROOT / "data" / "routing" / "cache_vial_v1.csv",
            version_cache_esperada="pedemonte-vial-v1",
            permitir_fallback=False,
        )
        self.reward = ConfiguracionRewardRL(
            modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA
        )

    def crear_env(self) -> PedemonteTemporalV4PlanEnv:
        return PedemonteTemporalV4PlanEnv(
            self.instancia,
            proveedor_viaje=self.proveedor,
            configuracion_reward=self.reward,
        )

    def test_dimensiones_separadas_historico_y_v4(self) -> None:
        historico = PedemontePlanEnv(
            self.instancia,
            proveedor_viaje=self.proveedor,
        )
        v4 = self.crear_env()
        try:
            obs_h, _ = historico.reset()
            obs_v4, _ = v4.reset()
            self.assertEqual(obs_h.shape, (276,))
            self.assertEqual(obs_v4.shape, (702,))
            self.assertTrue(v4.observation_space.contains(obs_v4))
        finally:
            historico.close()
            v4.close()

    def test_observacion_marca_este_como_mejor_despues_de_norte(self) -> None:
        env = self.crear_env()
        try:
            env.reset(seed=15_104)
            env.step(env.accion_de_pedido_id("B04-NORTE-TEMPRANO"))
            consecuencias = env.consecuencias_temporales_actuales
            self.assertEqual(
                consecuencias["B04-ESTE-MEDIO"].pedidos_tardios_finales,
                0,
            )
            self.assertGreaterEqual(
                consecuencias[
                    "B04-CERCANA-TARDE"
                ].pedidos_tardios_finales,
                1,
            )
        finally:
            env.close()

    def test_reward_acumulado_prefiere_secuencia_factible(self) -> None:
        riesgoso = self.crear_env()
        factible = self.crear_env()
        try:
            reward_r, info_r = ejecutar(
                riesgoso,
                (
                    "B04-NORTE-TEMPRANO",
                    "B04-CERCANA-TARDE",
                    "B04-ESTE-MEDIO",
                ),
            )
            reward_f, info_f = ejecutar(
                factible,
                (
                    "B04-NORTE-TEMPRANO",
                    "B04-ESTE-MEDIO",
                    "B04-CERCANA-TARDE",
                ),
            )
            self.assertGreater(reward_f, reward_r)
            self.assertEqual(info_f["pedidos_tardios_prefijo"], 0)
            self.assertGreaterEqual(info_r["pedidos_tardios_prefijo"], 1)
            self.assertTrue(info_f["factible_temporal_terminal"])
            self.assertFalse(info_r["factible_temporal_terminal"])
        finally:
            riesgoso.close()
            factible.close()

    def test_mascara_temporal_dura_desactivada(self) -> None:
        env = self.crear_env()
        try:
            env.reset(seed=15_104)
            env.step(env.accion_de_pedido_id("B04-NORTE-TEMPRANO"))
            self.assertTrue(
                bool(
                    env.action_masks()[
                        env.accion_de_pedido_id("B04-CERCANA-TARDE")
                    ]
                )
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
