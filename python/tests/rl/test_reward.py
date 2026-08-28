import unittest
from typing import Any
from planner.rl.base_env import PedemontePlanEnv
from planner.rl.reward import ConfiguracionRewardRL, ModoRewardRL
from tests.fixtures import crear_instancia_demo

def ejecutar_secuencia(env: PedemontePlanEnv, secuencia: list[str]) -> tuple[float, dict[str, Any]]:
    if not secuencia:
        raise ValueError('La secuencia no puede estar vacía.')
    env.reset(seed=97001)
    reward_final = 0.0
    info_final: dict[str, Any] = {}
    terminado = False
    for pedido_id in secuencia:
        _, reward, terminado, truncado, info = env.step(env.accion_de_pedido_id(pedido_id))
        if truncado:
            raise AssertionError('El episodio fue truncado.')
        reward_final = reward
        info_final = info
    if not terminado:
        raise AssertionError('La secuencia no terminó el episodio.')
    return (reward_final, info_final)

class RewardRLTest(unittest.TestCase):

    def crear_env_relativo(self) -> PedemontePlanEnv:
        return PedemontePlanEnv(crear_instancia_demo(), configuracion_reward=ConfiguracionRewardRL(modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA))

    def test_reward_conserva_orden_de_costos(self) -> None:
        env_bueno = self.crear_env_relativo()
        env_malo = self.crear_env_relativo()
        try:
            reward_bueno, _ = ejecutar_secuencia(env_bueno, ['P004', 'P003', 'P001', 'P002'])
            reward_malo, _ = ejecutar_secuencia(env_malo, ['P003', 'P004', 'P002', 'P001'])
            self.assertGreater(reward_bueno, reward_malo)
        finally:
            env_bueno.close()
            env_malo.close()
if __name__ == '__main__':
    unittest.main()
