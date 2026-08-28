import unittest
from random import Random
import numpy as np
from stable_baselines3.common.env_checker import check_env
from planner.rl.base_env import PedemontePlanEnv
from planner.core.schema import AlgoritmoPlanificacion
from planner.domain.validator import validar_plan
from tests.fixtures import crear_instancia_demo

class PedemontePlanEnvTest(unittest.TestCase):

    def test_cumple_api_gymnasium(self) -> None:
        env = PedemontePlanEnv(crear_instancia_demo())
        check_env(env, warn=True, skip_render_check=True)

    def test_episodios_aleatorios_enmascarados_validos(self) -> None:
        instancia = crear_instancia_demo()
        for seed in range(50):
            env = PedemontePlanEnv(instancia)
            env.reset(seed=seed)
            rng = Random(seed)
            terminado = False
            while not terminado:
                acciones_validas = np.flatnonzero(env.action_masks()).tolist()
                accion = int(rng.choice(acciones_validas))
                _, _, terminado, truncado, _ = env.step(accion)
                self.assertFalse(truncado)
            self.assertIsNotNone(env.ultimo_plan)
            assert env.ultimo_plan is not None
            validacion = validar_plan(instancia, env.ultimo_plan)
            self.assertTrue(validacion.valido, msg=f'Seed={seed}: ' + ' | '.join(validacion.errores))
if __name__ == '__main__':
    unittest.main()
