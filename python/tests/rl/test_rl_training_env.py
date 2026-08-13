import unittest
from random import Random

import numpy as np

from planner.rl.instance_generator import (
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)

from planner.rl.rl_reward import (
    ConfiguracionRewardRL,
    ModoRewardRL,
)

from planner.rl.rl_training_env import (
    PedemonteTrainingEnv,
)

from planner.domain.validator import (
    validar_plan,
)


class PedemonteTrainingEnvTest(
    unittest.TestCase
):
    def test_semillas_fijas_se_repiten_en_ciclo(
        self,
    ) -> None:
        env = PedemonteTrainingEnv(
            semillas_fijas=[
                92_001,
                92_002,
            ]
        )

        _, info_a = env.reset()

        _, info_b = env.reset()

        _, info_c = env.reset()

        self.assertEqual(
            info_a["seed_instancia"],
            92_001,
        )

        self.assertEqual(
            info_b["seed_instancia"],
            92_002,
        )

        self.assertEqual(
            info_c["seed_instancia"],
            92_001,
        )

        env.close()

    def test_espacios_permanecen_estables(
        self,
    ) -> None:
        env = PedemonteTrainingEnv(
            seed_base=93_000
        )

        action_space_original = (
            env.action_space
        )

        observation_space_original = (
            env.observation_space
        )

        for _ in range(10):
            observacion, _ = env.reset()

            self.assertEqual(
                observacion.shape,
                (276,),
            )

            self.assertTrue(
                env.observation_space.contains(
                    observacion
                )
            )

            self.assertEqual(
                int(
                    env.action_masks().sum()
                ),

                len(
                    env.instancia.pedidos
                ),
            )

            self.assertEqual(
                env.action_space,
                action_space_original,
            )

            self.assertEqual(
                env.observation_space,
                observation_space_original,
            )

        env.close()

    def test_episodios_variables_son_validos(
        self,
    ) -> None:
        env = PedemonteTrainingEnv(
            seed_base=94_000
        )

        rng = Random(
            94_000
        )

        ids_instancia: set[str] = set()

        for _ in range(30):
            env.reset()

            ids_instancia.add(
                env.instancia.instancia_id
            )

            terminado = False

            while not terminado:
                acciones_validas = (
                    np.flatnonzero(
                        env.action_masks()
                    )
                    .tolist()
                )

                accion = int(
                    rng.choice(
                        acciones_validas
                    )
                )

                (
                    _,
                    _,
                    terminado,
                    truncado,
                    _,
                ) = env.step(
                    accion
                )

                self.assertFalse(
                    truncado
                )

            self.assertIsNotNone(
                env.ultimo_plan
            )

            assert (
                env.ultimo_plan
                is not None
            )

            validacion = validar_plan(
                env.instancia,
                env.ultimo_plan,
            )

            self.assertTrue(
                validacion.valido,

                msg=" | ".join(
                    validacion.errores
                ),
            )

        self.assertGreater(
            len(ids_instancia),
            1,
        )

        env.close()

    def test_soporta_curriculum_hasta_doce(
        self,
    ) -> None:
        configuracion_generador = (
            ConfiguracionGeneradorInstancias(
                min_pedidos_finales=12,

                max_pedidos_finales=12,

                probabilidad_pedido_mayor_capacidad=(
                    0.0
                ),
            )
        )

        env = PedemonteTrainingEnv(
            generador=(
                GeneradorInstanciasRL(
                    configuracion_generador
                )
            ),

            seed_base=98_000,
        )

        for _ in range(10):
            observacion, _ = env.reset()

            self.assertEqual(
                observacion.shape,
                (276,),
            )

            self.assertTrue(
                env.observation_space.contains(
                    observacion
                )
            )

            self.assertEqual(
                len(
                    env.instancia.pedidos
                ),
                12,
            )

            self.assertEqual(
                int(
                    env.action_masks().sum()
                ),
                12,
            )

        env.close()

    def test_entorno_variable_produce_reward_relativo(
        self,
    ) -> None:
        env = PedemonteTrainingEnv(
            seed_base=99_000,

            configuracion_reward=(
                ConfiguracionRewardRL(
                    modo=(
                        ModoRewardRL
                        .VENTAJA_GREEDY_RELATIVA
                    )
                )
            ),
        )

        env.reset()

        terminado = False

        info_final: dict = {}

        while not terminado:
            acciones_validas = (
                np.flatnonzero(
                    env.action_masks()
                )
                .tolist()
            )

            accion = int(
                acciones_validas[0]
            )

            (
                _,
                _,
                terminado,
                truncado,
                info,
            ) = env.step(
                accion
            )

            self.assertFalse(
                truncado
            )

            info_final = info

        self.assertIn(
            "costo_greedy_referencia",
            info_final,
        )

        self.assertIn(
            "gap_relativo_greedy",
            info_final,
        )

        self.assertEqual(
            info_final["modo_reward"],

            ModoRewardRL
            .VENTAJA_GREEDY_RELATIVA
            .value,
        )

        env.close()


if __name__ == "__main__":
    unittest.main()