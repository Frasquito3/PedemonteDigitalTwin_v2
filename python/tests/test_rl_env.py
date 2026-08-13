import unittest
from random import Random

import numpy as np

# pyrefly: ignore [missing-import]
from stable_baselines3.common.env_checker import (
    check_env,
)

from planner.rl_env import (
    PedemontePlanEnv,
)

from planner.schema import (
    AlgoritmoPlanificacion,
)

from planner.validator import (
    validar_plan,
)

from tests.fixtures import (
    crear_instancia_demo,
)


class PedemontePlanEnvTest(
    unittest.TestCase
):
    def test_cumple_api_gymnasium(
        self,
    ) -> None:
        env = PedemontePlanEnv(
            crear_instancia_demo()
        )

        check_env(
            env,
            warn=True,
            skip_render_check=True,
        )

    def test_mascara_inicial_y_reduccion(
        self,
    ) -> None:
        env = PedemontePlanEnv(
            crear_instancia_demo()
        )

        observacion, _ = env.reset(
            seed=9001
        )

        self.assertTrue(
            env.observation_space.contains(
                observacion
            )
        )

        self.assertEqual(
            observacion.shape,
            (276,),
        )

        mascara_inicial = (
            env.action_masks()
        )

        self.assertEqual(
            int(
                mascara_inicial.sum()
            ),
            4,
        )

        accion_p001 = (
            env.accion_de_pedido_id(
                "P001"
            )
        )

        env.step(
            accion_p001
        )

        self.assertEqual(
            int(
                env.action_masks().sum()
            ),
            3,
        )

        self.assertFalse(
            bool(
                env.action_masks()[
                    accion_p001
                ]
            )
        )

    def test_permutacion_greedy_genera_plan_valido(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        env = PedemontePlanEnv(
            instancia
        )

        env.reset(
            seed=9001
        )

        secuencia = [
            "P004",
            "P003",
            "P001",
            "P002",
        ]

        terminado = False

        recompensa_final = 0.0

        for pedido_id in secuencia:
            accion = (
                env.accion_de_pedido_id(
                    pedido_id
                )
            )

            (
                _,
                recompensa,
                terminado,
                truncado,
                _,
            ) = env.step(
                accion
            )

            self.assertFalse(
                truncado
            )

            recompensa_final = (
                recompensa
            )

        self.assertTrue(
            terminado
        )

        self.assertIsNotNone(
            env.ultimo_plan
        )

        plan = env.ultimo_plan

        assert plan is not None

        validacion = validar_plan(
            instancia,
            plan,
        )

        self.assertTrue(
            validacion.valido,

            msg=" | ".join(
                validacion.errores
            ),
        )

        self.assertEqual(
            plan.algoritmo,
            AlgoritmoPlanificacion.RL,
        )

        self.assertLess(
            recompensa_final,
            0.0,
        )

        self.assertAlmostEqual(
            recompensa_final,

            -plan.costo_estimado
            / env.escala_reward,

            places=9,
        )

    def test_misma_secuencia_reproduce_costo(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        secuencia = [
            "P004",
            "P003",
            "P001",
            "P002",
        ]

        costos: list[float] = []

        for _ in range(2):
            env = PedemontePlanEnv(
                instancia
            )

            env.reset(
                seed=9001
            )

            for pedido_id in secuencia:
                env.step(
                    env.accion_de_pedido_id(
                        pedido_id
                    )
                )

            self.assertIsNotNone(
                env.ultimo_plan
            )

            assert (
                env.ultimo_plan
                is not None
            )

            costos.append(
                env.ultimo_plan
                .costo_estimado
            )

        self.assertAlmostEqual(
            costos[0],
            costos[1],
            places=12,
        )

    def test_episodios_aleatorios_enmascarados_validos(
        self,
    ) -> None:
        instancia = crear_instancia_demo()

        for seed in range(50):
            env = PedemontePlanEnv(
                instancia
            )

            env.reset(
                seed=seed
            )

            rng = Random(
                seed
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
                instancia,
                env.ultimo_plan,
            )

            self.assertTrue(
                validacion.valido,

                msg=(
                    f"Seed={seed}: "
                    + " | ".join(
                        validacion.errores
                    )
                ),
            )

    def test_accion_padding_se_penaliza(
        self,
    ) -> None:
        env = PedemontePlanEnv(
            crear_instancia_demo()
        )

        env.reset(
            seed=9001
        )

        (
            _,
            recompensa,
            terminado,
            truncado,
            info,
        ) = env.step(
            29
        )

        self.assertFalse(
            terminado
        )

        self.assertFalse(
            truncado
        )

        self.assertEqual(
            recompensa,
            -1.0,
        )

        self.assertFalse(
            info["accion_valida"]
        )

        self.assertEqual(
            info["motivo"],
            "ACCION_PADDING",
        )

        self.assertEqual(
            env.cantidad_seleccionados,
            0,
        )

    def test_rechaza_instancia_mayor_que_maximo(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            PedemontePlanEnv(
                crear_instancia_demo(),
                max_pedidos=3,
            )


if __name__ == "__main__":
    unittest.main()