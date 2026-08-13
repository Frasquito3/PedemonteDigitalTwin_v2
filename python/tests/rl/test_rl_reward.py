import unittest
from typing import Any

from planner.rl.rl_env import (
    PedemontePlanEnv,
)

from planner.rl.rl_reward import (
    ConfiguracionRewardRL,
    ModoRewardRL,
)

from tests.fixtures import (
    crear_instancia_demo,
)


def ejecutar_secuencia(
    env: PedemontePlanEnv,
    secuencia: list[str],
) -> tuple[
    float,
    dict[str, Any],
]:
    if not secuencia:
        raise ValueError(
            "La secuencia no puede estar vacía."
        )

    env.reset(
        seed=97_001
    )

    reward_final = 0.0

    info_final: dict[
        str,
        Any,
    ] = {}

    # Se inicializa antes del bucle para que tanto
    # Python como Pyrefly puedan garantizar que la
    # variable siempre existe.
    terminado = False

    for pedido_id in secuencia:
        (
            _,
            reward,
            terminado,
            truncado,
            info,
        ) = env.step(
            env.accion_de_pedido_id(
                pedido_id
            )
        )

        if truncado:
            raise AssertionError(
                "El episodio fue truncado."
            )

        reward_final = reward

        info_final = info

    if not terminado:
        raise AssertionError(
            "La secuencia no terminó "
            "el episodio."
        )

    return (
        reward_final,
        info_final,
    )


class RewardRLTest(
    unittest.TestCase
):
    def crear_env_relativo(
        self,
    ) -> PedemontePlanEnv:
        return PedemontePlanEnv(
            crear_instancia_demo(),

            configuracion_reward=(
                ConfiguracionRewardRL(
                    modo=(
                        ModoRewardRL
                        .VENTAJA_GREEDY_RELATIVA
                    )
                )
            ),
        )

    def test_plan_greedy_tiene_reward_cero(
        self,
    ) -> None:
        env = self.crear_env_relativo()

        try:
            reward, info = ejecutar_secuencia(
                env,

                [
                    "P004",
                    "P003",
                    "P001",
                    "P002",
                ],
            )

            self.assertAlmostEqual(
                reward,
                0.0,
                places=9,
            )

            self.assertAlmostEqual(
                info[
                    "gap_relativo_greedy"
                ],

                0.0,

                places=9,
            )

            self.assertEqual(
                info["modo_reward"],

                ModoRewardRL
                .VENTAJA_GREEDY_RELATIVA
                .value,
            )

        finally:
            env.close()

    def test_plan_peor_que_greedy_es_negativo(
        self,
    ) -> None:
        env = self.crear_env_relativo()

        try:
            reward, info = ejecutar_secuencia(
                env,

                [
                    "P003",
                    "P004",
                    "P002",
                    "P001",
                ],
            )

            self.assertLess(
                reward,
                0.0,
            )

            self.assertGreater(
                info[
                    "gap_relativo_greedy"
                ],

                0.0,
            )

        finally:
            env.close()

    def test_reward_conserva_orden_de_costos(
        self,
    ) -> None:
        env_bueno = self.crear_env_relativo()

        env_malo = self.crear_env_relativo()

        try:
            (
                reward_bueno,
                _,
            ) = ejecutar_secuencia(
                env_bueno,

                [
                    "P004",
                    "P003",
                    "P001",
                    "P002",
                ],
            )

            (
                reward_malo,
                _,
            ) = ejecutar_secuencia(
                env_malo,

                [
                    "P003",
                    "P004",
                    "P002",
                    "P001",
                ],
            )

            self.assertGreater(
                reward_bueno,
                reward_malo,
            )

        finally:
            env_bueno.close()

            env_malo.close()


if __name__ == "__main__":
    unittest.main()