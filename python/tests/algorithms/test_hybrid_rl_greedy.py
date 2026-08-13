import unittest

from planner.algorithms.greedy import (
    generar_plan_greedy,
)

from planner.algorithms.hybrid_rl_greedy import (
    FuentePlanHibrido,
    HybridRLGreedyPlanner,
    MotivoSeleccionHibrida,
)

from planner.algorithms.random_feasible import (
    generar_plan_random,
)

from planner.domain.validator import (
    validar_plan,
)

from planner.rl.instance_generator import (
    GeneradorInstanciasRL,
)

from tests.fixtures import (
    crear_instancia_demo,
)


class PlannerCostoAjustado:
    def __init__(
        self,
        delta: float,
    ) -> None:
        self.delta = delta

    def generar_plan(
        self,
        instancia,
    ):
        plan = generar_plan_greedy(
            instancia
        )

        plan.costo_estimado += (
            self.delta
        )

        return plan


class PlannerConExcepcion:
    def generar_plan(
        self,
        instancia,
    ):
        _ = instancia

        raise RuntimeError(
            "Fallo RL controlado."
        )


class PlannerRandomValido:
    def generar_plan(
        self,
        instancia,
    ):
        return generar_plan_random(
            instancia,

            seed=(
                instancia
                .seed_escenario
                + 7_001
            ),
        )


class HybridRLGreedyPlannerTest(
    unittest.TestCase
):
    def test_elige_rl_si_tiene_menor_costo(
        self,
    ) -> None:
        instancia = (
            crear_instancia_demo()
        )

        planner = (
            HybridRLGreedyPlanner(
                planner_rl=(
                    PlannerCostoAjustado(
                        delta=-10.0
                    )
                )
            )
        )

        plan = planner.generar_plan(
            instancia
        )

        decision = (
            planner.ultima_decision
        )

        self.assertIsNotNone(
            decision
        )

        assert decision is not None

        self.assertEqual(
            decision.fuente_seleccionada,

            FuentePlanHibrido.RL,
        )

        self.assertEqual(
            decision.motivo,

            MotivoSeleccionHibrida
            .RL_MENOR_COSTO,
        )

        self.assertLess(
            plan.costo_estimado,

            decision.costo_greedy,
        )

    def test_elige_greedy_si_rl_es_peor(
        self,
    ) -> None:
        instancia = (
            crear_instancia_demo()
        )

        planner = (
            HybridRLGreedyPlanner(
                planner_rl=(
                    PlannerCostoAjustado(
                        delta=10.0
                    )
                )
            )
        )

        plan = planner.generar_plan(
            instancia
        )

        decision = (
            planner.ultima_decision
        )

        self.assertIsNotNone(
            decision
        )

        assert decision is not None

        self.assertEqual(
            decision.fuente_seleccionada,

            FuentePlanHibrido
            .GREEDY,
        )

        self.assertEqual(
            decision.motivo,

            MotivoSeleccionHibrida
            .GREEDY_MENOR_O_IGUAL,
        )

        self.assertAlmostEqual(
            plan.costo_estimado,

            decision.costo_greedy,
            places=9,
        )

    def test_fallback_si_rl_lanza_excepcion(
        self,
    ) -> None:
        instancia = (
            crear_instancia_demo()
        )

        planner = (
            HybridRLGreedyPlanner(
                planner_rl=(
                    PlannerConExcepcion()
                )
            )
        )

        plan = planner.generar_plan(
            instancia
        )

        decision = (
            planner.ultima_decision
        )

        self.assertIsNotNone(
            decision
        )

        assert decision is not None

        self.assertEqual(
            decision.fuente_seleccionada,

            FuentePlanHibrido
            .GREEDY,
        )

        self.assertEqual(
            decision.motivo,

            MotivoSeleccionHibrida
            .RL_EXCEPCION,
        )

        self.assertTrue(
            decision.errores_rl
        )

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

    def test_plan_seleccionado_es_valido(
        self,
    ) -> None:
        instancia = (
            crear_instancia_demo()
        )

        planner = (
            HybridRLGreedyPlanner(
                planner_rl=(
                    PlannerRandomValido()
                )
            )
        )

        plan = planner.generar_plan(
            instancia
        )

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

    def test_nunca_empeora_greedy_en_50_instancias(
        self,
    ) -> None:
        generador = (
            GeneradorInstanciasRL()
        )

        planner = (
            HybridRLGreedyPlanner(
                planner_rl=(
                    PlannerRandomValido()
                )
            )
        )

        for seed in range(
            220_000,
            220_050,
        ):
            instancia = (
                generador.generar(
                    seed
                )
            )

            plan = planner.generar_plan(
                instancia
            )

            decision = (
                planner.ultima_decision
            )

            self.assertIsNotNone(
                decision
            )

            assert decision is not None

            self.assertLessEqual(
                plan.costo_estimado,

                decision.costo_greedy
                + 1e-9,

                msg=(
                    f"El híbrido empeoró Greedy "
                    f"en seed={seed}."
                ),
            )

            validacion = validar_plan(
                instancia,
                plan,
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


if __name__ == "__main__":
    unittest.main()