from __future__ import annotations

import unittest

from planner.algorithms.greedy import (
    generar_plan_greedy,
)
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanTurno,
    Turno,
)
from planner.domain.validator import (
    validar_plan,
)
from planner.integration.planner_selector import (
    ModoPlanificacion,
    SelectorPlanificadores,
    normalizar_modo,
)


class PlannerRLFalso:
    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        plan = generar_plan_greedy(
            instancia
        )

        plan.algoritmo = (
            AlgoritmoPlanificacion.RL
        )

        return plan


class PlannerSelectorTest(
    unittest.TestCase
):
    def crear_instancia(
        self,
    ) -> InstanciaTurno:
        return InstanciaTurno(
            instancia_id=(
                "SELECTOR-TEST"
            ),
            fecha_operacion=(
                "2026-08-19"
            ),
            turno=Turno.MANANA,
            pedidos=[
                PedidoInput(
                    pedido_id="P001",
                    pedido_original_id=(
                        "P001"
                    ),
                    numero_parte=1,
                    total_partes=1,
                    turno=(
                        Turno.MANANA
                    ),
                    latitud=-32.8488,
                    longitud=-60.7221,
                    unidades_capacidad=5,
                    requiere_volcador=False,
                    tiene_ventana_especifica=(
                        False
                    ),
                    hora_desde_min=450,
                    hora_hasta_min=720,
                ),
                PedidoInput(
                    pedido_id="P002",
                    pedido_original_id=(
                        "P002"
                    ),
                    numero_parte=1,
                    total_partes=1,
                    turno=(
                        Turno.MANANA
                    ),
                    latitud=-32.8502,
                    longitud=-60.7232,
                    unidades_capacidad=3,
                    requiere_volcador=False,
                    tiene_ventana_especifica=(
                        False
                    ),
                    hora_desde_min=450,
                    hora_hasta_min=720,
                ),
            ],
            lat_corralon=-32.8495006,
            lon_corralon=-60.722653,
            capacidad_camion=8,
            cantidad_camiones=2,
            hora_inicio_turno_min=450,
            hora_fin_objetivo_min=720,
            hora_fin_tolerancia_min=735,
            seed_escenario=4003,
            seed_ejecucion=1_004_003,
        )

    def crear_selector(
        self,
    ) -> SelectorPlanificadores:
        return SelectorPlanificadores(
            planner_rl=(
                PlannerRLFalso()
            ),
        )

    def exigir_valido(
        self,
        plan: PlanTurno,
    ) -> None:
        validacion = validar_plan(
            self.crear_instancia(),
            plan,
        )

        self.assertTrue(
            validacion.valido,
            validacion.errores,
        )

    def test_normaliza_alias_hybrid(
        self,
    ) -> None:
        self.assertEqual(
            normalizar_modo(
                "hybrid"
            ),
            ModoPlanificacion.HIBRIDO,
        )

    def test_greedy(
        self,
    ) -> None:
        plan = (
            self.crear_selector()
            .generar_plan(
                self.crear_instancia(),
                "GREEDY",
            )
        )

        self.assertEqual(
            plan.algoritmo,
            AlgoritmoPlanificacion.GREEDY,
        )

        self.exigir_valido(
            plan
        )

    def test_random(
        self,
    ) -> None:
        plan = (
            self.crear_selector()
            .generar_plan(
                self.crear_instancia(),
                "RANDOM",
            )
        )

        self.assertEqual(
            plan.algoritmo,
            AlgoritmoPlanificacion.RANDOM,
        )

        self.exigir_valido(
            plan
        )

    def test_ga(
        self,
    ) -> None:
        plan = (
            self.crear_selector()
            .generar_plan(
                self.crear_instancia(),
                "GA",
            )
        )

        self.assertEqual(
            plan.algoritmo,
            AlgoritmoPlanificacion.GA,
        )

        self.exigir_valido(
            plan
        )

    def test_rl_inyectado(
        self,
    ) -> None:
        plan = (
            self.crear_selector()
            .generar_plan(
                self.crear_instancia(),
                "RL",
            )
        )

        self.assertEqual(
            plan.algoritmo,
            AlgoritmoPlanificacion.RL,
        )

        self.exigir_valido(
            plan
        )

    def test_hibrido_registra_decision(
        self,
    ) -> None:
        selector = (
            self.crear_selector()
        )

        plan = selector.generar_plan(
            self.crear_instancia(),
            "HIBRIDO",
        )

        self.exigir_valido(
            plan
        )

        decision = (
            selector.ultima_decision
        )

        self.assertIsNotNone(
            decision
        )

        assert decision is not None

        self.assertEqual(
            decision.modo_solicitado,
            ModoPlanificacion.HIBRIDO,
        )

        self.assertIn(
            "fuente=",
            decision.detalle,
        )

        self.assertIn(
            "costo_ga=",
            decision.detalle,
        )

        self.assertIn(
            "costo_greedy=",
            decision.detalle,
        )

        self.assertIn(
            "costo_rl=",
            decision.detalle,
        )

    def test_rechaza_modo_desconocido(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            (
                self.crear_selector()
                .generar_plan(
                    self.crear_instancia(),
                    "NO_EXISTE",
                )
            )


if __name__ == "__main__":
    unittest.main()