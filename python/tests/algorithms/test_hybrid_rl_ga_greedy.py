from __future__ import annotations

import unittest

from planner.algorithms.greedy import generar_plan_greedy
from planner.algorithms.hybrid_rl_ga_greedy import (
    FuentePlanHibridoRobusto,
    HybridRLGAGreedyPlanner,
    MotivoSeleccionHibridaRobusta,
)
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanTurno,
    Turno,
)
from planner.domain.validator import validar_plan


class PlannerCostoAjustado:
    def __init__(self, delta: float) -> None:
        self.delta = delta

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        plan = generar_plan_greedy(instancia)
        plan.algoritmo = AlgoritmoPlanificacion.RL
        plan.costo_estimado += self.delta
        return plan


class PlannerConExcepcion:
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        raise RuntimeError("fallo controlado de RL")


class HybridRLGAGreedyPlannerTest(unittest.TestCase):
    def crear_instancia(self) -> InstanciaTurno:
        return InstanciaTurno(
            instancia_id="HIBRIDO-ROBUSTO-TEST",
            fecha_operacion="2026-08-24",
            turno=Turno.MANANA,
            pedidos=[
                PedidoInput(
                    pedido_id="P001",
                    pedido_original_id="P001",
                    numero_parte=1,
                    total_partes=1,
                    turno=Turno.MANANA,
                    latitud=-32.831000,
                    longitud=-60.719000,
                    unidades_capacidad=4,
                    requiere_volcador=False,
                    tiene_ventana_especifica=False,
                    hora_desde_min=450,
                    hora_hasta_min=720,
                ),
                PedidoInput(
                    pedido_id="P002",
                    pedido_original_id="P002",
                    numero_parte=1,
                    total_partes=1,
                    turno=Turno.MANANA,
                    latitud=-32.8595006,
                    longitud=-60.702653,
                    unidades_capacidad=4,
                    requiere_volcador=False,
                    tiene_ventana_especifica=False,
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
            seed_escenario=15105,
            seed_ejecucion=1_015_105,
        )

    def generar_ga_con_delta(
        self,
        instancia: InstanciaTurno,
        delta: float,
    ) -> PlanTurno:
        plan = generar_plan_greedy(instancia)
        plan.algoritmo = AlgoritmoPlanificacion.GA
        plan.costo_estimado += delta
        return plan

    def exigir_valido(
        self,
        instancia: InstanciaTurno,
        plan: PlanTurno,
    ) -> None:
        validacion = validar_plan(instancia, plan)
        self.assertTrue(
            validacion.valido,
            msg=" | ".join(validacion.errores),
        )

    def test_selecciona_ga_si_es_el_menor(self) -> None:
        instancia = self.crear_instancia()
        planner = HybridRLGAGreedyPlanner(
            planner_rl=PlannerCostoAjustado(delta=5.0),
            generador_ga=lambda actual: self.generar_ga_con_delta(
                actual,
                delta=-2.0,
            ),
        )

        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(
            decision.fuente_seleccionada,
            FuentePlanHibridoRobusto.GA,
        )
        self.assertEqual(
            decision.motivo,
            MotivoSeleccionHibridaRobusta.GA_MENOR_COSTO,
        )
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.GA)
        self.assertTrue(decision.cumple_garantia_greedy)
        self.assertTrue(decision.cumple_garantia_ga)
        self.exigir_valido(instancia, plan)

    def test_selecciona_rl_solo_si_supera_ga_y_greedy(self) -> None:
        instancia = self.crear_instancia()
        planner = HybridRLGAGreedyPlanner(
            planner_rl=PlannerCostoAjustado(delta=-3.0),
            generador_ga=lambda actual: self.generar_ga_con_delta(
                actual,
                delta=-2.0,
            ),
        )

        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(
            decision.fuente_seleccionada,
            FuentePlanHibridoRobusto.RL,
        )
        self.assertEqual(
            decision.motivo,
            MotivoSeleccionHibridaRobusta.RL_MENOR_COSTO,
        )
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.RL)
        self.assertTrue(decision.cumple_garantia_greedy)
        self.assertTrue(decision.cumple_garantia_ga)
        self.exigir_valido(instancia, plan)

    def test_prefiere_greedy_en_empate_total(self) -> None:
        instancia = self.crear_instancia()
        planner = HybridRLGAGreedyPlanner(
            planner_rl=PlannerCostoAjustado(delta=0.0),
            generador_ga=lambda actual: self.generar_ga_con_delta(
                actual,
                delta=0.0,
            ),
        )

        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(
            decision.fuente_seleccionada,
            FuentePlanHibridoRobusto.GREEDY,
        )
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.GREEDY)

    def test_fallback_de_rl_conserva_ga(self) -> None:
        instancia = self.crear_instancia()
        planner = HybridRLGAGreedyPlanner(
            planner_rl=PlannerConExcepcion(),
            generador_ga=lambda actual: self.generar_ga_con_delta(
                actual,
                delta=-2.0,
            ),
        )

        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(
            decision.fuente_seleccionada,
            FuentePlanHibridoRobusto.GA,
        )
        self.assertTrue(decision.errores_rl)
        self.assertFalse(decision.errores_ga)
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.GA)

    def test_fallback_doble_conserva_greedy(self) -> None:
        instancia = self.crear_instancia()

        def ga_con_excepcion(actual: InstanciaTurno) -> PlanTurno:
            raise RuntimeError("fallo controlado de GA")

        planner = HybridRLGAGreedyPlanner(
            planner_rl=PlannerConExcepcion(),
            generador_ga=ga_con_excepcion,
        )

        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(
            decision.fuente_seleccionada,
            FuentePlanHibridoRobusto.GREEDY,
        )
        self.assertTrue(decision.errores_ga)
        self.assertTrue(decision.errores_rl)
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.GREEDY)
        self.assertTrue(decision.cumple_garantia_greedy)
        self.assertIsNone(decision.cumple_garantia_ga)


if __name__ == "__main__":
    unittest.main()
