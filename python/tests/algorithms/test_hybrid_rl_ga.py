from __future__ import annotations

from types import SimpleNamespace
import unittest

from planner.algorithms.greedy import generar_plan_greedy
from planner.algorithms.hybrid_rl_ga import (
    FuenteResultadoHibrido,
    HybridRLGAPlanner,
    MotivoResultadoHibrido,
)
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanTurno,
    Turno,
)
from planner.domain.validator import validar_plan


class PlannerRLFalso:
    def __init__(self, delta: float = 0.0, error: Exception | None = None) -> None:
        self.delta = delta
        self.error = error
        self.ultima_decision = SimpleNamespace(fuente_seleccionada="EXTENSION")

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        if self.error is not None:
            raise self.error
        plan = generar_plan_greedy(instancia)
        plan.algoritmo = AlgoritmoPlanificacion.RL
        plan.costo_estimado += self.delta
        return plan


class HybridRLGAPlannerTest(unittest.TestCase):
    def crear_instancia(self) -> InstanciaTurno:
        return InstanciaTurno(
            instancia_id="HIBRIDO-RL-GA-TEST",
            fecha_operacion="2026-08-27",
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

    def plan_ga_con_delta(
        self,
        instancia: InstanciaTurno,
        delta: float,
    ) -> PlanTurno:
        plan = generar_plan_greedy(instancia)
        plan.algoritmo = AlgoritmoPlanificacion.GA
        plan.costo_estimado += delta
        return plan

    def exigir_valido(self, instancia: InstanciaTurno, plan: PlanTurno) -> None:
        validacion = validar_plan(instancia, plan)
        self.assertTrue(validacion.valido, msg=" | ".join(validacion.errores))

    def test_refina_con_ga_si_mejora_la_semilla_rl(self) -> None:
        instancia = self.crear_instancia()
        cromosomas_recibidos = []

        def generar_ga(actual: InstanciaTurno, cromosoma):
            cromosomas_recibidos.append(cromosoma)
            return self.plan_ga_con_delta(actual, delta=-2.0)

        planner = HybridRLGAPlanner(
            planner_rl=PlannerRLFalso(delta=0.0),
            generador_ga_refinado=generar_ga,
        )
        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(
            decision.resultado,
            FuenteResultadoHibrido.REFINADO_GA,
        )
        self.assertEqual(
            decision.motivo,
            MotivoResultadoHibrido.GA_MEJORA_SEMILLA_RL,
        )
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.GA)
        self.assertGreater(decision.mejora_absoluta, 0.0)
        self.assertEqual(decision.fuente_rl, "EXTENSION")
        self.assertEqual(len(cromosomas_recibidos), 1)
        self.assertEqual(set(cromosomas_recibidos[0]), {"P001", "P002"})
        self.exigir_valido(instancia, plan)

    def test_conserva_rl_si_ga_no_mejora(self) -> None:
        instancia = self.crear_instancia()
        planner = HybridRLGAPlanner(
            planner_rl=PlannerRLFalso(delta=0.0),
            generador_ga_refinado=lambda actual, _semilla: (
                self.plan_ga_con_delta(actual, delta=1.0)
            ),
        )
        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(
            decision.resultado,
            FuenteResultadoHibrido.SEMILLA_RL,
        )
        self.assertEqual(
            decision.motivo,
            MotivoResultadoHibrido.SEMILLA_RL_CONSERVADA,
        )
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.RL)
        self.assertEqual(decision.mejora_absoluta, 0.0)

    def test_fallo_ga_conserva_semilla_rl_y_lo_audita(self) -> None:
        instancia = self.crear_instancia()

        def ga_con_error(_actual, _semilla):
            raise RuntimeError("fallo controlado de GA")

        planner = HybridRLGAPlanner(
            planner_rl=PlannerRLFalso(delta=0.0),
            generador_ga_refinado=ga_con_error,
        )
        plan = planner.generar_plan(instancia)
        decision = planner.ultima_decision

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(plan.algoritmo, AlgoritmoPlanificacion.RL)
        self.assertEqual(
            decision.motivo,
            MotivoResultadoHibrido.GA_NO_EJECUTABLE,
        )
        self.assertTrue(decision.error_ga)

    def test_fallo_rl_impide_iniciar_el_hibrido(self) -> None:
        planner = HybridRLGAPlanner(
            planner_rl=PlannerRLFalso(error=RuntimeError("fallo RL")),
        )
        with self.assertRaisesRegex(RuntimeError, "semilla RL"):
            planner.generar_plan(self.crear_instancia())


if __name__ == "__main__":
    unittest.main()
