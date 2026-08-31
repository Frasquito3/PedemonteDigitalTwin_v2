from __future__ import annotations

import unittest

from planner.integration import selector_bridge as bridge
from planner.integration.alpyne_codec import codificar_instancia_alpyne
from planner.integration.planner_selector import ModoPlanificacion
from planner.routing.travel import ProveedorHaversineAjustado
from tests.fixtures import crear_instancia_demo
from tests.integration.test_estimated_comparison import (
    SelectorComparacionFalso,
)


class SelectorBridgeComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        bridge.reiniciar()

        self.instancia = crear_instancia_demo()
        self.vector = codificar_instancia_alpyne(
            self.instancia
        )

        self.selector = SelectorComparacionFalso()
        self.proveedor = ProveedorHaversineAjustado()

        bridge._selector = self.selector  # type: ignore[assignment]
        bridge._proveedor_viaje = self.proveedor

    def tearDown(self) -> None:
        bridge.reiniciar()

    def test_compara_y_conserva_los_cinco_planes(self) -> None:
        resultado = bridge.comparar_estimado_vector(
            self.vector,
            self.instancia.seed_escenario,
            self.instancia.seed_ejecucion,
        )

        self.assertEqual(resultado[1], 5.0)
        self.assertIn(
            "factibles=5",
            bridge.obtener_resumen_comparacion_estimada(),
        )

        for modo in ModoPlanificacion:
            plan = bridge.obtener_plan_comparacion_vector(
                modo.value
            )
            self.assertTrue(plan)

    def test_planificacion_individual_invalida_comparacion_anterior(self) -> None:
        bridge.comparar_estimado_vector(
            self.vector,
            self.instancia.seed_escenario,
            self.instancia.seed_ejecucion,
        )

        bridge.planificar_vector(
            self.vector,
            self.instancia.seed_escenario,
            self.instancia.seed_ejecucion,
            "GREEDY",
        )

        self.assertEqual(
            bridge.obtener_resumen_comparacion_estimada(),
            "SIN_COMPARACION",
        )

    def test_rechaza_recuperar_plan_con_error(self) -> None:
        self.selector.modo_con_error = ModoPlanificacion.GA

        bridge.comparar_estimado_vector(
            self.vector,
            self.instancia.seed_escenario,
            self.instancia.seed_ejecucion,
        )

        detalle = bridge.obtener_resultado_comparacion_estimado(
            "GA"
        )
        self.assertIn("factible=NO", detalle)

        with self.assertRaises(RuntimeError):
            bridge.obtener_plan_comparacion_vector("GA")


if __name__ == "__main__":
    unittest.main()
