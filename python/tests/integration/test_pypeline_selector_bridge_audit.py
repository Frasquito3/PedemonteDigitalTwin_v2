from __future__ import annotations

import unittest
from typing import cast

from planner.algorithms.greedy import generar_plan_greedy
from planner.core.config import ConfiguracionPlanificacion
from planner.integration import (
    pypeline_selector_bridge as bridge,
)
from planner.integration.planner_selector import (
    DecisionSelector,
    ModoPlanificacion,
    SelectorPlanificadores,
)
from planner.routing.travel import (
    ProveedorHaversineAjustado,
)


class SelectorAuditoriaFalso:
    def __init__(self) -> None:
        self.configuracion = ConfiguracionPlanificacion()
        self.proveedor_viaje = ProveedorHaversineAjustado()
        self.ultima_decision = None

    def generar_plan(
        self,
        instancia,
        modo_planificacion,
    ):
        plan = generar_plan_greedy(
            instancia,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
        )

        self.ultima_decision = DecisionSelector(
            instancia_id=instancia.instancia_id,
            modo_solicitado=ModoPlanificacion.GREEDY,
            algoritmo_resultante=plan.algoritmo,
            costo_estimado=plan.costo_estimado,
            tiempo_plan_ms=plan.tiempo_computo_ms,
            tiempo_selector_ms=plan.tiempo_computo_ms,
            detalle="",
        )

        return plan


class PypelineSelectorBridgeAuditTest(
    unittest.TestCase
):
    def setUp(self) -> None:
        bridge.reiniciar()

        selector = SelectorAuditoriaFalso()

        bridge._selector = cast(
            SelectorPlanificadores,
            selector,
        )
        bridge._proveedor_viaje = (
            selector.proveedor_viaje
        )
        bridge._resumen_proveedor = (
            "proveedor=HAVERSINE_AJUSTADA|"
            "version_viaje=haversine-ajustada-v1|"
            "cache_tramos=NA|fallback=NA"
        )

    def tearDown(self) -> None:
        bridge.reiniciar()

    def test_decision_incluye_auditoria_estimacion(
        self,
    ) -> None:
        vector = [
            1.0,
            0.0,
            1.0,
            8.0,
            2.0,
            -32.8495006,
            -60.722653,
            0.0,

            0.0,
            0.0,
            1.0,
            1.0,
            2.0,
            0.0,
            -32.831,
            -60.719,
            -1.0,
            -1.0,
        ]

        resultado = bridge.planificar_vector(
            vector,
            seed_escenario=6001,
            seed_ejecucion=1006001,
            modo_planificacion="GREEDY",
        )

        self.assertGreater(
            len(resultado),
            0,
        )

        auditoria = (
            bridge
            .obtener_ultima_auditoria_estimacion()
        )

        decision = bridge.obtener_ultima_decision()

        self.assertIn(
            "version=estimacion-costo-v2",
            auditoria,
        )
        self.assertIn(
            "espera_respuesta_cliente_min=3.333333",
            auditoria,
        )
        self.assertIn(
            "auditoria_estimacion=version=estimacion-costo-v2",
            decision,
        )
        self.assertIn(
            "costo_total=",
            decision,
        )


if __name__ == "__main__":
    unittest.main()
