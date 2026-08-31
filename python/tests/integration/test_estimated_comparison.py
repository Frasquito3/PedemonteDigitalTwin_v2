from __future__ import annotations

import unittest

from planner.algorithms.greedy import generar_plan_greedy
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PlanTurno,
)
from planner.integration.estimated_comparison import (
    CAMPOS_POR_METODO,
    MODOS_COMPARACION,
    TAMANO_CABECERA_COMPARACION,
    codificar_comparacion_estimada,
    ejecutar_comparacion_estimada,
    firmar_instancia_vector,
)
from planner.integration.planner_selector import (
    DecisionSelector,
    ModoPlanificacion,
    normalizar_modo,
)
from planner.routing.travel import ProveedorHaversineAjustado
from tests.fixtures import crear_instancia_demo


class SelectorComparacionFalso:
    def __init__(self) -> None:
        self.configuracion = ConfiguracionPlanificacion()
        self.proveedor_viaje = ProveedorHaversineAjustado()
        self.ultima_decision: DecisionSelector | None = None
        self.modo_con_error: ModoPlanificacion | None = None

    def generar_plan(
        self,
        instancia: InstanciaTurno,
        modo: ModoPlanificacion | str,
    ) -> PlanTurno:
        modo_normalizado = normalizar_modo(modo)

        if modo_normalizado == self.modo_con_error:
            raise RuntimeError("fallo controlado")

        plan = generar_plan_greedy(
            instancia,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
        )

        algoritmo_por_modo = {
            ModoPlanificacion.RL: AlgoritmoPlanificacion.RL,
            ModoPlanificacion.HIBRIDO: AlgoritmoPlanificacion.GA,
            ModoPlanificacion.GREEDY: AlgoritmoPlanificacion.GREEDY,
            ModoPlanificacion.RANDOM: AlgoritmoPlanificacion.RANDOM,
            ModoPlanificacion.GA: AlgoritmoPlanificacion.GA,
        }

        plan.algoritmo = algoritmo_por_modo[modo_normalizado]

        self.ultima_decision = DecisionSelector(
            instancia_id=instancia.instancia_id,
            modo_solicitado=modo_normalizado,
            algoritmo_resultante=plan.algoritmo,
            costo_estimado=plan.costo_estimado,
            tiempo_plan_ms=plan.tiempo_computo_ms,
            tiempo_selector_ms=plan.tiempo_computo_ms + 1.0,
            detalle=f"modo_prueba={modo_normalizado.value}",
        )

        return plan


class EstimatedComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.instancia = crear_instancia_demo()
        self.selector = SelectorComparacionFalso()
        self.decision_previa = DecisionSelector(
            instancia_id="ANTERIOR",
            modo_solicitado=ModoPlanificacion.RL,
            algoritmo_resultante=AlgoritmoPlanificacion.RL,
            costo_estimado=1.0,
            tiempo_plan_ms=1.0,
            tiempo_selector_ms=1.0,
            detalle="anterior",
        )
        self.selector.ultima_decision = self.decision_previa

    def test_ejecuta_los_cinco_metodos_y_restaura_decision(self) -> None:
        comparacion = ejecutar_comparacion_estimada(
            instancia=self.instancia,
            selector=self.selector,  # type: ignore[arg-type]
            proveedor_viaje=self.selector.proveedor_viaje,
            firma_instancia="firma-prueba",
        )

        self.assertEqual(
            tuple(
                resultado.modo_solicitado
                for resultado in comparacion.resultados
            ),
            MODOS_COMPARACION,
        )
        self.assertTrue(
            all(
                resultado.factible
                for resultado in comparacion.resultados
            )
        )
        self.assertTrue(
            all(
                resultado.plan_vector
                for resultado in comparacion.resultados
            )
        )
        self.assertIs(
            self.selector.ultima_decision,
            self.decision_previa,
        )

    def test_un_error_individual_no_cancela_los_demas(self) -> None:
        self.selector.modo_con_error = ModoPlanificacion.RANDOM

        comparacion = ejecutar_comparacion_estimada(
            instancia=self.instancia,
            selector=self.selector,  # type: ignore[arg-type]
            proveedor_viaje=self.selector.proveedor_viaje,
            firma_instancia="firma-prueba",
        )

        random_resultado = comparacion.obtener_resultado("RANDOM")
        self.assertFalse(random_resultado.factible)
        self.assertIn("fallo controlado", random_resultado.error)
        self.assertFalse(random_resultado.plan_vector)

        self.assertEqual(
            sum(
                1
                for resultado in comparacion.resultados
                if resultado.factible
            ),
            4,
        )

    def test_codificacion_tiene_longitud_estable(self) -> None:
        comparacion = ejecutar_comparacion_estimada(
            instancia=self.instancia,
            selector=self.selector,  # type: ignore[arg-type]
            proveedor_viaje=self.selector.proveedor_viaje,
            firma_instancia="firma-prueba",
        )

        vector = codificar_comparacion_estimada(comparacion)

        self.assertEqual(vector[0], 1.0)
        self.assertEqual(vector[1], 5.0)
        self.assertEqual(vector[2], float(CAMPOS_POR_METODO))
        self.assertEqual(
            len(vector),
            TAMANO_CABECERA_COMPARACION
            + len(MODOS_COMPARACION) * CAMPOS_POR_METODO,
        )

    def test_firma_depende_de_vector_y_semillas(self) -> None:
        vector = [1.0, 2.0, 3.0]

        firma_a = firmar_instancia_vector(vector, 10, 20)
        firma_b = firmar_instancia_vector(vector, 10, 20)
        firma_c = firmar_instancia_vector(vector, 10, 21)

        self.assertEqual(firma_a, firma_b)
        self.assertNotEqual(firma_a, firma_c)


if __name__ == "__main__":
    unittest.main()
