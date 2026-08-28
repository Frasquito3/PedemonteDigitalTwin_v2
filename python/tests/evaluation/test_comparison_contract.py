from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planner.algorithms.greedy import GreedyFeasiblePlanner
from planner.core.schema import AlgoritmoPlanificacion
from planner.evaluation.comparison_contract import (
    MODOS_COMPARACION,
    VERSION_CONTRATO_COMPARACION,
    escribir_contrato_comparacion,
    preparar_contrato_comparacion,
)
from planner.integration.planner_selector import SelectorPlanificadores
from planner.routing.objective import VERSION_AUDITORIA_COSTO
from planner.routing.travel import ProveedorHaversineAjustado
from tests.fixtures import crear_instancia_demo


class _PlannerGreedyComoRL:
    def __init__(self, proveedor) -> None:
        self._planner = GreedyFeasiblePlanner(
            proveedor_viaje=proveedor,
        )

    def generar_plan(self, instancia):
        plan = self._planner.generar_plan(instancia)
        plan.algoritmo = AlgoritmoPlanificacion.RL
        return plan


class _PlannerRLConError:
    def generar_plan(self, instancia):
        raise RuntimeError("fallo RL controlado")


class TestContratoComparacion(unittest.TestCase):
    def setUp(self) -> None:
        self.instancia = crear_instancia_demo()
        self.proveedor = ProveedorHaversineAjustado()

    def _selector(self, planner_rl) -> SelectorPlanificadores:
        return SelectorPlanificadores(
            planner_rl=planner_rl,
            proveedor_viaje=self.proveedor,
        )

    def test_genera_cinco_planes_en_orden_rl_primero(self) -> None:
        contrato = preparar_contrato_comparacion(
            self.instancia,
            selector=self._selector(
                _PlannerGreedyComoRL(self.proveedor)
            ),
            proveedor_viaje=self.proveedor,
        )

        self.assertEqual(
            contrato.version_contrato,
            VERSION_CONTRATO_COMPARACION,
        )
        self.assertEqual(
            contrato.version_objetivo,
            VERSION_AUDITORIA_COSTO,
        )
        self.assertEqual(
            contrato.orden_modos,
            tuple(modo.value for modo in MODOS_COMPARACION),
        )
        self.assertEqual(contrato.planes_ok, 5)
        self.assertEqual(contrato.planes_error, 0)

        for registro in contrato.planes:
            self.assertEqual(registro.estado, "OK")
            self.assertTrue(registro.plan_valido)
            self.assertEqual(
                registro.seed_escenario,
                self.instancia.seed_escenario,
            )
            self.assertEqual(
                registro.seed_ejecucion,
                self.instancia.seed_ejecucion,
            )
            self.assertTrue(registro.firma_ruta)
            self.assertTrue(registro.camiones)
            self.assertTrue(registro.plan_vector)
            self.assertIsNotNone(registro.costo_estimado)
            self.assertIsNotNone(registro.tiempo_plan_ms)

    def test_registra_semillas_derivadas_y_fuente_hibrida(self) -> None:
        contrato = preparar_contrato_comparacion(
            self.instancia,
            selector=self._selector(
                _PlannerGreedyComoRL(self.proveedor)
            ),
            proveedor_viaje=self.proveedor,
        )
        por_modo = {
            registro.modo_solicitado: registro
            for registro in contrato.planes
        }

        self.assertEqual(
            por_modo["RL"].seed_planificacion,
            self.instancia.seed_escenario,
        )
        self.assertEqual(
            por_modo["RANDOM"].seed_planificacion,
            self.instancia.seed_escenario + 7001,
        )
        self.assertEqual(
            por_modo["GA"].seed_planificacion,
            self.instancia.seed_escenario + 8001,
        )
        self.assertIsNone(
            por_modo["GREEDY"].seed_planificacion
        )
        self.assertEqual(
            [
                (semilla.componente, semilla.valor)
                for semilla in por_modo["HIBRIDO"].semillas_componentes
            ],
            [
                ("RL", self.instancia.seed_escenario),
                ("GA", self.instancia.seed_escenario + 8001),
            ],
        )
        hibrido = por_modo["HIBRIDO"]
        self.assertNotEqual(
            hibrido.fuente_seleccionada,
            "GREEDY",
        )
        self.assertIn(
            hibrido.fuente_seleccionada,
            {"RL", "GA"},
        )
        self.assertIn(
            hibrido.motivo_seleccion,
            {
                "GA_MEJORA_SEMILLA_RL",
                "SEMILLA_RL_CONSERVADA",
                "GA_NO_EJECUTABLE",
            },
        )
        self.assertIn(
            "arquitectura=RL_GA_SEEDED",
            hibrido.detalle_decision,
        )
        self.assertIn(
            "fuente_rl=RL",
            hibrido.detalle_decision,
        )

    def test_error_rl_impide_hibrido_pero_no_los_clasicos(self) -> None:
        contrato = preparar_contrato_comparacion(
            self.instancia,
            selector=self._selector(_PlannerRLConError()),
            proveedor_viaje=self.proveedor,
        )
        por_modo = {
            registro.modo_solicitado: registro
            for registro in contrato.planes
        }

        self.assertEqual(por_modo["RL"].estado, "ERROR")
        self.assertIn("fallo RL controlado", por_modo["RL"].error)

        # El híbrido vigente necesita una semilla RL ejecutable. Si RL falla,
        # el híbrido también debe fallar de forma explícita y nunca recurrir a
        # Greedy como sustituto silencioso.
        self.assertEqual(por_modo["HIBRIDO"].estado, "ERROR")
        self.assertIn(
            "No se pudo obtener la semilla RL del híbrido",
            por_modo["HIBRIDO"].error,
        )
        self.assertIn(
            "fallo RL controlado",
            por_modo["HIBRIDO"].error,
        )

        for modo_clasico in ("GA", "GREEDY", "RANDOM"):
            self.assertEqual(por_modo[modo_clasico].estado, "OK")

        self.assertEqual(contrato.planes_ok, 3)
        self.assertEqual(contrato.planes_error, 2)

    def test_escribe_json_y_csv(self) -> None:
        contrato = preparar_contrato_comparacion(
            self.instancia,
            selector=self._selector(
                _PlannerGreedyComoRL(self.proveedor)
            ),
            proveedor_viaje=self.proveedor,
        )

        with tempfile.TemporaryDirectory() as temporal:
            rutas = escribir_contrato_comparacion(
                contrato,
                temporal,
            )
            self.assertEqual(
                set(rutas),
                {"contrato_json", "planes_csv"},
            )
            for ruta in rutas.values():
                self.assertTrue(Path(ruta).is_file())
                self.assertGreater(Path(ruta).stat().st_size, 0)

            with rutas["contrato_json"].open(
                "r",
                encoding="utf-8",
            ) as archivo:
                datos = json.load(archivo)

            self.assertEqual(
                datos["orden_modos"][0],
                "RL",
            )
            self.assertEqual(len(datos["planes"]), 5)
            self.assertEqual(
                datos["planes"][0]["modo_solicitado"],
                "RL",
            )


if __name__ == "__main__":
    unittest.main()
