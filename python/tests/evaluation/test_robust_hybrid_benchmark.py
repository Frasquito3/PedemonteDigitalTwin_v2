from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planner.algorithms.greedy import generar_plan_greedy
from planner.core.schema import AlgoritmoPlanificacion, InstanciaTurno, PlanTurno
from planner.evaluation.robust_hybrid_benchmark import (
    ejecutar_benchmark_hibrido_robusto,
    escribir_resultados_benchmark_hibrido_robusto,
)
from planner.evaluation.rl_stress_instances import crear_casos_stress_rl
from planner.routing.travel import ProveedorHaversineAjustado


class PlannerRLIgualGreedy:
    def __init__(self, proveedor: ProveedorHaversineAjustado) -> None:
        self.proveedor = proveedor

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        plan = generar_plan_greedy(
            instancia,
            proveedor_viaje=self.proveedor,
        )
        plan.algoritmo = AlgoritmoPlanificacion.RL
        return plan


class RobustHybridBenchmarkTest(unittest.TestCase):
    def crear_resultado(self):
        proveedor = ProveedorHaversineAjustado()
        casos = crear_casos_stress_rl(
            cantidad_por_estrato=1,
            seed_base=19_100,
        )
        return ejecutar_benchmark_hibrido_robusto(
            casos,
            proveedor_viaje=proveedor,
            planners_rl={
                "FAKE": PlannerRLIgualGreedy(proveedor),
            },
            exigir_sin_fallback=True,
        )

    def test_genera_una_fila_por_caso_y_modelo(self) -> None:
        resultado = self.crear_resultado()
        self.assertEqual(resultado.cantidad_casos, 5)
        self.assertEqual(resultado.cantidad_modelos, 1)
        self.assertEqual(resultado.cantidad_filas, 5)
        self.assertEqual(len(resultado.corridas), 5)

    def test_cumple_garantias_frente_a_greedy_y_ga(self) -> None:
        resultado = self.crear_resultado()
        for fila in resultado.corridas:
            self.assertTrue(fila.cumple_garantia_greedy)
            self.assertIsNot(fila.cumple_garantia_ga, False)
            self.assertEqual(fila.fallbacks_matriz, 0)

        resumen = resultado.resumenes[0]
        self.assertEqual(resumen.violaciones_garantia_greedy, 0)
        self.assertEqual(resumen.violaciones_garantia_ga, 0)

    def test_escribe_csv_y_json(self) -> None:
        resultado = self.crear_resultado()
        with tempfile.TemporaryDirectory() as temporal:
            rutas = escribir_resultados_benchmark_hibrido_robusto(
                resultado,
                temporal,
            )
            for ruta in rutas.values():
                self.assertTrue(Path(ruta).is_file())

            datos = json.loads(
                Path(rutas["benchmark_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(datos["cantidad_filas"], 5)
            self.assertEqual(len(datos["corridas"]), 5)


if __name__ == "__main__":
    unittest.main()
