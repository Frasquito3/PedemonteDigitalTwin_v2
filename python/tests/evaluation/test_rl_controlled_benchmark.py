from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from planner.algorithms.ga import ConfiguracionGA
from planner.algorithms.greedy import GreedyFeasiblePlanner
from planner.evaluation.classic_instances import (
    crear_casos_benchmark_clasico,
)
from planner.evaluation.rl_controlled_benchmark import (
    ConfiguracionBenchmarkRLControlado,
    MetadatosModeloRL,
    ejecutar_benchmark_rl_controlado,
    escribir_resultados_benchmark_rl_controlado,
)
from planner.routing.travel import ProveedorHaversineAjustado


class _PlannerGreedyComoRL:
    def __init__(self, proveedor) -> None:
        self._planner = GreedyFeasiblePlanner(
            proveedor_viaje=proveedor,
        )

    def generar_plan(self, instancia):
        return self._planner.generar_plan(instancia)


class _PlannerConError:
    def generar_plan(self, instancia):
        raise RuntimeError("fallo RL controlado")


class TestBenchmarkRLControlado(unittest.TestCase):
    def setUp(self) -> None:
        self.caso = (crear_casos_benchmark_clasico()[0],)
        self.proveedor = ProveedorHaversineAjustado()
        self.configuracion = ConfiguracionBenchmarkRLControlado(
            configuracion_ga=ConfiguracionGA(
                tamano_poblacion=8,
                generaciones=8,
                tamano_elite=2,
                tamano_torneo=3,
                generaciones_sin_mejora_max=3,
            ),
            seed_ga=101,
            exigir_sin_fallback=True,
        )

    def test_rl_equivalente_a_greedy_y_hibrido_cumple_garantia(self):
        resultado = ejecutar_benchmark_rl_controlado(
            self.caso,
            proveedor_viaje=self.proveedor,
            planners_rl={
                "FAKE": _PlannerGreedyComoRL(self.proveedor),
            },
            metadatos_modelos={
                "FAKE": MetadatosModeloRL(
                    alias="FAKE",
                    ruta_modelo="fake.zip",
                    sha256="abc123",
                ),
            },
            configuracion_benchmark=self.configuracion,
        )

        self.assertEqual(len(resultado.corridas), 4)
        self.assertEqual(len(resultado.resumenes), 1)
        resumen = resultado.resumenes[0]
        self.assertEqual(resumen.rl_estado, "OK")
        self.assertEqual(resumen.hibrido_estado, "OK")
        self.assertTrue(resumen.hibrido_cumple_garantia)
        self.assertAlmostEqual(
            resumen.costo_rl,
            resumen.costo_greedy,
            places=6,
        )

    def test_error_rl_no_aborta_y_hibrido_hace_fallback(self):
        resultado = ejecutar_benchmark_rl_controlado(
            self.caso,
            proveedor_viaje=self.proveedor,
            planners_rl={"ERROR": _PlannerConError()},
            metadatos_modelos={
                "ERROR": MetadatosModeloRL(
                    alias="ERROR",
                    ruta_modelo="error.zip",
                    sha256="def456",
                ),
            },
            configuracion_benchmark=self.configuracion,
        )

        filas = {
            fila.algoritmo: fila
            for fila in resultado.corridas
            if fila.modelo_alias == "ERROR"
        }
        self.assertEqual(filas["RL"].estado, "ERROR")
        self.assertFalse(filas["RL"].plan_valido)
        self.assertEqual(
            filas["HIBRIDO"].estado,
            "FALLBACK_GREEDY",
        )
        self.assertEqual(filas["HIBRIDO"].fuente_hibrida, "GREEDY")
        self.assertTrue(resultado.resumenes[0].hibrido_cumple_garantia)

    def test_escritura_csv_y_json(self):
        resultado = ejecutar_benchmark_rl_controlado(
            self.caso,
            proveedor_viaje=self.proveedor,
            planners_rl={
                "FAKE": _PlannerGreedyComoRL(self.proveedor),
            },
            metadatos_modelos={
                "FAKE": MetadatosModeloRL(
                    alias="FAKE",
                    ruta_modelo="fake.zip",
                    sha256="abc123",
                ),
            },
            configuracion_benchmark=self.configuracion,
        )

        with tempfile.TemporaryDirectory() as temporal:
            rutas = escribir_resultados_benchmark_rl_controlado(
                resultado,
                temporal,
            )
            self.assertEqual(set(rutas), {
                "corridas_csv",
                "resumen_csv",
                "benchmark_json",
            })
            for ruta in rutas.values():
                self.assertTrue(Path(ruta).is_file())
                self.assertGreater(Path(ruta).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
