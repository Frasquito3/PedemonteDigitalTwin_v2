from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from planner.algorithms.ga import ConfiguracionGA
from planner.algorithms.greedy import GreedyFeasiblePlanner
from planner.evaluation.rl_controlled_benchmark import (
    ConfiguracionBenchmarkRLControlado,
    MetadatosModeloRL,
)
from planner.evaluation.rl_stress_benchmark import (
    ejecutar_benchmark_rl_stress,
    escribir_resultados_benchmark_rl_stress,
)
from planner.evaluation.rl_stress_instances import (
    ESTRATOS_STRESS_RL,
    crear_casos_stress_rl,
)
from planner.routing.travel import ProveedorHaversineAjustado


class _PlannerGreedyComoRL:
    def __init__(self, proveedor) -> None:
        self._planner = GreedyFeasiblePlanner(
            proveedor_viaje=proveedor,
        )

    def generar_plan(self, instancia):
        return self._planner.generar_plan(instancia)


class TestBenchmarkRLStress(unittest.TestCase):
    def test_generacion_deterministica_y_estratificada(self):
        casos_a = crear_casos_stress_rl(
            cantidad_por_estrato=2,
            seed_base=20_000,
        )
        casos_b = crear_casos_stress_rl(
            cantidad_por_estrato=2,
            seed_base=20_000,
        )
        self.assertEqual(len(casos_a), 10)
        self.assertEqual(
            {caso.categoria for caso in casos_a},
            set(ESTRATOS_STRESS_RL),
        )
        self.assertEqual(
            [caso.instancia for caso in casos_a],
            [caso.instancia for caso in casos_b],
        )
        self.assertTrue(all(len(c.instancia.pedidos) <= 8 for c in casos_a))

    def test_resumen_con_rl_equivalente_a_greedy(self):
        proveedor = ProveedorHaversineAjustado()
        casos = crear_casos_stress_rl(
            cantidad_por_estrato=1,
            seed_base=21_000,
        )
        resultado = ejecutar_benchmark_rl_stress(
            casos,
            proveedor_viaje=proveedor,
            planners_rl={
                "FAKE": _PlannerGreedyComoRL(proveedor),
            },
            metadatos_modelos={
                "FAKE": MetadatosModeloRL(
                    alias="FAKE",
                    ruta_modelo="fake.zip",
                    sha256="abc123",
                ),
            },
            configuracion_benchmark=ConfiguracionBenchmarkRLControlado(
                configuracion_ga=ConfiguracionGA(
                    tamano_poblacion=8,
                    generaciones=8,
                    tamano_elite=2,
                    tamano_torneo=3,
                    generaciones_sin_mejora_max=3,
                ),
                seed_ga=101,
            ),
        )
        self.assertEqual(resultado.cantidad_casos, 5)
        self.assertEqual(resultado.cantidad_filas, 20)
        resumen = resultado.resumen_modelos[0]
        self.assertEqual(resumen.rl_empata_greedy, 5)
        self.assertEqual(resumen.rl_pierde_greedy, 0)
        self.assertEqual(resumen.violaciones_garantia_hibrida, 0)

    def test_escritura_de_cinco_archivos(self):
        proveedor = ProveedorHaversineAjustado()
        casos = crear_casos_stress_rl(
            cantidad_por_estrato=1,
            seed_base=22_000,
        )
        resultado = ejecutar_benchmark_rl_stress(
            casos,
            proveedor_viaje=proveedor,
            planners_rl={
                "FAKE": _PlannerGreedyComoRL(proveedor),
            },
            metadatos_modelos={
                "FAKE": MetadatosModeloRL(
                    alias="FAKE",
                    ruta_modelo="fake.zip",
                    sha256="abc123",
                ),
            },
            configuracion_benchmark=ConfiguracionBenchmarkRLControlado(
                configuracion_ga=ConfiguracionGA(
                    tamano_poblacion=8,
                    generaciones=8,
                    tamano_elite=2,
                    tamano_torneo=3,
                    generaciones_sin_mejora_max=3,
                ),
                seed_ga=101,
            ),
        )
        with tempfile.TemporaryDirectory() as temporal:
            rutas = escribir_resultados_benchmark_rl_stress(
                resultado,
                temporal,
            )
            self.assertEqual(len(rutas), 5)
            for ruta in rutas.values():
                archivo = Path(ruta)
                self.assertTrue(archivo.is_file())
                self.assertGreater(archivo.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
