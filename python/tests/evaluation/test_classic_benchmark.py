from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from planner.algorithms.ga import ConfiguracionGA
from planner.domain.validator import validar_instancia
from planner.evaluation.classic_benchmark import (
    ConfiguracionBenchmarkClasico,
    ejecutar_benchmark_clasico,
    escribir_resultados_benchmark,
)
from planner.evaluation.classic_instances import (
    crear_casos_benchmark_clasico,
)
from planner.routing.travel import ProveedorHaversineAjustado


class ClassicBenchmarkTest(unittest.TestCase):
    def test_casos_formales_son_validos_y_cubren_categorias(self) -> None:
        casos = crear_casos_benchmark_clasico()

        self.assertEqual(6, len(casos))
        self.assertEqual(
            {
                "SIMPLE",
                "CARGA_PARALELA",
                "MULTIVIAJE",
                "VENTANAS",
                "VOLCADOR",
                "SPLIT",
            },
            {caso.categoria for caso in casos},
        )

        for caso in casos:
            with self.subTest(caso=caso.caso_id):
                self.assertEqual(
                    [],
                    validar_instancia(caso.instancia),
                )

    def test_benchmark_pequeno_valida_planes_y_ga_no_supera_greedy(
        self,
    ) -> None:
        casos = crear_casos_benchmark_clasico()[:2]
        configuracion = ConfiguracionBenchmarkClasico(
            seeds_estocasticas=(17,),
            configuracion_ga=ConfiguracionGA(
                tamano_poblacion=8,
                generaciones=4,
                tamano_elite=1,
                tamano_torneo=2,
                generaciones_sin_mejora_max=2,
            ),
            exigir_sin_fallback=True,
        )

        resultado = ejecutar_benchmark_clasico(
            casos,
            proveedor_viaje=ProveedorHaversineAjustado(),
            configuracion_benchmark=configuracion,
        )

        self.assertEqual(6, len(resultado.corridas))
        self.assertTrue(
            all(corrida.plan_valido for corrida in resultado.corridas)
        )
        self.assertTrue(
            all(corrida.fallbacks_matriz == 0 for corrida in resultado.corridas)
        )

        resumenes_ga = [
            resumen
            for resumen in resultado.resumenes
            if resumen.algoritmo == "GA"
        ]
        self.assertEqual(2, len(resumenes_ga))
        self.assertTrue(
            all(
                resumen.cumple_ga_no_peor_greedy is True
                for resumen in resumenes_ga
            )
        )

    def test_resultados_se_escriben_en_csv_y_json(self) -> None:
        caso = crear_casos_benchmark_clasico()[0]
        configuracion = ConfiguracionBenchmarkClasico(
            seeds_estocasticas=(23,),
            configuracion_ga=ConfiguracionGA(
                tamano_poblacion=8,
                generaciones=3,
                tamano_elite=1,
                tamano_torneo=2,
                generaciones_sin_mejora_max=2,
            ),
        )
        resultado = ejecutar_benchmark_clasico(
            (caso,),
            proveedor_viaje=ProveedorHaversineAjustado(),
            configuracion_benchmark=configuracion,
        )

        with tempfile.TemporaryDirectory() as directorio:
            rutas = escribir_resultados_benchmark(
                resultado,
                directorio,
            )

            for ruta in rutas.values():
                self.assertTrue(ruta.is_file())

            with Path(rutas["corridas_csv"]).open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as archivo:
                filas = list(csv.DictReader(archivo))

            self.assertEqual(3, len(filas))
            self.assertEqual(
                {"GREEDY", "RANDOM", "GA"},
                {fila["algoritmo"] for fila in filas},
            )

            with Path(rutas["benchmark_json"]).open(
                "r",
                encoding="utf-8",
            ) as archivo:
                documento = json.load(archivo)

            self.assertEqual(
                "benchmark-clasico-v1",
                documento["version_benchmark"],
            )
            self.assertEqual(
                "estimacion-costo-v3",
                documento["version_objetivo"],
            )
            self.assertEqual(3, len(documento["corridas"]))


if __name__ == "__main__":
    unittest.main()
