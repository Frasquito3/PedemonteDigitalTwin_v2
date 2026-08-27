from __future__ import annotations

import json
import tempfile
import unittest

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from planner.evaluation.comparison_execution import (
    ORDEN_MODOS_ESPERADO,
)
from planner.evaluation.comparison_suite import (
    VERSION_SUITE_COMPARACION,
    CasoContratoComparacion,
    ConfiguracionSuiteComparacion,
    escribir_resultado_suite_comparacion,
    ejecutar_suite_comparacion,
)


@dataclass
class _FakeExecutor:
    fallar_identificador: str = ""
    identificadores: list[str] = field(default_factory=list)

    def ejecutar_vectores(
        self,
        *,
        instancia_vector,
        plan_vector,
        seed_ejecucion,
        cantidad_pedidos,
        cantidad_viajes,
        identificador_corrida,
    ):
        self.identificadores.append(identificador_corrida)

        if (
            self.fallar_identificador
            and self.fallar_identificador in identificador_corrida
        ):
            raise RuntimeError("fallo controlado de suite")

        costo = float(plan_vector[3])
        return SimpleNamespace(
            modelo="modelo.zip",
            java="java.exe",
            estado_final="EngineState.FINISHED",
            stop_condition=True,
            observacion_final={
                "costoTotal": costo,
                "tareasEntregadas": cantidad_pedidos,
                "tareasNoEntregadas": 0,
                "viajesTotales": cantidad_viajes,
                "tiempoSimuladoMin": 40.0 + costo / 100.0,
                "mensaje": "EJECUCIÓN FINALIZADA",
            },
        )


def _contrato(
    indice: int,
    costos: dict[str, float] | None = None,
) -> dict:
    costos_modo = costos or {
        "RL": 90.0,
        "GA": 80.0,
        "GREEDY": 100.0,
        "RANDOM": 110.0,
        "HIBRIDO": 80.0,
    }
    planes = []
    codigos = {
        "RL": 0.0,
        "GA": 1.0,
        "GREEDY": 2.0,
        "RANDOM": 3.0,
        "HIBRIDO": 1.0,
    }

    seed_escenario = 15100 + indice
    seed_ejecucion = seed_escenario + 10000

    for orden, modo in enumerate(ORDEN_MODOS_ESPERADO, start=1):
        algoritmo = "GA" if modo == "HIBRIDO" else modo
        costo = costos_modo[modo] + indice
        planes.append(
            {
                "orden": orden,
                "modo_solicitado": modo,
                "algoritmo_resultante": algoritmo,
                "estado": "OK",
                "fuente_seleccionada": algoritmo,
                "firma_ruta": f"firma-{indice}-{modo}",
                "seed_escenario": seed_escenario,
                "seed_planificacion": (
                    None if modo == "GREEDY" else seed_escenario + orden
                ),
                "seed_ejecucion": seed_ejecucion,
                "costo_estimado": costo,
                "tiempo_plan_ms": float(orden),
                "tiempo_selector_ms": float(orden) + 0.1,
                "camiones": [
                    {
                        "camion_id": 0,
                        "viajes": [
                            {
                                "numero_viaje": 1,
                                "pedido_ids": ["P0", "P1"],
                            }
                        ],
                    },
                    {
                        "camion_id": 1,
                        "viajes": [],
                    },
                ],
                "plan_vector": [
                    1.0,
                    2.0,
                    codigos[modo],
                    costo,
                    float(orden),
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                ],
            }
        )

    return {
        "version_contrato": "comparacion-anylogic-v1",
        "version_objetivo": "estimacion-costo-v3",
        "instancia_id": f"BENCH-B{indice:02d}",
        "cantidad_pedidos": 2,
        "seed_escenario": seed_escenario,
        "seed_ejecucion": seed_ejecucion,
        "fuente_viaje": "VIAL_CACHE",
        "version_viaje": "cache-test-v1",
        "planes_ok": 5,
        "planes_error": 0,
        "orden_modos": list(ORDEN_MODOS_ESPERADO),
        "instancia_vector": [
            1.0,
            0.0,
            2.0,
            8.0,
            2.0,
            -32.8,
            -60.7,
            0.0,
        ],
        "planes": planes,
    }


def _convertir_listas_a_tuplas(valor):
    if isinstance(valor, dict):
        return {
            clave: _convertir_listas_a_tuplas(elemento)
            for clave, elemento in valor.items()
        }
    if isinstance(valor, list):
        return tuple(
            _convertir_listas_a_tuplas(elemento)
            for elemento in valor
        )
    return valor


def _casos(cantidad: int = 6) -> list[CasoContratoComparacion]:
    return [
        CasoContratoComparacion(
            caso_id=f"B{indice:02d}_CASO",
            categoria=f"CAT_{indice}",
            descripcion=f"Caso {indice}",
            contrato=_contrato(indice),
        )
        for indice in range(1, cantidad + 1)
    ]


class TestComparisonSuite(unittest.TestCase):
    def test_ejecuta_seis_casos_y_treinta_corridas(self) -> None:
        ejecutor = _FakeExecutor()
        resultado = ejecutar_suite_comparacion(
            _casos(),
            ejecutor=ejecutor,
        )

        self.assertEqual(
            resultado.version_suite,
            VERSION_SUITE_COMPARACION,
        )
        self.assertEqual(resultado.cantidad_casos, 6)
        self.assertEqual(resultado.corridas_esperadas, 30)
        self.assertEqual(resultado.corridas_ok, 30)
        self.assertEqual(resultado.corridas_error, 0)
        self.assertEqual(len(resultado.corridas), 30)
        self.assertEqual(len(ejecutor.identificadores), 30)
        self.assertEqual(len(set(ejecutor.identificadores)), 30)
        self.assertTrue(
            ejecutor.identificadores[0].startswith("B01_CASO_01_RL")
        )


    def test_acepta_contratos_en_memoria_con_tuplas(self) -> None:
        casos = [
            CasoContratoComparacion(
                caso_id=caso.caso_id,
                categoria=caso.categoria,
                descripcion=caso.descripcion,
                contrato=_convertir_listas_a_tuplas(
                    caso.contrato
                ),
            )
            for caso in _casos()
        ]

        resultado = ejecutar_suite_comparacion(
            casos,
            ejecutor=_FakeExecutor(),
        )

        self.assertEqual(resultado.corridas_ok, 30)
        self.assertEqual(resultado.corridas_error, 0)

    def test_ranking_y_comparaciones_preservan_rl_primero(self) -> None:
        resultado = ejecutar_suite_comparacion(
            _casos(),
            ejecutor=_FakeExecutor(),
        )

        corridas_b01 = [
            corrida
            for corrida in resultado.corridas
            if corrida.caso_id == "B01_CASO"
        ]
        self.assertEqual(
            tuple(corrida.modo_solicitado for corrida in corridas_b01),
            ORDEN_MODOS_ESPERADO,
        )

        por_modo = {
            corrida.modo_solicitado: corrida
            for corrida in corridas_b01
        }
        self.assertEqual(por_modo["GA"].ranking_caso, 1)
        self.assertEqual(por_modo["HIBRIDO"].ranking_caso, 1)
        self.assertEqual(por_modo["RL"].ranking_caso, 2)
        self.assertEqual(por_modo["GREEDY"].ranking_caso, 3)
        self.assertEqual(por_modo["RANDOM"].ranking_caso, 4)
        self.assertEqual(por_modo["GA"].comparacion_vs_rl, "MEJOR")
        self.assertEqual(por_modo["RL"].comparacion_vs_rl, "EMPATE")
        self.assertEqual(
            por_modo["GREEDY"].comparacion_vs_greedy,
            "EMPATE",
        )

        resumen_ga = next(
            resumen
            for resumen in resultado.resumen_algoritmos
            if resumen.modo_solicitado == "GA"
        )
        self.assertEqual(resumen_ga.primeros_puestos, 6)
        self.assertEqual(resumen_ga.victorias_vs_rl, 6)
        self.assertEqual(resumen_ga.derrotas_vs_rl, 0)

    def test_error_aislado_conserva_las_otras_corridas(self) -> None:
        resultado = ejecutar_suite_comparacion(
            _casos(),
            ejecutor=_FakeExecutor(
                fallar_identificador="B03_CASO_02_GA"
            ),
        )

        self.assertEqual(resultado.corridas_ok, 29)
        self.assertEqual(resultado.corridas_error, 1)
        error = next(
            corrida
            for corrida in resultado.corridas
            if corrida.estado_ejecucion == "ERROR"
        )
        self.assertEqual(error.caso_id, "B03_CASO")
        self.assertEqual(error.modo_solicitado, "GA")
        self.assertIn("fallo controlado", error.error_ejecucion)

    def test_rechaza_casos_duplicados_y_suite_incompleta(self) -> None:
        duplicados = _casos()
        duplicados[1] = duplicados[0]

        with self.assertRaisesRegex(ValueError, "no pueden repetirse"):
            ejecutar_suite_comparacion(
                duplicados,
                ejecutor=_FakeExecutor(),
            )

        with self.assertRaisesRegex(ValueError, "exige 6 casos"):
            ejecutar_suite_comparacion(
                _casos(2),
                ejecutor=_FakeExecutor(),
            )

        parcial = ejecutar_suite_comparacion(
            _casos(2),
            ejecutor=_FakeExecutor(),
            configuracion=ConfiguracionSuiteComparacion(
                exigir_seis_casos=False,
            ),
        )
        self.assertEqual(parcial.corridas_esperadas, 10)

    def test_escribe_json_y_tres_csv(self) -> None:
        resultado = ejecutar_suite_comparacion(
            _casos(),
            ejecutor=_FakeExecutor(),
        )

        with tempfile.TemporaryDirectory() as temporal:
            rutas = escribir_resultado_suite_comparacion(
                resultado,
                temporal,
            )

            self.assertEqual(
                set(rutas),
                {
                    "suite_json",
                    "corridas_csv",
                    "casos_csv",
                    "algoritmos_csv",
                },
            )
            for ruta in rutas.values():
                self.assertTrue(Path(ruta).is_file())
                self.assertGreater(Path(ruta).stat().st_size, 0)

            with rutas["suite_json"].open(
                "r",
                encoding="utf-8",
            ) as archivo:
                datos = json.load(archivo)

            self.assertEqual(datos["corridas_ok"], 30)
            self.assertEqual(datos["corridas"][0]["modo_solicitado"], "RL")


if __name__ == "__main__":
    unittest.main()
