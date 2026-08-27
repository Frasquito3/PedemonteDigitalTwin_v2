from __future__ import annotations

import json
import tempfile
import unittest

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

from planner.evaluation.comparison_execution import (
    ORDEN_MODOS_ESPERADO,
    VERSION_EJECUCION_COMPARACION,
    ConfiguracionEjecucionComparacion,
    escribir_resultado_ejecucion_comparacion,
    ejecutar_contrato_comparacion,
)


@dataclass
class _FakeExecutor:
    fallar_modo: str = ""

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
        modo = identificador_corrida.split("_", 1)[1]
        if modo == self.fallar_modo:
            raise RuntimeError("fallo controlado")

        costo = float(plan_vector[3]) - 1.0
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
                "tiempoSimuladoMin": 42.0,
                "mensaje": "EJECUCIÓN FINALIZADA",
            },
        )


def _contrato_valido() -> dict[str, Any]:
    planes = []
    codigos = {
        "RL": 0.0,
        "GA": 1.0,
        "GREEDY": 2.0,
        "RANDOM": 3.0,
        "HIBRIDO": 1.0,
    }

    for orden, modo in enumerate(ORDEN_MODOS_ESPERADO, start=1):
        planes.append(
            {
                "orden": orden,
                "modo_solicitado": modo,
                "algoritmo_resultante": (
                    "GA" if modo == "HIBRIDO" else modo
                ),
                "estado": "OK",
                "fuente_seleccionada": (
                    "GA" if modo == "HIBRIDO" else modo
                ),
                "firma_ruta": f"firma-{modo}",
                "seed_escenario": 15105,
                "seed_planificacion": (
                    None if modo == "GREEDY" else 15105 + orden
                ),
                "seed_ejecucion": 25105,
                "costo_estimado": 100.0 + orden,
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
                    100.0 + orden,
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
        "instancia_id": "BENCH-B05-VOLCADOR",
        "cantidad_pedidos": 2,
        "seed_escenario": 15105,
        "seed_ejecucion": 25105,
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
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            -32.8,
            -60.7,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            -32.8,
            -60.7,
            -1.0,
            -1.0,
        ],
        "planes": planes,
    }


def _convertir_listas_a_tuplas(valor: Any) -> Any:
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


class TestEjecucionComparacion(unittest.TestCase):
    def test_ejecuta_cinco_planes_en_orden_rl_primero(self) -> None:
        resultado = ejecutar_contrato_comparacion(
            _contrato_valido(),
            ejecutor=_FakeExecutor(),
        )

        self.assertEqual(
            resultado.version_ejecucion,
            VERSION_EJECUCION_COMPARACION,
        )
        self.assertEqual(resultado.orden_modos, ORDEN_MODOS_ESPERADO)
        self.assertTrue(resultado.common_random_numbers)
        self.assertTrue(resultado.proceso_nuevo_por_plan)
        self.assertEqual(resultado.ejecuciones_ok, 5)
        self.assertEqual(resultado.ejecuciones_error, 0)
        self.assertEqual(
            tuple(
                registro.modo_solicitado
                for registro in resultado.registros
            ),
            ORDEN_MODOS_ESPERADO,
        )
        self.assertTrue(
            all(
                registro.seed_ejecucion == 25105
                for registro in resultado.registros
            )
        )


    def test_acepta_contrato_en_memoria_con_tuplas(self) -> None:
        contrato = cast(
            Mapping[str, Any],
            _convertir_listas_a_tuplas(
                _contrato_valido()
            ),
        )

        resultado = ejecutar_contrato_comparacion(
            contrato,
            ejecutor=_FakeExecutor(),
        )

        self.assertEqual(resultado.ejecuciones_ok, 5)
        self.assertEqual(resultado.ejecuciones_error, 0)
        self.assertEqual(
            resultado.orden_modos,
            ORDEN_MODOS_ESPERADO,
        )

    def test_error_aislado_no_impide_ejecutar_resto(self) -> None:
        resultado = ejecutar_contrato_comparacion(
            _contrato_valido(),
            ejecutor=_FakeExecutor(fallar_modo="GA"),
        )

        self.assertEqual(resultado.ejecuciones_ok, 4)
        self.assertEqual(resultado.ejecuciones_error, 1)
        ga = next(
            registro
            for registro in resultado.registros
            if registro.modo_solicitado == "GA"
        )
        self.assertEqual(ga.estado_ejecucion, "ERROR")
        self.assertIn("fallo controlado", ga.error_ejecucion)

    def test_rechaza_seed_ejecucion_distinta(self) -> None:
        contrato = _contrato_valido()
        contrato["planes"][0]["seed_ejecucion"] = 999

        with self.assertRaisesRegex(
            ValueError,
            "seed_ejecucion",
        ):
            ejecutar_contrato_comparacion(
                contrato,
                ejecutor=_FakeExecutor(),
            )

    def test_rechaza_orden_sin_rl_primero(self) -> None:
        contrato = _contrato_valido()
        contrato["orden_modos"] = [
            "GA",
            "RL",
            "GREEDY",
            "RANDOM",
            "HIBRIDO",
        ]

        with self.assertRaisesRegex(
            ValueError,
            "orden de comparación",
        ):
            ejecutar_contrato_comparacion(
                contrato,
                ejecutor=_FakeExecutor(),
            )

    def test_escribe_json_y_csv(self) -> None:
        resultado = ejecutar_contrato_comparacion(
            _contrato_valido(),
            ejecutor=_FakeExecutor(),
            configuracion=ConfiguracionEjecucionComparacion(),
        )

        with tempfile.TemporaryDirectory() as temporal:
            rutas = escribir_resultado_ejecucion_comparacion(
                resultado,
                temporal,
            )

            self.assertEqual(
                set(rutas),
                {"ejecucion_json", "ejecucion_csv"},
            )

            for ruta in rutas.values():
                self.assertTrue(Path(ruta).is_file())
                self.assertGreater(Path(ruta).stat().st_size, 0)

            with rutas["ejecucion_json"].open(
                "r",
                encoding="utf-8",
            ) as archivo:
                datos = json.load(archivo)

            self.assertEqual(datos["ejecuciones_ok"], 5)
            self.assertEqual(
                datos["registros"][0]["modo_solicitado"],
                "RL",
            )


if __name__ == "__main__":
    unittest.main()
