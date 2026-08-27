from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from planner.evaluation.comparison_service_audit import (
    VERSION_AUDITORIA_SERVICIO,
    auditar_suite_servicio,
    escribir_auditoria_servicio,
)
from planner.routing.travel import ProveedorHaversineAjustado


def _corrida(
    *,
    modo: str,
    costo: float,
    entregadas: int,
    no_entregadas: int,
    orden: int,
) -> dict:
    algoritmo = "GREEDY" if modo == "HIBRIDO" else modo
    return {
        "caso_id": "B04_VENTANAS",
        "categoria": "VENTANAS",
        "descripcion": "Caso ventanas",
        "instancia_id": "BENCH-B04-VENTANAS",
        "orden_modo": orden,
        "modo_solicitado": modo,
        "algoritmo_resultante": algoritmo,
        "fuente_seleccionada": algoritmo,
        "firma_ruta": f"firma-{modo}",
        "seed_escenario": 15104,
        "seed_planificacion": None if modo == "GREEDY" else 15104 + orden,
        "seed_ejecucion": 25104,
        "estado_ejecucion": "OK",
        "error_ejecucion": "",
        "costo_estimado": costo,
        "costo_real": costo,
        "diferencia_costo_real_estimado": 0.0,
        "error_relativo_estimacion_pct": 0.0,
        "tiempo_plan_ms": float(orden),
        "tiempo_selector_ms": float(orden),
        "tiempo_simulado_min": 100.0,
        "tareas_entregadas": entregadas,
        "tareas_no_entregadas": no_entregadas,
        "viajes_totales": 1,
        "estado_final_motor": "EngineState.FINISHED",
        "stop_condition": True,
        "mensaje_anylogic": "EJECUCIÓN FINALIZADA",
    }


def _suite() -> dict:
    corridas = [
        _corrida(
            modo="RL",
            costo=15000.0,
            entregadas=2,
            no_entregadas=1,
            orden=1,
        ),
        _corrida(modo="GA", costo=200.0, entregadas=3, no_entregadas=0, orden=2),
        _corrida(modo="GREEDY", costo=210.0, entregadas=3, no_entregadas=0, orden=3),
        _corrida(modo="RANDOM", costo=500.0, entregadas=3, no_entregadas=0, orden=4),
        _corrida(modo="HIBRIDO", costo=200.0, entregadas=3, no_entregadas=0, orden=5),
    ]
    return {
        "version_suite": "comparacion-anylogic-suite-v1",
        "corridas_esperadas": 5,
        "orden_modos": ["RL", "GA", "GREEDY", "RANDOM", "HIBRIDO"],
        "common_random_numbers_por_caso": True,
        "proceso_nuevo_por_plan": True,
        "version_viaje": "cache-test",
        "version_objetivo": "estimacion-costo-v3",
        "casos": [
            {
                "caso_id": "B04_VENTANAS",
                "cantidad_pedidos": 3,
            }
        ],
        "corridas": corridas,
    }


def _contrato_b04() -> dict:
    pedidos_rl = [
        "B04-NORTE-TEMPRANO",
        "B04-CERCANA-TARDE",
        "B04-ESTE-MEDIO",
    ]
    pedidos_correctos = [
        "B04-NORTE-TEMPRANO",
        "B04-ESTE-MEDIO",
        "B04-CERCANA-TARDE",
    ]
    planes = []
    for orden, modo in enumerate(
        ["RL", "GA", "GREEDY", "RANDOM", "HIBRIDO"],
        start=1,
    ):
        algoritmo = "GREEDY" if modo == "HIBRIDO" else modo
        pedidos = pedidos_rl if modo == "RL" else pedidos_correctos
        planes.append(
            {
                "orden": orden,
                "modo_solicitado": modo,
                "algoritmo_resultante": algoritmo,
                "estado": "OK",
                "costo_estimado": 200.0,
                "tiempo_plan_ms": 1.0,
                "camiones": [
                    {
                        "camion_id": 0,
                        "viajes": [
                            {
                                "numero_viaje": 1,
                                "pedido_ids": pedidos,
                            }
                        ],
                    },
                    {"camion_id": 1, "viajes": []},
                ],
            }
        )
    return {"planes": planes}


class TestComparisonServiceAudit(unittest.TestCase):
    def test_separa_estado_tecnico_y_servicio(self) -> None:
        resultado = auditar_suite_servicio(_suite())
        self.assertEqual(resultado.version_auditoria, VERSION_AUDITORIA_SERVICIO)
        self.assertEqual(resultado.ejecuciones_tecnicas_ok, 5)
        self.assertEqual(resultado.servicios_completos, 4)
        self.assertEqual(resultado.servicios_incompletos, 1)

        rl = next(
            corrida
            for corrida in resultado.corridas
            if corrida.modo_solicitado == "RL"
        )
        self.assertEqual(rl.estado_ejecucion, "OK")
        self.assertEqual(rl.estado_servicio, "INCOMPLETA")
        self.assertAlmostEqual(rl.nivel_servicio_pct or 0.0, 66.6666666667)
        self.assertFalse(rl.elegible_ranking)
        self.assertIsNone(rl.ranking_caso)

    def test_excluye_incompletas_del_ranking_y_comparaciones(self) -> None:
        resultado = auditar_suite_servicio(_suite())
        por_modo = {
            corrida.modo_solicitado: corrida
            for corrida in resultado.corridas
        }
        self.assertEqual(por_modo["GA"].ranking_caso, 1)
        self.assertEqual(por_modo["HIBRIDO"].ranking_caso, 1)
        self.assertEqual(por_modo["GREEDY"].ranking_caso, 2)
        self.assertEqual(por_modo["RL"].comparacion_vs_greedy, "NO_DISPONIBLE")
        self.assertEqual(por_modo["GA"].comparacion_vs_rl, "NO_DISPONIBLE")

        caso = resultado.casos[0]
        self.assertEqual(caso.estado_servicio_rl, "INCOMPLETA")
        self.assertIsNone(caso.ranking_rl)
        self.assertEqual(caso.modos_mejor_costo_completo, ("GA", "HIBRIDO"))

    def test_auditoria_ventanas_detecta_riesgo_rl(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            carpeta = raiz / "B04_VENTANAS"
            carpeta.mkdir(parents=True)
            (carpeta / "comparison_contract.json").write_text(
                json.dumps(_contrato_b04()),
                encoding="utf-8",
            )

            resultado = auditar_suite_servicio(
                _suite(),
                contratos_dir=raiz,
                proveedor_viaje=ProveedorHaversineAjustado(),
            )

        resumen_rl = next(
            resumen
            for resumen in resultado.resumen_planes_ventanas
            if resumen.modo_solicitado == "RL"
        )
        self.assertEqual(resumen_rl.estado_servicio_real, "INCOMPLETA")
        self.assertGreaterEqual(resumen_rl.llegadas_tardias_estimadas, 1)
        self.assertIn(
            "B04-ESTE-MEDIO",
            resumen_rl.pedidos_riesgo_rechazo,
        )

    def test_escribe_siete_archivos(self) -> None:
        resultado = auditar_suite_servicio(_suite())
        with tempfile.TemporaryDirectory() as temporal:
            rutas = escribir_auditoria_servicio(resultado, temporal)
            self.assertEqual(len(rutas), 7)
            self.assertTrue(all(ruta.is_file() for ruta in rutas.values()))


if __name__ == "__main__":
    unittest.main()
