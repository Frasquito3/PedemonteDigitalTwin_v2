import json
import tempfile
import unittest
from pathlib import Path

from planner.rl.high_demand_policy_validation import (
    ResumenValidacionCompletaV4,
    crear_bateria_validacion_completa_v4,
    es_mejor_validacion_completa_v4,
    validar_evidencia_holdout_16d7,
    validar_origen_extension_v4,
)


class TemporalV4FullValidationTest(unittest.TestCase):
    def _resumen(
        self,
        *,
        clasicos_tardios: int = 0,
        guard_3_8_tardios: int = 0,
        guard_9_10_tardios: int = 0,
        objetivo_12_sin_riesgo: int = 8,
        objetivo_12_tardios: int = 4,
        general_sin_riesgo: int = 8,
        objetivo_total_sin_riesgo: int = 16,
    ) -> ResumenValidacionCompletaV4:
        return ResumenValidacionCompletaV4(
            timestep=100_000,
            clasicos_pedidos_tardios=clasicos_tardios,
            clasicos_tardanza_total_min=float(clasicos_tardios),
            clasicos_regresiones_costo=0,
            guard_3_8_totales=16,
            guard_3_8_sin_riesgo=16 - min(guard_3_8_tardios, 16),
            guard_3_8_pedidos_tardios=guard_3_8_tardios,
            guard_3_8_tardanza_total_min=float(guard_3_8_tardios),
            guard_3_8_costos_extremos=0,
            guard_9_10_totales=8,
            guard_9_10_sin_riesgo=8 - min(guard_9_10_tardios, 8),
            guard_9_10_pedidos_tardios=guard_9_10_tardios,
            guard_9_10_tardanza_total_min=float(guard_9_10_tardios),
            guard_9_10_costos_extremos=0,
            objetivo_11_totales=12,
            objetivo_11_sin_riesgo=8,
            objetivo_11_pedidos_tardios=4,
            objetivo_11_tardanza_total_min=20.0,
            objetivo_12_totales=12,
            objetivo_12_sin_riesgo=objetivo_12_sin_riesgo,
            objetivo_12_pedidos_tardios=objetivo_12_tardios,
            objetivo_12_tardanza_total_min=float(objetivo_12_tardios * 10),
            objetivo_general_11_12_totales=12,
            objetivo_general_11_12_sin_riesgo=general_sin_riesgo,
            objetivo_general_11_12_pedidos_tardios=4,
            objetivo_general_11_12_tardanza_total_min=20.0,
            objetivo_11_12_totales=24,
            objetivo_11_12_sin_riesgo=objetivo_total_sin_riesgo,
            objetivo_11_12_pedidos_tardios=8,
            objetivo_11_12_tardanza_total_min=60.0,
            objetivo_11_12_costos_extremos=0,
            gap_costo_mediano_vs_greedy_pct=0.0,
            resumenes_grupo=(),
            resumenes_estrato=(),
            casos=(),
        )

    def test_clasicos_dominan_mejora_exactos_12(self) -> None:
        actual = self._resumen(objetivo_12_sin_riesgo=8)
        candidata = self._resumen(
            clasicos_tardios=1,
            objetivo_12_sin_riesgo=12,
            objetivo_12_tardios=0,
        )
        self.assertFalse(
            es_mejor_validacion_completa_v4(candidata, actual)
        )

    def test_guard_3_8_domina_mejora_exactos_12(self) -> None:
        actual = self._resumen(objetivo_12_sin_riesgo=8)
        candidata = self._resumen(
            guard_3_8_tardios=1,
            objetivo_12_sin_riesgo=12,
            objetivo_12_tardios=0,
        )
        self.assertFalse(
            es_mejor_validacion_completa_v4(candidata, actual)
        )

    def test_exactos_12_dominan_general_11_12(self) -> None:
        actual = self._resumen(
            objetivo_12_sin_riesgo=8,
            general_sin_riesgo=12,
        )
        candidata = self._resumen(
            objetivo_12_sin_riesgo=9,
            objetivo_12_tardios=3,
            general_sin_riesgo=6,
        )
        self.assertTrue(
            es_mejor_validacion_completa_v4(candidata, actual)
        )

    def test_bateria_incluye_diez_estratos_y_exactos_12(self) -> None:
        bateria = crear_bateria_validacion_completa_v4(
            cantidad_por_estrato=1,
            seed_inicio=273_000,
        )
        self.assertEqual(len(bateria.casos), 13)
        self.assertEqual(len(bateria.semillas_sinteticas), 10)
        estratos = {caso.estrato for caso in bateria.casos}
        self.assertIn("CLASICO_GUARD", estratos)
        self.assertIn("OBJETIVO_GENERAL_12", estratos)
        exactos_12 = [
            caso
            for caso in bateria.casos
            if caso.grupo == "OBJETIVO_12"
        ]
        self.assertEqual(len(exactos_12), 2)
        self.assertTrue(
            all(len(caso.instancia.pedidos) == 12 for caso in exactos_12)
        )
        self.assertLess(max(bateria.semillas_sinteticas), 274_000)

    def test_rechaza_semilla_de_holdout_anterior(self) -> None:
        with self.assertRaises(ValueError):
            crear_bateria_validacion_completa_v4(
                cantidad_por_estrato=1,
                seed_inicio=272_000,
            )

    def test_valida_evidencia_holdout_candidata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "holdout.json"
            ruta.write_text(
                json.dumps(
                    {
                        "metadatos": {
                            "casos_sinteticos": 160,
                            "seed_min_usada": 272_000,
                            "modelo_promovido": False,
                            "modo_solo_clasicos": False,
                        },
                        "veredicto": {
                            "estado": "CANDIDATO_ENTRENAMIENTO_COMPLETO",
                            "criterios": {
                                "sin_errores": True,
                                "mejora_12": True,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            datos = validar_evidencia_holdout_16d7(ruta)
            self.assertEqual(
                datos["veredicto"]["estado"],
                "CANDIDATO_ENTRENAMIENTO_COMPLETO",
            )

    def test_rechaza_holdout_no_candidato(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "holdout.json"
            ruta.write_text(
                json.dumps(
                    {
                        "metadatos": {
                            "casos_sinteticos": 160,
                            "seed_min_usada": 272_000,
                            "modelo_promovido": False,
                            "modo_solo_clasicos": False,
                        },
                        "veredicto": {
                            "estado": "PROMETEDOR_CON_AJUSTES_PENDIENTES",
                            "criterios": {"sin_errores": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validar_evidencia_holdout_16d7(ruta)

    def test_valida_origen_extension_68288(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            config = {
                "version_run": (
                    "pedemonte-rl-temporal-v4-extension-9-12-v1"
                ),
                "observacion": 702,
                "acciones": 30,
                "reward_modificado": False,
                "observacion_modificada": False,
                "mascara_temporal_dura": False,
                "continuacion_entre_etapas": "EXTERNAL_BEST_9_12",
                "modelo_promovido": False,
            }
            seleccion = {
                "criterio": (
                    "VALIDACION_EXTERNA_9_12_LEXICOGRAFICA_V4_EXTENSION"
                ),
                "modelo_promovido": False,
            }
            resumen = {
                "timestep": 68_288,
                "clasicos_pedidos_tardios": 0,
                "guard_3_8_sin_riesgo": 16,
                "guard_3_8_totales": 16,
            }
            rutas = []
            for nombre, contenido in (
                ("config.json", config),
                ("selection.json", seleccion),
                ("summary.json", resumen),
            ):
                ruta = raiz / nombre
                ruta.write_text(json.dumps(contenido), encoding="utf-8")
                rutas.append(ruta)

            cargada, elegida, resumen_cargado = (
                validar_origen_extension_v4(*rutas)
            )
            self.assertEqual(cargada["observacion"], 702)
            self.assertFalse(elegida["modelo_promovido"])
            self.assertEqual(resumen_cargado["timestep"], 68_288)


if __name__ == "__main__":
    unittest.main()
