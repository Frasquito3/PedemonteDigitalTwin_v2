import json
import tempfile
import unittest
from pathlib import Path

from planner.rl.balanced_policy_validation import (
    ResumenValidacionExtensionV4,
    crear_bateria_validacion_extension_v4,
    es_mejor_validacion_extension_v4,
    validar_origen_v4_quick,
)


class TemporalV4ExtensionValidationTest(unittest.TestCase):
    def _resumen(
        self,
        *,
        clasicos_tardios: int = 0,
        regresiones_clasicas: int = 0,
        objetivo_sin_riesgo: int = 8,
        objetivo_tardanza: float = 100.0,
        guard_sin_riesgo: int = 8,
    ) -> ResumenValidacionExtensionV4:
        return ResumenValidacionExtensionV4(
            timestep=1000,
            clasicos_pedidos_tardios=clasicos_tardios,
            clasicos_tardanza_total_min=float(clasicos_tardios),
            clasicos_regresiones_costo=regresiones_clasicas,
            objetivo_9_12_totales=16,
            objetivo_9_12_sin_riesgo=objetivo_sin_riesgo,
            objetivo_9_12_tardanza_total_min=objetivo_tardanza,
            objetivo_9_12_costos_extremos=0,
            guard_3_8_totales=16,
            guard_3_8_sin_riesgo=guard_sin_riesgo,
            guard_3_8_tardanza_total_min=0.0,
            guard_3_8_costos_extremos=0,
            gap_costo_mediano_vs_greedy_pct=0.0,
            resumenes_estrato=(),
            casos=(),
        )

    def test_preservar_clasicos_domina_mejora_9_12(self) -> None:
        actual = self._resumen(objetivo_sin_riesgo=8)
        candidata = self._resumen(
            clasicos_tardios=1,
            objetivo_sin_riesgo=16,
        )
        self.assertFalse(
            es_mejor_validacion_extension_v4(candidata, actual)
        )

    def test_factibilidad_9_12_domina_guard_3_8(self) -> None:
        actual = self._resumen(
            objetivo_sin_riesgo=8,
            guard_sin_riesgo=16,
        )
        candidata = self._resumen(
            objetivo_sin_riesgo=9,
            guard_sin_riesgo=12,
        )
        self.assertTrue(
            es_mejor_validacion_extension_v4(candidata, actual)
        )

    def test_bateria_incluye_clasicos_y_ocho_estratos(self) -> None:
        bateria = crear_bateria_validacion_extension_v4(
            cantidad_por_estrato=1,
            seed_inicio=271_000,
        )
        self.assertEqual(len(bateria.casos), 11)
        estratos = {item.estrato for item in bateria.casos}
        self.assertIn("CLASICO_GUARD", estratos)
        self.assertIn("OBJETIVO_CONFLICTIVO_11_12", estratos)
        self.assertIn("GUARD_GENERAL_6_8", estratos)
        self.assertEqual(len(bateria.semillas_sinteticas), 8)

    def test_valida_origen_quick_externo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            config = {
                "version_entorno": "pedemonte-rl-temporal-v4",
                "quick": True,
                "continuacion_entre_etapas": "EXTERNAL_BEST",
                "modelo_historico_sobrescrito": False,
                "modelo_v3_sobrescrito": False,
                "temporal": {"usar_mascara_temporal_dura": False},
            }
            seleccion = {
                "criterio": "VALIDACION_EXTERNA_LEXICOGRAFICA_V4",
                "modelo_promovido": False,
            }
            ruta_config = raiz / "config.json"
            ruta_seleccion = raiz / "selection.json"
            ruta_config.write_text(json.dumps(config), encoding="utf-8")
            ruta_seleccion.write_text(
                json.dumps(seleccion),
                encoding="utf-8",
            )
            cargada, elegida = validar_origen_v4_quick(
                ruta_config,
                ruta_seleccion,
            )
            self.assertTrue(cargada["quick"])
            self.assertFalse(elegida["modelo_promovido"])


if __name__ == "__main__":
    unittest.main()
