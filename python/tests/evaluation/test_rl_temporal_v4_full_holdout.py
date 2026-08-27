from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from planner.evaluation.rl_temporal_v4_full_holdout import (
    ESTRATOS_FASE_16D7,
    MODO_GREEDY,
    MODO_RL_HISTORICO,
    MODO_RL_TEMPORAL_V4_EXTENSION,
    MODO_RL_TEMPORAL_V4_FULL,
    MODO_RL_TEMPORAL_V4_QUICK,
    ORDEN_MODOS,
    RegistroHoldoutFull,
    comparar_lexicografico,
    construir_veredicto,
    crear_casos_sinteticos_finales,
    escribir_resultados,
    resumir_casos,
    resumir_registros,
)


class TestRLTemporalV4FullHoldout(unittest.TestCase):
    def _registro(
        self,
        caso_id: str,
        modo: str,
        *,
        grupo: str = "HOLDOUT_SINTETICO",
        estrato: str = "GENERAL_3_5",
        cantidad: int = 3,
        tardios: int = 0,
        tardanza: float = 0.0,
        costo: float = 100.0,
        estado: str = "OK",
    ) -> RegistroHoldoutFull:
        return RegistroHoldoutFull(
            grupo=grupo,
            caso_id=caso_id,
            categoria="TEST",
            descripcion="Caso de prueba",
            instancia_id=f"I-{caso_id}",
            seed_escenario=274000,
            cantidad_pedidos=cantidad,
            cantidad_objetivo=cantidad,
            patron_conflictivo="CONFLICTIVO" in estrato,
            banda_pedidos=estrato,
            estrato=estrato,
            modo=modo,
            estado=estado,
            error="" if estado == "OK" else "error",
            firma_plan="c0:v1[P1]",
            secuencia_generacion="P1",
            pedidos_tardios_estimados=tardios if estado == "OK" else None,
            tardanza_estimada_min=tardanza if estado == "OK" else None,
            sin_riesgo_temporal_estimado=(
                estado == "OK" and tardios == 0 and tardanza == 0.0
            ),
            costo_estimado=costo if estado == "OK" else None,
            costo_recalculado=costo if estado == "OK" else None,
            diferencia_costo_recalculado=0.0 if estado == "OK" else None,
            espera_ventana_estimada_min=0.0 if estado == "OK" else None,
            exceso_tolerancia_estimado_min=0.0 if estado == "OK" else None,
            duracion_operacion_estimada_min=100.0 if estado == "OK" else None,
            viajes_totales=1 if estado == "OK" else None,
            tiempo_plan_ms=1.0 if estado == "OK" else None,
        )

    def _caso(
        self,
        caso_id: str,
        *,
        grupo: str,
        estrato: str,
        cantidad: int,
        historico: tuple[int, float, float],
        quick: tuple[int, float, float],
        extension: tuple[int, float, float],
        full: tuple[int, float, float],
        greedy: tuple[int, float, float],
    ) -> list[RegistroHoldoutFull]:
        valores = {
            MODO_RL_HISTORICO: historico,
            MODO_RL_TEMPORAL_V4_QUICK: quick,
            MODO_RL_TEMPORAL_V4_EXTENSION: extension,
            MODO_RL_TEMPORAL_V4_FULL: full,
            MODO_GREEDY: greedy,
        }
        registros = [
            self._registro(
                caso_id, modo, grupo=grupo, estrato=estrato, cantidad=cantidad,
                tardios=v[0], tardanza=v[1], costo=v[2],
            )
            for modo, v in valores.items()
        ]
        por_modo = {r.modo: r for r in registros}
        salida = []
        for r in registros:
            greedy_r = por_modo[MODO_GREEDY]
            gap_greedy = 100.0 * (
                (r.costo_recalculado or 0.0) - (greedy_r.costo_recalculado or 0.0)
            ) / (greedy_r.costo_recalculado or 1.0)
            ext_r = por_modo[MODO_RL_TEMPORAL_V4_EXTENSION]
            gap_ext = 100.0 * (
                (r.costo_recalculado or 0.0) - (ext_r.costo_recalculado or 0.0)
            ) / (ext_r.costo_recalculado or 1.0)
            salida.append(
                RegistroHoldoutFull(
                    **{
                        **r.__dict__,
                        "comparacion_vs_historico": comparar_lexicografico(
                            r, por_modo[MODO_RL_HISTORICO]
                        ),
                        "comparacion_vs_quick": comparar_lexicografico(
                            r, por_modo[MODO_RL_TEMPORAL_V4_QUICK]
                        ),
                        "comparacion_vs_extension": comparar_lexicografico(r, ext_r),
                        "comparacion_vs_full": comparar_lexicografico(
                            r, por_modo[MODO_RL_TEMPORAL_V4_FULL]
                        ),
                        "comparacion_vs_greedy": comparar_lexicografico(r, greedy_r),
                        "gap_costo_vs_historico_pct": 0.0,
                        "gap_costo_vs_quick_pct": 0.0,
                        "gap_costo_vs_extension_pct": gap_ext,
                        "gap_costo_vs_greedy_pct": gap_greedy,
                    }
                )
            )
        return salida

    def _candidato(self, *, falla_b04: bool = False, mejora_12: bool = True):
        registros: list[RegistroHoldoutFull] = []
        b04_full = (1, 5.0, 100.0) if falla_b04 else (0, 0.0, 100.0)
        registros += self._caso(
            "B04_VENTANAS", grupo="CLASICO", estrato="CLASICO", cantidad=3,
            historico=(1, 5.0, 500.0), quick=(0, 0.0, 100.0),
            extension=(0, 0.0, 100.0), full=b04_full, greedy=(0, 0.0, 100.0),
        )
        for cid in ("B05_VOLCADOR", "B06_SPLIT"):
            registros += self._caso(
                cid, grupo="CLASICO", estrato="CLASICO", cantidad=3,
                historico=(0, 0.0, 120.0), quick=(0, 0.0, 100.0),
                extension=(0, 0.0, 100.0), full=(0, 0.0, 100.0),
                greedy=(0, 0.0, 110.0),
            )
        for i, cantidad in enumerate((3, 6, 9, 10)):
            registros += self._caso(
                f"G-{cantidad}", grupo="HOLDOUT_SINTETICO",
                estrato="GENERAL_3_5" if cantidad <= 5 else (
                    "GENERAL_6_8" if cantidad <= 8 else "GENERAL_9_10"
                ),
                cantidad=cantidad,
                historico=(0, 0.0, 110.0), quick=(0, 0.0, 105.0),
                extension=(0, 0.0, 100.0), full=(0, 0.0, 95.0),
                greedy=(0, 0.0, 100.0),
            )
        registros += self._caso(
            "G-11", grupo="HOLDOUT_SINTETICO", estrato="GENERAL_11_12",
            cantidad=11,
            historico=(2, 30.0, 300.0), quick=(1, 20.0, 250.0),
            extension=(1, 10.0, 200.0), full=(0, 0.0, 90.0),
            greedy=(0, 0.0, 100.0),
        )
        registros += self._caso(
            "G-12", grupo="HOLDOUT_SINTETICO", estrato="GENERAL_11_12",
            cantidad=12,
            historico=(2, 30.0, 300.0), quick=(1, 20.0, 250.0),
            extension=(1, 10.0, 200.0),
            full=((0, 0.0, 90.0) if mejora_12 else (1, 10.0, 190.0)),
            greedy=(0, 0.0, 100.0),
        )
        return registros

    def test_incluye_cinco_modos(self) -> None:
        self.assertEqual(len(ORDEN_MODOS), 5)
        self.assertIn(MODO_RL_TEMPORAL_V4_FULL, ORDEN_MODOS)

    def test_comparacion_prioriza_pedidos_tardios(self) -> None:
        candidato = self._registro(
            "C1", MODO_RL_TEMPORAL_V4_FULL, tardios=0, tardanza=100.0, costo=1000.0
        )
        referencia = self._registro(
            "C1", MODO_RL_TEMPORAL_V4_EXTENSION, tardios=1, tardanza=1.0, costo=1.0
        )
        self.assertEqual(comparar_lexicografico(candidato, referencia), "MEJOR")

    def test_comparacion_prioriza_tardanza_sobre_costo(self) -> None:
        candidato = self._registro(
            "C2", MODO_RL_TEMPORAL_V4_FULL, tardios=1, tardanza=5.0, costo=1000.0
        )
        referencia = self._registro(
            "C2", MODO_RL_TEMPORAL_V4_EXTENSION, tardios=1, tardanza=10.0, costo=1.0
        )
        self.assertEqual(comparar_lexicografico(candidato, referencia), "MEJOR")

    def test_generador_final_ocho_estratos_y_exactos_12(self) -> None:
        casos = crear_casos_sinteticos_finales(
            casos_por_estrato=2, seed_inicio=274000
        )
        self.assertEqual(len(casos), 2 * len(ESTRATOS_FASE_16D7))
        self.assertEqual(len({c.estrato for c in casos}), 8)
        self.assertTrue(all(c.instancia.seed_escenario >= 274000 for c in casos))
        exactos = [c for c in casos if len(c.instancia.pedidos) == 12]
        self.assertEqual(len(exactos), 2)

    def test_rechaza_semilla_anterior_a_274000(self) -> None:
        with self.assertRaises(ValueError):
            crear_casos_sinteticos_finales(
                casos_por_estrato=1, seed_inicio=273999
            )

    def test_veredicto_candidato_promocion_manual(self) -> None:
        registros = self._candidato()
        globales, _, segmentos = resumir_registros(registros)
        veredicto = construir_veredicto(registros, globales, segmentos)
        self.assertEqual(veredicto["estado"], "CANDIDATO_PROMOCION_MANUAL")
        self.assertFalse(veredicto["modelo_promovido"])

    def test_veredicto_prometedor_si_no_mejora_12(self) -> None:
        registros = self._candidato(mejora_12=False)
        globales, _, segmentos = resumir_registros(registros)
        veredicto = construir_veredicto(registros, globales, segmentos)
        self.assertEqual(veredicto["estado"], "PROMETEDOR_NO_PROMOVER")

    def test_veredicto_rechaza_si_falla_b04(self) -> None:
        registros = self._candidato(falla_b04=True)
        globales, _, segmentos = resumir_registros(registros)
        veredicto = construir_veredicto(registros, globales, segmentos)
        self.assertEqual(veredicto["estado"], "NO_RECOMENDADO_PARA_PROMOCION")

    def test_escribe_siete_salidas(self) -> None:
        registros = self._candidato()
        globales, estratos, segmentos = resumir_registros(registros)
        casos = resumir_casos(registros)
        veredicto = construir_veredicto(registros, globales, segmentos)
        with tempfile.TemporaryDirectory() as tmp:
            rutas = escribir_resultados(
                tmp,
                metadatos={"fase": "16D.9", "modelo_promovido": False},
                registros=registros,
                resumen_global=globales,
                resumen_estratos=estratos,
                resumen_segmentos=segmentos,
                casos=casos,
                clasicos={},
                veredicto=veredicto,
                semillas=[],
            )
            self.assertEqual(len(rutas), 7)
            self.assertTrue(all(Path(r).is_file() for r in rutas.values()))


if __name__ == "__main__":
    unittest.main()
