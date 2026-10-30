import unittest
from typing import Any

from planner.core.schema import PedidoInput, Turno
from planner.domain.preprocess import preprocesar_pedidos


def crear_pedido(pedido_id: str, unidades: int, volcador: bool = False) -> PedidoInput:
    return PedidoInput(
        pedido_id=pedido_id,
        pedido_original_id=pedido_id,
        numero_parte=1,
        total_partes=1,
        turno=Turno.MANANA,
        latitud=-32.85,
        longitud=-60.72,
        unidades_capacidad=unidades,
        requiere_volcador=volcador,
        tiene_ventana_especifica=False,
        hora_desde_min=450,
        hora_hasta_min=720,
    )


class PreprocessTest(unittest.TestCase):
    def test_no_divide_pedido_igual_a_capacidad(self) -> None:
        resultado = preprocesar_pedidos([crear_pedido("P8", 8)], capacidad_camion=8)
        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0].pedido_id, "P8")
        self.assertEqual(resultado[0].unidades_capacidad, 8)
        self.assertEqual(resultado[0].total_partes, 1)

    def test_divide_nueve_en_ocho_y_uno(self) -> None:
        resultado = preprocesar_pedidos([crear_pedido("P9", 9)], capacidad_camion=8)
        self.assertEqual([pedido.unidades_capacidad for pedido in resultado], [8, 1])
        self.assertEqual([pedido.pedido_id for pedido in resultado], ["P9-P1", "P9-P2"])

    def test_divide_diecisiete_en_tres_partes(self) -> None:
        resultado = preprocesar_pedidos([crear_pedido("P17", 17)], capacidad_camion=8)
        self.assertEqual([pedido.unidades_capacidad for pedido in resultado], [8, 8, 1])
        self.assertTrue(all(pedido.pedido_original_id == "P17" for pedido in resultado))

    def test_todas_las_partes_heredan_volcador(self) -> None:
        resultado = preprocesar_pedidos([crear_pedido("PV", 11, volcador=True)], capacidad_camion=8)
        self.assertTrue(all(pedido.requiere_volcador for pedido in resultado))


class PreprocessMixedLoadTest(unittest.TestCase):
    def _split(self, total: int, volcador: Any, *, id: str = "MX"):
        return preprocesar_pedidos(
            [crear_pedido(id, total, volcador=volcador > 0)],
            capacidad_camion=8,
            unidades_volcador_por_pedido={id: volcador},
        )

    def _firma(self, pedidos):
        return [(p.pedido_id, p.unidades_capacidad, p.requiere_volcador) for p in pedidos]

    def test_mixto_nueve_con_cuatro_volcador(self):
        partes = self._split(9, 4)
        self.assertEqual(
            self._firma(partes),
            [("MX-P1", 5, False), ("MX-P2", 4, True)],
        )
        self.assertEqual([p.numero_parte for p in partes], [1, 2])
        self.assertTrue(all(p.total_partes == 2 for p in partes))
        self.assertTrue(all(p.pedido_original_id == "MX" for p in partes))

    def test_mixto_ocho_con_tres_volcador_se_divide(self):
        self.assertEqual(
            self._firma(self._split(8, 3)),
            [("MX-P1", 5, False), ("MX-P2", 3, True)],
        )

    def test_mixto_veinte_con_once_volcador(self):
        self.assertEqual(
            self._firma(self._split(20, 11)),
            [
                ("MX-P1", 8, False),
                ("MX-P2", 1, False),
                ("MX-P3", 8, True),
                ("MX-P4", 3, True),
            ],
        )

    def test_sin_volcador_conserva_unico_id(self):
        self.assertEqual(
            self._firma(self._split(6, 0)),
            [("MX", 6, False)],
        )

    def test_todo_volcador_conserva_unico_id(self):
        self.assertEqual(
            self._firma(self._split(6, 6)),
            [("MX", 6, True)],
        )

    def test_todo_volcador_conserva_split_heredado(self):
        self.assertEqual(
            self._firma(self._split(11, 11)),
            [("MX-P1", 8, True), ("MX-P2", 3, True)],
        )

    def test_multiples_pedidos_con_mapa_parcial(self):
        partes = preprocesar_pedidos(
            [crear_pedido("A", 9, volcador=True), crear_pedido("B", 3, volcador=False)],
            8,
            unidades_volcador_por_pedido={"A": 4},
        )
        self.assertEqual(
            self._firma(partes),
            [
                ("A-P1", 5, False),
                ("A-P2", 4, True),
                ("B", 3, False),
            ],
        )

    def test_preserva_metadatos_de_cada_parte(self):
        original = crear_pedido("DIRECCION", 9, volcador=True)
        original.cliente = "Cliente real"
        original.direccion = "Calle 123"
        original.barrio = "Centro"
        original.observaciones = "Entrega acordada"
        original.tiene_ventana_especifica = True
        original.hora_desde_min = 510
        original.hora_hasta_min = 580

        partes = preprocesar_pedidos(
            [original],
            8,
            unidades_volcador_por_pedido={"DIRECCION": 4},
        )
        self.assertEqual(
            [
                (
                    p.cliente,
                    p.direccion,
                    p.barrio,
                    p.observaciones,
                    p.hora_desde_min,
                    p.hora_hasta_min,
                )
                for p in partes
            ],
            [("Cliente real", "Calle 123", "Centro", "Entrega acordada", 510, 580)] * 2,
        )
        self.assertFalse(partes[0].requiere_volcador)
        self.assertTrue(partes[1].requiere_volcador)
        self.assertEqual(sum(p.unidades_capacidad for p in partes), 9)

    def test_rechaza_cantidad_negativa(self):
        with self.assertRaisesRegex(ValueError, "entre 0 y 9"):
            self._split(9, -1)

    def test_rechaza_cantidad_superior_total(self):
        with self.assertRaisesRegex(ValueError, "entre 0 y 9"):
            self._split(9, 10)

    def test_rechaza_cantidad_fraccionaria(self):
        with self.assertRaisesRegex(ValueError, "debe ser un entero"):
            self._split(9, 1.5)

    def test_rechaza_booleano_como_cantidad(self):
        with self.assertRaisesRegex(ValueError, "debe ser un entero"):
            self._split(9, True)

    def test_rechaza_checkbox_desmarcado_con_volcador(self):
        with self.assertRaisesRegex(ValueError, "no coincide"):
            preprocesar_pedidos(
                [crear_pedido("X", 9, volcador=False)],
                8,
                unidades_volcador_por_pedido={"X": 4},
            )

    def test_rechaza_checkbox_marcado_con_cero(self):
        with self.assertRaisesRegex(ValueError, "no coincide"):
            preprocesar_pedidos(
                [crear_pedido("X", 9, volcador=True)],
                8,
                unidades_volcador_por_pedido={"X": 0},
            )

    def test_rechaza_mapa_con_pedido_inexistente(self):
        with self.assertRaisesRegex(ValueError, "IDs desconocidos"):
            preprocesar_pedidos(
                [crear_pedido("X", 2)],
                8,
                unidades_volcador_por_pedido={"FANTASMA": 1},
            )

    def test_rechaza_ids_duplicados_producidos_por_split(self):
        with self.assertRaisesRegex(ValueError, "IDs duplicados"):
            preprocesar_pedidos(
                [crear_pedido("X", 9, volcador=True), crear_pedido("X-P1", 1)],
                8,
                unidades_volcador_por_pedido={"X": 4},
            )

    def test_rechaza_preprocesamiento_doble(self):
        partes = self._split(9, 4)
        with self.assertRaisesRegex(ValueError, "dos veces"):
            preprocesar_pedidos(partes, 8)

    def test_instancia_no_muta_original(self):
        from dataclasses import replace

        from planner.domain.preprocess import preprocesar_instancia

        from tests.fixtures import crear_instancia_demo

        demo = crear_instancia_demo()
        pedido = replace(
            demo.pedidos[0],
            pedido_id="MIX",
            pedido_original_id="MIX",
            unidades_capacidad=9,
            requiere_volcador=True,
            numero_parte=1,
            total_partes=1,
        )
        instancia = replace(demo, pedidos=[pedido])
        salida = preprocesar_instancia(
            instancia,
            unidades_volcador_por_pedido={"MIX": 4},
        )

        self.assertIsNot(instancia, salida)
        self.assertEqual(len(instancia.pedidos), 1)
        self.assertEqual(
            [(p.unidades_capacidad, p.requiere_volcador) for p in salida.pedidos],
            [(5, False), (4, True)],
        )


if __name__ == "__main__":
    unittest.main()
