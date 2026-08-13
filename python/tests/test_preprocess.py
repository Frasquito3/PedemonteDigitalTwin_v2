import unittest

from planner.preprocess import (
    preprocesar_pedidos,
)

from planner.schema import (
    PedidoInput,
    Turno,
)


def crear_pedido(
    pedido_id: str,
    unidades: int,
    volcador: bool = False,
) -> PedidoInput:
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
    def test_no_divide_pedido_igual_a_capacidad(
        self,
    ) -> None:
        resultado = preprocesar_pedidos(
            [crear_pedido("P8", 8)],

            capacidad_camion=8,
        )

        self.assertEqual(
            len(resultado),
            1,
        )

        self.assertEqual(
            resultado[0].pedido_id,
            "P8",
        )

        self.assertEqual(
            resultado[0].unidades_capacidad,
            8,
        )

        self.assertEqual(
            resultado[0].total_partes,
            1,
        )

    def test_divide_nueve_en_ocho_y_uno(
        self,
    ) -> None:
        resultado = preprocesar_pedidos(
            [crear_pedido("P9", 9)],

            capacidad_camion=8,
        )

        self.assertEqual(
            [
                pedido.unidades_capacidad
                for pedido in resultado
            ],

            [8, 1],
        )

        self.assertEqual(
            [
                pedido.pedido_id
                for pedido in resultado
            ],

            ["P9-P1", "P9-P2"],
        )

    def test_divide_diecisiete_en_tres_partes(
        self,
    ) -> None:
        resultado = preprocesar_pedidos(
            [crear_pedido("P17", 17)],

            capacidad_camion=8,
        )

        self.assertEqual(
            [
                pedido.unidades_capacidad
                for pedido in resultado
            ],

            [8, 8, 1],
        )

        self.assertTrue(
            all(
                pedido.pedido_original_id
                == "P17"

                for pedido in resultado
            )
        )

    def test_todas_las_partes_heredan_volcador(
        self,
    ) -> None:
        resultado = preprocesar_pedidos(
            [
                crear_pedido(
                    "PV",
                    11,
                    volcador=True,
                )
            ],

            capacidad_camion=8,
        )

        self.assertTrue(
            all(
                pedido.requiere_volcador

                for pedido in resultado
            )
        )


if __name__ == "__main__":
    unittest.main()