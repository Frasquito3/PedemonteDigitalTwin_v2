from __future__ import annotations

import unittest

from planner.algorithms.greedy import GreedyFeasiblePlanner
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import PedidoInput, Turno
from planner.routing.decoder import estimar_fin_viaje
from planner.routing.objective import (
    estimar_espera_cliente,
    tiempo_espera_respuesta_cliente_esperado_min,
)
from planner.routing.travel import (
    FuenteMatrizViaje,
    MatrizViaje,
)


class ClientWaitEstimationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configuracion = ConfiguracionPlanificacion()

        self.pedido = PedidoInput(
            pedido_id="P1",
            pedido_original_id="P1",
            numero_parte=1,
            total_partes=1,
            turno=Turno.MANANA,
            latitud=-32.831,
            longitud=-60.719,
            unidades_capacidad=1,
            requiere_volcador=False,
            tiene_ventana_especifica=True,
            hora_desde_min=480,
            hora_hasta_min=540,
        )

    def test_media_triangular_coincide_con_anylogic(
        self,
    ) -> None:
        self.assertAlmostEqual(
            tiempo_espera_respuesta_cliente_esperado_min(
                self.configuracion
            ),
            10.0 / 3.0,
        )

    def test_llegada_temprana_suma_ventana_y_respuesta(
        self,
    ) -> None:
        estimacion = estimar_espera_cliente(
            self.pedido,
            minuto_llegada=470.0,
            configuracion=self.configuracion,
        )

        self.assertAlmostEqual(
            estimacion.tiempo_espera_ventana_min,
            10.0,
        )
        self.assertAlmostEqual(
            estimacion.tiempo_espera_respuesta_min,
            10.0 / 3.0,
        )
        self.assertAlmostEqual(
            estimacion.minuto_inicio_descarga,
            480.0 + 10.0 / 3.0,
        )

    def test_llegada_en_ventana_solo_espera_respuesta(
        self,
    ) -> None:
        estimacion = estimar_espera_cliente(
            self.pedido,
            minuto_llegada=500.0,
            configuracion=self.configuracion,
        )

        self.assertEqual(
            estimacion.tiempo_espera_ventana_min,
            0.0,
        )
        self.assertAlmostEqual(
            estimacion.tiempo_espera_respuesta_min,
            10.0 / 3.0,
        )
        self.assertAlmostEqual(
            estimacion.minuto_inicio_descarga,
            500.0 + 10.0 / 3.0,
        )

    def test_llegada_tardia_no_agrega_respuesta(
        self,
    ) -> None:
        estimacion = estimar_espera_cliente(
            self.pedido,
            minuto_llegada=541.0,
            configuracion=self.configuracion,
        )

        self.assertEqual(
            estimacion.tiempo_espera_ventana_min,
            0.0,
        )
        self.assertEqual(
            estimacion.tiempo_espera_respuesta_min,
            0.0,
        )
        self.assertEqual(
            estimacion.minuto_inicio_descarga,
            541.0,
        )

    def test_greedy_y_decoder_incluyen_la_misma_espera(
        self,
    ) -> None:
        pedido_turno = PedidoInput(
            pedido_id="P1",
            pedido_original_id="P1",
            numero_parte=1,
            total_partes=1,
            turno=Turno.MANANA,
            latitud=-32.831,
            longitud=-60.719,
            unidades_capacidad=1,
            requiere_volcador=False,
            tiene_ventana_especifica=False,
            hora_desde_min=450,
            hora_hasta_min=720,
        )

        matriz = MatrizViaje(
            nodo_ids=["DEPOT", "P1"],
            indice_por_id={
                "DEPOT": 0,
                "P1": 1,
            },
            distancia_metros=[
                [0.0, 1000.0],
                [2000.0, 0.0],
            ],
            tiempo_base_min=[
                [0.0, 10.0],
                [20.0, 0.0],
            ],
            fuente=FuenteMatrizViaje.VIAL_CACHE,
            version_fuente="prueba-vial-v1",
        )

        pedidos_por_id = {
            "P1": pedido_turno,
        }

        fin_decoder = estimar_fin_viaje(
            pedido_ids=["P1"],
            minuto_inicio=450.0,
            pedidos_por_id=pedidos_por_id,
            matriz=matriz,
            configuracion=self.configuracion,
        )

        fin_greedy = GreedyFeasiblePlanner(
            configuracion=self.configuracion
        )._estimar_fin_viaje(
            pedido_ids=["P1"],
            minuto_inicio=450.0,
            pedidos_por_id=pedidos_por_id,
            matriz=matriz,
        )

        self.assertAlmostEqual(
            fin_decoder,
            495.00392156862745,
        )
        self.assertAlmostEqual(
            fin_greedy,
            fin_decoder,
        )

    def test_configuracion_rechaza_triangular_invalida(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "min <= moda <= max",
        ):
            ConfiguracionPlanificacion(
                cliente_espera_respuesta_min=4.0,
                cliente_espera_respuesta_moda=2.0,
                cliente_espera_respuesta_max=8.0,
            )


if __name__ == "__main__":
    unittest.main()
