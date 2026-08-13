from __future__ import annotations

import math
import unittest

from dataclasses import replace

from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    Turno,
    ViajePlan,
)

from planner.integration.alpyne_codec import (
    ErrorContratoAlpyne,
    PROTOCOL_VERSION,
    codificar_instancia_alpyne,
    codificar_plan_alpyne,
)


class AlpyneCodecTest(
    unittest.TestCase
):
    def crear_instancia(
        self,
    ) -> InstanciaTurno:
        pedidos = [
            PedidoInput(
                pedido_id="P001",
                pedido_original_id="P001",
                numero_parte=1,
                total_partes=1,
                turno=Turno.MANANA,
                latitud=-32.848,
                longitud=-60.721,
                unidades_capacidad=3,
                requiere_volcador=False,
                tiene_ventana_especifica=False,
                hora_desde_min=450,
                hora_hasta_min=720,
            ),
            PedidoInput(
                pedido_id="P002",
                pedido_original_id="P002",
                numero_parte=1,
                total_partes=1,
                turno=Turno.MANANA,
                latitud=-32.851,
                longitud=-60.724,
                unidades_capacidad=5,
                requiere_volcador=False,
                tiene_ventana_especifica=True,
                hora_desde_min=500,
                hora_hasta_min=650,
            ),
        ]

        return InstanciaTurno(
            instancia_id="TEST-ALPYNE",
            fecha_operacion="2026-08-13",
            turno=Turno.MANANA,
            pedidos=pedidos,
            lat_corralon=-32.8495006,
            lon_corralon=-60.722653,
            capacidad_camion=8,
            cantidad_camiones=2,
            hora_inicio_turno_min=450,
            hora_fin_objetivo_min=720,
            hora_fin_tolerancia_min=735,
            seed_escenario=100,
            seed_ejecucion=1_000_100,
        )

    def crear_plan(
        self,
    ) -> PlanTurno:
        return PlanTurno(
            instancia_id="TEST-ALPYNE",
            algoritmo=(
                AlgoritmoPlanificacion
                .GREEDY
            ),
            camiones=[
                PlanCamion(
                    camion_id=0,
                    viajes=[
                        ViajePlan(
                            numero_viaje=1,
                            pedido_ids=[
                                "P001",
                                "P002",
                            ],
                        )
                    ],
                ),
                PlanCamion(
                    camion_id=1,
                    viajes=[],
                ),
            ],
            costo_estimado=123.5,
            tiempo_computo_ms=2.25,
        )

    def test_codifica_cabecera_instancia(
        self,
    ) -> None:
        vector = (
            codificar_instancia_alpyne(
                self.crear_instancia()
            )
        )

        self.assertEqual(
            vector[:8],
            [
                float(
                    PROTOCOL_VERSION
                ),
                0.0,
                2.0,
                8.0,
                2.0,
                -32.8495006,
                -60.722653,
                0.0,
            ],
        )

        self.assertEqual(
            len(
                vector
            ),
            28,
        )

    def test_pedido_sin_ventana_usa_menos_uno(
        self,
    ) -> None:
        vector = (
            codificar_instancia_alpyne(
                self.crear_instancia()
            )
        )

        base_primer_pedido = 8

        self.assertEqual(
            vector[
                base_primer_pedido + 8
            ],
            -1.0,
        )

        self.assertEqual(
            vector[
                base_primer_pedido + 9
            ],
            -1.0,
        )

    def test_pedido_con_ventana_conserva_limites(
        self,
    ) -> None:
        vector = (
            codificar_instancia_alpyne(
                self.crear_instancia()
            )
        )

        base_segundo_pedido = (
            8 + 10
        )

        self.assertEqual(
            vector[
                base_segundo_pedido + 8
            ],
            500.0,
        )

        self.assertEqual(
            vector[
                base_segundo_pedido + 9
            ],
            650.0,
        )

    def test_partes_comparten_indice_original(
        self,
    ) -> None:
        parte_1 = PedidoInput(
            pedido_id="P010-P1",
            pedido_original_id="P010",
            numero_parte=1,
            total_partes=2,
            turno=Turno.MANANA,
            latitud=-32.848,
            longitud=-60.721,
            unidades_capacidad=8,
            requiere_volcador=True,
            tiene_ventana_especifica=False,
            hora_desde_min=450,
            hora_hasta_min=720,
        )

        parte_2 = replace(
            parte_1,
            pedido_id="P010-P2",
            numero_parte=2,
            unidades_capacidad=2,
        )

        instancia = replace(
            self.crear_instancia(),
            pedidos=[
                parte_1,
                parte_2,
            ],
        )

        vector = (
            codificar_instancia_alpyne(
                instancia
            )
        )

        original_1 = vector[
            8 + 1
        ]

        original_2 = vector[
            8 + 10 + 1
        ]

        self.assertEqual(
            original_1,
            original_2,
        )

    def test_codifica_plan_completo(
        self,
    ) -> None:
        vector = codificar_plan_alpyne(
            self.crear_instancia(),
            self.crear_plan(),
        )

        self.assertEqual(
            vector[:5],
            [
                float(
                    PROTOCOL_VERSION
                ),
                2.0,
                2.0,
                123.5,
                2.25,
            ],
        )

        self.assertEqual(
            vector[5:],
            [
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                2.0,
                1.0,
            ],
        )

    def test_rechaza_plan_invalido(
        self,
    ) -> None:
        plan = self.crear_plan()

        plan.camiones[0].viajes[0].pedido_ids = [
            "P001",
        ]

        with self.assertRaises(
            ErrorContratoAlpyne
        ):
            codificar_plan_alpyne(
                self.crear_instancia(),
                plan,
            )

    def test_rechaza_numeracion_no_consecutiva(
        self,
    ) -> None:
        plan = self.crear_plan()

        plan.camiones[0].viajes[0].numero_viaje = 2

        with self.assertRaises(
            ErrorContratoAlpyne
        ):
            codificar_plan_alpyne(
                self.crear_instancia(),
                plan,
            )

    def test_rechaza_costo_no_finito(
        self,
    ) -> None:
        plan = self.crear_plan()

        plan.costo_estimado = (
            math.inf
        )

        with self.assertRaises(
            ErrorContratoAlpyne
        ):
            codificar_plan_alpyne(
                self.crear_instancia(),
                plan,
            )


if __name__ == "__main__":
    unittest.main()