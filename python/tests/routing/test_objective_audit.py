from __future__ import annotations

import unittest

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    Turno,
    ViajePlan,
)
from planner.routing.objective import (
    VERSION_AUDITORIA_COSTO,
    evaluar_plan_estimado,
    serializar_auditoria_estimacion,
)
from planner.routing.travel import (
    FuenteMatrizViaje,
    MatrizViaje,
)


class ObjectiveAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configuracion = ConfiguracionPlanificacion()

        self.instancia = InstanciaTurno(
            instancia_id="AUDITORIA-001",
            fecha_operacion="2026-08-24",
            turno=Turno.MANANA,
            pedidos=[
                PedidoInput(
                    pedido_id="P1",
                    pedido_original_id="P1",
                    numero_parte=1,
                    total_partes=1,
                    turno=Turno.MANANA,
                    latitud=-32.831,
                    longitud=-60.719,
                    unidades_capacidad=2,
                    requiere_volcador=False,
                    tiene_ventana_especifica=True,
                    hora_desde_min=470,
                    hora_hasta_min=600,
                )
            ],
            lat_corralon=-32.8495006,
            lon_corralon=-60.722653,
            capacidad_camion=8,
            cantidad_camiones=2,
            hora_inicio_turno_min=450,
            hora_fin_objetivo_min=720,
            hora_fin_tolerancia_min=735,
            seed_escenario=6001,
            seed_ejecucion=1006001,
        )

        self.plan = PlanTurno(
            instancia_id=self.instancia.instancia_id,
            algoritmo=AlgoritmoPlanificacion.GREEDY,
            camiones=[
                PlanCamion(
                    camion_id=0,
                    viajes=[
                        ViajePlan(
                            numero_viaje=1,
                            pedido_ids=["P1"],
                        )
                    ],
                ),
                PlanCamion(
                    camion_id=1,
                    viajes=[],
                ),
            ],
        )

        self.matriz = MatrizViaje(
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

    def test_expone_tiempos_y_costos_sin_cambiar_total(
        self,
    ) -> None:
        estimacion = evaluar_plan_estimado(
            self.instancia,
            self.plan,
            self.matriz,
            self.configuracion,
        )

        self.assertAlmostEqual(
            estimacion.tiempo_carga_total_min,
            4.9411764705882355,
        )
        self.assertAlmostEqual(
            estimacion.tiempo_viaje_total_min,
            36.0,
        )
        self.assertAlmostEqual(
            estimacion.tiempo_espera_ventana_total_min,
            3.058823529411768,
        )
        self.assertAlmostEqual(
            estimacion
            .tiempo_espera_respuesta_cliente_total_min,
            10.0 / 3.0,
        )
        self.assertAlmostEqual(
            estimacion.tiempo_descarga_total_min,
            3.4,
        )
        self.assertAlmostEqual(
            estimacion.duracion_operacion_min,
            50.733333333333334,
        )
        self.assertAlmostEqual(
            estimacion.distancia_total_km,
            3.0,
        )
        self.assertAlmostEqual(
            estimacion.costo_operacion,
            50.733333333333334,
        )
        self.assertAlmostEqual(
            estimacion.costo_distancia,
            6.0,
        )
        self.assertAlmostEqual(
            estimacion.costo_viajes,
            5.0,
        )
        self.assertAlmostEqual(
            estimacion.costo_desbalance,
            25.366666666666667,
        )
        self.assertAlmostEqual(
            estimacion.costo_total,
            87.1,
        )

    def test_serializacion_incluye_espera_esperada_cliente(
        self,
    ) -> None:
        estimacion = evaluar_plan_estimado(
            self.instancia,
            self.plan,
            self.matriz,
            self.configuracion,
        )

        resumen = serializar_auditoria_estimacion(
            estimacion
        )

        self.assertIn(
            f"version={VERSION_AUDITORIA_COSTO}",
            resumen,
        )
        self.assertIn(
            "espera_respuesta_cliente_min=3.333333",
            resumen,
        )
        self.assertIn(
            "costo_operacion=50.733333",
            resumen,
        )
        self.assertIn(
            "costo_total=87.100000",
            resumen,
        )


if __name__ == "__main__":
    unittest.main()
