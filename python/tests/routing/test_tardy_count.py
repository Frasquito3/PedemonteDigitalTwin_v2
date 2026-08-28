from __future__ import annotations
import unittest
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import AlgoritmoPlanificacion, InstanciaTurno, PedidoInput, PlanCamion, PlanTurno, Turno, ViajePlan
from planner.routing.objective import evaluar_plan_estimado, serializar_auditoria_estimacion
from planner.routing.travel import construir_matriz_viaje

class TestTardyCount(unittest.TestCase):

    def _evaluar(self, hora_hasta_min: int):
        pedido = PedidoInput(pedido_id='P1', pedido_original_id='P1', numero_parte=1, total_partes=1, turno=Turno.MANANA, latitud=-32.8495006, longitud=-60.722653, unidades_capacidad=1, requiere_volcador=False, tiene_ventana_especifica=True, hora_desde_min=450, hora_hasta_min=hora_hasta_min)
        instancia = InstanciaTurno(instancia_id='TEST-TARDIOS', fecha_operacion='2026-08-27', turno=Turno.MANANA, pedidos=[pedido], lat_corralon=-32.8495006, lon_corralon=-60.722653, capacidad_camion=8, cantidad_camiones=2, hora_inicio_turno_min=450, hora_fin_objetivo_min=720, hora_fin_tolerancia_min=735, seed_escenario=1, seed_ejecucion=2)
        plan = PlanTurno(instancia_id=instancia.instancia_id, algoritmo=AlgoritmoPlanificacion.MANUAL_TEST, camiones=[PlanCamion(camion_id=0, viajes=[ViajePlan(numero_viaje=1, pedido_ids=['P1'])]), PlanCamion(camion_id=1)])
        configuracion = ConfiguracionPlanificacion()
        matriz = construir_matriz_viaje(instancia, configuracion)
        return evaluar_plan_estimado(instancia, plan, matriz, configuracion)

    def test_cuenta_llegada_tardia(self) -> None:
        estimacion = self._evaluar(hora_hasta_min=451)
        self.assertEqual(estimacion.pedidos_tardios, 1)
        self.assertGreater(estimacion.tardanza_total_min, 0.0)
        self.assertIn('pedidos_tardios=1', serializar_auditoria_estimacion(estimacion))

    def test_no_cuenta_llegada_en_horario(self) -> None:
        estimacion = self._evaluar(hora_hasta_min=720)
        self.assertEqual(estimacion.pedidos_tardios, 0)
        self.assertEqual(estimacion.tardanza_total_min, 0.0)
if __name__ == '__main__':
    unittest.main()
