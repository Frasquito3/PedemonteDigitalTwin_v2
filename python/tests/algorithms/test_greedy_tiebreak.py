from __future__ import annotations
import unittest
from planner.algorithms.greedy import generar_plan_greedy
from planner.core.schema import InstanciaTurno, PedidoInput, Turno
from planner.routing.travel import ProveedorHaversineAjustado
COORDENADA_NORTE = (-32.831, -60.719)
COORDENADA_ESTE = (-32.8595006, -60.702653)
COORDENADA_CERCANA = (-32.841, -60.721)

class GreedyTieBreakTest(unittest.TestCase):

    def _pedido(self, pedido_id: str, coordenada: tuple[float, float]) -> PedidoInput:
        return PedidoInput(pedido_id=pedido_id, pedido_original_id=pedido_id, numero_parte=1, total_partes=1, turno=Turno.MANANA, latitud=coordenada[0], longitud=coordenada[1], unidades_capacidad=8, requiere_volcador=False, tiene_ventana_especifica=False, hora_desde_min=450, hora_hasta_min=720, cliente=pedido_id, direccion=f'Dirección {pedido_id}', barrio='Prueba desempate operacional', observaciones='')

    def _instancia(self, instancia_id: str, ids: dict[str, str]) -> InstanciaTurno:
        return InstanciaTurno(instancia_id=instancia_id, fecha_operacion='2026-08-24', turno=Turno.MANANA, pedidos=[self._pedido(ids['NORTE'], COORDENADA_NORTE), self._pedido(ids['ESTE'], COORDENADA_ESTE), self._pedido(ids['CERCANA'], COORDENADA_CERCANA)], lat_corralon=-32.8495006, lon_corralon=-60.722653, capacidad_camion=8, cantidad_camiones=2, hora_inicio_turno_min=450, hora_fin_objetivo_min=720, hora_fin_tolerancia_min=735, seed_escenario=15103, seed_ejecucion=25103)

    def _firma_semantica(self, plan, semantica_por_id: dict[str, str]) -> tuple:
        return tuple((tuple((tuple((semantica_por_id[pedido_id] for pedido_id in viaje.pedido_ids)) for viaje in camion.viajes)) for camion in sorted(plan.camiones, key=lambda actual: actual.camion_id)))

    def test_renombrar_pedidos_no_cambia_la_decision_operativa(self) -> None:
        ids_numericos = {'NORTE': 'MULTIVIAJE-001', 'ESTE': 'MULTIVIAJE-002', 'CERCANA': 'MULTIVIAJE-003'}
        ids_lexicos_invertidos = {'NORTE': 'Z-NORTE', 'ESTE': 'M-ESTE', 'CERCANA': 'A-CERCANA'}
        instancia_a = self._instancia('GREEDY-TIEBREAK-A', ids_numericos)
        instancia_b = self._instancia('GREEDY-TIEBREAK-B', ids_lexicos_invertidos)
        proveedor = ProveedorHaversineAjustado()
        plan_a = generar_plan_greedy(instancia_a, proveedor_viaje=proveedor)
        plan_b = generar_plan_greedy(instancia_b, proveedor_viaje=proveedor)
        semantica_a = {pedido_id: nombre for nombre, pedido_id in ids_numericos.items()}
        semantica_b = {pedido_id: nombre for nombre, pedido_id in ids_lexicos_invertidos.items()}
        self.assertEqual(self._firma_semantica(plan_a, semantica_a), self._firma_semantica(plan_b, semantica_b))
        self.assertAlmostEqual(plan_a.costo_estimado, plan_b.costo_estimado, places=9)
if __name__ == '__main__':
    unittest.main()
