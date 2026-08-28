from __future__ import annotations
import unittest
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import AlgoritmoPlanificacion, InstanciaTurno, PedidoInput, PlanCamion, PlanTurno, Turno, ViajePlan
from planner.routing.operational import ReservaEmpleadosCorralon, simular_plan_operativo_estimado, tiempo_carga_estimado_min
from planner.routing.travel import FuenteMatrizViaje, MatrizViaje

class LoadingResourceEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.configuracion = ConfiguracionPlanificacion(cliente_espera_respuesta_min=0.0, cliente_espera_respuesta_moda=0.0, cliente_espera_respuesta_max=0.0, descarga_setup_min=0.0, descarga_min_por_unidad=0.0, trafico_factor_normal=1.0, trafico_factor_pico_manana=1.0, trafico_factor_pico_tarde=1.0)

    def _pedido(self, pedido_id: str, unidades: int=1) -> PedidoInput:
        return PedidoInput(pedido_id=pedido_id, pedido_original_id=pedido_id, numero_parte=1, total_partes=1, turno=Turno.MANANA, latitud=-32.84, longitud=-60.72, unidades_capacidad=unidades, requiere_volcador=False, tiene_ventana_especifica=False, hora_desde_min=450, hora_hasta_min=720)

    def _instancia(self, pedidos: list[PedidoInput]) -> InstanciaTurno:
        return InstanciaTurno(instancia_id='CARGA-DINAMICA', fecha_operacion='2026-08-24', turno=Turno.MANANA, pedidos=pedidos, lat_corralon=-32.8495006, lon_corralon=-60.722653, capacidad_camion=8, cantidad_camiones=2, hora_inicio_turno_min=450, hora_fin_objetivo_min=720, hora_fin_tolerancia_min=735, seed_escenario=1, seed_ejecucion=2)

    def _matriz(self, pedidos: list[PedidoInput], tiempos: dict[tuple[str, str], float] | None=None) -> MatrizViaje:
        ids = ['DEPOT'] + [pedido.pedido_id for pedido in pedidos]
        indice = {nodo_id: posicion for posicion, nodo_id in enumerate(ids)}
        n = len(ids)
        distancias = [[0.0 for _ in range(n)] for _ in range(n)]
        tiempos_base = [[0.0 for _ in range(n)] for _ in range(n)]
        for origen in ids:
            for destino in ids:
                if origen == destino:
                    continue
                tiempo = tiempos.get((origen, destino), 10.0) if tiempos is not None else 10.0
                i = indice[origen]
                j = indice[destino]
                tiempos_base[i][j] = tiempo
                distancias[i][j] = tiempo * 1000.0
        return MatrizViaje(nodo_ids=ids, indice_por_id=indice, distancia_metros=distancias, tiempo_base_min=tiempos_base, fuente=FuenteMatrizViaje.VIAL_CACHE, version_fuente='test-carga-dinamica')

    def test_un_camion_activo_y_otro_sin_trabajo_usa_cuatro_personas(self) -> None:
        pedidos = [self._pedido('P1', unidades=2)]
        instancia = self._instancia(pedidos)
        plan = PlanTurno(instancia_id=instancia.instancia_id, algoritmo=AlgoritmoPlanificacion.GREEDY, camiones=[PlanCamion(camion_id=0, viajes=[ViajePlan(numero_viaje=1, pedido_ids=['P1'])]), PlanCamion(camion_id=1)])
        resultado = simular_plan_operativo_estimado(instancia, plan, self._matriz(pedidos), self.configuracion)
        carga = resultado.cargas[0]
        self.assertEqual(carga.personas_estimadas, 4)
        self.assertEqual(carga.empleados_corralon_asignados, 2)
        self.assertTrue(carga.chofer_ayudante_asignado)
        self.assertAlmostEqual(carga.duracion_min, tiempo_carga_estimado_min(2, self.configuracion, personas=4))

    def test_dos_camiones_cargando_simultaneamente_reciben_dos_personas(self) -> None:
        pedidos = [self._pedido('P1'), self._pedido('P2')]
        instancia = self._instancia(pedidos)
        plan = PlanTurno(instancia_id=instancia.instancia_id, algoritmo=AlgoritmoPlanificacion.GREEDY, camiones=[PlanCamion(camion_id=0, viajes=[ViajePlan(1, ['P1'])]), PlanCamion(camion_id=1, viajes=[ViajePlan(1, ['P2'])])])
        resultado = simular_plan_operativo_estimado(instancia, plan, self._matriz(pedidos), self.configuracion)
        self.assertEqual(len(resultado.cargas), 2)
        self.assertEqual([carga.personas_estimadas for carga in resultado.cargas], [2, 2])
        self.assertEqual([carga.empleados_corralon_asignados for carga in resultado.cargas], [1, 1])

    def test_reserva_de_proveedor_reduce_empleados_disponibles(self) -> None:
        pedidos = [self._pedido('P1', unidades=2)]
        instancia = self._instancia(pedidos)
        plan = PlanTurno(instancia_id=instancia.instancia_id, algoritmo=AlgoritmoPlanificacion.GREEDY, camiones=[PlanCamion(camion_id=0, viajes=[ViajePlan(1, ['P1'])]), PlanCamion(camion_id=1)])
        resultado = simular_plan_operativo_estimado(instancia, plan, self._matriz(pedidos), self.configuracion, reservas_empleados=(ReservaEmpleadosCorralon(minuto_inicio=440.0, minuto_fin=470.0, empleados_ocupados=2),))
        carga = resultado.cargas[0]
        self.assertEqual(carga.personas_estimadas, 2)
        self.assertEqual(carga.empleados_corralon_asignados, 0)
        self.assertTrue(carga.chofer_ayudante_asignado)
if __name__ == '__main__':
    unittest.main()
