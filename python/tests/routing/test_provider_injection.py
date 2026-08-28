from __future__ import annotations
import unittest
from dataclasses import dataclass, field
from planner.algorithms.ga import ConfiguracionGA, generar_plan_ga
from planner.algorithms.greedy import generar_plan_greedy
from planner.algorithms.random_feasible import generar_plan_random
from planner.core.config import ConfiguracionPlanificacion
from planner.integration.planner_selector import ModoPlanificacion, SelectorPlanificadores
from planner.routing.travel import Coordenada, FuenteMatrizViaje, ResultadoTramoViaje
from tests.fixtures import crear_instancia_demo

@dataclass
class ProveedorViajePrueba:
    distancia_metros: float = 2500.0
    tiempo_base_min: float = 6.0
    llamadas: list[tuple[Coordenada, Coordenada]] = field(default_factory=list)

    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.VIAL_CACHE

    @property
    def version(self) -> str:
        return 'proveedor-prueba-v1'

    def calcular_tramo(self, origen: Coordenada, destino: Coordenada, configuracion: ConfiguracionPlanificacion) -> ResultadoTramoViaje:
        _ = configuracion
        self.llamadas.append((origen, destino))
        return ResultadoTramoViaje(distancia_metros=self.distancia_metros, tiempo_base_min=self.tiempo_base_min, fuente=self.fuente)

class InyeccionProveedorAlgoritmosTest(unittest.TestCase):

    def setUp(self) -> None:
        self.instancia = crear_instancia_demo()

    def test_ga_usa_el_mismo_proveedor_en_matriz_y_semilla_greedy(self) -> None:
        proveedor = ProveedorViajePrueba()
        plan = generar_plan_ga(self.instancia, seed=1234, configuracion_ga=ConfiguracionGA(tamano_poblacion=8, generaciones=3, tamano_elite=2, tamano_torneo=2, generaciones_sin_mejora_max=2), proveedor_viaje=proveedor)
        cantidad_nodos = len(self.instancia.pedidos) + 1
        llamadas_por_matriz = cantidad_nodos * (cantidad_nodos - 1)
        self.assertGreaterEqual(len(proveedor.llamadas), llamadas_por_matriz * 2)
        self.assertGreaterEqual(plan.costo_estimado, 0.0)

    def test_selector_inyecta_el_mismo_proveedor_en_modos_clasicos(self) -> None:
        proveedor = ProveedorViajePrueba()
        selector = SelectorPlanificadores(proveedor_viaje=proveedor)
        for modo in (ModoPlanificacion.GREEDY, ModoPlanificacion.RANDOM, ModoPlanificacion.GA):
            llamadas_antes = len(proveedor.llamadas)
            plan = selector.generar_plan(self.instancia, modo)
            self.assertGreater(len(proveedor.llamadas), llamadas_antes)
            self.assertGreaterEqual(plan.costo_estimado, 0.0)
if __name__ == '__main__':
    unittest.main()
