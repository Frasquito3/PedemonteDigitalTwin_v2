import unittest
from planner.core.config import ConfiguracionPlanificacion
from planner.routing.travel import FuenteMatrizViaje, ProveedorHaversineAjustado, ResultadoTramoViaje, construir_matriz_viaje, distancia_haversine_metros
from tests.fixtures import crear_instancia_demo

class ProveedorConstantePrueba:

    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.VIAL_LOCAL

    @property
    def version(self) -> str:
        return 'proveedor-constante-test-v1'

    def calcular_tramo(self, origen: tuple[float, float], destino: tuple[float, float], configuracion: ConfiguracionPlanificacion) -> ResultadoTramoViaje:
        del origen
        del destino
        del configuracion
        return ResultadoTramoViaje(distancia_metros=1234.0, tiempo_base_min=5.5, fuente=self.fuente, uso_fallback=True, advertencia='Fallback de prueba.')

class ProveedorInvalidoPrueba:

    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.VIAL_LOCAL

    @property
    def version(self) -> str:
        return 'proveedor-invalido-test-v1'

    def calcular_tramo(self, origen: tuple[float, float], destino: tuple[float, float], configuracion: ConfiguracionPlanificacion) -> ResultadoTramoViaje:
        del origen
        del destino
        del configuracion
        return ResultadoTramoViaje(distancia_metros=-1.0, tiempo_base_min=1.0, fuente=self.fuente)

class MatrizViajeTest(unittest.TestCase):

    def test_acepta_proveedor_intercambiable_y_registra_metadata(self) -> None:
        instancia = crear_instancia_demo()
        configuracion = ConfiguracionPlanificacion()
        matriz = construir_matriz_viaje(instancia, configuracion, proveedor=ProveedorConstantePrueba())
        cantidad_nodos = 1 + len(instancia.pedidos)
        cantidad_tramos_no_diagonales = cantidad_nodos * (cantidad_nodos - 1)
        self.assertEqual(matriz.fuente, FuenteMatrizViaje.VIAL_LOCAL)
        self.assertEqual(matriz.version_fuente, 'proveedor-constante-test-v1')
        self.assertEqual(matriz.cantidad_fallbacks, cantidad_tramos_no_diagonales)
        self.assertTrue(matriz.usa_fallback)
        self.assertEqual(matriz.advertencias, ('Fallback de prueba.',))
        self.assertEqual(matriz.distancia(configuracion.id_nodo_corralon, instancia.pedidos[0].pedido_id), 1234.0)
        self.assertEqual(matriz.tiempo_base(configuracion.id_nodo_corralon, instancia.pedidos[0].pedido_id), 5.5)
        self.assertEqual(matriz.resumen_fuente(), f'fuente=VIAL_LOCAL|version=proveedor-constante-test-v1|fallbacks={cantidad_tramos_no_diagonales}')
if __name__ == '__main__':
    unittest.main()
