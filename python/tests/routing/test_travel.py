import unittest

from planner.core.config import ConfiguracionPlanificacion
from planner.routing.travel import (
    FuenteMatrizViaje,
    ProveedorHaversineAjustado,
    ResultadoTramoViaje,
    construir_matriz_viaje,
    distancia_haversine_metros,
)
from tests.fixtures import crear_instancia_demo


class ProveedorConstantePrueba:
    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.VIAL_LOCAL

    @property
    def version(self) -> str:
        return "proveedor-constante-test-v1"

    def calcular_tramo(
        self,
        origen: tuple[float, float],
        destino: tuple[float, float],
        configuracion: ConfiguracionPlanificacion,
    ) -> ResultadoTramoViaje:
        del origen
        del destino
        del configuracion

        return ResultadoTramoViaje(
            distancia_metros=1234.0,
            tiempo_base_min=5.5,
            fuente=self.fuente,
            uso_fallback=True,
            advertencia="Fallback de prueba.",
        )


class ProveedorInvalidoPrueba:
    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.VIAL_LOCAL

    @property
    def version(self) -> str:
        return "proveedor-invalido-test-v1"

    def calcular_tramo(
        self,
        origen: tuple[float, float],
        destino: tuple[float, float],
        configuracion: ConfiguracionPlanificacion,
    ) -> ResultadoTramoViaje:
        del origen
        del destino
        del configuracion

        return ResultadoTramoViaje(
            distancia_metros=-1.0,
            tiempo_base_min=1.0,
            fuente=self.fuente,
        )


class MatrizViajeTest(unittest.TestCase):
    def test_proveedor_predeterminado_conserva_baseline(
        self,
    ) -> None:
        instancia = crear_instancia_demo()
        configuracion = ConfiguracionPlanificacion()

        matriz = construir_matriz_viaje(
            instancia,
            configuracion,
        )

        pedido = instancia.pedidos[0]

        distancia_esperada = (
            distancia_haversine_metros(
                instancia.lat_corralon,
                instancia.lon_corralon,
                pedido.latitud,
                pedido.longitud,
            )
            * configuracion.factor_urbano_distancia
        )

        tiempo_esperado = (
            distancia_esperada
            / 1000.0
            / configuracion.velocidad_base_kmh
            * 60.0
        )

        self.assertAlmostEqual(
            matriz.distancia(
                configuracion.id_nodo_corralon,
                pedido.pedido_id,
            ),
            distancia_esperada,
            places=9,
        )

        self.assertAlmostEqual(
            matriz.tiempo_base(
                configuracion.id_nodo_corralon,
                pedido.pedido_id,
            ),
            tiempo_esperado,
            places=9,
        )

        self.assertEqual(
            matriz.fuente,
            FuenteMatrizViaje.HAVERSINE_AJUSTADA,
        )

        self.assertEqual(
            matriz.version_fuente,
            ProveedorHaversineAjustado().version,
        )

        self.assertEqual(
            matriz.cantidad_fallbacks,
            0,
        )

        self.assertFalse(
            matriz.usa_fallback
        )

    def test_acepta_proveedor_intercambiable_y_registra_metadata(
        self,
    ) -> None:
        instancia = crear_instancia_demo()
        configuracion = ConfiguracionPlanificacion()

        matriz = construir_matriz_viaje(
            instancia,
            configuracion,
            proveedor=ProveedorConstantePrueba(),
        )

        cantidad_nodos = 1 + len(
            instancia.pedidos
        )

        cantidad_tramos_no_diagonales = (
            cantidad_nodos
            * (cantidad_nodos - 1)
        )

        self.assertEqual(
            matriz.fuente,
            FuenteMatrizViaje.VIAL_LOCAL,
        )

        self.assertEqual(
            matriz.version_fuente,
            "proveedor-constante-test-v1",
        )

        self.assertEqual(
            matriz.cantidad_fallbacks,
            cantidad_tramos_no_diagonales,
        )

        self.assertTrue(
            matriz.usa_fallback
        )

        self.assertEqual(
            matriz.advertencias,
            ("Fallback de prueba.",),
        )

        self.assertEqual(
            matriz.distancia(
                configuracion.id_nodo_corralon,
                instancia.pedidos[0].pedido_id,
            ),
            1234.0,
        )

        self.assertEqual(
            matriz.tiempo_base(
                configuracion.id_nodo_corralon,
                instancia.pedidos[0].pedido_id,
            ),
            5.5,
        )

        self.assertEqual(
            matriz.resumen_fuente(),
            "fuente=VIAL_LOCAL"
            "|version=proveedor-constante-test-v1"
            f"|fallbacks={cantidad_tramos_no_diagonales}",
        )

    def test_rechaza_resultados_invalidos_del_proveedor(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "distancia inválida",
        ):
            construir_matriz_viaje(
                crear_instancia_demo(),
                ConfiguracionPlanificacion(),
                proveedor=ProveedorInvalidoPrueba(),
            )


if __name__ == "__main__":
    unittest.main()
