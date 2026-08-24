from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from planner.core.config import ConfiguracionPlanificacion
from planner.routing.travel import (
    FuenteMatrizViaje,
    construir_matriz_viaje,
)
from planner.routing.vial_cache import (
    COLUMNAS_CACHE_VIAL,
    ProveedorVialCachePersistente,
)


VERSION_CACHE = "pedemonte-vial-test-v1"


class ProveedorVialCachePersistenteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directorio_temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.directorio_temporal.cleanup)

        self.ruta_cache = (
            Path(self.directorio_temporal.name)
            / "cache_vial.csv"
        )

        self.configuracion = ConfiguracionPlanificacion()

        self.origen = (-32.8495006, -60.722653)
        self.destino = (-32.8310000, -60.719000)

    def test_lee_tramo_dirigido_y_conserva_tiempo_explicito(
        self,
    ) -> None:
        self._escribir_cache(
            [
                self._fila(
                    origen=self.origen,
                    destino=self.destino,
                    distancia_metros=3203.46,
                    tiempo_base_min=8.25,
                ),
                self._fila(
                    origen=self.destino,
                    destino=self.origen,
                    distancia_metros=3410.50,
                    tiempo_base_min=8.90,
                ),
            ]
        )

        proveedor = self._crear_proveedor()

        ida = proveedor.calcular_tramo(
            self.origen,
            self.destino,
            self.configuracion,
        )

        vuelta = proveedor.calcular_tramo(
            self.destino,
            self.origen,
            self.configuracion,
        )

        self.assertEqual(
            ida.fuente,
            FuenteMatrizViaje.VIAL_CACHE,
        )
        self.assertAlmostEqual(ida.distancia_metros, 3203.46)
        self.assertAlmostEqual(ida.tiempo_base_min, 8.25)
        self.assertFalse(ida.uso_fallback)

        self.assertAlmostEqual(vuelta.distancia_metros, 3410.50)
        self.assertAlmostEqual(vuelta.tiempo_base_min, 8.90)

        self.assertEqual(
            proveedor.estadisticas.cantidad_tramos,
            2,
        )
        self.assertIn(
            "vial-cache-csv-v1:pedemonte-vial-test-v1",
            proveedor.version,
        )

    def test_deriva_tiempo_desde_distancia_si_cache_no_lo_informa(
        self,
    ) -> None:
        self._escribir_cache(
            [
                self._fila(
                    origen=self.origen,
                    destino=self.destino,
                    distancia_metros=3000.0,
                    tiempo_base_min=None,
                    fuente_tiempo="VELOCIDAD_BASE_CONFIG",
                )
            ]
        )

        proveedor = self._crear_proveedor()

        resultado = proveedor.calcular_tramo(
            self.origen,
            self.destino,
            self.configuracion,
        )

        tiempo_esperado = (
            3000.0
            / 1000.0
            / self.configuracion.velocidad_base_kmh
            * 60.0
        )

        self.assertAlmostEqual(
            resultado.tiempo_base_min,
            tiempo_esperado,
        )
        self.assertFalse(resultado.uso_fallback)

    def test_cache_miss_usa_fallback_y_lo_declara(
        self,
    ) -> None:
        self._escribir_cache([])

        proveedor = self._crear_proveedor(
            permitir_fallback=True,
        )

        resultado = proveedor.calcular_tramo(
            self.origen,
            self.destino,
            self.configuracion,
        )

        self.assertTrue(resultado.uso_fallback)
        self.assertGreater(resultado.distancia_metros, 0.0)
        self.assertGreater(resultado.tiempo_base_min, 0.0)
        self.assertIn(
            "caché vial no contiene todos los tramos",
            resultado.advertencia,
        )

    def test_cache_miss_estricto_lanza_error(
        self,
    ) -> None:
        self._escribir_cache([])

        proveedor = self._crear_proveedor(
            permitir_fallback=False,
        )

        with self.assertRaisesRegex(
            KeyError,
            "no contiene el tramo dirigido",
        ):
            proveedor.calcular_tramo(
                self.origen,
                self.destino,
                self.configuracion,
            )

    def test_coordenadas_iguales_devuelven_cero_sin_fallback(
        self,
    ) -> None:
        self._escribir_cache([])

        proveedor = self._crear_proveedor(
            permitir_fallback=False,
        )

        resultado = proveedor.calcular_tramo(
            self.origen,
            self.origen,
            self.configuracion,
        )

        self.assertEqual(resultado.distancia_metros, 0.0)
        self.assertEqual(resultado.tiempo_base_min, 0.0)
        self.assertFalse(resultado.uso_fallback)


    def test_matriz_registra_hits_y_fallbacks_de_cache(
        self,
    ) -> None:
        from planner.core.schema import InstanciaTurno
        from tests.fixtures import crear_instancia_demo

        instancia_base = crear_instancia_demo()
        pedido = instancia_base.pedidos[0]

        instancia = InstanciaTurno(
            instancia_id="CACHE-TEST-001",
            fecha_operacion=instancia_base.fecha_operacion,
            turno=instancia_base.turno,
            pedidos=[pedido],
            lat_corralon=instancia_base.lat_corralon,
            lon_corralon=instancia_base.lon_corralon,
            capacidad_camion=instancia_base.capacidad_camion,
            cantidad_camiones=instancia_base.cantidad_camiones,
            hora_inicio_turno_min=(
                instancia_base.hora_inicio_turno_min
            ),
            hora_fin_objetivo_min=(
                instancia_base.hora_fin_objetivo_min
            ),
            hora_fin_tolerancia_min=(
                instancia_base.hora_fin_tolerancia_min
            ),
            seed_escenario=instancia_base.seed_escenario,
            seed_ejecucion=instancia_base.seed_ejecucion,
        )

        origen = (
            instancia.lat_corralon,
            instancia.lon_corralon,
        )
        destino = (pedido.latitud, pedido.longitud)

        self._escribir_cache(
            [
                self._fila(
                    origen=origen,
                    destino=destino,
                    distancia_metros=2500.0,
                    tiempo_base_min=6.0,
                )
            ]
        )

        proveedor = self._crear_proveedor(
            permitir_fallback=True,
        )

        matriz = construir_matriz_viaje(
            instancia,
            self.configuracion,
            proveedor=proveedor,
        )

        self.assertEqual(
            matriz.fuente,
            FuenteMatrizViaje.VIAL_CACHE,
        )
        self.assertEqual(matriz.cantidad_fallbacks, 1)
        self.assertTrue(matriz.usa_fallback)
        self.assertEqual(len(matriz.advertencias), 1)
        self.assertAlmostEqual(
            matriz.distancia(
                self.configuracion.id_nodo_corralon,
                pedido.pedido_id,
            ),
            2500.0,
        )

    def test_rechaza_version_incompatible(self) -> None:
        fila = self._fila(
            origen=self.origen,
            destino=self.destino,
            distancia_metros=1000.0,
            tiempo_base_min=2.0,
        )
        fila["version_cache"] = "otra-version"

        self._escribir_cache([fila])

        with self.assertRaisesRegex(
            ValueError,
            "Versión de caché inesperada",
        ):
            self._crear_proveedor()

    def _crear_proveedor(
        self,
        *,
        permitir_fallback: bool = True,
    ) -> ProveedorVialCachePersistente:
        return ProveedorVialCachePersistente(
            self.ruta_cache,
            version_cache_esperada=VERSION_CACHE,
            precision_coordenadas=6,
            permitir_fallback=permitir_fallback,
        )

    def _escribir_cache(
        self,
        filas: list[dict[str, str]],
    ) -> None:
        with self.ruta_cache.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as archivo:
            escritor = csv.DictWriter(
                archivo,
                fieldnames=COLUMNAS_CACHE_VIAL,
            )
            escritor.writeheader()
            escritor.writerows(filas)

    @staticmethod
    def _fila(
        *,
        origen: tuple[float, float],
        destino: tuple[float, float],
        distancia_metros: float,
        tiempo_base_min: float | None,
        fuente_tiempo: str = "ANYLOGIC_ONLINE",
    ) -> dict[str, str]:
        return {
            "version_cache": VERSION_CACHE,
            "lat_origen": str(origen[0]),
            "lon_origen": str(origen[1]),
            "lat_destino": str(destino[0]),
            "lon_destino": str(destino[1]),
            "distancia_metros": str(distancia_metros),
            "tiempo_base_min": (
                ""
                if tiempo_base_min is None
                else str(tiempo_base_min)
            ),
            "fuente_distancia": "ANYLOGIC_ONLINE",
            "fuente_tiempo": fuente_tiempo,
        }


if __name__ == "__main__":
    unittest.main()
