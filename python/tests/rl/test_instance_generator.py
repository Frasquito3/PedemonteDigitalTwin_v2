from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from planner.data.real_demand import (
    CatalogoDemandaReal,
    ParticionDemandaReal,
    PuntoDemandaReal,
    SEED_DIVISION_DEMANDA_REAL_V1,
)
from planner.domain.validator import (
    validar_instancia,
)
from planner.rl.instance_generator import (
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
    ModoDemandaGeografica,
)


def firma_instancia(
    instancia,
) -> tuple:
    return (
        instancia.turno,

        tuple(
            (
                pedido.pedido_id,
                pedido.pedido_original_id,
                pedido.unidades_capacidad,
                pedido.requiere_volcador,
                pedido.hora_desde_min,
                pedido.hora_hasta_min,
                pedido.latitud,
                pedido.longitud,
                pedido.direccion,
                pedido.barrio,
                pedido.observaciones,
            )

            for pedido in instancia.pedidos
        ),
    )


def crear_catalogo_prueba(
    cantidad: int = 20,
) -> CatalogoDemandaReal:
    registros = [
        PuntoDemandaReal(
            registro_id=f"DG-TEST-{indice:03d}",
            calle=f"Calle {indice}",
            altura=str(100 + indice),
            ciudad=(
                "Granadero Baigorria"
                if indice % 2 == 0
                else "Rosario"
            ),
            barrio=f"Barrio {indice % 4}",
            latitud=-32.8500000 - indice * 0.001,
            longitud=-60.7200000 - indice * 0.001,
            distancia_corralon_km=float(indice + 1),
            direccion_osm=(
                f"Calle {indice} {100 + indice}"
            ),
            clave_direccion_fuente=(
                f"calle {indice}|{100 + indice}"
            ),
            frecuencia_direccion_fuente=(
                1 + indice % 3
            ),
        )
        for indice in range(cantidad)
    ]

    return CatalogoDemandaReal(
        registros=registros,
        ruta_fuente=Path("catalogo_prueba.csv"),
    )


class GeneradorInstanciasRLTest(
    unittest.TestCase
):
    def test_misma_seed_reproduce_instancia(
        self,
    ) -> None:
        generador = (
            GeneradorInstanciasRL()
        )

        instancia_a = generador.generar(
            91_001
        )

        instancia_b = generador.generar(
            91_001
        )

        self.assertEqual(
            firma_instancia(instancia_a),
            firma_instancia(instancia_b),
        )

    def test_distintas_seeds_generan_diversidad(
        self,
    ) -> None:
        generador = (
            GeneradorInstanciasRL()
        )

        firmas = {
            firma_instancia(
                generador.generar(seed)
            )

            for seed in range(
                91_000,
                91_010,
            )
        }

        self.assertGreater(
            len(firmas),
            1,
        )

    def test_instancias_generadas_son_validas(
        self,
    ) -> None:
        generador = (
            GeneradorInstanciasRL()
        )

        for seed in range(
            91_000,
            91_100,
        ):
            instancia = generador.generar(
                seed
            )

            errores = validar_instancia(
                instancia
            )

            self.assertFalse(
                errores,

                msg=(
                    f"Seed={seed}: "
                    + " | ".join(errores)
                ),
            )

            self.assertGreaterEqual(
                len(instancia.pedidos),
                4,
            )

            self.assertLessEqual(
                len(instancia.pedidos),
                8,
            )

            for pedido in instancia.pedidos:
                self.assertGreater(
                    pedido.unidades_capacidad,
                    0,
                )

                self.assertLessEqual(
                    pedido.unidades_capacidad,
                    instancia.capacidad_camion,
                )

                self.assertGreaterEqual(
                    pedido.hora_desde_min,
                    instancia.hora_inicio_turno_min,
                )

                self.assertLessEqual(
                    pedido.hora_hasta_min,
                    instancia.hora_fin_objetivo_min,
                )

                self.assertLess(
                    pedido.hora_desde_min,
                    pedido.hora_hasta_min,
                )

    def test_modo_sintetico_sigue_siendo_predeterminado(
        self,
    ) -> None:
        generador = GeneradorInstanciasRL()

        self.assertEqual(
            generador.configuracion
            .modo_demanda_geografica,
            ModoDemandaGeografica.SINTETICA,
        )

        self.assertIsNone(
            generador.catalogo_demanda_real
        )

        instancia = generador.generar(
            91_111
        )

        self.assertTrue(
            all(
                not pedido.direccion
                and not pedido.barrio
                and not pedido.observaciones
                for pedido in instancia.pedidos
            )
        )

    def test_modo_real_reproduce_instancia_con_misma_seed(
        self,
    ) -> None:
        catalogo = crear_catalogo_prueba()

        configuracion = (
            ConfiguracionGeneradorInstancias(
                min_pedidos_finales=4,
                max_pedidos_finales=4,
                probabilidad_pedido_mayor_capacidad=0.0,
                modo_demanda_geografica=(
                    ModoDemandaGeografica.REAL
                ),
            )
        )

        generador = GeneradorInstanciasRL(
            configuracion=configuracion,
            catalogo_demanda_real=catalogo,
        )

        instancia_a = generador.generar(
            92_001
        )

        instancia_b = generador.generar(
            92_001
        )

        self.assertEqual(
            firma_instancia(instancia_a),
            firma_instancia(instancia_b),
        )

    def test_modo_real_usa_coordenadas_y_datos_del_catalogo(
        self,
    ) -> None:
        catalogo = crear_catalogo_prueba()

        configuracion = (
            ConfiguracionGeneradorInstancias(
                min_pedidos_finales=6,
                max_pedidos_finales=6,
                probabilidad_pedido_mayor_capacidad=0.0,
                modo_demanda_geografica=(
                    ModoDemandaGeografica.REAL
                ),
            )
        )

        generador = GeneradorInstanciasRL(
            configuracion=configuracion,
            catalogo_demanda_real=catalogo,
        )

        instancia = generador.generar(
            92_002
        )

        coordenadas_catalogo = {
            (
                punto.latitud,
                punto.longitud,
            )
            for punto in catalogo.registros
        }

        for pedido in instancia.pedidos:
            self.assertIn(
                (
                    pedido.latitud,
                    pedido.longitud,
                ),
                coordenadas_catalogo,
            )

            self.assertTrue(
                pedido.direccion
            )

            self.assertTrue(
                pedido.barrio
            )

            self.assertIn(
                "FUENTE_DEMANDA_REAL=",
                pedido.observaciones,
            )

            self.assertIn(
                "CIUDAD=",
                pedido.observaciones,
            )

    def test_catalogo_csv_se_carga_una_sola_vez(
        self,
    ) -> None:
        catalogo = crear_catalogo_prueba()

        configuracion = (
            ConfiguracionGeneradorInstancias(
                min_pedidos_finales=4,
                max_pedidos_finales=4,
                probabilidad_pedido_mayor_capacidad=0.0,
                modo_demanda_geografica=(
                    ModoDemandaGeografica.REAL
                ),
                ruta_demanda_real=(
                    "data/processed/"
                    "demanda_geografica_v1.csv"
                ),
            )
        )

        with patch.object(
            CatalogoDemandaReal,
            "desde_csv",
            return_value=catalogo,
        ) as cargar_catalogo:
            generador = GeneradorInstanciasRL(
                configuracion=configuracion
            )

            generador.generar(
                92_003
            )

            generador.generar(
                92_004
            )

        cargar_catalogo.assert_called_once()

    def test_modo_real_particion_train_usa_solo_train(
        self,
    ) -> None:
        catalogo = crear_catalogo_prueba(
            cantidad=60
        )

        configuracion = (
            ConfiguracionGeneradorInstancias(
                min_pedidos_finales=6,
                max_pedidos_finales=6,
                probabilidad_pedido_mayor_capacidad=0.0,
                modo_demanda_geografica=(
                    ModoDemandaGeografica.REAL
                ),
                particion_demanda_real=(
                    ParticionDemandaReal.ENTRENAMIENTO
                ),
                seed_division_demanda_real=(
                    SEED_DIVISION_DEMANDA_REAL_V1
                ),
            )
        )

        generador = GeneradorInstanciasRL(
            configuracion=configuracion,
            catalogo_demanda_real=catalogo,
        )

        catalogo_efectivo = (
            generador.catalogo_demanda_real
        )

        self.assertIsNotNone(
            catalogo_efectivo
        )

        assert catalogo_efectivo is not None

        self.assertEqual(
            catalogo_efectivo
            .cantidad_direcciones_fuente_unicas(),
            48,
        )

        coordenadas_train = {
            (punto.latitud, punto.longitud)
            for punto in catalogo_efectivo.registros
        }

        instancia = generador.generar(
            92_101
        )

        self.assertTrue(
            all(
                (pedido.latitud, pedido.longitud)
                in coordenadas_train
                for pedido in instancia.pedidos
            )
        )

    def test_particiones_reales_del_generador_no_comparten_direcciones(
        self,
    ) -> None:
        catalogo = crear_catalogo_prueba(
            cantidad=60
        )

        catalogos_por_particion: dict[
            ParticionDemandaReal,
            CatalogoDemandaReal,
        ] = {}

        for particion in ParticionDemandaReal:
            configuracion = (
                ConfiguracionGeneradorInstancias(
                    min_pedidos_finales=3,
                    max_pedidos_finales=3,
                    probabilidad_pedido_mayor_capacidad=0.0,
                    modo_demanda_geografica=(
                        ModoDemandaGeografica.REAL
                    ),
                    particion_demanda_real=particion,
                )
            )

            generador = GeneradorInstanciasRL(
                configuracion=configuracion,
                catalogo_demanda_real=catalogo,
            )

            catalogo_efectivo = (
                generador.catalogo_demanda_real
            )

            self.assertIsNotNone(
                catalogo_efectivo
            )

            assert catalogo_efectivo is not None

            catalogos_por_particion[particion] = (
                catalogo_efectivo
            )

        claves_train = (
            catalogos_por_particion[
                ParticionDemandaReal.ENTRENAMIENTO
            ].claves_direccion_fuente
        )

        claves_validation = (
            catalogos_por_particion[
                ParticionDemandaReal.VALIDACION
            ].claves_direccion_fuente
        )

        claves_test = (
            catalogos_por_particion[
                ParticionDemandaReal.PRUEBA
            ].claves_direccion_fuente
        )

        self.assertFalse(
            claves_train & claves_validation
        )
        self.assertFalse(
            claves_train & claves_test
        )
        self.assertFalse(
            claves_validation & claves_test
        )

    def test_modo_sintetico_rechaza_particion_real(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "particion_demanda_real",
        ):
            ConfiguracionGeneradorInstancias(
                modo_demanda_geografica=(
                    ModoDemandaGeografica.SINTETICA
                ),
                particion_demanda_real=(
                    ParticionDemandaReal.PRUEBA
                ),
            )

    def test_modo_real_sin_particion_conserva_catalogo_completo(
        self,
    ) -> None:
        catalogo = crear_catalogo_prueba(
            cantidad=20
        )

        generador = GeneradorInstanciasRL(
            configuracion=(
                ConfiguracionGeneradorInstancias(
                    min_pedidos_finales=4,
                    max_pedidos_finales=4,
                    probabilidad_pedido_mayor_capacidad=0.0,
                    modo_demanda_geografica=(
                        ModoDemandaGeografica.REAL
                    ),
                )
            ),
            catalogo_demanda_real=catalogo,
        )

        self.assertIs(
            generador.catalogo_demanda_real_completo,
            catalogo,
        )

        self.assertIs(
            generador.catalogo_demanda_real,
            catalogo,
        )

        self.assertIsNone(
            generador.division_demanda_real
        )

    def test_modo_real_sin_reemplazo_requiere_catalogo_suficiente(
        self,
    ) -> None:
        catalogo = crear_catalogo_prueba(
            cantidad=3
        )

        configuracion = (
            ConfiguracionGeneradorInstancias(
                min_pedidos_finales=4,
                max_pedidos_finales=4,
                modo_demanda_geografica=(
                    ModoDemandaGeografica.REAL
                ),
                muestreo_demanda_real_con_reemplazo=False,
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "no tiene suficientes registros",
        ):
            GeneradorInstanciasRL(
                configuracion=configuracion,
                catalogo_demanda_real=catalogo,
            )


if __name__ == "__main__":
    unittest.main()