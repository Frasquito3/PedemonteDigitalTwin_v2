from __future__ import annotations

import csv
import random
import tempfile
import unittest

from pathlib import Path

from planner.data.real_demand import (
    CatalogoDemandaReal,
    ParticionDemandaReal,
)


COLUMNAS = [
    "registro_id",
    "calle",
    "altura",
    "ciudad",
    "barrio",
    "latitud",
    "longitud",
    "distancia_corralon_km",
    "direccion_osm",
    "clave_direccion_fuente",
    "frecuencia_direccion_fuente",
    "estado_calidad",
    "motivo_revision",
]


class RealDemandTest(unittest.TestCase):
    def test_carga_solo_aptos_y_conserva_repeticiones(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_valido(ruta)

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            self.assertEqual(
                len(catalogo),
                4,
            )

            claves = [
                registro.clave_direccion_fuente
                for registro in catalogo.registros
            ]

            self.assertEqual(
                claves.count(
                    "calle a|100|granadero baigorria"
                ),
                2,
            )

            self.assertNotIn(
                "DG-REVISION",
                {
                    registro.registro_id
                    for registro in catalogo.registros
                },
            )

    def test_muestreo_es_determinista_con_misma_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_valido(ruta)

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            muestra_1 = catalogo.muestrear_con_seed(
                cantidad=3,
                seed=6001,
            )

            muestra_2 = catalogo.muestrear_con_seed(
                cantidad=3,
                seed=6001,
            )

            self.assertEqual(
                [
                    punto.registro_id
                    for punto in muestra_1
                ],
                [
                    punto.registro_id
                    for punto in muestra_2
                ],
            )

    def test_muestreo_sin_reemplazo_no_repite_filas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_valido(ruta)

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            muestra = catalogo.muestrear(
                cantidad=4,
                rng=random.Random(12),
                con_reemplazo=False,
            )

            ids = [
                punto.registro_id
                for punto in muestra
            ]

            self.assertEqual(
                len(ids),
                len(set(ids)),
            )

    def test_muestreo_con_reemplazo_admite_mas_registros(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_valido(ruta)

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            muestra = catalogo.muestrear(
                cantidad=20,
                rng=random.Random(99),
                con_reemplazo=True,
            )

            self.assertEqual(
                len(muestra),
                20,
            )

            ids_validos = {
                punto.registro_id
                for punto in catalogo.registros
            }

            self.assertTrue(
                all(
                    punto.registro_id in ids_validos
                    for punto in muestra
                )
            )

    def test_rechaza_cantidades_invalidas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_valido(ruta)

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            with self.assertRaises(ValueError):
                catalogo.muestrear(
                    cantidad=-1,
                    rng=random.Random(1),
                )

            with self.assertRaises(ValueError):
                catalogo.muestrear(
                    cantidad=5,
                    rng=random.Random(1),
                    con_reemplazo=False,
                )

    def test_rechaza_dataset_con_columnas_faltantes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "incompleto.csv"

            with ruta.open(
                mode="w",
                encoding="utf-8",
                newline="",
            ) as archivo:
                escritor = csv.DictWriter(
                    archivo,
                    fieldnames=[
                        "registro_id",
                        "latitud",
                    ],
                )

                escritor.writeheader()
                escritor.writerow(
                    {
                        "registro_id": "DG-1",
                        "latitud": "-32.85",
                    }
                )

            with self.assertRaisesRegex(
                ValueError,
                "Faltan columnas requeridas",
            ):
                CatalogoDemandaReal.desde_csv(
                    ruta
                )

    def test_division_agrupa_repeticiones_sin_fuga(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_valido(ruta)

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            division = (
                catalogo
                .dividir_por_direccion_fuente(
                    seed=15_2026
                )
            )

            division.validar_sin_fuga()

            claves_por_particion = {
                particion: (
                    division
                    .catalogo_para(
                        particion
                    )
                    .claves_direccion_fuente
                )
                for particion
                in ParticionDemandaReal
            }

            clave_repetida = (
                "calle a|100|granadero baigorria"
            )

            particiones_con_clave = [
                particion
                for particion, claves
                in claves_por_particion.items()
                if clave_repetida in claves
            ]

            self.assertEqual(
                len(particiones_con_clave),
                1,
            )

            catalogo_repetido = (
                division.catalogo_para(
                    particiones_con_clave[0]
                )
            )

            self.assertEqual(
                sum(
                    1
                    for registro
                    in catalogo_repetido.registros
                    if (
                        registro
                        .clave_direccion_fuente
                        == clave_repetida
                    )
                ),
                2,
            )

            self.assertEqual(
                sum(
                    len(
                        division.catalogo_para(
                            particion
                        )
                    )
                    for particion
                    in ParticionDemandaReal
                ),
                len(catalogo),
            )

    def test_division_es_determinista_con_misma_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_con_claves_unicas(
                ruta=ruta,
                cantidad=20,
            )

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            division_a = (
                catalogo
                .dividir_por_direccion_fuente(
                    seed=8_001
                )
            )

            division_b = (
                catalogo
                .dividir_por_direccion_fuente(
                    seed=8_001
                )
            )

            for particion in ParticionDemandaReal:
                self.assertEqual(
                    division_a
                    .catalogo_para(
                        particion
                    )
                    .claves_direccion_fuente,

                    division_b
                    .catalogo_para(
                        particion
                    )
                    .claves_direccion_fuente,
                )

    def test_division_80_10_10_por_claves(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_con_claves_unicas(
                ruta=ruta,
                cantidad=20,
            )

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            division = (
                catalogo
                .dividir_por_direccion_fuente(
                    seed=8_002
                )
            )

            resumen = division.resumen()

            self.assertEqual(
                resumen["TRAIN"][
                    "direcciones_fuente_unicas"
                ],
                16,
            )

            self.assertEqual(
                resumen["VALIDATION"][
                    "direcciones_fuente_unicas"
                ],
                2,
            )

            self.assertEqual(
                resumen["TEST"][
                    "direcciones_fuente_unicas"
                ],
                2,
            )

    def test_division_cambia_con_otra_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_con_claves_unicas(
                ruta=ruta,
                cantidad=20,
            )

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            division_a = (
                catalogo
                .dividir_por_direccion_fuente(
                    seed=8_003
                )
            )

            division_b = (
                catalogo
                .dividir_por_direccion_fuente(
                    seed=8_004
                )
            )

            self.assertNotEqual(
                division_a
                .entrenamiento
                .claves_direccion_fuente,

                division_b
                .entrenamiento
                .claves_direccion_fuente,
            )

    def test_division_no_depende_del_orden_de_filas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_con_claves_unicas(
                ruta=ruta,
                cantidad=20,
            )

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            catalogo_invertido = CatalogoDemandaReal(
                registros=tuple(
                    reversed(
                        catalogo.registros
                    )
                ),
                ruta_fuente=(
                    catalogo.ruta_fuente
                ),
            )

            division_original = (
                catalogo
                .dividir_por_direccion_fuente(
                    seed=8_005
                )
            )

            division_invertida = (
                catalogo_invertido
                .dividir_por_direccion_fuente(
                    seed=8_005
                )
            )

            for particion in ParticionDemandaReal:
                self.assertEqual(
                    division_original
                    .catalogo_para(
                        particion
                    )
                    .claves_direccion_fuente,

                    division_invertida
                    .catalogo_para(
                        particion
                    )
                    .claves_direccion_fuente,
                )

    def test_division_rechaza_proporciones_invalidas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demanda.csv"

            self._escribir_dataset_valido(ruta)

            catalogo = (
                CatalogoDemandaReal.desde_csv(ruta)
            )

            with self.assertRaisesRegex(
                ValueError,
                "deben sumar 1.0",
            ):
                catalogo.dividir_por_direccion_fuente(
                    proporcion_entrenamiento=0.70,
                    proporcion_validacion=0.10,
                    proporcion_prueba=0.10,
                )

    def _escribir_dataset_valido(
        self,
        ruta: Path,
    ) -> None:
        filas = [
            self._crear_fila(
                registro_id="DG-0001",
                calle="Calle A",
                altura="100",
                ciudad="Granadero Baigorria",
                clave=(
                    "calle a|100|granadero baigorria"
                ),
                frecuencia=2,
            ),
            self._crear_fila(
                registro_id="DG-0002",
                calle="Calle A",
                altura="100",
                ciudad="Granadero Baigorria",
                clave=(
                    "calle a|100|granadero baigorria"
                ),
                frecuencia=2,
            ),
            self._crear_fila(
                registro_id="DG-0003",
                calle="Calle B",
                altura="200",
                ciudad="Rosario",
                clave="calle b|200|rosario",
                frecuencia=1,
            ),
            self._crear_fila(
                registro_id="DG-0004",
                calle="Calle C",
                altura="300",
                ciudad="Funes",
                clave="calle c|300|funes",
                frecuencia=1,
            ),
            self._crear_fila(
                registro_id="DG-REVISION",
                calle="Calle Lejana",
                altura="1",
                ciudad="Rafaela",
                clave="calle lejana|1|rafaela",
                frecuencia=1,
                estado="REVISAR_DISTANCIA",
            ),
        ]

        self._escribir_filas(
            ruta=ruta,
            filas=filas,
        )

    def _escribir_dataset_con_claves_unicas(
        self,
        ruta: Path,
        cantidad: int,
    ) -> None:
        filas = [
            self._crear_fila(
                registro_id=(
                    f"DG-{indice:04d}"
                ),
                calle=(
                    f"Calle {indice}"
                ),
                altura=str(
                    100
                    + indice
                ),
                ciudad="Granadero Baigorria",
                clave=(
                    f"calle {indice}|"
                    f"{100 + indice}|"
                    "granadero baigorria"
                ),
                frecuencia=1,
            )
            for indice in range(
                1,
                cantidad + 1,
            )
        ]

        self._escribir_filas(
            ruta=ruta,
            filas=filas,
        )

    def _escribir_filas(
        self,
        ruta: Path,
        filas: list[dict[str, str]],
    ) -> None:
        with ruta.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as archivo:
            escritor = csv.DictWriter(
                archivo,
                fieldnames=COLUMNAS,
            )

            escritor.writeheader()
            escritor.writerows(filas)

    def _crear_fila(
        self,
        registro_id: str,
        calle: str,
        altura: str,
        ciudad: str,
        clave: str,
        frecuencia: int,
        estado: str = "APTO_ENTRENAMIENTO",
    ) -> dict[str, str]:
        return {
            "registro_id": registro_id,
            "calle": calle,
            "altura": altura,
            "ciudad": ciudad,
            "barrio": "Barrio de prueba",
            "latitud": "-32.8500000",
            "longitud": "-60.7200000",
            "distancia_corralon_km": "1.25",
            "direccion_osm": (
                f"{calle} {altura}, {ciudad}"
            ),
            "clave_direccion_fuente": clave,
            "frecuencia_direccion_fuente": str(
                frecuencia
            ),
            "estado_calidad": estado,
            "motivo_revision": "",
        }


if __name__ == "__main__":
    unittest.main()