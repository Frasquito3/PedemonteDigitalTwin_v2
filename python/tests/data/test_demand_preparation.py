from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from planner.data.demand_preparation import (
    clave_direccion,
    preparar_dataset,
)


class DemandPreparationTest(unittest.TestCase):
    def test_clave_direccion_normaliza_acentos_y_espacios(
        self,
    ) -> None:
        self.assertEqual(
            clave_direccion(
                "  José   Hernández ",
                " 2027 ",
                "Granadero Baigorria",
            ),
            (
                "jose hernandez|2027|"
                "granadero baigorria"
            ),
        )

    def test_preparacion_conserva_repeticiones_y_marca_lejanos(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directorio:
            raiz = Path(directorio)
            entrada = raiz / "entrada.csv"
            salida = raiz / "salida.csv"
            revision = raiz / "revision.csv"
            metadata = raiz / "metadata.json"

            with entrada.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as archivo:
                escritor = csv.DictWriter(
                    archivo,
                    fieldnames=(
                        "calle",
                        "altura",
                        "latitud",
                        "longitud",
                        "ciudad_detectada",
                        "barrio",
                        "direccion_osm",
                    ),
                )
                escritor.writeheader()
                escritor.writerows(
                    (
                        {
                            "calle": "José Hernández",
                            "altura": "2027",
                            "latitud": "-32.8548543",
                            "longitud": "-60.7224465",
                            "ciudad_detectada": (
                                "Granadero Baigorria"
                            ),
                            "barrio": "No especificado",
                            "direccion_osm": "Dirección A",
                        },
                        {
                            "calle": "Jose Hernandez",
                            "altura": "2027",
                            "latitud": "-32.8548543",
                            "longitud": "-60.7224465",
                            "ciudad_detectada": (
                                "Granadero Baigorria"
                            ),
                            "barrio": "No especificado",
                            "direccion_osm": "Dirección A",
                        },
                        {
                            "calle": "Lejana",
                            "altura": "1",
                            "latitud": "-31.2414936",
                            "longitud": "-61.4720621",
                            "ciudad_detectada": "Rafaela",
                            "barrio": "Italia",
                            "direccion_osm": "Dirección B",
                        },
                    )
                )

            resultado = preparar_dataset(
                ruta_entrada=entrada,
                ruta_dataset=salida,
                ruta_revision=revision,
                ruta_metadata=metadata,
                distancia_max_entrenamiento_km=30.0,
            )

            conteos = resultado["conteos"]
            self.assertIsInstance(conteos, dict)
            assert isinstance(conteos, dict)

            self.assertEqual(
                conteos["registros_totales"],
                3,
            )
            self.assertEqual(
                conteos["registros_aptos_entrenamiento"],
                2,
            )
            self.assertEqual(
                conteos["registros_revision"],
                1,
            )
            self.assertEqual(
                conteos["direcciones_fuente_unicas"],
                2,
            )

            with salida.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as archivo:
                filas = list(csv.DictReader(archivo))

            self.assertEqual(len(filas), 3)
            self.assertEqual(
                filas[0]["frecuencia_direccion_fuente"],
                "2",
            )
            self.assertEqual(
                filas[1]["frecuencia_direccion_fuente"],
                "2",
            )
            self.assertEqual(
                filas[2]["estado_calidad"],
                "REVISAR_DISTANCIA",
            )

            with revision.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as archivo:
                filas_revision = list(
                    csv.DictReader(archivo)
                )

            self.assertEqual(len(filas_revision), 1)
            self.assertEqual(
                filas_revision[0]["ciudad"],
                "Rafaela",
            )

            metadata_leida = json.loads(
                metadata.read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata_leida["version"],
                "1.0.0",
            )


if __name__ == "__main__":
    unittest.main()