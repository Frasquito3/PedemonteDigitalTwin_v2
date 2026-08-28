from __future__ import annotations
import csv
import random
import tempfile
import unittest
from pathlib import Path
from planner.data.real_demand import CatalogoDemandaReal, ParticionDemandaReal
COLUMNAS = ['registro_id', 'calle', 'altura', 'ciudad', 'barrio', 'latitud', 'longitud', 'distancia_corralon_km', 'direccion_osm', 'clave_direccion_fuente', 'frecuencia_direccion_fuente', 'estado_calidad', 'motivo_revision']

class RealDemandTest(unittest.TestCase):

    def test_carga_solo_aptos_y_conserva_repeticiones(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'demanda.csv'
            self._escribir_dataset_valido(ruta)
            catalogo = CatalogoDemandaReal.desde_csv(ruta)
            self.assertEqual(len(catalogo), 4)
            claves = [registro.clave_direccion_fuente for registro in catalogo.registros]
            self.assertEqual(claves.count('calle a|100|granadero baigorria'), 2)
            self.assertNotIn('DG-REVISION', {registro.registro_id for registro in catalogo.registros})

    def test_division_agrupa_repeticiones_sin_fuga(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'demanda.csv'
            self._escribir_dataset_valido(ruta)
            catalogo = CatalogoDemandaReal.desde_csv(ruta)
            division = catalogo.dividir_por_direccion_fuente(seed=152026)
            division.validar_sin_fuga()
            claves_por_particion = {particion: division.catalogo_para(particion).claves_direccion_fuente for particion in ParticionDemandaReal}
            clave_repetida = 'calle a|100|granadero baigorria'
            particiones_con_clave = [particion for particion, claves in claves_por_particion.items() if clave_repetida in claves]
            self.assertEqual(len(particiones_con_clave), 1)
            catalogo_repetido = division.catalogo_para(particiones_con_clave[0])
            self.assertEqual(sum((1 for registro in catalogo_repetido.registros if registro.clave_direccion_fuente == clave_repetida)), 2)
            self.assertEqual(sum((len(division.catalogo_para(particion)) for particion in ParticionDemandaReal)), len(catalogo))

    def test_division_no_depende_del_orden_de_filas(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'demanda.csv'
            self._escribir_dataset_con_claves_unicas(ruta=ruta, cantidad=20)
            catalogo = CatalogoDemandaReal.desde_csv(ruta)
            catalogo_invertido = CatalogoDemandaReal(registros=tuple(reversed(catalogo.registros)), ruta_fuente=catalogo.ruta_fuente)
            division_original = catalogo.dividir_por_direccion_fuente(seed=8005)
            division_invertida = catalogo_invertido.dividir_por_direccion_fuente(seed=8005)
            for particion in ParticionDemandaReal:
                self.assertEqual(division_original.catalogo_para(particion).claves_direccion_fuente, division_invertida.catalogo_para(particion).claves_direccion_fuente)

    def _escribir_dataset_valido(self, ruta: Path) -> None:
        filas = [self._crear_fila(registro_id='DG-0001', calle='Calle A', altura='100', ciudad='Granadero Baigorria', clave='calle a|100|granadero baigorria', frecuencia=2), self._crear_fila(registro_id='DG-0002', calle='Calle A', altura='100', ciudad='Granadero Baigorria', clave='calle a|100|granadero baigorria', frecuencia=2), self._crear_fila(registro_id='DG-0003', calle='Calle B', altura='200', ciudad='Rosario', clave='calle b|200|rosario', frecuencia=1), self._crear_fila(registro_id='DG-0004', calle='Calle C', altura='300', ciudad='Funes', clave='calle c|300|funes', frecuencia=1), self._crear_fila(registro_id='DG-REVISION', calle='Calle Lejana', altura='1', ciudad='Rafaela', clave='calle lejana|1|rafaela', frecuencia=1, estado='REVISAR_DISTANCIA')]
        self._escribir_filas(ruta=ruta, filas=filas)

    def _escribir_dataset_con_claves_unicas(self, ruta: Path, cantidad: int) -> None:
        filas = [self._crear_fila(registro_id=f'DG-{indice:04d}', calle=f'Calle {indice}', altura=str(100 + indice), ciudad='Granadero Baigorria', clave=f'calle {indice}|{100 + indice}|granadero baigorria', frecuencia=1) for indice in range(1, cantidad + 1)]
        self._escribir_filas(ruta=ruta, filas=filas)

    def _escribir_filas(self, ruta: Path, filas: list[dict[str, str]]) -> None:
        with ruta.open(mode='w', encoding='utf-8', newline='') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
            escritor.writeheader()
            escritor.writerows(filas)

    def _crear_fila(self, registro_id: str, calle: str, altura: str, ciudad: str, clave: str, frecuencia: int, estado: str='APTO_ENTRENAMIENTO') -> dict[str, str]:
        return {'registro_id': registro_id, 'calle': calle, 'altura': altura, 'ciudad': ciudad, 'barrio': 'Barrio de prueba', 'latitud': '-32.8500000', 'longitud': '-60.7200000', 'distancia_corralon_km': '1.25', 'direccion_osm': f'{calle} {altura}, {ciudad}', 'clave_direccion_fuente': clave, 'frecuencia_direccion_fuente': str(frecuencia), 'estado_calidad': estado, 'motivo_revision': ''}
if __name__ == '__main__':
    unittest.main()
