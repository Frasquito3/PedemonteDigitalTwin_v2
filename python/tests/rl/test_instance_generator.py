from __future__ import annotations
import unittest
from pathlib import Path
from unittest.mock import patch
from planner.rl.training_data import CatalogoDemandaReal, ParticionDemandaReal, PuntoDemandaReal, SEED_DIVISION_DEMANDA_REAL
from planner.domain.validator import validar_instancia
from planner.rl.instance_generator import ConfiguracionGeneradorInstancias, GeneradorInstanciasRL, ModoDemandaGeografica

def firma_instancia(instancia) -> tuple:
    return (instancia.turno, tuple(((pedido.pedido_id, pedido.pedido_original_id, pedido.unidades_capacidad, pedido.requiere_volcador, pedido.hora_desde_min, pedido.hora_hasta_min, pedido.latitud, pedido.longitud, pedido.direccion, pedido.barrio, pedido.observaciones) for pedido in instancia.pedidos)))

def crear_catalogo_prueba(cantidad: int=20) -> CatalogoDemandaReal:
    registros = [PuntoDemandaReal(registro_id=f'DG-TEST-{indice:03d}', calle=f'Calle {indice}', altura=str(100 + indice), ciudad='Granadero Baigorria' if indice % 2 == 0 else 'Rosario', barrio=f'Barrio {indice % 4}', latitud=-32.85 - indice * 0.001, longitud=-60.72 - indice * 0.001, distancia_corralon_km=float(indice + 1), direccion_osm=f'Calle {indice} {100 + indice}', clave_direccion_fuente=f'calle {indice}|{100 + indice}', frecuencia_direccion_fuente=1 + indice % 3) for indice in range(cantidad)]
    return CatalogoDemandaReal(registros=registros, ruta_fuente=Path('catalogo_prueba.csv'))

class GeneradorInstanciasRLTest(unittest.TestCase):

    def test_misma_seed_reproduce_instancia(self) -> None:
        generador = GeneradorInstanciasRL()
        instancia_a = generador.generar(91001)
        instancia_b = generador.generar(91001)
        self.assertEqual(firma_instancia(instancia_a), firma_instancia(instancia_b))

    def test_instancias_generadas_son_validas(self) -> None:
        generador = GeneradorInstanciasRL()
        for seed in range(91000, 91100):
            instancia = generador.generar(seed)
            errores = validar_instancia(instancia)
            self.assertFalse(errores, msg=f'Seed={seed}: ' + ' | '.join(errores))
            self.assertGreaterEqual(len(instancia.pedidos), 4)
            self.assertLessEqual(len(instancia.pedidos), 8)
            for pedido in instancia.pedidos:
                self.assertGreater(pedido.unidades_capacidad, 0)
                self.assertLessEqual(pedido.unidades_capacidad, instancia.capacidad_camion)
                self.assertGreaterEqual(pedido.hora_desde_min, instancia.hora_inicio_turno_min)
                self.assertLessEqual(pedido.hora_hasta_min, instancia.hora_fin_objetivo_min)
                self.assertLess(pedido.hora_desde_min, pedido.hora_hasta_min)

    def test_particiones_reales_del_generador_no_comparten_direcciones(self) -> None:
        catalogo = crear_catalogo_prueba(cantidad=60)
        catalogos_por_particion: dict[ParticionDemandaReal, CatalogoDemandaReal] = {}
        for particion in ParticionDemandaReal:
            configuracion = ConfiguracionGeneradorInstancias(min_pedidos_finales=3, max_pedidos_finales=3, probabilidad_pedido_mayor_capacidad=0.0, modo_demanda_geografica=ModoDemandaGeografica.REAL, particion_demanda_real=particion)
            generador = GeneradorInstanciasRL(configuracion=configuracion, catalogo_demanda_real=catalogo)
            catalogo_efectivo = generador.catalogo_demanda_real
            self.assertIsNotNone(catalogo_efectivo)
            assert catalogo_efectivo is not None
            catalogos_por_particion[particion] = catalogo_efectivo
        claves_train = catalogos_por_particion[ParticionDemandaReal.ENTRENAMIENTO].claves_direccion_fuente
        claves_validation = catalogos_por_particion[ParticionDemandaReal.VALIDACION].claves_direccion_fuente
        claves_test = catalogos_por_particion[ParticionDemandaReal.PRUEBA].claves_direccion_fuente
        self.assertFalse(claves_train & claves_validation)
        self.assertFalse(claves_train & claves_test)
        self.assertFalse(claves_validation & claves_test)
if __name__ == '__main__':
    unittest.main()
