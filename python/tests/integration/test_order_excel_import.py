from __future__ import annotations
from datetime import time
from pathlib import Path
from tempfile import TemporaryDirectory
from openpyxl import Workbook
import pytest
from planner.integration.order_excel_import import ErrorImportacionExcel, formatear_minutos_hora, importar_pedidos_excel, parsear_hora_a_minutos
ENCABEZADOS = ['pedido_id', 'cliente', 'direccion', 'barrio', 'unidades', 'latitud', 'longitud', 'requiere_volcador', 'tiene_ventana', 'hora_desde', 'hora_hasta', 'observaciones']

def _crear_planilla(ruta: Path, filas: list[list[object]]) -> None:
    libro = Workbook()
    hoja = libro.active
    hoja.title = 'Pedidos'
    hoja.append(ENCABEZADOS)
    for fila in filas:
        hoja.append(fila)
    libro.save(ruta)
    libro.close()

def test_importa_hora_texto_y_hora_excel() -> None:
    with TemporaryDirectory() as temporal:
        ruta = Path(temporal) / 'pedidos.xlsx'
        _crear_planilla(ruta, [['P-1', 'Cliente 1', 'Dirección 1', 'Centro', 3, -32.85, -60.72, 'NO', 'SI', '07:45', time(9, 30), ''], ['P-2', 'Cliente 2', 'Dirección 2', 'Norte', 2, -32.84, -60.71, 'SI', 'NO', None, None, '']])
        resultado = importar_pedidos_excel(ruta, turno='MANANA')
        assert len(resultado.pedidos) == 2
        assert resultado.tareas_estimadas == 2
        assert resultado.pedidos[0].hora_desde_min == 465
        assert resultado.pedidos[0].hora_hasta_min == 570
        assert resultado.pedidos[1].hora_desde_min == 450
        assert resultado.pedidos[1].hora_hasta_min == 720

def test_split_estima_tareas_y_respeta_maximo() -> None:
    with TemporaryDirectory() as temporal:
        ruta = Path(temporal) / 'pedidos.xlsx'
        _crear_planilla(ruta, [['P-1', '', '', '', 17, -32.85, -60.72, 'SI', 'NO', None, None, '']])
        resultado = importar_pedidos_excel(ruta, turno='MANANA', capacidad_camion=8, max_tareas=3)
        assert resultado.tareas_estimadas == 3
        with pytest.raises(ErrorImportacionExcel):
            importar_pedidos_excel(ruta, turno='MANANA', capacidad_camion=8, max_tareas=2)

def test_rechaza_id_duplicado_y_ventana_fuera_de_turno() -> None:
    with TemporaryDirectory() as temporal:
        ruta = Path(temporal) / 'pedidos.xlsx'
        _crear_planilla(ruta, [['P-1', '', '', '', 2, -32.85, -60.72, 'NO', 'SI', '07:00', '08:00', ''], ['P-1', '', '', '', 2, -32.84, -60.71, 'NO', 'NO', None, None, '']])
        with pytest.raises(ErrorImportacionExcel) as captura:
            importar_pedidos_excel(ruta, turno='MANANA')
        mensajes = [error.mensaje for error in captura.value.errores]
        assert any(('dentro del turno' in mensaje for mensaje in mensajes))
        assert any(('ID duplicado' in mensaje for mensaje in mensajes))
