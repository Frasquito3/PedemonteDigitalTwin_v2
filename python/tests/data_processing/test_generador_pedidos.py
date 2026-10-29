from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

from planner.integration.order_excel_import import importar_pedidos_excel
from scripts.data_processing.generar_pedidos_dia import (
    cargar_y_analizar_historico,
    generar_pedidos_para_un_dia,
    ejecutar_generacion,
    parsear_rango_pedidos,
    DEFAULT_HISTORICO_CSV,
)

PYTHON_ROOT = Path(__file__).resolve().parents[2]


def test_parsear_rango_pedidos():
    assert parsear_rango_pedidos("10") == (10, 10)
    assert parsear_rango_pedidos("8-12") == (8, 12)
    with pytest.raises(ValueError):
        parsear_rango_pedidos("12-8")
    with pytest.raises(ValueError):
        parsear_rango_pedidos("0")


def test_cargar_y_analizar_historico_real():
    assert DEFAULT_HISTORICO_CSV.exists(), "El dataset demanda_geografica.csv debe existir"
    ubicaciones = cargar_y_analizar_historico(DEFAULT_HISTORICO_CSV)
    assert len(ubicaciones) > 100
    primera = ubicaciones[0]
    assert primera.latitud != 0.0
    assert primera.longitud != 0.0
    assert primera.peso_frecuencia >= 1.0


def test_generar_pedidos_un_dia_manana_y_tarde():
    ubicaciones = cargar_y_analizar_historico(DEFAULT_HISTORICO_CSV)

    # Test Turno MAÑANA
    pedidos_manana = generar_pedidos_para_un_dia(
        ubicaciones=ubicaciones,
        cantidad_pedidos=6,
        turno="MANANA",
        dia_numero=1,
    )
    assert len(pedidos_manana) == 6
    for p in pedidos_manana:
        assert p.turno == "MANANA"
        assert p.unidades >= 1
        if p.requiere_volcador == "SI":
            assert p.unidades_volcador == p.unidades
        else:
            assert p.unidades_volcador == 0

        # Verificar rango de horas: 07:30 a 12:00
        assert "07:30" <= p.hora_desde <= "12:00"
        assert "07:30" <= p.hora_hasta <= "12:00"
        assert p.hora_desde < p.hora_hasta

    # Test Turno TARDE
    pedidos_tarde = generar_pedidos_para_un_dia(
        ubicaciones=ubicaciones,
        cantidad_pedidos=4,
        turno="TARDE",
        dia_numero=2,
    )
    assert len(pedidos_tarde) == 4
    for p in pedidos_tarde:
        assert p.turno == "TARDE"
        # Rango de horas: 14:00 a 17:00
        assert "14:00" <= p.hora_desde <= "17:00"
        assert "14:00" <= p.hora_hasta <= "17:00"
        assert p.hora_desde < p.hora_hasta


def test_ejecutar_generacion_n_dias_y_compatibilidad_excel():
    with TemporaryDirectory() as temp_dir:
        salida_dir = Path(temp_dir)
        dias_a_generar = 3

        # 1. Prueba con timestamp activado (por defecto)
        resultados = ejecutar_generacion(
            dias_n=dias_a_generar,
            rango_pedidos=(5, 7),
            ruta_historico=DEFAULT_HISTORICO_CSV,
            directorio_salida=salida_dir,
            formato="ambos",
            turno="MANANA",
            crear_consolidado=True,
            seed=1234,
            usar_timestamp=True,
        )

        assert len(resultados) == dias_a_generar
        carpeta_timestamp = resultados[0][0].parent
        assert carpeta_timestamp != salida_dir
        assert carpeta_timestamp.parent == salida_dir

        for i in range(1, dias_a_generar + 1):
            csv_path = carpeta_timestamp / f"pedidos_dia_{i:02d}.csv"
            xlsx_path = carpeta_timestamp / f"pedidos_dia_{i:02d}.xlsx"

            assert csv_path.exists(), f"Debe existir {csv_path}"
            assert xlsx_path.exists(), f"Debe existir {xlsx_path}"

            # Validar compatibilidad estricta con el importador de AnyLogic
            res_importacion = importar_pedidos_excel(xlsx_path, turno="MANANA")
            assert len(res_importacion.pedidos) >= 5
            assert res_importacion.tareas_estimadas >= len(res_importacion.pedidos)

        # Validar archivos consolidados
        assert (carpeta_timestamp / "pedidos_dia_consolidado.csv").exists()
        assert (carpeta_timestamp / "pedidos_dia_consolidado.xlsx").exists()

        # 2. Prueba con timestamp desactivado
        resultados_sin_ts = ejecutar_generacion(
            dias_n=1,
            rango_pedidos=(5, 5),
            ruta_historico=DEFAULT_HISTORICO_CSV,
            directorio_salida=salida_dir,
            formato="csv",
            turno="MANANA",
            prefijo="pedidos_directo",
            seed=42,
            usar_timestamp=False,
        )
        assert (salida_dir / "pedidos_directo_01.csv").exists()

