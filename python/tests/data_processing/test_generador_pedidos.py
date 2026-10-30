from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from planner.integration.order_excel_import import importar_pedidos_excel
from scripts.data_processing.generar_pedidos_dia import (
    DEFAULT_CANT_PEDIDOS_CSV,
    DEFAULT_GEOCODING_CSV,
    DEFAULT_HISTORICO_CSV,
    DEFAULT_PROBABILIDADES_CSV,
    NOMBRE_METADATA_GENERACION,
    cargar_cantidades_pedidos_turnos,
    cargar_probabilidades_demanda,
    cargar_y_analizar_historico,
    ejecutar_generacion,
    generar_pedidos_para_un_dia,
    parsear_rango_pedidos,
    sortear_cantidad_pedidos_turno,
)
from scripts.simulation.run_simulated_comparison import (
    NOMBRE_METADATA_EXPERIMENTO,
    NOMBRE_RESUMEN_CSV,
    NOMBRE_RESUMEN_JSON,
    ejecutar_experimento_comparacion,
)

PYTHON_ROOT = Path(__file__).resolve().parents[2]


def test_parsear_rango_pedidos():
    assert parsear_rango_pedidos("10") == (10, 10)
    assert parsear_rango_pedidos("8-12") == (8, 12)
    assert parsear_rango_pedidos("historico") is None
    assert parsear_rango_pedidos("AUTO") is None
    with pytest.raises(ValueError):
        parsear_rango_pedidos("12-8")
    with pytest.raises(ValueError):
        parsear_rango_pedidos("0")


def test_cantidades_pedidos_turnos_y_reglas_negocio():
    import random

    assert DEFAULT_CANT_PEDIDOS_CSV.exists(), "Debe existir data/raw/cant_pedidos_turnos.csv"
    assert DEFAULT_GEOCODING_CSV.exists(), "Debe existir direcciones_geolocalizadas.csv"
    assert DEFAULT_PROBABILIDADES_CSV.exists(), "Debe existir probabilidades_demanda.csv"

    conteos = cargar_cantidades_pedidos_turnos(DEFAULT_CANT_PEDIDOS_CSV)
    assert len(conteos) == 70
    assert min(conteos) == 7
    assert max(conteos) == 32

    por_cb, por_b = cargar_probabilidades_demanda(DEFAULT_PROBABILIDADES_CSV)
    assert len(por_cb) > 0 and len(por_b) > 0

    ubicaciones = cargar_y_analizar_historico(
        DEFAULT_GEOCODING_CSV,
        ruta_probabilidades=DEFAULT_PROBABILIDADES_CSV,
    )
    assert len(ubicaciones) > 500
    coords_exactas_historico = {(round(u.latitud, 6), round(u.longitud, 6)) for u in ubicaciones}

    rng = random.Random(20261029)
    tipos_ventana_vistos: set[str] = set()
    unidades_por_turno: list[int] = []
    volcadores_totales = 0
    n_turnos_muestra = 80

    for idx_turno in range(1, n_turnos_muestra + 1):
        cant = sortear_cantidad_pedidos_turno(conteos, rng)
        assert 7 <= cant <= 32
        turno_str = "MANANA" if idx_turno % 2 == 1 else "TARDE"
        inicio_str = "07:30" if turno_str == "MANANA" else "14:00"
        fin_str = "12:00" if turno_str == "MANANA" else "17:00"

        pedidos = generar_pedidos_para_un_dia(
            ubicaciones=ubicaciones,
            cantidad_pedidos=cant,
            turno=turno_str,
            dia_numero=idx_turno,
            rng=rng,
            max_tareas=None,
        )
        assert len(pedidos) == cant
        unidades_por_turno.append(sum(p.unidades for p in pedidos))

        for p in pedidos:
            assert set(p.a_dict().keys()) == {
                "pedido_id",
                "latitud",
                "longitud",
                "unidades",
                "requiere_volcador",
                "tiene_ventana",
                "hora_desde",
                "hora_hasta",
                "turno",
            }
            # 1. Unidades siempre enteras positivas
            assert isinstance(p.unidades, int) and p.unidades >= 1
            # 2. Nunca asignar la latitud y longitud exacta de un pedido histórico
            assert (p.latitud, p.longitud) not in coords_exactas_historico
            # 3. Volcador coherente
            assert p.requiere_volcador in ("SI", "NO")
            if p.requiere_volcador == "SI":
                volcadores_totales += 1
            # 4. Validar las 4 categorías de ventana horaria
            if p.tiene_ventana == "NO":
                tipos_ventana_vistos.add("SIN_VENTANA")
                assert p.hora_desde == inicio_str and p.hora_hasta == fin_str
            elif p.hora_desde == inicio_str and p.hora_hasta < fin_str:
                tipos_ventana_vistos.add("ANTES_DE")
            elif p.hora_desde > inicio_str and p.hora_hasta == fin_str:
                tipos_ventana_vistos.add("DESPUES_DE")
            elif inicio_str < p.hora_desde < p.hora_hasta < fin_str:
                tipos_ventana_vistos.add("ENTRE_X_Y")
            else:
                pytest.fail(f"Ventana inválida: {p.tiene_ventana} {p.hora_desde}-{p.hora_hasta}")

    # Deben aparecer las 4 modalidades de ventana horaria
    assert tipos_ventana_vistos == {
        "SIN_VENTANA",
        "ANTES_DE",
        "DESPUES_DE",
        "ENTRE_X_Y",
    }
    # Promedio de unidades por turno en torno a ~48 unidades (~6 camiones llenos por turno)
    promedio_unidades_turno = sum(unidades_por_turno) / n_turnos_muestra
    assert 36.0 <= promedio_unidades_turno <= 64.0
    # Volcador ~1 cada 2 días = 1 cada 4 turnos (80 turnos -> esperado ~20 volcadores, rango razonable 8..36)
    assert 6 <= volcadores_totales <= 38


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
        assert p.requiere_volcador in ("SI", "NO")

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
        assert carpeta_timestamp.name.startswith("seed_1234_")

        ruta_metadata = carpeta_timestamp / NOMBRE_METADATA_GENERACION
        assert ruta_metadata.is_file()
        metadata = json.loads(ruta_metadata.read_text(encoding="utf-8"))
        assert metadata["seed_inicial"] == 1234
        assert len(metadata["turnos"]) == dias_a_generar
        assert all(t["seed_escenario"] > 0 for t in metadata["turnos"])
        assert all(t["tareas_estimadas"] <= 12 for t in metadata["turnos"])

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
        _ = ejecutar_generacion(
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


def test_experimento_5_algoritmos_3_repeticiones_y_carpeta_con_semilla():
    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        gen_dir = base_dir / "generados"
        res_dir = base_dir / "resultados"

        # 1. Generar 2 turnos con semilla aleatoria (seed=None) y verificar que queda grabada
        generados = ejecutar_generacion(
            dias_n=2,
            rango_pedidos=(4, 6),
            ruta_historico=DEFAULT_HISTORICO_CSV,
            directorio_salida=gen_dir,
            formato="ambos",
            turno="MIXTO",
            seed=None,
            usar_timestamp=True,
        )
        carpeta_turnos = generados[0][0].parent
        meta_gen = json.loads(
            (carpeta_turnos / NOMBRE_METADATA_GENERACION).read_text(encoding="utf-8")
        )
        seed_gen = meta_gen["seed_inicial"]
        assert isinstance(seed_gen, int) and seed_gen > 0
        assert carpeta_turnos.name.startswith(f"seed_{seed_gen}_")

        # 2. Correr los 5 algoritmos 3 veces sobre cada turno generado (con semilla aleatoria grabada)
        meta_exp = ejecutar_experimento_comparacion(
            raiz_python=PYTHON_ROOT,
            directorio_turnos=carpeta_turnos,
            repeticiones=3,
            seed=None,
            directorio_salida=res_dir,
            solo_estimado=True,
        )

        assert meta_exp["completa"] is True
        seed_exp = meta_exp["seed_inicial"]
        assert isinstance(seed_exp, int) and seed_exp > 0
        assert meta_exp["seed_generacion_origen"] == seed_gen

        carpeta_exp = Path(meta_exp["carpeta_experimento"])
        assert carpeta_exp.name.startswith(f"seed_{seed_exp}_")
        assert (carpeta_exp / NOMBRE_METADATA_EXPERIMENTO).is_file()
        assert (carpeta_exp / NOMBRE_RESUMEN_CSV).is_file()
        assert (carpeta_exp / NOMBRE_RESUMEN_JSON).is_file()

        filas_resumen = json.loads((carpeta_exp / NOMBRE_RESUMEN_JSON).read_text(encoding="utf-8"))
        # 2 turnos * 3 repeticiones * 5 algoritmos = 30 ejecuciones registradas
        assert len(filas_resumen) == 2 * 3 * 5
        assert all(f["factible_estimado"] is True for f in filas_resumen)
        assert all(f["seed_algoritmo"] > 0 for f in filas_resumen)

        # Verificar que las 3 repeticiones de un mismo turno usaron 3 semillas distintas
        turnos_eval = meta_exp["turnos_evaluados"]
        assert len(turnos_eval) == 2
        for turno_info in turnos_eval:
            reps = turno_info["repeticiones"]
            assert len(reps) == 3
            seeds_ejec = {r["seed_ejecucion"] for r in reps}
            assert len(seeds_ejec) == 3
            for r in reps:
                assert set(r["semillas_por_metodo"].keys()) == {
                    "RL",
                    "HIBRIDO",
                    "GREEDY",
                    "RANDOM",
                    "GA",
                }
                assert (carpeta_exp / r["archivo_resultado"]).is_file()
