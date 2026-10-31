from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from scripts.data_processing.estadistica_demanda import (
    calcular_probabilidades_demanda,
    calcular_resumen_ciudades,
)
from scripts.data_processing.extraccion_ocr import (
    cargar_progreso as cargar_progreso_ocr,
)
from scripts.data_processing.extraccion_ocr import (
    cargar_resultado as cargar_resultado_ocr,
)
from scripts.data_processing.extraccion_ocr import (
    guardar_progreso as guardar_progreso_ocr,
)
from scripts.data_processing.extraccion_ocr import (
    guardar_resultado as guardar_resultado_ocr,
)
from scripts.data_processing.extraccion_ocr import (
    obtener_cliente_genai,
)
from scripts.data_processing.geocoding import (
    cargar_progreso as cargar_progreso_geo,
)
from scripts.data_processing.geocoding import (
    guardar_progreso as guardar_progreso_geo,
)
from scripts.data_processing.geocoding import (
    normalizar_ciudad_osm,
)

PYTHON_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PYTHON_ROOT.parent


def test_env_example_y_gitignore_configurados():
    """Verifica que .env.example exista y que .gitignore bloquee la fuga de .env."""
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.exists(), "Falta el archivo .env.example en la raíz"

    contenido_example = env_example.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in contenido_example

    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists(), "Falta .gitignore en la raíz"
    lineas_ignore = gitignore.read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in lineas_ignore]


def test_calcular_probabilidades_demanda_mock():
    """Valida el cálculo de frecuencias y suma de probabilidades de demanda."""
    datos = {
        "ciudad": ["Granadero Baigorria", "Granadero Baigorria", "Rosario", "Granadero Baigorria"],
        "barrio": ["Los Robles", "Los Robles", "Centro", "San Miguel"],
    }
    df = pd.DataFrame(datos)

    resultado = calcular_probabilidades_demanda(df)

    assert "cantidad_pedidos" in resultado.columns
    assert "probabilidad_demanda" in resultado.columns
    assert len(resultado) == 3

    # El total de pedidos debe ser 4 y las probabilidades sumar 1.0
    assert resultado["cantidad_pedidos"].sum() == 4
    assert pytest.approx(resultado["probabilidad_demanda"].sum(), abs=1e-3) == 1.0

    fila_top = resultado.iloc[0]
    assert fila_top["barrio"] == "Los Robles"
    assert fila_top["cantidad_pedidos"] == 2
    assert fila_top["probabilidad_demanda"] == 0.5


def test_calcular_resumen_ciudades_mock():
    """Valida el porcentaje agregado por ciudad."""
    datos = {
        "ciudad": ["Granadero Baigorria", "Granadero Baigorria", "Rosario", "Granadero Baigorria"],
        "barrio": ["B1", "B2", "B3", "B4"],
    }
    df = pd.DataFrame(datos)

    resumen = calcular_resumen_ciudades(df)

    assert "ciudad" in resumen.columns
    assert "porcentaje" in resumen.columns
    assert len(resumen) == 2
    assert pytest.approx(resumen["porcentaje"].sum(), abs=1e-2) == 100.0


def test_archivos_de_datos_procesados_existen():
    """Comprueba que los datasets procesados y salidas de OCR existan y sean válidos."""
    json_ocr = PYTHON_ROOT / "data" / "processed" / "extraccion" / "direcciones.json"
    csv_geo = PYTHON_ROOT / "data" / "processed" / "geocoding" / "direcciones_geolocalizadas.csv"
    csv_prob = PYTHON_ROOT / "data" / "processed" / "estadistica" / "probabilidades_demanda.csv"
    csv_resumen = PYTHON_ROOT / "data" / "processed" / "estadistica" / "resumen_ciudades.csv"
    img_dir = PYTHON_ROOT / "data" / "raw" / "hojas_pedidos"

    assert json_ocr.exists(), "Falta direcciones.json en data/processed/extraccion"
    assert csv_geo.exists(), "Falta direcciones_geolocalizadas.csv en data/processed/geocoding"
    assert csv_prob.exists(), "Falta probabilidades_demanda.csv en data/processed/estadistica"
    assert csv_resumen.exists(), "Falta resumen_ciudades.csv en data/processed/estadistica"
    assert img_dir.exists(), "Falta la carpeta raw/hojas_pedidos"

    # Verificar que el JSON tenga contenido válido
    with open(json_ocr, encoding="utf-8") as f:
        direcciones = json.load(f)
    assert len(direcciones) > 0

    # Verificar que el CSV tenga registros con coordenadas
    df_geo = pd.read_csv(csv_geo)
    assert len(df_geo) > 0
    assert "latitud" in df_geo.columns
    assert "longitud" in df_geo.columns
    assert df_geo["latitud"].notna().all()


def test_normalizar_ciudad_osm():
    """Verifica la normalización de cadenas OpenStreetMap a municipios estándar."""
    osm_sample = "1476, Callao, San Miguel, Granadero Baigorria, Municipio de Granadero Baigorria, Gran Rosario"
    assert normalizar_ciudad_osm(osm_sample, "Rosario") == "Granadero Baigorria"

    osm_ibarlucea = "Ruta 34, Ybarlucea, Municipio de Ybarlucea, Santa Fe"
    assert normalizar_ciudad_osm(osm_ibarlucea, "Ibarlucea") == "Ibarlucea"


def test_validacion_api_key_inexistente(monkeypatch):
    """Verifica que se lance ValueError explicativo si no se define la API Key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No se encontró una API Key válida"):
        obtener_cliente_genai()


def test_escritura_atomica_ocr_y_geocoding_preserva_archivo_ante_fallo():
    """Verifica que las escrituras atómicas persistan sin dejar .tmp y preserven el archivo previo ante fallos."""
    with TemporaryDirectory() as temporal_str:
        temporal = Path(temporal_str)
        ruta_progreso_ocr = temporal / "progreso_ocr.json"
        ruta_resultado_ocr = temporal / "direcciones.json"
        ruta_progreso_geo = temporal / "progreso_geo.json"

        guardar_progreso_ocr(ruta_progreso_ocr, {"img_02.jpg", "img_01.jpg"})
        assert cargar_progreso_ocr(ruta_progreso_ocr) == {"img_01.jpg", "img_02.jpg"}

        datos_iniciales = [{"calle": "San Martín", "altura": "1200"}]
        guardar_resultado_ocr(ruta_resultado_ocr, datos_iniciales)
        assert cargar_resultado_ocr(ruta_resultado_ocr) == datos_iniciales

        guardar_progreso_geo(ruta_progreso_geo, 1, datos_iniciales, [])
        estado_geo = cargar_progreso_geo(ruta_progreso_geo)
        assert estado_geo["procesados"] == 1
        assert estado_geo["validos"] == datos_iniciales

        # Si ocurre un error de serialización a mitad de escritura, el archivo original queda intacto
        # y no quedan archivos temporales (.tmp) huérfanos.
        with pytest.raises(TypeError):
            guardar_resultado_ocr(ruta_resultado_ocr, [{"invalido": object()}])  # type: ignore[dict-item]

        assert cargar_resultado_ocr(ruta_resultado_ocr) == datos_iniciales
        archivos_tmp = list(temporal.glob("*.tmp"))
        assert archivos_tmp == []
