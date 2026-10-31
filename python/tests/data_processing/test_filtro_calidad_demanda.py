"""Verifica que el generador excluya ubicaciones marcadas para revision."""

from pathlib import Path

from scripts.data_processing.generar_pedidos_dia import (
    cargar_y_analizar_historico,
)


def test_historico_excluye_ubicaciones_en_revision(tmp_path: Path) -> None:
    archivo = tmp_path / "demanda_geografica.csv"
    archivo.write_text(
        "latitud,longitud,ciudad,barrio,estado_calidad\n"
        "-32.8500,-60.7200,Granadero Baigorria,Centro,APTO_ENTRENAMIENTO\n"
        "-31.2414936,-61.4720621,Rafaela,Italia,REVISAR_DISTANCIA\n",
        encoding="utf-8",
    )

    ubicaciones = cargar_y_analizar_historico(
        archivo,
        ruta_probabilidades=tmp_path / "probabilidades_inexistentes.csv",
    )

    assert len(ubicaciones) == 1
    assert ubicaciones[0].latitud == -32.8500
    assert ubicaciones[0].longitud == -60.7200
