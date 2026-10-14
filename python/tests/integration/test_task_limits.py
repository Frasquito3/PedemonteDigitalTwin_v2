from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

import pytest
from openpyxl import Workbook

from planner.integration.alpyne_codec import MAX_PEDIDOS_ALPYNE
from planner.integration.instance_vector_codec import MAX_PEDIDOS
from planner.integration.order_excel_import import (
    ErrorImportacionExcel,
    importar_pedidos_excel,
)
from planner.integration.order_import_bridge import (
    importar_pedidos_excel_pypeline,
    importar_pedidos_excel_pypeline_v2,
)


def _crear_excel(ruta: Path, cantidad: int) -> None:
    libro = Workbook()
    hoja = libro.active
    assert hoja is not None
    hoja.title = "Pedidos"
    hoja.append([
        "pedido_id", "cliente", "direccion", "barrio", "unidades",
        "latitud", "longitud", "requiere_volcador", "tiene_ventana",
        "hora_desde", "hora_hasta", "observaciones",
    ])
    for numero in range(1, cantidad + 1):
        hoja.append([
            f"ORD-{numero:03d}", "Cliente", "Calle", "Centro", 1,
            -32.85, -60.72, "NO", "NO", None, None, "",
        ])
    libro.save(ruta)
    libro.close()


def test_constantes_tareas_compartidas() -> None:
    assert MAX_PEDIDOS_ALPYNE == MAX_PEDIDOS == 20


@pytest.mark.parametrize("cantidad,permitido", [(20, True), (21, False)])
def test_excel_tiene_el_mismo_limite_por_defecto(
    cantidad: int, permitido: bool,
) -> None:
    with TemporaryDirectory() as temporal:
        ruta = Path(temporal) / "pedidos.xlsx"
        _crear_excel(ruta, cantidad)
        if permitido:
            resultado = importar_pedidos_excel(ruta, turno="MANANA")
            assert resultado.tareas_estimadas == 20
        else:
            with pytest.raises(ErrorImportacionExcel, match="máximo"):
                importar_pedidos_excel(ruta, turno="MANANA")


@pytest.mark.parametrize(
    "funcion,version",
    [
        (importar_pedidos_excel_pypeline, "PEDXLSX1"),
        (importar_pedidos_excel_pypeline_v2, "PEDXLSX2"),
    ],
)
@pytest.mark.parametrize("cantidad,permitido", [(20, True), (21, False)])
def test_protocolos_excel_respetan_el_limite_por_defecto(
    funcion, version: str, cantidad: int, permitido: bool,
) -> None:
    with TemporaryDirectory() as temporal:
        ruta = Path(temporal) / "pedidos.xlsx"
        _crear_excel(ruta, cantidad)
        salida = funcion(str(ruta), "MANANA")
        if permitido:
            assert salida.startswith(f"{version}|OK|20|20|")
        else:
            assert salida.startswith(f"{version}|ERROR|")
            assert "máximo" in unquote(salida)
