"""Puente de importación de pedidos Excel hacia AnyLogic vía Pypeline.

Este módulo serializa el resultado de la importación y validación de pedidos a formatos
de texto delimitados por tuberías (`PEDXLSX1` y `PEDXLSX2`), con codificación de
caracteres URL-encoding segura para la comunicación interproceso con AnyLogic.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from planner.integration.alpyne_codec import MAX_PEDIDOS_ALPYNE
from planner.integration.order_excel_import import (
    ErrorImportacionExcel,
    ResultadoImportacionExcel,
    importar_pedidos_excel,
)

__all__ = [
    "PROTOCOL_VERSION",
    "importar_pedidos_excel_pypeline",
    "importar_pedidos_excel_pypeline_v2",
]

PROTOCOL_VERSION = "PEDXLSX1"


def importar_pedidos_excel_pypeline(
    archivo: str,
    turno: str,
    capacidad_camion: int = 8,
    max_tareas: int = MAX_PEDIDOS_ALPYNE,
) -> str:
    """Importa pedidos y serializa la respuesta bajo el protocolo PEDXLSX1 para AnyLogic.

    Encabezado OK:
        PEDXLSX1|OK|originales|tareas|filas_ignoradas|hoja|archivo
    Pedido:
        P|fila|id|cliente|direccion|barrio|unidades|lat|lon|volcador|ventana|desde_min|hasta_min|observaciones
    Encabezado ERROR:
        PEDXLSX1|ERROR|cantidad
    Error:
        E|fila|campo|mensaje
    """
    try:
        resultado = importar_pedidos_excel(
            archivo,
            turno=turno,
            capacidad_camion=capacidad_camion,
            max_tareas=max_tareas,
        )
    except ErrorImportacionExcel as ex:
        lineas = [f"{PROTOCOL_VERSION}|ERROR|{len(ex.errores)}"]
        lineas.extend(
            f"E|{error.fila}|{_codificar(error.campo)}|{_codificar(error.mensaje)}"
            for error in ex.errores
        )
        return "\n".join(lineas)
    except Exception as ex:
        return f"{PROTOCOL_VERSION}|ERROR|1\nE|0|{_codificar('archivo')}|{_codificar(f'{type(ex).__name__}: {ex}')}"

    return _serializar_ok(resultado)


def importar_pedidos_excel_pypeline_v2(
    archivo: str,
    turno: str,
    capacidad_camion: int = 8,
    max_tareas: int = MAX_PEDIDOS_ALPYNE,
) -> str:
    """Importa pedidos bajo el protocolo extendido PEDXLSX2 con soporte de carga mixta."""
    version = "PEDXLSX2"
    try:
        resultado = importar_pedidos_excel(
            archivo,
            turno=turno,
            capacidad_camion=capacidad_camion,
            max_tareas=max_tareas,
        )
    except ErrorImportacionExcel as ex:
        return "\n".join(
            [f"{version}|ERROR|{len(ex.errores)}"]
            + [f"E|{e.fila}|{_codificar(e.campo)}|{_codificar(e.mensaje)}" for e in ex.errores]
        )
    except Exception as ex:
        return f"{version}|ERROR|1\nE|0|archivo|{_codificar(f'{type(ex).__name__}: {ex}')}"

    lineas = [
        "|".join(
            [
                version,
                "OK",
                str(len(resultado.pedidos)),
                str(resultado.tareas_estimadas),
                str(resultado.filas_ignoradas),
                _codificar(resultado.hoja),
                _codificar(str(resultado.archivo)),
            ]
        )
    ]
    for pedido in resultado.pedidos:
        lineas.append(
            "|".join(
                [
                    "P",
                    str(pedido.fila_excel),
                    _codificar(pedido.pedido_id),
                    _codificar(pedido.cliente),
                    _codificar(pedido.direccion),
                    _codificar(pedido.barrio),
                    str(pedido.unidades),
                    repr(pedido.latitud),
                    repr(pedido.longitud),
                    "1" if pedido.requiere_volcador else "0",
                    "1" if pedido.tiene_ventana else "0",
                    str(pedido.hora_desde_min),
                    str(pedido.hora_hasta_min),
                    _codificar(pedido.observaciones),
                    str(pedido.unidades_volcador),
                ]
            )
        )
    return "\n".join(lineas)


def _serializar_ok(resultado: ResultadoImportacionExcel) -> str:
    for pedido in resultado.pedidos:
        if 0 < pedido.unidades_volcador < pedido.unidades:
            return (
                f"{PROTOCOL_VERSION}|ERROR|1\n"
                f"E|{pedido.fila_excel}|{_codificar('unidades_volcador')}|"
                + _codificar(
                    "La importación de carga mixta todavía no está habilitada en AnyLogic. "
                    "Falta actualizar el contrato de importación."
                )
            )

    lineas = [
        "|".join(
            [
                PROTOCOL_VERSION,
                "OK",
                str(len(resultado.pedidos)),
                str(resultado.tareas_estimadas),
                str(resultado.filas_ignoradas),
                _codificar(resultado.hoja),
                _codificar(str(resultado.archivo)),
            ]
        )
    ]

    for pedido in resultado.pedidos:
        lineas.append(
            "|".join(
                [
                    "P",
                    str(pedido.fila_excel),
                    _codificar(pedido.pedido_id),
                    _codificar(pedido.cliente),
                    _codificar(pedido.direccion),
                    _codificar(pedido.barrio),
                    str(pedido.unidades),
                    repr(pedido.latitud),
                    repr(pedido.longitud),
                    "1" if pedido.requiere_volcador else "0",
                    "1" if pedido.tiene_ventana else "0",
                    str(pedido.hora_desde_min),
                    str(pedido.hora_hasta_min),
                    _codificar(pedido.observaciones),
                ]
            )
        )

    return "\n".join(lineas)


def _codificar(valor: str | Path) -> str:
    return quote(str(valor), safe="", encoding="utf-8", errors="strict")
