from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from planner.integration.order_excel_import import (
    ErrorImportacionExcel,
    ResultadoImportacionExcel,
    importar_pedidos_excel,
)


PROTOCOL_VERSION = "PEDXLSX1"


def importar_pedidos_excel_pypeline(
    archivo: str,
    turno: str,
    capacidad_camion: int = 8,
    max_tareas: int = 30,
) -> str:
    """
    Devuelve un protocolo de texto simple y seguro para AnyLogic.

    Encabezado OK:
        PEDXLSX1|OK|originales|tareas|filas_ignoradas|hoja

    Pedido:
        P|fila|id|cliente|direccion|barrio|unidades|lat|lon|
          volcador|ventana|desde_min|hasta_min|observaciones

    Encabezado ERROR:
        PEDXLSX1|ERROR|cantidad

    Error:
        E|fila|campo|mensaje

    Los campos textuales están codificados con URL encoding UTF-8.
    """
    try:
        resultado = importar_pedidos_excel(
            archivo,
            turno=turno,
            capacidad_camion=capacidad_camion,
            max_tareas=max_tareas,
        )

    except ErrorImportacionExcel as ex:
        lineas = [
            (
                f"{PROTOCOL_VERSION}|ERROR|"
                f"{len(ex.errores)}"
            )
        ]

        lineas.extend(
            "|".join(
                [
                    "E",
                    str(error.fila),
                    _codificar(error.campo),
                    _codificar(error.mensaje),
                ]
            )
            for error in ex.errores
        )

        return "\n".join(lineas)

    except Exception as ex:
        return "\n".join(
            [
                f"{PROTOCOL_VERSION}|ERROR|1",
                "|".join(
                    [
                        "E",
                        "0",
                        _codificar("archivo"),
                        _codificar(
                            f"{type(ex).__name__}: {ex}"
                        ),
                    ]
                ),
            ]
        )

    return _serializar_ok(resultado)


def _serializar_ok(
    resultado: ResultadoImportacionExcel,
) -> str:
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
    return quote(
        str(valor),
        safe="",
        encoding="utf-8",
        errors="strict",
    )
