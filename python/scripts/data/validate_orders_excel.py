from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


RAIZ_PYTHON = Path(__file__).resolve().parents[2]

if str(RAIZ_PYTHON) not in sys.path:
    sys.path.insert(0, str(RAIZ_PYTHON))

from planner.integration.order_excel_import import (  # noqa: E402
    ErrorImportacionExcel,
    formatear_minutos_hora,
    importar_pedidos_excel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Valida una planilla .xlsx de pedidos "
            "antes de importarla en AnyLogic."
        )
    )
    parser.add_argument("archivo", type=Path)
    parser.add_argument(
        "--turno",
        choices=["MANANA", "TARDE"],
        required=True,
    )
    parser.add_argument(
        "--capacidad-camion",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-tareas",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="como_json",
    )
    argumentos = parser.parse_args()

    try:
        resultado = importar_pedidos_excel(
            argumentos.archivo,
            turno=argumentos.turno,
            capacidad_camion=(
                argumentos.capacidad_camion
            ),
            max_tareas=argumentos.max_tareas,
        )

    except ErrorImportacionExcel as ex:
        if argumentos.como_json:
            print(
                json.dumps(
                    {
                        "estado": "ERROR",
                        "errores": [
                            {
                                "fila": error.fila,
                                "campo": error.campo,
                                "mensaje": error.mensaje,
                            }
                            for error in ex.errores
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print("RESULTADO: PLANILLA_INVALIDA")
            for error in ex.errores:
                print(
                    f"- Fila {error.fila} | "
                    f"{error.campo}: {error.mensaje}"
                )

        return 2

    if argumentos.como_json:
        print(
            json.dumps(
                {
                    "estado": "OK",
                    "archivo": str(resultado.archivo),
                    "hoja": resultado.hoja,
                    "turno": resultado.turno,
                    "pedidos_originales": len(
                        resultado.pedidos
                    ),
                    "tareas_estimadas": (
                        resultado.tareas_estimadas
                    ),
                    "filas_ignoradas": (
                        resultado.filas_ignoradas
                    ),
                    "pedidos": [
                        {
                            "fila": pedido.fila_excel,
                            "pedido_id": pedido.pedido_id,
                            "unidades": pedido.unidades,
                            "ventana": (
                                None
                                if not pedido.tiene_ventana
                                else {
                                    "desde": (
                                        formatear_minutos_hora(
                                            pedido.hora_desde_min
                                        )
                                    ),
                                    "hasta": (
                                        formatear_minutos_hora(
                                            pedido.hora_hasta_min
                                        )
                                    ),
                                }
                            ),
                        }
                        for pedido in resultado.pedidos
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("RESULTADO: PLANILLA_VALIDA")
        print(f"Archivo: {resultado.archivo}")
        print(f"Hoja: {resultado.hoja}")
        print(f"Turno: {resultado.turno}")
        print(
            "Pedidos originales: "
            f"{len(resultado.pedidos)}"
        )
        print(
            "Tareas después del split: "
            f"{resultado.tareas_estimadas}"
        )
        print(
            "Filas vacías ignoradas: "
            f"{resultado.filas_ignoradas}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
