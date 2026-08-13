from dataclasses import replace
from math import ceil

from planner.core.schema import InstanciaTurno, PedidoInput


def preprocesar_pedidos(
    pedidos: list[PedidoInput],
    capacidad_camion: int,
) -> list[PedidoInput]:
    if capacidad_camion <= 0:
        raise ValueError(
            "capacidad_camion debe ser > 0."
        )

    ids_originales: set[str] = set()

    resultado: list[PedidoInput] = []

    for pedido in pedidos:
        if not pedido.pedido_id.strip():
            raise ValueError(
                "Existe un pedido con pedido_id vacío."
            )

        if pedido.pedido_id in ids_originales:
            raise ValueError(
                "pedido_id duplicado antes del "
                "preprocesamiento: "
                f"{pedido.pedido_id}"
            )

        ids_originales.add(
            pedido.pedido_id
        )

        if pedido.unidades_capacidad <= 0:
            raise ValueError(
                f"El pedido {pedido.pedido_id} "
                "debe tener unidades > 0."
            )

        if (
            pedido.numero_parte != 1
            or pedido.total_partes != 1
        ):
            raise ValueError(
                f"El pedido {pedido.pedido_id} "
                "ya parece preprocesado. "
                "No debe aplicarse split dos veces."
            )

        # =================================================
        # PEDIDO QUE YA CABE
        # =================================================

        if (
            pedido.unidades_capacidad
            <= capacidad_camion
        ):
            resultado.append(
                replace(
                    pedido,
                    pedido_original_id=(
                        pedido.pedido_original_id
                        or pedido.pedido_id
                    ),
                    numero_parte=1,
                    total_partes=1,
                )
            )

            continue

        # =================================================
        # PEDIDO MAYOR QUE LA CAPACIDAD
        # =================================================

        total_partes = ceil(
            pedido.unidades_capacidad
            / capacidad_camion
        )

        unidades_restantes = (
            pedido.unidades_capacidad
        )

        for numero_parte in range(
            1,
            total_partes + 1,
        ):
            unidades_parte = min(
                capacidad_camion,
                unidades_restantes,
            )

            resultado.append(
                replace(
                    pedido,
                    pedido_id=(
                        f"{pedido.pedido_id}"
                        f"-P{numero_parte}"
                    ),
                    pedido_original_id=pedido.pedido_id,
                    numero_parte=numero_parte,
                    total_partes=total_partes,
                    unidades_capacidad=unidades_parte,
                )
            )

            unidades_restantes -= (
                unidades_parte
            )

    ids_generados = [
        pedido.pedido_id
        for pedido in resultado
    ]

    if len(ids_generados) != len(set(ids_generados)):
        raise ValueError(
            "El split generó IDs duplicados. "
            "Revisar los pedido_id de entrada."
        )

    return resultado


def preprocesar_instancia(
    instancia: InstanciaTurno,
) -> InstanciaTurno:
    return replace(
        instancia,
        pedidos=preprocesar_pedidos(
            instancia.pedidos,
            instancia.capacidad_camion,
        ),
    )