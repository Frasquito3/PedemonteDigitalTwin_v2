from __future__ import annotations

from dataclasses import replace
from math import ceil
from typing import Mapping

from planner.core.schema import InstanciaTurno, PedidoInput


def preprocesar_pedidos(
    pedidos: list[PedidoInput],
    capacidad_camion: int,
    *,
    unidades_volcador_por_pedido: Mapping[str, int] | None = None,
) -> list[PedidoInput]:
    """
    Convierte pedidos originales en tareas de hasta `capacidad_camion`.

    Compatibilidad:
    - Sin el mapa opcional, conserva la semántica antigua: si un pedido
      requiere volcador, TODAS sus unidades requieren volcador.
    - Con el mapa, un pedido mixto se separa en tareas normales y
      tareas volcador. Cada tarea mantiene el booleano existente.

    No modifica el schema de PedidoInput ni el vector AnyLogic-Python.
    """
    if capacidad_camion <= 0:
        raise ValueError("capacidad_camion debe ser > 0.")

    cantidades_volcador = (
        {}
        if unidades_volcador_por_pedido is None
        else dict(unidades_volcador_por_pedido)
    )

    ids_originales: set[str] = set()
    resultado: list[PedidoInput] = []

    for pedido in pedidos:
        pedido_id = pedido.pedido_id

        if not pedido_id.strip():
            raise ValueError("Existe un pedido con pedido_id vacío.")

        if pedido_id in ids_originales:
            raise ValueError(
                "pedido_id duplicado antes del preprocesamiento: "
                f"{pedido_id}"
            )
        ids_originales.add(pedido_id)

        if pedido.unidades_capacidad <= 0:
            raise ValueError(
                f"El pedido {pedido_id} debe tener unidades > 0."
            )

        if pedido.numero_parte != 1 or pedido.total_partes != 1:
            raise ValueError(
                f"El pedido {pedido_id} ya parece preprocesado. "
                "No debe aplicarse split dos veces."
            )

        unidades_totales = pedido.unidades_capacidad

        if pedido_id in cantidades_volcador:
            unidades_volcador = cantidades_volcador[pedido_id]

            if (
                isinstance(unidades_volcador, bool)
                or not isinstance(unidades_volcador, int)
            ):
                raise ValueError(
                    f"El pedido {pedido_id}: unidades_volcador "
                    "debe ser un entero."
                )

            if not 0 <= unidades_volcador <= unidades_totales:
                raise ValueError(
                    f"El pedido {pedido_id}: unidades_volcador "
                    f"debe estar entre 0 y {unidades_totales}."
                )

            if pedido.requiere_volcador != (unidades_volcador > 0):
                raise ValueError(
                    f"El pedido {pedido_id}: requiere_volcador "
                    "no coincide con unidades_volcador."
                )
        else:
            # Pedidos anteriores: el booleano se aplicaba a todo el pedido.
            unidades_volcador = (
                unidades_totales if pedido.requiere_volcador else 0
            )

        unidades_normales = unidades_totales - unidades_volcador

        # Cada grupo es homogéneo. Los normales se enumeran primero;
        # los volcadores después, pero el planificador decide los viajes.
        grupos: list[tuple[int, bool]] = [
            (unidades_normales, False),
            (unidades_volcador, True),
        ]

        total_partes = sum(
            ceil(unidades / capacidad_camion)
            for unidades, _ in grupos
            if unidades > 0
        )

        numero_parte = 0

        for unidades_grupo, es_volcador in grupos:
            unidades_restantes = unidades_grupo

            while unidades_restantes > 0:
                numero_parte += 1
                unidades_parte = min(
                    capacidad_camion,
                    unidades_restantes,
                )

                resultado.append(
                    replace(
                        pedido,
                        pedido_id=(
                            pedido_id
                            if total_partes == 1
                            else f"{pedido_id}-P{numero_parte}"
                        ),
                        pedido_original_id=(
                            (pedido.pedido_original_id or pedido_id)
                            if total_partes == 1
                            else pedido_id
                        ),
                        numero_parte=numero_parte,
                        total_partes=total_partes,
                        unidades_capacidad=unidades_parte,
                        requiere_volcador=es_volcador,
                    )
                )

                unidades_restantes -= unidades_parte

    desconocidos = (
        set(cantidades_volcador) - ids_originales
    )
    if desconocidos:
        raise ValueError(
            "unidades_volcador_por_pedido contiene IDs desconocidos: "
            + ", ".join(sorted(desconocidos))
        )

    ids_generados = [pedido.pedido_id for pedido in resultado]
    if len(ids_generados) != len(set(ids_generados)):
        raise ValueError(
            "El split generó IDs duplicados. "
            "Revisar los pedido_id de entrada."
        )

    return resultado


def preprocesar_instancia(
    instancia: InstanciaTurno,
    *,
    unidades_volcador_por_pedido: Mapping[str, int] | None = None,
) -> InstanciaTurno:
    return replace(
        instancia,
        pedidos=preprocesar_pedidos(
            instancia.pedidos,
            instancia.capacidad_camion,
            unidades_volcador_por_pedido=unidades_volcador_por_pedido,
        ),
    )
