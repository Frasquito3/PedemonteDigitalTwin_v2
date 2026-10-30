"""Preprocesamiento y partición (split) de pedidos según capacidad vehicular y requerimiento de descarga."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import ceil

from planner.core.schema import InstanciaTurno, PedidoInput

__all__ = ["preprocesar_pedidos", "preprocesar_instancia"]


def _validar_capacidad(capacidad_camion: int) -> None:
    """Verifica que la capacidad nominal del camión sea estrictamente positiva."""
    if capacidad_camion <= 0:
        raise ValueError("capacidad_camion debe ser > 0.")


def _validar_pedido_base(pedido: PedidoInput, ids_vistos: set[str]) -> None:
    """Comprueba las invariantes básicas de un pedido antes de su particionamiento."""
    pedido_id = pedido.pedido_id

    if not pedido_id.strip():
        raise ValueError("Existe un pedido con pedido_id vacío.")

    if pedido_id in ids_vistos:
        raise ValueError(f"pedido_id duplicado antes del preprocesamiento: {pedido_id}")

    if pedido.unidades_capacidad <= 0:
        raise ValueError(f"El pedido {pedido_id} debe tener unidades > 0.")

    if pedido.numero_parte != 1 or pedido.total_partes != 1:
        raise ValueError(
            f"El pedido {pedido_id} ya parece preprocesado. No debe aplicarse split dos veces."
        )


def _obtener_distribucion_volcador(
    pedido: PedidoInput,
    cantidades_volcador: Mapping[str, int],
) -> tuple[int, int]:
    """Determina la cantidad de unidades normales y de volcador de un pedido."""
    pedido_id = pedido.pedido_id
    unidades_totales = pedido.unidades_capacidad

    if pedido_id in cantidades_volcador:
        unidades_volcador = cantidades_volcador[pedido_id]

        if isinstance(unidades_volcador, bool) or not isinstance(unidades_volcador, int):
            raise ValueError(f"El pedido {pedido_id}: unidades_volcador debe ser un entero.")

        if not 0 <= unidades_volcador <= unidades_totales:
            raise ValueError(
                f"El pedido {pedido_id}: unidades_volcador debe estar entre 0 y {unidades_totales}."
            )

        if pedido.requiere_volcador != (unidades_volcador > 0):
            raise ValueError(
                f"El pedido {pedido_id}: requiere_volcador no coincide con unidades_volcador."
            )
    else:
        # Compatibilidad histórica: el booleano aplica a la totalidad del pedido
        unidades_volcador = unidades_totales if pedido.requiere_volcador else 0

    unidades_normales = unidades_totales - unidades_volcador
    return unidades_normales, unidades_volcador


def _particionar_pedido(
    pedido: PedidoInput,
    capacidad_camion: int,
    unidades_normales: int,
    unidades_volcador: int,
) -> list[PedidoInput]:
    """Subdivide un pedido en fracciones homogéneas que no superen la capacidad del camión."""
    pedido_id = pedido.pedido_id
    grupos: list[tuple[int, bool]] = [
        (unidades_normales, False),
        (unidades_volcador, True),
    ]

    total_partes = sum(ceil(unidades / capacidad_camion) for unidades, _ in grupos if unidades > 0)

    partes_generadas: list[PedidoInput] = []
    numero_parte = 0

    for unidades_grupo, es_volcador in grupos:
        unidades_restantes = unidades_grupo

        while unidades_restantes > 0:
            numero_parte += 1
            unidades_parte = min(capacidad_camion, unidades_restantes)

            partes_generadas.append(
                replace(
                    pedido,
                    pedido_id=(pedido_id if total_partes == 1 else f"{pedido_id}-P{numero_parte}"),
                    pedido_original_id=(
                        (pedido.pedido_original_id or pedido_id) if total_partes == 1 else pedido_id
                    ),
                    numero_parte=numero_parte,
                    total_partes=total_partes,
                    unidades_capacidad=unidades_parte,
                    requiere_volcador=es_volcador,
                )
            )
            unidades_restantes -= unidades_parte

    return partes_generadas


def _validar_consistencia_post_split(
    cantidades_volcador: Mapping[str, int],
    ids_originales: set[str],
    resultado: list[PedidoInput],
) -> None:
    """Verifica que el mapa de volcadores no tenga remanentes y los IDs generados sean únicos."""
    desconocidos = set(cantidades_volcador) - ids_originales
    if desconocidos:
        raise ValueError(
            "unidades_volcador_por_pedido contiene IDs desconocidos: "
            + ", ".join(sorted(desconocidos))
        )

    ids_generados = [p.pedido_id for p in resultado]
    if len(ids_generados) != len(set(ids_generados)):
        raise ValueError("El split generó IDs duplicados. Revisar los pedido_id de entrada.")


def preprocesar_pedidos(
    pedidos: list[PedidoInput],
    capacidad_camion: int,
    *,
    unidades_volcador_por_pedido: Mapping[str, int] | None = None,
) -> list[PedidoInput]:
    """Convierte pedidos originales en fracciones homogéneas de hasta `capacidad_camion`.

    Si una orden excede la capacidad vehicular, se subdivide en varias partes.
    Si se proporciona `unidades_volcador_por_pedido`, los pedidos mixtos se separan
    en entregas convencionales y entregas con volcador, respetando las restricciones
    de descarga.

    Args:
        pedidos: Colección de pedidos originales a procesar.
        capacidad_camion: Capacidad de carga máxima admisible por camión.
        unidades_volcador_por_pedido: Mapeo opcional de identificador de pedido
            a la cantidad de unidades que requieren camión volcador.

    Returns:
        Lista de pedidos procesados listos para la planificación.

    Raises:
        ValueError: Si la capacidad es inválida, se detectan IDs duplicados o vacíos,
            o existen inconsistencias con las cantidades declaradas.
    """
    _validar_capacidad(capacidad_camion)
    cantidades_volcador = dict(unidades_volcador_por_pedido or {})

    ids_originales: set[str] = set()
    resultado: list[PedidoInput] = []

    for pedido in pedidos:
        _validar_pedido_base(pedido, ids_originales)
        ids_originales.add(pedido.pedido_id)

        unidades_normales, unidades_volcador = _obtener_distribucion_volcador(
            pedido, cantidades_volcador
        )
        partes = _particionar_pedido(pedido, capacidad_camion, unidades_normales, unidades_volcador)
        resultado.extend(partes)

    _validar_consistencia_post_split(cantidades_volcador, ids_originales, resultado)
    return resultado


def preprocesar_instancia(
    instancia: InstanciaTurno,
    *,
    unidades_volcador_por_pedido: Mapping[str, int] | None = None,
) -> InstanciaTurno:
    """Aplica la partición de pedidos sobre una instancia de turno completa.

    Args:
        instancia: Escenario original de planificación.
        unidades_volcador_por_pedido: Mapeo opcional de unidades que requieren volcador.

    Returns:
        Nueva instancia de turno con su lista de pedidos preprocesada.
    """
    return replace(
        instancia,
        pedidos=preprocesar_pedidos(
            instancia.pedidos,
            instancia.capacidad_camion,
            unidades_volcador_por_pedido=unidades_volcador_por_pedido,
        ),
    )
