"""Puente Pypeline para la exportación de hojas de ruta desde AnyLogic.

Este módulo recibe vectores de instancia y plan directamente desde el entorno
de ejecución de AnyLogic, reconstruye las estructuras `InstanciaTurno` y `PlanTurno`
sin alterar la solución generada y delega la exportación formateada a Excel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import isclose, isfinite
from pathlib import Path

from planner.core.schema import InstanciaTurno, PedidoInput, PlanCamion, PlanTurno, ViajePlan
from planner.domain.validator import validar_plan
from planner.integration import selector_bridge
from planner.integration.alpyne_codec import CODIGO_ALGORITMO, PROTOCOL_VERSION
from planner.integration.instance_vector_codec import decodificar_instancia_vector
from planner.integration.planner_selector import normalizar_modo
from planner.integration.route_excel_archive import exportar_hoja_ruta_archivada
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import construir_matriz_viaje

__all__ = [
    "exportar_hoja_ruta_desde_anylogic",
]


def exportar_hoja_ruta_desde_anylogic(
    instancia_vector: Sequence[float],
    plan_vector: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
    modo_solicitado: str,
    instancia_id: str,
    fecha_operacion: str,
    datos_pedidos: Sequence[Sequence[str]],
    raiz_proyecto: str | Path | None = None,
) -> str:
    """Exporta el plan generado desde AnyLogic sin recalcular la solución.

    Args:
        instancia_vector: Vector plano de la instancia recibido de AnyLogic.
        plan_vector: Vector plano del plan a exportar.
        seed_escenario: Semilla del escenario.
        seed_ejecucion: Semilla de la ejecución.
        modo_solicitado: Modo de ruteo utilizado ('RL', 'GA', 'GREEDY', etc.).
        instancia_id: Identificador textual de la instancia.
        fecha_operacion: Fecha de operación para la cabecera.
        datos_pedidos: Matriz con campos descriptivos de pedidos:
            [pedido_id, pedido_original_id, cliente, direccion, barrio, observaciones].
        raiz_proyecto: Directorio raíz opcional para la carpeta exports/.

    Returns:
        Ruta absoluta como cadena al archivo Excel generado.
    """
    selector = selector_bridge._selector
    proveedor = selector_bridge._proveedor_viaje

    if selector is None or proveedor is None:
        raise RuntimeError("El selector Python debe estar inicializado antes de exportar.")

    modo = normalizar_modo(modo_solicitado)
    instancia = decodificar_instancia_vector(instancia_vector, seed_escenario, seed_ejecucion)

    if len(datos_pedidos) != len(instancia.pedidos):
        raise ValueError("Los datos descriptivos no coinciden con la cantidad de pedidos.")

    pedidos_actualizados: list[PedidoInput] = []
    for pedido, datos in zip(instancia.pedidos, datos_pedidos):
        if len(datos) != 6:
            raise ValueError("Cada pedido debe tener seis campos descriptivos.")

        pedido_id = _texto(datos[0])
        original_id = _texto(datos[1]) or pedido_id
        if not pedido_id:
            raise ValueError("Pedido con ID vacío.")

        pedidos_actualizados.append(
            replace(
                pedido,
                pedido_id=pedido_id,
                pedido_original_id=original_id,
                cliente=_texto(datos[2]),
                direccion=_texto(datos[3]),
                barrio=_texto(datos[4]),
                observaciones=_texto(datos[5]),
            )
        )

    instancia = replace(
        instancia,
        instancia_id=_texto(instancia_id) or instancia.instancia_id,
        fecha_operacion=_texto(fecha_operacion),
        pedidos=pedidos_actualizados,
    )

    plan = _reconstruir_plan(instancia, plan_vector)
    configuracion = selector.configuracion
    matriz = construir_matriz_viaje(instancia, configuracion, proveedor=proveedor)
    estimacion = evaluar_plan_estimado(instancia, plan, matriz, configuracion)

    if not isclose(plan.costo_estimado, estimacion.costo_total, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError(
            "La estimación del plan no coincide con el costo calculado originalmente. Se canceló la exportación."
        )

    ruta = exportar_hoja_ruta_archivada(
        instancia,
        plan,
        matriz,
        configuracion,
        modo_solicitado=modo.value,
        raiz_proyecto=raiz_proyecto,
    )
    return str(ruta)


def _reconstruir_plan(instancia: InstanciaTurno, plan_vector: Sequence[float]) -> PlanTurno:
    valores = [float(valor) for valor in plan_vector]
    if len(valores) < 5:
        raise ValueError("planVector incompleto.")

    version = _entero_exacto(valores[0], "version")
    cantidad = _entero_exacto(valores[1], "cantidad")
    codigo = _entero_exacto(valores[2], "algoritmo")

    if version != PROTOCOL_VERSION:
        raise ValueError("Versión de planVector no soportada.")
    if cantidad != len(instancia.pedidos):
        raise ValueError("La cantidad de asignaciones no coincide con los pedidos de la instancia.")
    if len(valores) != 5 + 4 * cantidad:
        raise ValueError("Longitud inválida de planVector.")

    algoritmos_por_codigo = {
        codigo_algoritmo: algoritmo for algoritmo, codigo_algoritmo in CODIGO_ALGORITMO.items()
    }
    if codigo not in algoritmos_por_codigo:
        raise ValueError("Algoritmo de planVector desconocido.")

    costo = valores[3]
    tiempo_computo = valores[4]
    if not isfinite(costo) or costo < 0 or not isfinite(tiempo_computo) or tiempo_computo < 0:
        raise ValueError("Métricas inválidas en planVector.")

    viajes_por_camion: dict[int, dict[int, list[tuple[int, str]]]] = {
        camion_id: {} for camion_id in range(instancia.cantidad_camiones)
    }
    indices_utilizados: set[int] = set()

    for indice_asignacion in range(cantidad):
        posicion = 5 + 4 * indice_asignacion
        camion_id = _entero_exacto(valores[posicion], "camionId")
        numero_viaje = _entero_exacto(valores[posicion + 1], "numeroViaje")
        orden = _entero_exacto(valores[posicion + 2], "orden")
        indice_pedido = _entero_exacto(valores[posicion + 3], "indicePedido")

        if camion_id not in viajes_por_camion:
            raise ValueError("Camión fuera de rango.")
        if numero_viaje <= 0 or orden <= 0:
            raise ValueError("Viaje u orden inválido.")
        if not 0 <= indice_pedido < cantidad:
            raise ValueError("Índice de pedido fuera de rango.")
        if indice_pedido in indices_utilizados:
            raise ValueError("Pedido asignado más de una vez.")

        indices_utilizados.add(indice_pedido)
        pedido_id = instancia.pedidos[indice_pedido].pedido_id
        viajes_por_camion[camion_id].setdefault(numero_viaje, []).append((orden, pedido_id))

    camiones: list[PlanCamion] = []
    for camion_id in range(instancia.cantidad_camiones):
        viajes: list[ViajePlan] = []
        agrupados = viajes_por_camion[camion_id]
        numeros = sorted(agrupados)

        if numeros != list(range(1, len(numeros) + 1)):
            raise ValueError(f"Numeración no consecutiva en camión {camion_id}.")

        for numero_viaje in numeros:
            asignaciones = sorted(agrupados[numero_viaje], key=lambda elemento: elemento[0])
            ordenes = [orden for orden, _ in asignaciones]
            if ordenes != list(range(1, len(asignaciones) + 1)):
                raise ValueError("Órdenes de visita no consecutivos.")

            viajes.append(
                ViajePlan(numero_viaje=numero_viaje, pedido_ids=[pid for _, pid in asignaciones])
            )

        camiones.append(PlanCamion(camion_id=camion_id, viajes=viajes))

    plan = PlanTurno(
        instancia_id=instancia.instancia_id,
        algoritmo=algoritmos_por_codigo[codigo],
        camiones=camiones,
        costo_estimado=costo,
        tiempo_computo_ms=tiempo_computo,
    )

    validacion = validar_plan(instancia, plan)
    if not validacion.valido:
        raise ValueError("Plan reconstruido inválido: " + " | ".join(validacion.errores))

    return plan


def _entero_exacto(valor: float, campo: str) -> int:
    numero = float(valor)
    if not isfinite(numero):
        raise ValueError(f"{campo} no es finito.")
    entero = round(numero)
    if abs(numero - entero) > 1e-9:
        raise ValueError(f"{campo} debe ser entero.")
    return int(entero)


def _texto(valor: object) -> str:
    return "" if valor is None else str(valor).strip()
