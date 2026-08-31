from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Sequence

from planner.integration.alpyne_codec import CODIGO_ALGORITMO
from planner.integration.estimated_comparison import firmar_instancia_vector
from planner.integration import selector_bridge
from planner.integration.simulated_comparison import (
    ComparacionSimulada,
    ESTADO_ERROR,
    ESTADO_FINALIZADO,
    ESTADO_OMITIDO,
    METODOS_COMPARACION_SIMULADA,
    ejecutar_comparacion_simulada,
    normalizar_metodo_simulado,
)


VERSION_PROTOCOLO_COMPARACION_SIMULADA = 1
CAMPOS_POR_METODO_SIMULADO = 21
TAMANO_CABECERA_COMPARACION_SIMULADA = 4
VALOR_NO_DISPONIBLE = -1.0

CODIGO_METODO_SIMULADO: dict[str, int] = {
    metodo: indice
    for indice, metodo in enumerate(METODOS_COMPARACION_SIMULADA)
}

CODIGO_ESTADO_SIMULADO: dict[str, int] = {
    ESTADO_FINALIZADO: 1,
    ESTADO_ERROR: 2,
    ESTADO_OMITIDO: 3,
}

CODIGO_ALGORITMO_APLICADO: dict[str, int] = {
    algoritmo.value: codigo
    for algoritmo, codigo in CODIGO_ALGORITMO.items()
}

_ultima_comparacion_simulada: ComparacionSimulada | None = None


def comparar_simulado_vector(
    instancia_vector: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
    modelo_exportado: str,
    raiz_python: str,
    identificadores_pedidos: str,
    instancia_id: str,
    fecha_operacion: str,
    proveedores_habilitados: bool = True,
    timeout_segundos_por_metodo: int = 240,
    horizonte_simulacion_min: float = 600.0,
) -> list[float]:
    """
    Ejecuta los planes ya almacenados por la comparación estimada.

    Cada método corre en un proceso AnyLogic externo, nuevo e independiente.
    La función rechaza cualquier diferencia entre la firma estimada y la firma
    de la instancia que se intenta simular.
    """
    global _ultima_comparacion_simulada

    _ultima_comparacion_simulada = None

    vector = _vector_finito_no_vacio(
        instancia_vector,
        "instancia_vector",
    )

    firma_estimada = (
        selector_bridge.obtener_firma_comparacion_estimada()
    )

    if firma_estimada == "SIN_COMPARACION":
        raise RuntimeError(
            "No existe una comparación estimada disponible."
        )

    firma_recibida = firmar_instancia_vector(
        vector,
        int(seed_escenario),
        int(seed_ejecucion),
    )

    if firma_estimada != firma_recibida:
        raise RuntimeError(
            "La firma de la instancia no coincide con la comparación "
            "estimada almacenada. "
            f"estimada={firma_estimada}, recibida={firma_recibida}."
        )

    planes_por_metodo: dict[str, list[float]] = {}

    for metodo in METODOS_COMPARACION_SIMULADA:
        try:
            planes_por_metodo[metodo] = list(
                selector_bridge.obtener_plan_comparacion_vector(
                    metodo
                )
            )
        except RuntimeError:
            # Un método no factible se representa como OMITIDO dentro del
            # controlador de comparación simulada.
            continue

    comparacion = ejecutar_comparacion_simulada(
        modelo_exportado=Path(modelo_exportado),
        raiz_python=Path(raiz_python),
        instancia_vector=vector,
        planes_por_metodo=planes_por_metodo,
        identificadores_pedidos=str(
            identificadores_pedidos
        ).strip(),
        instancia_id=str(instancia_id).strip(),
        fecha_operacion=str(fecha_operacion).strip(),
        seed_escenario=int(seed_escenario),
        seed_ejecucion=int(seed_ejecucion),
        proveedores_habilitados=bool(
            proveedores_habilitados
        ),
        timeout_segundos_por_metodo=int(
            timeout_segundos_por_metodo
        ),
        horizonte_simulacion_min=float(
            horizonte_simulacion_min
        ),
        continuar_ante_error=True,
    )

    if comparacion.firma_instancia != firma_estimada:
        raise RuntimeError(
            "La comparación simulada devolvió una firma diferente. "
            f"estimada={firma_estimada}, "
            f"simulada={comparacion.firma_instancia}."
        )

    _ultima_comparacion_simulada = comparacion

    return codificar_comparacion_simulada(comparacion)


def codificar_comparacion_simulada(
    comparacion: ComparacionSimulada,
) -> list[float]:
    if comparacion is None:
        raise ValueError("comparacion no puede ser null.")

    vector: list[float] = [
        float(VERSION_PROTOCOLO_COMPARACION_SIMULADA),
        float(len(comparacion.resultados)),
        float(CAMPOS_POR_METODO_SIMULADO),
        float(comparacion.tiempo_total_segundos),
    ]

    for fila in comparacion.resultados:
        metodo = normalizar_metodo_simulado(
            fila.metodo_solicitado
        )

        codigo_estado = CODIGO_ESTADO_SIMULADO.get(
            fila.estado,
            0,
        )

        if fila.resultado is None:
            vector.extend(
                [
                    float(CODIGO_METODO_SIMULADO[metodo]),
                    float(codigo_estado),
                    VALOR_NO_DISPONIBLE,
                ]
                + [VALOR_NO_DISPONIBLE] * 17
                + [float(fila.tiempo_motor_segundos)]
            )
            continue

        resultado = fila.resultado
        codigo_algoritmo = CODIGO_ALGORITMO_APLICADO.get(
            resultado.algoritmo_aplicado,
            -1,
        )

        vector.extend(
            [
                float(CODIGO_METODO_SIMULADO[metodo]),
                float(codigo_estado),
                float(codigo_algoritmo),
                float(resultado.costo_total),
                float(resultado.distancia_total_km),
                float(resultado.duracion_simulada_min),
                float(resultado.viajes_totales),
                float(resultado.tardanza_total_min),
                float(resultado.diferencia_fin_camiones_min),
                float(resultado.tareas_entregadas),
                float(resultado.tareas_no_entregadas),
                float(resultado.ocupacion_global_pct),
                float(resultado.costo_tareas_no_entregadas),
                float(
                    resultado
                    .costo_pedidos_originales_incompletos
                ),
                float(resultado.costo_tardanza),
                float(resultado.costo_exceso_tolerancia),
                float(resultado.costo_operacion),
                float(resultado.costo_distancia),
                float(resultado.costo_viajes),
                float(resultado.costo_desbalance),
                float(fila.tiempo_motor_segundos),
            ]
        )

    longitud_esperada = (
        TAMANO_CABECERA_COMPARACION_SIMULADA
        + len(comparacion.resultados)
        * CAMPOS_POR_METODO_SIMULADO
    )

    if len(vector) != longitud_esperada:
        raise AssertionError(
            "Longitud inesperada del protocolo simulado. "
            f"esperada={longitud_esperada}, recibida={len(vector)}."
        )

    if not all(isfinite(valor) for valor in vector):
        raise AssertionError(
            "El protocolo simulado contiene valores no finitos."
        )

    return vector


def obtener_resumen_comparacion_simulada() -> str:
    if _ultima_comparacion_simulada is None:
        return "SIN_COMPARACION_SIMULADA"

    return _ultima_comparacion_simulada.resumen()


def obtener_firma_comparacion_simulada() -> str:
    if _ultima_comparacion_simulada is None:
        return "SIN_COMPARACION_SIMULADA"

    return _ultima_comparacion_simulada.firma_instancia


def obtener_estado_metodo_comparacion_simulada(
    metodo: str,
) -> str:
    if _ultima_comparacion_simulada is None:
        return "SIN_COMPARACION_SIMULADA"

    fila = _ultima_comparacion_simulada.obtener_resultado(
        metodo
    )

    error = (
        fila.error
        .replace("|", "/")
        .replace("\r", " ")
        .replace("\n", " ")
    )

    algoritmo = (
        "NO_DISPONIBLE"
        if fila.resultado is None
        else fila.resultado.algoritmo_aplicado
    )

    return (
        f"metodo={fila.metodo_solicitado}"
        f"|estado={fila.estado}"
        f"|algoritmo={algoritmo}"
        f"|error={error}"
    )


def limpiar_comparacion_simulada() -> str:
    global _ultima_comparacion_simulada

    _ultima_comparacion_simulada = None

    return "OK|COMPARACION_SIMULADA_LIMPIA"


def _vector_finito_no_vacio(
    valores: Sequence[float],
    nombre: str,
) -> list[float]:
    if valores is None:
        raise ValueError(f"{nombre} no puede ser null.")

    vector = [float(valor) for valor in valores]

    if not vector:
        raise ValueError(f"{nombre} no puede estar vacío.")

    if not all(isfinite(valor) for valor in vector):
        raise ValueError(
            f"{nombre} debe contener solo valores finitos."
        )

    return vector
