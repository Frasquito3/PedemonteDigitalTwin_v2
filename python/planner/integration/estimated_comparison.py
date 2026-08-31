from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isclose
from struct import pack
from time import perf_counter
from typing import Sequence

from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
)
from planner.domain.validator import validar_plan
from planner.integration.alpyne_codec import (
    CODIGO_ALGORITMO,
    codificar_plan_alpyne,
)
from planner.integration.planner_selector import (
    DecisionSelector,
    ModoPlanificacion,
    SelectorPlanificadores,
    normalizar_modo,
)
from planner.routing.objective import (
    EstimacionPlan,
    evaluar_plan_estimado,
)
from planner.routing.travel import (
    ProveedorViaje,
    construir_matriz_viaje,
)


VERSION_PROTOCOLO_COMPARACION = 1

MODOS_COMPARACION: tuple[ModoPlanificacion, ...] = (
    ModoPlanificacion.RL,
    ModoPlanificacion.HIBRIDO,
    ModoPlanificacion.GREEDY,
    ModoPlanificacion.RANDOM,
    ModoPlanificacion.GA,
)

CODIGO_MODO_COMPARACION: dict[ModoPlanificacion, int] = {
    modo: indice
    for indice, modo in enumerate(MODOS_COMPARACION)
}

# Cada bloque de método se codifica en este orden estable.
# Los valores no disponibles se representan con -1.0.
CAMPOS_POR_METODO = 20
TAMANO_CABECERA_COMPARACION = 4
VALOR_NO_DISPONIBLE = -1.0


@dataclass(frozen=True)
class ResultadoMetodoComparacion:
    modo_solicitado: ModoPlanificacion
    algoritmo_resultante: AlgoritmoPlanificacion | None
    factible: bool
    error: str
    detalle: str
    costo_estimado: float
    costo_tareas_no_entregadas: float
    costo_pedidos_originales_incompletos: float
    costo_tardanza: float
    costo_exceso_tolerancia: float
    costo_operacion: float
    costo_distancia: float
    costo_viajes: float
    costo_desbalance: float
    distancia_total_km: float
    duracion_operacion_min: float
    viajes_totales: int
    pedidos_tardios: int
    tardanza_total_min: float
    diferencia_fin_camiones_min: float
    tiempo_plan_ms: float
    tiempo_selector_ms: float
    plan_vector: tuple[float, ...]


@dataclass(frozen=True)
class ComparacionEstimada:
    instancia_id: str
    firma_instancia: str
    resultados: tuple[ResultadoMetodoComparacion, ...]
    tiempo_total_ms: float

    def obtener_resultado(
        self,
        modo: ModoPlanificacion | str,
    ) -> ResultadoMetodoComparacion:
        modo_normalizado = normalizar_modo(modo)

        for resultado in self.resultados:
            if resultado.modo_solicitado == modo_normalizado:
                return resultado

        raise KeyError(
            "La comparación no contiene el método "
            f"{modo_normalizado.value}."
        )



def firmar_instancia_vector(
    instancia_vector: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
) -> str:
    """Genera una huella estable del contrato recibido desde AnyLogic."""
    digest = sha256()

    digest.update(
        pack(">q", int(seed_escenario))
    )
    digest.update(
        pack(">q", int(seed_ejecucion))
    )

    for valor in instancia_vector:
        digest.update(
            pack(">d", float(valor))
        )

    return digest.hexdigest()



def ejecutar_comparacion_estimada(
    instancia: InstanciaTurno,
    selector: SelectorPlanificadores,
    proveedor_viaje: ProveedorViaje,
    firma_instancia: str,
) -> ComparacionEstimada:
    """
    Ejecuta los cinco métodos sobre una misma instancia y una única matriz.

    Cada método se aísla: un error individual queda registrado y no impide
    obtener resultados de los demás. La decisión previa del selector se
    restaura al finalizar para no alterar el flujo de planificación normal.
    """
    inicio_total = perf_counter()

    matriz = construir_matriz_viaje(
        instancia,
        selector.configuracion,
        proveedor=proveedor_viaje,
    )

    decision_anterior = selector.ultima_decision
    resultados: list[ResultadoMetodoComparacion] = []

    try:
        for modo in MODOS_COMPARACION:
            resultados.append(
                _ejecutar_metodo(
                    instancia=instancia,
                    selector=selector,
                    matriz=matriz,
                    modo=modo,
                )
            )
    finally:
        selector.ultima_decision = decision_anterior

    tiempo_total_ms = (
        perf_counter() - inicio_total
    ) * 1000.0

    return ComparacionEstimada(
        instancia_id=instancia.instancia_id,
        firma_instancia=firma_instancia,
        resultados=tuple(resultados),
        tiempo_total_ms=tiempo_total_ms,
    )



def codificar_comparacion_estimada(
    comparacion: ComparacionEstimada,
) -> list[float]:
    vector: list[float] = [
        float(VERSION_PROTOCOLO_COMPARACION),
        float(len(comparacion.resultados)),
        float(CAMPOS_POR_METODO),
        float(comparacion.tiempo_total_ms),
    ]

    for resultado in comparacion.resultados:
        algoritmo_codigo = (
            VALOR_NO_DISPONIBLE
            if resultado.algoritmo_resultante is None
            else float(
                CODIGO_ALGORITMO[
                    resultado.algoritmo_resultante
                ]
            )
        )

        vector.extend(
            [
                float(
                    CODIGO_MODO_COMPARACION[
                        resultado.modo_solicitado
                    ]
                ),
                1.0 if resultado.factible else 0.0,
                algoritmo_codigo,
                _valor_o_no_disponible(
                    resultado.costo_estimado,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_tareas_no_entregadas,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_pedidos_originales_incompletos,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_tardanza,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_exceso_tolerancia,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_operacion,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_distancia,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_viajes,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.costo_desbalance,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.distancia_total_km,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.duracion_operacion_min,
                    resultado.factible,
                ),
                (
                    float(resultado.viajes_totales)
                    if resultado.factible
                    else VALOR_NO_DISPONIBLE
                ),
                (
                    float(resultado.pedidos_tardios)
                    if resultado.factible
                    else VALOR_NO_DISPONIBLE
                ),
                _valor_o_no_disponible(
                    resultado.tardanza_total_min,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.diferencia_fin_camiones_min,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.tiempo_plan_ms,
                    resultado.factible,
                ),
                _valor_o_no_disponible(
                    resultado.tiempo_selector_ms,
                    resultado.factible,
                ),
            ]
        )

    longitud_esperada = (
        TAMANO_CABECERA_COMPARACION
        + len(comparacion.resultados)
        * CAMPOS_POR_METODO
    )

    if len(vector) != longitud_esperada:
        raise AssertionError(
            "Longitud inesperada del protocolo de comparación. "
            f"esperada={longitud_esperada}, recibida={len(vector)}."
        )

    return vector



def serializar_resumen_comparacion(
    comparacion: ComparacionEstimada,
) -> str:
    factibles = sum(
        1
        for resultado in comparacion.resultados
        if resultado.factible
    )

    errores = len(comparacion.resultados) - factibles

    return (
        "OK"
        f"|version={VERSION_PROTOCOLO_COMPARACION}"
        f"|firma={comparacion.firma_instancia}"
        f"|metodos={len(comparacion.resultados)}"
        f"|factibles={factibles}"
        f"|errores={errores}"
        f"|tiempo_total_ms={comparacion.tiempo_total_ms:.6f}"
    )



def serializar_resultado_metodo(
    resultado: ResultadoMetodoComparacion,
) -> str:
    algoritmo = (
        "NO_DISPONIBLE"
        if resultado.algoritmo_resultante is None
        else resultado.algoritmo_resultante.value
    )

    error = (
        resultado.error
        .replace("|", "/")
        .replace("\n", " ")
    )

    detalle = (
        resultado.detalle
        .replace("|", "/")
        .replace("\n", " ")
    )

    return (
        f"modo={resultado.modo_solicitado.value}"
        f"|factible={'SI' if resultado.factible else 'NO'}"
        f"|algoritmo={algoritmo}"
        f"|costo={resultado.costo_estimado}"
        "|costo_tareas_no_entregadas="
        f"{resultado.costo_tareas_no_entregadas}"
        "|costo_pedidos_incompletos="
        f"{resultado.costo_pedidos_originales_incompletos}"
        f"|costo_tardanza={resultado.costo_tardanza}"
        "|costo_exceso_tolerancia="
        f"{resultado.costo_exceso_tolerancia}"
        f"|costo_operacion={resultado.costo_operacion}"
        f"|costo_distancia={resultado.costo_distancia}"
        f"|costo_viajes={resultado.costo_viajes}"
        f"|costo_desbalance={resultado.costo_desbalance}"
        f"|distancia_km={resultado.distancia_total_km}"
        f"|duracion_min={resultado.duracion_operacion_min}"
        f"|viajes={resultado.viajes_totales}"
        f"|pedidos_tardios={resultado.pedidos_tardios}"
        f"|tardanza_min={resultado.tardanza_total_min}"
        f"|desbalance_min={resultado.diferencia_fin_camiones_min}"
        f"|tiempo_plan_ms={resultado.tiempo_plan_ms}"
        f"|tiempo_selector_ms={resultado.tiempo_selector_ms}"
        f"|error={error or 'NINGUNO'}"
        f"|detalle={detalle or 'NINGUNO'}"
    )



def _ejecutar_metodo(
    *,
    instancia: InstanciaTurno,
    selector: SelectorPlanificadores,
    matriz,
    modo: ModoPlanificacion,
) -> ResultadoMetodoComparacion:
    try:
        plan = selector.generar_plan(
            instancia,
            modo,
        )

        validacion = validar_plan(
            instancia,
            plan,
        )

        if not validacion.valido:
            raise RuntimeError(
                "Plan inválido: "
                + " | ".join(validacion.errores)
            )

        estimacion = evaluar_plan_estimado(
            instancia,
            plan,
            matriz,
            selector.configuracion,
        )

        if not isclose(
            plan.costo_estimado,
            estimacion.costo_total,
            rel_tol=1e-10,
            abs_tol=1e-7,
        ):
            raise RuntimeError(
                "El costo del plan no coincide con la evaluación común. "
                f"plan={plan.costo_estimado}, "
                f"evaluacion={estimacion.costo_total}."
            )

        decision = selector.ultima_decision

        return _resultado_exitoso(
            modo=modo,
            decision=decision,
            plan_algoritmo=plan.algoritmo,
            estimacion=estimacion,
            plan_vector=tuple(
                codificar_plan_alpyne(
                    instancia,
                    plan,
                )
            ),
            tiempo_plan_ms=plan.tiempo_computo_ms,
        )

    except Exception as exc:
        return ResultadoMetodoComparacion(
            modo_solicitado=modo,
            algoritmo_resultante=None,
            factible=False,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
            detalle="",
            costo_estimado=VALOR_NO_DISPONIBLE,
            costo_tareas_no_entregadas=VALOR_NO_DISPONIBLE,
            costo_pedidos_originales_incompletos=(
                VALOR_NO_DISPONIBLE
            ),
            costo_tardanza=VALOR_NO_DISPONIBLE,
            costo_exceso_tolerancia=VALOR_NO_DISPONIBLE,
            costo_operacion=VALOR_NO_DISPONIBLE,
            costo_distancia=VALOR_NO_DISPONIBLE,
            costo_viajes=VALOR_NO_DISPONIBLE,
            costo_desbalance=VALOR_NO_DISPONIBLE,
            distancia_total_km=VALOR_NO_DISPONIBLE,
            duracion_operacion_min=VALOR_NO_DISPONIBLE,
            viajes_totales=-1,
            pedidos_tardios=-1,
            tardanza_total_min=VALOR_NO_DISPONIBLE,
            diferencia_fin_camiones_min=VALOR_NO_DISPONIBLE,
            tiempo_plan_ms=VALOR_NO_DISPONIBLE,
            tiempo_selector_ms=VALOR_NO_DISPONIBLE,
            plan_vector=(),
        )



def _resultado_exitoso(
    *,
    modo: ModoPlanificacion,
    decision: DecisionSelector | None,
    plan_algoritmo: AlgoritmoPlanificacion,
    estimacion: EstimacionPlan,
    plan_vector: tuple[float, ...],
    tiempo_plan_ms: float,
) -> ResultadoMetodoComparacion:
    tiempo_selector_ms = (
        tiempo_plan_ms
        if decision is None
        else decision.tiempo_selector_ms
    )

    detalle = (
        ""
        if decision is None
        else decision.detalle
    )

    return ResultadoMetodoComparacion(
        modo_solicitado=modo,
        algoritmo_resultante=plan_algoritmo,
        factible=True,
        error="",
        detalle=detalle,
        costo_estimado=estimacion.costo_total,
        costo_tareas_no_entregadas=0.0,
        costo_pedidos_originales_incompletos=0.0,
        costo_tardanza=estimacion.costo_tardanza,
        costo_exceso_tolerancia=(
            estimacion.costo_exceso_tolerancia
        ),
        costo_operacion=estimacion.costo_operacion,
        costo_distancia=estimacion.costo_distancia,
        costo_viajes=estimacion.costo_viajes,
        costo_desbalance=estimacion.costo_desbalance,
        distancia_total_km=estimacion.distancia_total_km,
        duracion_operacion_min=(
            estimacion.duracion_operacion_min
        ),
        viajes_totales=estimacion.viajes_totales,
        pedidos_tardios=estimacion.pedidos_tardios,
        tardanza_total_min=estimacion.tardanza_total_min,
        diferencia_fin_camiones_min=(
            estimacion.diferencia_fin_camiones_min
        ),
        tiempo_plan_ms=tiempo_plan_ms,
        tiempo_selector_ms=tiempo_selector_ms,
        plan_vector=plan_vector,
    )



def _valor_o_no_disponible(
    valor: float,
    disponible: bool,
) -> float:
    return (
        float(valor)
        if disponible
        else VALOR_NO_DISPONIBLE
    )
