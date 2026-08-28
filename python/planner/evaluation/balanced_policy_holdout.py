from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from json import dumps, loads
from math import isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

from planner.core.base import PlanificadorTurno
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.evaluation.classic_instances import crear_casos_benchmark_clasico
from planner.rl.instance_generator import (
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)
from planner.rl.policy_instance_generator import (
    ConfiguracionGeneradorTemporalV4,
    GeneradorInstanciasTemporalV4RL,
)
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import ProveedorViaje, construir_matriz_viaje


VERSION_EVALUACION = "rl-temporal-v4-extension-holdout-v1"
MODO_RL_HISTORICO = "RL_HISTORICO"
MODO_RL_TEMPORAL_V4_QUICK = "RL_TEMPORAL_V4_QUICK"
MODO_RL_TEMPORAL_V4_EXTENSION = "RL_TEMPORAL_V4_EXTENSION_9_12"
MODO_GREEDY = "GREEDY"
ORDEN_MODOS = (
    MODO_RL_HISTORICO,
    MODO_RL_TEMPORAL_V4_QUICK,
    MODO_RL_TEMPORAL_V4_EXTENSION,
    MODO_GREEDY,
)
TOLERANCIA = 1e-9
UMBRAL_COSTO_EXTREMO_PCT = 500.0
SEED_HOLDOUT_FORMAL_MINIMO = 272_000
TOLERANCIA_PRESERVACION_3_8_PP = 5.0
TOLERANCIA_PROMETEDOR_3_8_PP = 10.0
TOLERANCIA_COSTO_CLASICOS_PCT = 10.0


@dataclass(frozen=True)
class DefinicionEstrato:
    nombre: str
    min_pedidos: int
    max_pedidos: int
    conflictivo: bool


ESTRATOS_FASE_16D7 = (
    DefinicionEstrato("CONFLICTIVO_3_5", 3, 5, True),
    DefinicionEstrato("GENERAL_3_5", 3, 5, False),
    DefinicionEstrato("CONFLICTIVO_6_8", 6, 8, True),
    DefinicionEstrato("GENERAL_6_8", 6, 8, False),
    DefinicionEstrato("CONFLICTIVO_9_10", 9, 10, True),
    DefinicionEstrato("GENERAL_9_10", 9, 10, False),
    DefinicionEstrato("CONFLICTIVO_11_12", 11, 12, True),
    DefinicionEstrato("GENERAL_11_12", 11, 12, False),
)


@dataclass(frozen=True)
class CasoHoldoutExtension:
    grupo: str
    caso_id: str
    categoria: str
    descripcion: str
    instancia: InstanciaTurno
    patron_conflictivo: bool = False
    banda_pedidos: str = ""
    estrato: str = ""
    cantidad_objetivo: int | None = None


@dataclass(frozen=True)
class RegistroHoldoutExtension:
    grupo: str
    caso_id: str
    categoria: str
    descripcion: str
    instancia_id: str
    seed_escenario: int
    cantidad_pedidos: int
    cantidad_objetivo: int | None
    patron_conflictivo: bool
    banda_pedidos: str
    estrato: str
    modo: str
    estado: str
    error: str
    firma_plan: str
    secuencia_generacion: str
    pedidos_tardios_estimados: int | None
    tardanza_estimada_min: float | None
    sin_riesgo_temporal_estimado: bool | None
    costo_estimado: float | None
    costo_recalculado: float | None
    diferencia_costo_recalculado: float | None
    espera_ventana_estimada_min: float | None
    exceso_tolerancia_estimado_min: float | None
    duracion_operacion_estimada_min: float | None
    viajes_totales: int | None
    tiempo_plan_ms: float | None
    comparacion_vs_historico: str = "NO_DISPONIBLE"
    comparacion_vs_quick: str = "NO_DISPONIBLE"
    comparacion_vs_extension: str = "NO_DISPONIBLE"
    comparacion_vs_greedy: str = "NO_DISPONIBLE"
    gap_costo_vs_historico_pct: float | None = None
    gap_costo_vs_quick_pct: float | None = None
    gap_costo_vs_greedy_pct: float | None = None


@dataclass(frozen=True)
class ResumenModoExtension:
    grupo: str
    alcance: str
    modo: str
    casos_totales: int
    casos_ok: int
    casos_error: int
    casos_sin_riesgo: int
    tasa_sin_riesgo_pct: float | None
    pedidos_tardios_media: float | None
    pedidos_tardios_mediana: float | None
    pedidos_tardios_p95: float | None
    tardanza_media_min: float | None
    tardanza_mediana_min: float | None
    tardanza_p95_min: float | None
    costo_medio: float | None
    costo_mediano: float | None
    comparables_vs_historico: int
    victorias_vs_historico: int
    empates_vs_historico: int
    derrotas_vs_historico: int
    comparables_vs_quick: int
    victorias_vs_quick: int
    empates_vs_quick: int
    derrotas_vs_quick: int
    comparables_vs_greedy: int
    victorias_vs_greedy: int
    empates_vs_greedy: int
    derrotas_vs_greedy: int
    gap_costo_mediano_vs_greedy_pct: float | None
    gap_costo_p95_vs_greedy_pct: float | None
    costos_extremos_vs_greedy: int


def _numero_finito(valor: Any) -> float | None:
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        numero = float(valor)
        return numero if isfinite(numero) else None
    return None


def percentil(valores: Sequence[float], porcentaje: float) -> float | None:
    if not valores:
        return None
    if not 0.0 <= porcentaje <= 100.0:
        raise ValueError("porcentaje debe estar entre 0 y 100.")

    ordenados = sorted(float(valor) for valor in valores)
    if len(ordenados) == 1:
        return ordenados[0]

    posicion = (len(ordenados) - 1) * porcentaje / 100.0
    inferior = int(posicion)
    superior = min(inferior + 1, len(ordenados) - 1)
    fraccion = posicion - inferior
    return ordenados[inferior] * (1.0 - fraccion) + ordenados[superior] * fraccion


def _gap_porcentual(candidato: float | None, referencia: float | None) -> float | None:
    if candidato is None or referencia is None:
        return None
    if abs(referencia) <= TOLERANCIA:
        return 0.0 if abs(candidato) <= TOLERANCIA else None
    return (candidato - referencia) / referencia * 100.0


def comparar_lexicografico(
    candidato: RegistroHoldoutExtension,
    referencia: RegistroHoldoutExtension,
) -> str:
    """Compara pedidos tardíos, tardanza total y costo, en ese orden."""

    if candidato.estado != "OK" or referencia.estado != "OK":
        return "NO_DISPONIBLE"

    valores_candidato = (
        candidato.pedidos_tardios_estimados,
        candidato.tardanza_estimada_min,
        candidato.costo_recalculado,
    )
    valores_referencia = (
        referencia.pedidos_tardios_estimados,
        referencia.tardanza_estimada_min,
        referencia.costo_recalculado,
    )
    if any(valor is None for valor in valores_candidato + valores_referencia):
        return "NO_DISPONIBLE"

    tardios_candidato = int(candidato.pedidos_tardios_estimados or 0)
    tardios_referencia = int(referencia.pedidos_tardios_estimados or 0)
    if tardios_candidato < tardios_referencia:
        return "MEJOR"
    if tardios_candidato > tardios_referencia:
        return "PEOR"

    tardanza_candidato = float(candidato.tardanza_estimada_min or 0.0)
    tardanza_referencia = float(referencia.tardanza_estimada_min or 0.0)
    if tardanza_candidato < tardanza_referencia - TOLERANCIA:
        return "MEJOR"
    if tardanza_candidato > tardanza_referencia + TOLERANCIA:
        return "PEOR"

    costo_candidato = float(candidato.costo_recalculado or 0.0)
    costo_referencia = float(referencia.costo_recalculado or 0.0)
    if costo_candidato < costo_referencia - TOLERANCIA:
        return "MEJOR"
    if costo_candidato > costo_referencia + TOLERANCIA:
        return "PEOR"
    return "EMPATE"


def firma_plan(plan: PlanTurno) -> str:
    partes: list[str] = []
    for camion in sorted(plan.camiones, key=lambda item: item.camion_id):
        viajes: list[str] = []
        for viaje in sorted(camion.viajes, key=lambda item: item.numero_viaje):
            pedidos = ">".join(viaje.pedido_ids)
            viajes.append(f"v{viaje.numero_viaje}[{pedidos}]")
        partes.append(
            f"c{camion.camion_id}:" + ("/".join(viajes) if viajes else "SIN_VIAJES")
        )
    return "||".join(partes)


def secuencia_generacion(planificador: PlanificadorTurno) -> str:
    valor = getattr(planificador, "ultima_permutacion", ())
    if not isinstance(valor, (list, tuple)):
        return ""
    return ">".join(str(item) for item in valor)


def _registro_error(
    caso: CasoHoldoutExtension,
    modo: str,
    exc: Exception,
) -> RegistroHoldoutExtension:
    return RegistroHoldoutExtension(
        grupo=caso.grupo,
        caso_id=caso.caso_id,
        categoria=caso.categoria,
        descripcion=caso.descripcion,
        instancia_id=caso.instancia.instancia_id,
        seed_escenario=caso.instancia.seed_escenario,
        cantidad_pedidos=len(caso.instancia.pedidos),
        cantidad_objetivo=caso.cantidad_objetivo,
        patron_conflictivo=caso.patron_conflictivo,
        banda_pedidos=caso.banda_pedidos,
        estrato=caso.estrato,
        modo=modo,
        estado="ERROR",
        error=f"{type(exc).__name__}: {exc}",
        firma_plan="",
        secuencia_generacion="",
        pedidos_tardios_estimados=None,
        tardanza_estimada_min=None,
        sin_riesgo_temporal_estimado=None,
        costo_estimado=None,
        costo_recalculado=None,
        diferencia_costo_recalculado=None,
        espera_ventana_estimada_min=None,
        exceso_tolerancia_estimado_min=None,
        duracion_operacion_estimada_min=None,
        viajes_totales=None,
        tiempo_plan_ms=None,
    )


def evaluar_caso_holdout(
    caso: CasoHoldoutExtension,
    planificadores: Mapping[str, PlanificadorTurno],
    *,
    configuracion: ConfiguracionPlanificacion | None = None,
    proveedor_viaje: ProveedorViaje | None = None,
) -> list[RegistroHoldoutExtension]:
    configuracion_efectiva = configuracion or ConfiguracionPlanificacion()
    matriz = construir_matriz_viaje(
        caso.instancia,
        configuracion_efectiva,
        proveedor=proveedor_viaje,
    )
    registros: list[RegistroHoldoutExtension] = []

    for modo in ORDEN_MODOS:
        planificador = planificadores.get(modo)
        if planificador is None:
            registros.append(
                _registro_error(caso, modo, ValueError(f"Falta el planificador {modo}."))
            )
            continue

        try:
            plan = planificador.generar_plan(caso.instancia)
            validacion = validar_plan(caso.instancia, plan)
            if not validacion.valido:
                raise RuntimeError("Plan inválido: " + " | ".join(validacion.errores))

            estimacion = evaluar_plan_estimado(
                caso.instancia,
                plan,
                matriz,
                configuracion_efectiva,
            )
            costo_plan = _numero_finito(plan.costo_estimado)
            costo_recalculado = _numero_finito(estimacion.costo_total)
            tardanza = _numero_finito(estimacion.tardanza_total_min)
            pedidos_tardios = int(estimacion.pedidos_tardios)
            diferencia = None
            if costo_plan is not None and costo_recalculado is not None:
                diferencia = costo_recalculado - costo_plan

            registros.append(
                RegistroHoldoutExtension(
                    grupo=caso.grupo,
                    caso_id=caso.caso_id,
                    categoria=caso.categoria,
                    descripcion=caso.descripcion,
                    instancia_id=caso.instancia.instancia_id,
                    seed_escenario=caso.instancia.seed_escenario,
                    cantidad_pedidos=len(caso.instancia.pedidos),
                    cantidad_objetivo=caso.cantidad_objetivo,
                    patron_conflictivo=caso.patron_conflictivo,
                    banda_pedidos=caso.banda_pedidos,
                    estrato=caso.estrato,
                    modo=modo,
                    estado="OK",
                    error="",
                    firma_plan=firma_plan(plan),
                    secuencia_generacion=secuencia_generacion(planificador),
                    pedidos_tardios_estimados=pedidos_tardios,
                    tardanza_estimada_min=tardanza,
                    sin_riesgo_temporal_estimado=(
                        pedidos_tardios == 0
                        and tardanza is not None
                        and tardanza <= TOLERANCIA
                    ),
                    costo_estimado=costo_plan,
                    costo_recalculado=costo_recalculado,
                    diferencia_costo_recalculado=diferencia,
                    espera_ventana_estimada_min=_numero_finito(
                        estimacion.tiempo_espera_ventana_total_min
                    ),
                    exceso_tolerancia_estimado_min=_numero_finito(
                        estimacion.exceso_tolerancia_min
                    ),
                    duracion_operacion_estimada_min=_numero_finito(
                        estimacion.duracion_operacion_min
                    ),
                    viajes_totales=estimacion.viajes_totales,
                    tiempo_plan_ms=_numero_finito(plan.tiempo_computo_ms),
                )
            )
        except Exception as exc:  # noqa: BLE001 - la auditoría registra cada fallo
            registros.append(_registro_error(caso, modo, exc))

    por_modo = {registro.modo: registro for registro in registros}
    referencias = {
        "historico": por_modo.get(MODO_RL_HISTORICO),
        "quick": por_modo.get(MODO_RL_TEMPORAL_V4_QUICK),
        "extension": por_modo.get(MODO_RL_TEMPORAL_V4_EXTENSION),
        "greedy": por_modo.get(MODO_GREEDY),
    }
    salida: list[RegistroHoldoutExtension] = []

    for registro in registros:
        historico = referencias["historico"]
        quick = referencias["quick"]
        extension = referencias["extension"]
        greedy = referencias["greedy"]
        salida.append(
            replace(
                registro,
                comparacion_vs_historico=(
                    comparar_lexicografico(registro, historico)
                    if historico is not None
                    else "NO_DISPONIBLE"
                ),
                comparacion_vs_quick=(
                    comparar_lexicografico(registro, quick)
                    if quick is not None
                    else "NO_DISPONIBLE"
                ),
                comparacion_vs_extension=(
                    comparar_lexicografico(registro, extension)
                    if extension is not None
                    else "NO_DISPONIBLE"
                ),
                comparacion_vs_greedy=(
                    comparar_lexicografico(registro, greedy)
                    if greedy is not None
                    else "NO_DISPONIBLE"
                ),
                gap_costo_vs_historico_pct=(
                    _gap_porcentual(
                        registro.costo_recalculado,
                        historico.costo_recalculado,
                    )
                    if historico is not None
                    else None
                ),
                gap_costo_vs_quick_pct=(
                    _gap_porcentual(
                        registro.costo_recalculado,
                        quick.costo_recalculado,
                    )
                    if quick is not None
                    else None
                ),
                gap_costo_vs_greedy_pct=(
                    _gap_porcentual(
                        registro.costo_recalculado,
                        greedy.costo_recalculado,
                    )
                    if greedy is not None
                    else None
                ),
            )
        )

    return salida


def evaluar_casos_holdout(
    casos: Sequence[CasoHoldoutExtension],
    planificadores: Mapping[str, PlanificadorTurno],
    *,
    configuracion: ConfiguracionPlanificacion | None = None,
    proveedor_viaje: ProveedorViaje | None = None,
) -> list[RegistroHoldoutExtension]:
    salida: list[RegistroHoldoutExtension] = []
    for caso in casos:
        salida.extend(
            evaluar_caso_holdout(
                caso,
                planificadores,
                configuracion=configuracion,
                proveedor_viaje=proveedor_viaje,
            )
        )
    return salida


def crear_casos_clasicos() -> list[CasoHoldoutExtension]:
    return [
        CasoHoldoutExtension(
            grupo="CLASICO",
            caso_id=caso.caso_id,
            categoria=caso.categoria,
            descripcion=caso.descripcion,
            instancia=caso.instancia,
            patron_conflictivo=(caso.caso_id == "B04_VENTANAS"),
            banda_pedidos="CLASICO",
            estrato="CLASICO",
            cantidad_objetivo=len(caso.instancia.pedidos),
        )
        for caso in crear_casos_benchmark_clasico()
    ]


def _tiene_patron_v4(instancia: InstanciaTurno) -> bool:
    return any(
        "PATRON_TEMPORAL_CONFLICTIVO_V4" in pedido.observaciones
        for pedido in instancia.pedidos
    )


def _crear_generador_estrato(
    cantidad_objetivo: int,
    conflictivo: bool,
) -> GeneradorInstanciasTemporalV4RL:
    base = GeneradorInstanciasRL(
        ConfiguracionGeneradorInstancias(
            min_pedidos_finales=cantidad_objetivo,
            max_pedidos_finales=cantidad_objetivo,
            probabilidad_volcador=0.15,
            probabilidad_ventana_especifica=0.90,
            probabilidad_pedido_mayor_capacidad=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        )
    )
    return GeneradorInstanciasTemporalV4RL(
        base,
        ConfiguracionGeneradorTemporalV4(
            probabilidad_patron_ventanas_conflictivas=(1.0 if conflictivo else 0.0)
        ),
    )


def crear_casos_sinteticos(
    *,
    casos_por_estrato: int = 20,
    seed_inicio: int = SEED_HOLDOUT_FORMAL_MINIMO,
) -> list[CasoHoldoutExtension]:
    if casos_por_estrato <= 0:
        raise ValueError("casos_por_estrato debe ser > 0.")
    if seed_inicio < SEED_HOLDOUT_FORMAL_MINIMO:
        raise ValueError(
            f"La Fase 16D.7 exige seed_inicio >= {SEED_HOLDOUT_FORMAL_MINIMO}."
        )

    casos: list[CasoHoldoutExtension] = []
    seed = int(seed_inicio)

    for definicion in ESTRATOS_FASE_16D7:
        generadores = {
            cantidad: _crear_generador_estrato(cantidad, definicion.conflictivo)
            for cantidad in range(definicion.min_pedidos, definicion.max_pedidos + 1)
        }
        aceptados = 0
        intentos = 0
        limite_intentos = max(5_000, casos_por_estrato * 500)

        while aceptados < casos_por_estrato:
            if intentos >= limite_intentos:
                raise RuntimeError(
                    f"No fue posible completar el estrato {definicion.nombre} "
                    f"después de {limite_intentos} intentos."
                )

            cantidad_objetivo = definicion.min_pedidos + (
                aceptados
                % (definicion.max_pedidos - definicion.min_pedidos + 1)
            )
            seed_actual = seed
            seed += 1
            intentos += 1
            instancia = generadores[cantidad_objetivo].generar(seed_actual)

            if _tiene_patron_v4(instancia) != definicion.conflictivo:
                continue
            if len(instancia.pedidos) != cantidad_objetivo:
                continue

            casos.append(
                CasoHoldoutExtension(
                    grupo="HOLDOUT_SINTETICO",
                    caso_id=f"HOLDOUT16D7-{definicion.nombre}-{seed_actual}",
                    categoria=(
                        "PATRON_CONFLICTIVO"
                        if definicion.conflictivo
                        else "TEMPORAL_GENERAL"
                    ),
                    descripcion=(
                        "Caso independiente de la Fase 16D.7, no utilizado en "
                        "entrenamiento ni selección externa de checkpoints."
                    ),
                    instancia=instancia,
                    patron_conflictivo=definicion.conflictivo,
                    banda_pedidos=(
                        f"{definicion.min_pedidos}_{definicion.max_pedidos}"
                    ),
                    estrato=definicion.nombre,
                    cantidad_objetivo=cantidad_objetivo,
                )
            )
            aceptados += 1

    return casos


def _resumir_seleccion(
    registros: Sequence[RegistroHoldoutExtension],
    *,
    grupo: str,
    alcance: str,
    modo: str,
    filtro: Callable[[RegistroHoldoutExtension], bool],
) -> ResumenModoExtension:
    seleccion = [
        registro
        for registro in registros
        if registro.grupo == grupo and registro.modo == modo and filtro(registro)
    ]
    ok = [registro for registro in seleccion if registro.estado == "OK"]
    tardios = [
        float(registro.pedidos_tardios_estimados)
        for registro in ok
        if registro.pedidos_tardios_estimados is not None
    ]
    tardanzas = [
        float(registro.tardanza_estimada_min)
        for registro in ok
        if registro.tardanza_estimada_min is not None
    ]
    costos = [
        float(registro.costo_recalculado)
        for registro in ok
        if registro.costo_recalculado is not None
    ]
    gaps_greedy = [
        float(registro.gap_costo_vs_greedy_pct)
        for registro in ok
        if registro.gap_costo_vs_greedy_pct is not None
    ]
    comparaciones_historico = [
        registro.comparacion_vs_historico
        for registro in ok
        if registro.comparacion_vs_historico != "NO_DISPONIBLE"
    ]
    comparaciones_quick = [
        registro.comparacion_vs_quick
        for registro in ok
        if registro.comparacion_vs_quick != "NO_DISPONIBLE"
    ]
    comparaciones_greedy = [
        registro.comparacion_vs_greedy
        for registro in ok
        if registro.comparacion_vs_greedy != "NO_DISPONIBLE"
    ]
    sin_riesgo = sum(
        1 for registro in ok if registro.sin_riesgo_temporal_estimado is True
    )

    return ResumenModoExtension(
        grupo=grupo,
        alcance=alcance,
        modo=modo,
        casos_totales=len(seleccion),
        casos_ok=len(ok),
        casos_error=len(seleccion) - len(ok),
        casos_sin_riesgo=sin_riesgo,
        tasa_sin_riesgo_pct=(100.0 * sin_riesgo / len(ok) if ok else None),
        pedidos_tardios_media=(float(mean(tardios)) if tardios else None),
        pedidos_tardios_mediana=(float(median(tardios)) if tardios else None),
        pedidos_tardios_p95=percentil(tardios, 95.0),
        tardanza_media_min=(float(mean(tardanzas)) if tardanzas else None),
        tardanza_mediana_min=(float(median(tardanzas)) if tardanzas else None),
        tardanza_p95_min=percentil(tardanzas, 95.0),
        costo_medio=(float(mean(costos)) if costos else None),
        costo_mediano=(float(median(costos)) if costos else None),
        comparables_vs_historico=len(comparaciones_historico),
        victorias_vs_historico=comparaciones_historico.count("MEJOR"),
        empates_vs_historico=comparaciones_historico.count("EMPATE"),
        derrotas_vs_historico=comparaciones_historico.count("PEOR"),
        comparables_vs_quick=len(comparaciones_quick),
        victorias_vs_quick=comparaciones_quick.count("MEJOR"),
        empates_vs_quick=comparaciones_quick.count("EMPATE"),
        derrotas_vs_quick=comparaciones_quick.count("PEOR"),
        comparables_vs_greedy=len(comparaciones_greedy),
        victorias_vs_greedy=comparaciones_greedy.count("MEJOR"),
        empates_vs_greedy=comparaciones_greedy.count("EMPATE"),
        derrotas_vs_greedy=comparaciones_greedy.count("PEOR"),
        gap_costo_mediano_vs_greedy_pct=(
            float(median(gaps_greedy)) if gaps_greedy else None
        ),
        gap_costo_p95_vs_greedy_pct=percentil(gaps_greedy, 95.0),
        costos_extremos_vs_greedy=sum(
            1 for gap in gaps_greedy if gap > UMBRAL_COSTO_EXTREMO_PCT
        ),
    )


def resumir_registros(
    registros: Sequence[RegistroHoldoutExtension],
) -> tuple[
    list[ResumenModoExtension],
    list[ResumenModoExtension],
    list[ResumenModoExtension],
]:
    resumen_global: list[ResumenModoExtension] = []
    resumen_estratos: list[ResumenModoExtension] = []
    resumen_segmentos: list[ResumenModoExtension] = []

    for grupo in sorted({registro.grupo for registro in registros}):
        for modo in ORDEN_MODOS:
            resumen_global.append(
                _resumir_seleccion(
                    registros,
                    grupo=grupo,
                    alcance="TODOS",
                    modo=modo,
                    filtro=lambda _registro: True,
                )
            )

    estratos = sorted(
        {
            registro.estrato
            for registro in registros
            if registro.grupo == "HOLDOUT_SINTETICO" and registro.estrato
        }
    )
    for estrato in estratos:
        for modo in ORDEN_MODOS:
            resumen_estratos.append(
                _resumir_seleccion(
                    registros,
                    grupo="HOLDOUT_SINTETICO",
                    alcance=estrato,
                    modo=modo,
                    filtro=lambda registro, nombre=estrato: registro.estrato == nombre,
                )
            )

    segmentos: tuple[tuple[str, Callable[[RegistroHoldoutExtension], bool]], ...] = (
        ("PEDIDOS_3_8", lambda registro: 3 <= registro.cantidad_pedidos <= 8),
        ("PEDIDOS_9_12", lambda registro: 9 <= registro.cantidad_pedidos <= 12),
        ("PEDIDOS_9_10", lambda registro: 9 <= registro.cantidad_pedidos <= 10),
        ("PEDIDOS_11_12", lambda registro: 11 <= registro.cantidad_pedidos <= 12),
        ("PEDIDOS_12", lambda registro: registro.cantidad_pedidos == 12),
    )
    for alcance, filtro in segmentos:
        for modo in ORDEN_MODOS:
            resumen_segmentos.append(
                _resumir_seleccion(
                    registros,
                    grupo="HOLDOUT_SINTETICO",
                    alcance=alcance,
                    modo=modo,
                    filtro=filtro,
                )
            )

    return resumen_global, resumen_estratos, resumen_segmentos


def _buscar_resumen(
    resumenes: Sequence[ResumenModoExtension],
    *,
    grupo: str,
    alcance: str,
    modo: str,
) -> ResumenModoExtension | None:
    return next(
        (
            resumen
            for resumen in resumenes
            if resumen.grupo == grupo
            and resumen.alcance == alcance
            and resumen.modo == modo
        ),
        None,
    )


def _buscar_registro(
    registros: Sequence[RegistroHoldoutExtension],
    caso_id: str,
    modo: str,
) -> RegistroHoldoutExtension | None:
    return next(
        (
            registro
            for registro in registros
            if registro.caso_id == caso_id and registro.modo == modo
        ),
        None,
    )


def _comparacion_no_regresiva_clasica(
    candidato: RegistroHoldoutExtension | None,
    referencia: RegistroHoldoutExtension | None,
    *,
    tolerancia_costo_pct: float = TOLERANCIA_COSTO_CLASICOS_PCT,
) -> bool:
    if candidato is None or referencia is None:
        return False
    if candidato.estado != "OK" or referencia.estado != "OK":
        return False
    if (
        candidato.pedidos_tardios_estimados is None
        or referencia.pedidos_tardios_estimados is None
        or candidato.tardanza_estimada_min is None
        or referencia.tardanza_estimada_min is None
        or candidato.costo_recalculado is None
        or referencia.costo_recalculado is None
    ):
        return False
    if candidato.pedidos_tardios_estimados > referencia.pedidos_tardios_estimados:
        return False
    if candidato.pedidos_tardios_estimados < referencia.pedidos_tardios_estimados:
        return True
    if candidato.tardanza_estimada_min > referencia.tardanza_estimada_min + TOLERANCIA:
        return False
    if candidato.tardanza_estimada_min < referencia.tardanza_estimada_min - TOLERANCIA:
        return True
    limite = referencia.costo_recalculado * (1.0 + tolerancia_costo_pct / 100.0)
    return candidato.costo_recalculado <= limite + TOLERANCIA


def _supera_segmento(
    candidato: ResumenModoExtension | None,
    referencia: ResumenModoExtension | None,
    *,
    referencia_modo: str,
) -> bool:
    if candidato is None or referencia is None:
        return False
    if candidato.casos_error or referencia.casos_error:
        return False
    if (
        candidato.tasa_sin_riesgo_pct is None
        or referencia.tasa_sin_riesgo_pct is None
    ):
        return False

    if referencia_modo == MODO_RL_HISTORICO:
        victorias = candidato.victorias_vs_historico
        derrotas = candidato.derrotas_vs_historico
    elif referencia_modo == MODO_RL_TEMPORAL_V4_QUICK:
        victorias = candidato.victorias_vs_quick
        derrotas = candidato.derrotas_vs_quick
    else:
        raise ValueError(f"Referencia no soportada: {referencia_modo}")

    return (
        candidato.tasa_sin_riesgo_pct + TOLERANCIA
        >= referencia.tasa_sin_riesgo_pct
        and victorias > derrotas
    )


def analizar_clasicos(
    registros: Sequence[RegistroHoldoutExtension],
) -> dict[str, Any]:
    salida: dict[str, Any] = {}
    for caso_id in ("B04_VENTANAS", "B05_VOLCADOR", "B06_SPLIT"):
        modos: dict[str, Any] = {}
        for modo in ORDEN_MODOS:
            registro = _buscar_registro(registros, caso_id, modo)
            modos[modo] = (
                {
                    "estado": registro.estado,
                    "error": registro.error,
                    "pedidos_tardios": registro.pedidos_tardios_estimados,
                    "tardanza_total_min": registro.tardanza_estimada_min,
                    "costo_recalculado": registro.costo_recalculado,
                    "firma_plan": registro.firma_plan,
                    "secuencia_generacion": registro.secuencia_generacion,
                }
                if registro is not None
                else {"estado": "FALTANTE"}
            )
        extension = _buscar_registro(
            registros, caso_id, MODO_RL_TEMPORAL_V4_EXTENSION
        )
        quick = _buscar_registro(registros, caso_id, MODO_RL_TEMPORAL_V4_QUICK)
        historico = _buscar_registro(registros, caso_id, MODO_RL_HISTORICO)
        salida[caso_id] = {
            "modos": modos,
            "extension_vs_quick": (
                comparar_lexicografico(extension, quick)
                if extension is not None and quick is not None
                else "NO_DISPONIBLE"
            ),
            "extension_vs_historico": (
                comparar_lexicografico(extension, historico)
                if extension is not None and historico is not None
                else "NO_DISPONIBLE"
            ),
        }
    return salida


def construir_veredicto(
    registros: Sequence[RegistroHoldoutExtension],
    resumen_global: Sequence[ResumenModoExtension],
    resumen_segmentos: Sequence[ResumenModoExtension],
) -> dict[str, Any]:
    extension_global = _buscar_resumen(
        resumen_global,
        grupo="HOLDOUT_SINTETICO",
        alcance="TODOS",
        modo=MODO_RL_TEMPORAL_V4_EXTENSION,
    )
    quick_global = _buscar_resumen(
        resumen_global,
        grupo="HOLDOUT_SINTETICO",
        alcance="TODOS",
        modo=MODO_RL_TEMPORAL_V4_QUICK,
    )
    historico_global = _buscar_resumen(
        resumen_global,
        grupo="HOLDOUT_SINTETICO",
        alcance="TODOS",
        modo=MODO_RL_HISTORICO,
    )

    def resumen_segmento(alcance: str, modo: str) -> ResumenModoExtension | None:
        return _buscar_resumen(
            resumen_segmentos,
            grupo="HOLDOUT_SINTETICO",
            alcance=alcance,
            modo=modo,
        )

    b04_extension = _buscar_registro(
        registros, "B04_VENTANAS", MODO_RL_TEMPORAL_V4_EXTENSION
    )
    b04_cero = bool(
        b04_extension is not None
        and b04_extension.estado == "OK"
        and b04_extension.pedidos_tardios_estimados == 0
        and b04_extension.tardanza_estimada_min is not None
        and b04_extension.tardanza_estimada_min <= TOLERANCIA
    )

    clasicos_sin_regresion: dict[str, bool] = {}
    for caso_id in ("B05_VOLCADOR", "B06_SPLIT"):
        extension = _buscar_registro(
            registros, caso_id, MODO_RL_TEMPORAL_V4_EXTENSION
        )
        quick = _buscar_registro(
            registros, caso_id, MODO_RL_TEMPORAL_V4_QUICK
        )
        historico = _buscar_registro(registros, caso_id, MODO_RL_HISTORICO)
        clasicos_sin_regresion[caso_id] = (
            _comparacion_no_regresiva_clasica(extension, quick)
            and _comparacion_no_regresiva_clasica(extension, historico)
        )

    sin_errores = bool(
        registros
        and all(registro.estado == "OK" for registro in registros)
        and extension_global is not None
        and extension_global.casos_error == 0
    )

    ext_3_8 = resumen_segmento("PEDIDOS_3_8", MODO_RL_TEMPORAL_V4_EXTENSION)
    quick_3_8 = resumen_segmento("PEDIDOS_3_8", MODO_RL_TEMPORAL_V4_QUICK)
    ext_9_12 = resumen_segmento("PEDIDOS_9_12", MODO_RL_TEMPORAL_V4_EXTENSION)
    quick_9_12 = resumen_segmento("PEDIDOS_9_12", MODO_RL_TEMPORAL_V4_QUICK)
    hist_9_12 = resumen_segmento("PEDIDOS_9_12", MODO_RL_HISTORICO)
    ext_11_12 = resumen_segmento("PEDIDOS_11_12", MODO_RL_TEMPORAL_V4_EXTENSION)
    quick_11_12 = resumen_segmento("PEDIDOS_11_12", MODO_RL_TEMPORAL_V4_QUICK)
    hist_11_12 = resumen_segmento("PEDIDOS_11_12", MODO_RL_HISTORICO)
    ext_12 = resumen_segmento("PEDIDOS_12", MODO_RL_TEMPORAL_V4_EXTENSION)
    quick_12 = resumen_segmento("PEDIDOS_12", MODO_RL_TEMPORAL_V4_QUICK)
    hist_12 = resumen_segmento("PEDIDOS_12", MODO_RL_HISTORICO)

    supera_9_12_quick = _supera_segmento(
        ext_9_12, quick_9_12, referencia_modo=MODO_RL_TEMPORAL_V4_QUICK
    )
    supera_9_12_historico = _supera_segmento(
        ext_9_12, hist_9_12, referencia_modo=MODO_RL_HISTORICO
    )
    mejora_11_12_quick = _supera_segmento(
        ext_11_12, quick_11_12, referencia_modo=MODO_RL_TEMPORAL_V4_QUICK
    )
    mejora_11_12_historico = _supera_segmento(
        ext_11_12, hist_11_12, referencia_modo=MODO_RL_HISTORICO
    )
    mejora_12_quick = _supera_segmento(
        ext_12, quick_12, referencia_modo=MODO_RL_TEMPORAL_V4_QUICK
    )
    mejora_12_historico = _supera_segmento(
        ext_12, hist_12, referencia_modo=MODO_RL_HISTORICO
    )

    preserva_3_8 = bool(
        ext_3_8 is not None
        and quick_3_8 is not None
        and ext_3_8.tasa_sin_riesgo_pct is not None
        and quick_3_8.tasa_sin_riesgo_pct is not None
        and ext_3_8.tasa_sin_riesgo_pct
        >= quick_3_8.tasa_sin_riesgo_pct - TOLERANCIA_PRESERVACION_3_8_PP
        and ext_3_8.victorias_vs_quick >= ext_3_8.derrotas_vs_quick
    )
    preserva_3_8_prometedor = bool(
        ext_3_8 is not None
        and quick_3_8 is not None
        and ext_3_8.tasa_sin_riesgo_pct is not None
        and quick_3_8.tasa_sin_riesgo_pct is not None
        and ext_3_8.tasa_sin_riesgo_pct
        >= quick_3_8.tasa_sin_riesgo_pct - TOLERANCIA_PROMETEDOR_3_8_PP
    )
    balance_global_positivo = bool(
        extension_global is not None
        and extension_global.victorias_vs_historico
        > extension_global.derrotas_vs_historico
    )
    balance_global_no_negativo = bool(
        extension_global is not None
        and extension_global.victorias_vs_historico
        >= extension_global.derrotas_vs_historico
    )
    reduce_costos_extremos = bool(
        extension_global is not None
        and quick_global is not None
        and historico_global is not None
        and (
            extension_global.costos_extremos_vs_greedy == 0
            or (
                extension_global.costos_extremos_vs_greedy
                < quick_global.costos_extremos_vs_greedy
                and extension_global.costos_extremos_vs_greedy
                < historico_global.costos_extremos_vs_greedy
            )
        )
    )

    criterios = {
        "b04_tardanza_cero": b04_cero,
        "b05_sin_regresion": clasicos_sin_regresion["B05_VOLCADOR"],
        "b06_sin_regresion": clasicos_sin_regresion["B06_SPLIT"],
        "supera_quick_en_9_12": supera_9_12_quick,
        "supera_historico_en_9_12": supera_9_12_historico,
        "preserva_3_8": preserva_3_8,
        "mas_victorias_que_derrotas_vs_historico": balance_global_positivo,
        "reduce_costos_extremos": reduce_costos_extremos,
        "mejora_11_12_vs_quick": mejora_11_12_quick,
        "mejora_11_12_vs_historico": mejora_11_12_historico,
        "mejora_12_vs_quick": mejora_12_quick,
        "mejora_12_vs_historico": mejora_12_historico,
        "sin_errores": sin_errores,
    }

    if all(criterios.values()):
        estado = "CANDIDATO_ENTRENAMIENTO_COMPLETO"
    elif (
        b04_cero
        and clasicos_sin_regresion["B05_VOLCADOR"]
        and clasicos_sin_regresion["B06_SPLIT"]
        and sin_errores
        and preserva_3_8_prometedor
        and balance_global_no_negativo
        and (supera_9_12_quick or supera_9_12_historico)
        and (
            mejora_11_12_quick
            or mejora_11_12_historico
            or mejora_12_quick
            or mejora_12_historico
        )
    ):
        estado = "PROMETEDOR_CON_AJUSTES_PENDIENTES"
    else:
        estado = "NO_JUSTIFICA_ENTRENAMIENTO_COMPLETO"

    return {
        "estado": estado,
        "criterios": criterios,
        "observaciones": {
            "extension_global": asdict(extension_global) if extension_global else None,
            "quick_global": asdict(quick_global) if quick_global else None,
            "historico_global": asdict(historico_global) if historico_global else None,
            "extension_3_8": asdict(ext_3_8) if ext_3_8 else None,
            "quick_3_8": asdict(quick_3_8) if quick_3_8 else None,
            "extension_9_12": asdict(ext_9_12) if ext_9_12 else None,
            "quick_9_12": asdict(quick_9_12) if quick_9_12 else None,
            "historico_9_12": asdict(hist_9_12) if hist_9_12 else None,
            "extension_11_12": asdict(ext_11_12) if ext_11_12 else None,
            "extension_12": asdict(ext_12) if ext_12 else None,
        },
        "umbrales": {
            "preservacion_3_8_max_caida_pp": TOLERANCIA_PRESERVACION_3_8_PP,
            "prometedor_3_8_max_caida_pp": TOLERANCIA_PROMETEDOR_3_8_PP,
            "costo_extremo_desde_pct": UMBRAL_COSTO_EXTREMO_PCT,
            "tolerancia_costo_clasicos_pct": TOLERANCIA_COSTO_CLASICOS_PCT,
        },
        "modelo_promovido": False,
        "nota": (
            "El veredicto no promueve ni copia modelos. Sólo decide si la "
            "extensión justifica una etapa posterior de entrenamiento completo."
        ),
    }


def resumir_casos(
    registros: Sequence[RegistroHoldoutExtension],
) -> list[dict[str, Any]]:
    salida: list[dict[str, Any]] = []
    claves = sorted({(registro.grupo, registro.caso_id) for registro in registros})
    slugs = {
        MODO_RL_HISTORICO: "historico",
        MODO_RL_TEMPORAL_V4_QUICK: "quick",
        MODO_RL_TEMPORAL_V4_EXTENSION: "extension",
        MODO_GREEDY: "greedy",
    }

    for grupo, caso_id in claves:
        seleccion = [
            registro
            for registro in registros
            if registro.grupo == grupo and registro.caso_id == caso_id
        ]
        if not seleccion:
            continue
        primero = seleccion[0]
        por_modo = {registro.modo: registro for registro in seleccion}
        fila: dict[str, Any] = {
            "grupo": grupo,
            "caso_id": caso_id,
            "categoria": primero.categoria,
            "instancia_id": primero.instancia_id,
            "seed_escenario": primero.seed_escenario,
            "cantidad_pedidos": primero.cantidad_pedidos,
            "cantidad_objetivo": primero.cantidad_objetivo,
            "patron_conflictivo": primero.patron_conflictivo,
            "banda_pedidos": primero.banda_pedidos,
            "estrato": primero.estrato,
        }
        for modo in ORDEN_MODOS:
            registro = por_modo.get(modo)
            slug = slugs[modo]
            fila[f"estado_{slug}"] = registro.estado if registro else "FALTANTE"
            fila[f"error_{slug}"] = registro.error if registro else ""
            fila[f"pedidos_tardios_{slug}"] = (
                registro.pedidos_tardios_estimados if registro else None
            )
            fila[f"tardanza_{slug}_min"] = (
                registro.tardanza_estimada_min if registro else None
            )
            fila[f"costo_{slug}"] = registro.costo_recalculado if registro else None
            fila[f"firma_{slug}"] = registro.firma_plan if registro else ""
            fila[f"secuencia_{slug}"] = (
                registro.secuencia_generacion if registro else ""
            )

        extension = por_modo.get(MODO_RL_TEMPORAL_V4_EXTENSION)
        fila["comparacion_extension_vs_historico"] = (
            extension.comparacion_vs_historico
            if extension
            else "NO_DISPONIBLE"
        )
        fila["comparacion_extension_vs_quick"] = (
            extension.comparacion_vs_quick if extension else "NO_DISPONIBLE"
        )
        fila["comparacion_extension_vs_greedy"] = (
            extension.comparacion_vs_greedy if extension else "NO_DISPONIBLE"
        )
        fila["gap_costo_extension_vs_greedy_pct"] = (
            extension.gap_costo_vs_greedy_pct if extension else None
        )
        salida.append(fila)

    return salida


def hash_archivo(ruta: str | Path) -> str:
    digest = sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _leer_objeto_json(ruta: str | Path, nombre: str) -> dict[str, Any]:
    path = Path(ruta)
    if not path.is_file():
        raise FileNotFoundError(f"No existe {nombre}: {path}")
    contenido = loads(path.read_text(encoding="utf-8"))
    if not isinstance(contenido, dict):
        raise ValueError(f"{nombre} debe contener un objeto JSON.")
    return contenido


def _semillas_json(contenido: Mapping[str, Any]) -> set[int]:
    salida: set[int] = set()

    def recorrer(valor: Any, clave: str = "") -> None:
        if isinstance(valor, dict):
            for subclave, subcontenido in valor.items():
                recorrer(subcontenido, str(subclave))
        elif isinstance(valor, list) and "semilla" in clave.lower():
            for item in valor:
                if isinstance(item, int) and not isinstance(item, bool):
                    salida.add(int(item))

    recorrer(contenido)
    return salida


def validar_metadatos_fase16d7(
    *,
    quick_model: str | Path,
    quick_config: str | Path,
    quick_selection: str | Path,
    extension_model: str | Path,
    extension_config: str | Path,
    extension_selection: str | Path,
    extension_summary: str | Path,
    historical_model: str | Path,
    seed_inicio: int,
    semillas_holdout: Sequence[int] = (),
    quick_holdout_result: str | Path | None = None,
) -> dict[str, Any]:
    if seed_inicio < SEED_HOLDOUT_FORMAL_MINIMO:
        raise ValueError(
            f"La Fase 16D.7 exige seed_inicio >= {SEED_HOLDOUT_FORMAL_MINIMO}."
        )

    rutas_modelo = {
        "historico": Path(historical_model),
        "quick": Path(quick_model),
        "extension": Path(extension_model),
    }
    for nombre, ruta in rutas_modelo.items():
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo {nombre}: {ruta}")

    quick_cfg = _leer_objeto_json(quick_config, "la configuración quick v4")
    quick_sel = _leer_objeto_json(quick_selection, "la selección quick v4")
    ext_cfg = _leer_objeto_json(extension_config, "la configuración de extensión")
    ext_sel = _leer_objeto_json(extension_selection, "la selección de extensión")
    ext_summary = _leer_objeto_json(extension_summary, "el resumen final de extensión")

    if quick_cfg.get("version_entorno") != "pedemonte-rl-temporal-v4":
        raise ValueError("version_entorno quick v4 inválida.")
    if quick_cfg.get("quick") is not True:
        raise ValueError("El modelo base debe ser el quick temporal v4.")
    if quick_cfg.get("modelo_historico_sobrescrito") is not False:
        raise ValueError("La configuración quick no preservó el modelo histórico.")
    temporal = quick_cfg.get("temporal")
    if not isinstance(temporal, dict) or temporal.get("usar_mascara_temporal_dura") is not False:
        raise ValueError("La máscara temporal dura quick debe estar desactivada.")
    if quick_sel.get("criterio") != "VALIDACION_EXTERNA_LEXICOGRAFICA_V4":
        raise ValueError("La selección quick no usa el criterio externo esperado.")
    if quick_sel.get("modelo_promovido") is not False:
        raise ValueError("El modelo quick no debe estar promovido.")

    if ext_cfg.get("version_run") != "pedemonte-rl-temporal-v4-extension-9-12-v1":
        raise ValueError("version_run de la extensión inválida.")
    if ext_cfg.get("reward_modificado") is not False:
        raise ValueError("La extensión declara un reward modificado.")
    if ext_cfg.get("observacion_modificada") is not False:
        raise ValueError("La extensión declara una observación modificada.")
    if ext_cfg.get("mascara_temporal_dura") is not False:
        raise ValueError("La máscara temporal dura de la extensión debe estar desactivada.")
    if ext_cfg.get("continuacion_entre_etapas") != "EXTERNAL_BEST_9_12":
        raise ValueError("La extensión no continuó desde EXTERNAL_BEST_9_12.")
    for clave in (
        "modelo_historico_sobrescrito",
        "modelo_v3_sobrescrito",
        "modelo_v4_quick_sobrescrito",
        "modelo_promovido",
    ):
        if ext_cfg.get(clave) is not False:
            raise ValueError(f"La extensión no preserva la garantía {clave}=false.")

    if ext_sel.get("criterio") != "VALIDACION_EXTERNA_9_12_LEXICOGRAFICA_V4_EXTENSION":
        raise ValueError("La selección final de extensión usa un criterio inesperado.")
    if ext_sel.get("modelo_promovido") is not False:
        raise ValueError("La extensión no debe estar promovida.")
    if ext_sel.get("modelo_v4_quick_sobrescrito") is not False:
        raise ValueError("La selección de extensión sobrescribió el quick.")
    if int(ext_summary.get("timestep", -1)) != 68_288:
        raise ValueError("El resumen final no corresponde al checkpoint acumulado 68288.")

    hashes = {nombre: hash_archivo(ruta) for nombre, ruta in rutas_modelo.items()}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("Los modelos histórico, quick y extensión deben ser archivos distintos.")
    hash_base_declarado = ext_cfg.get("sha256_modelo_base")
    if hash_base_declarado and hash_base_declarado != hashes["quick"]:
        raise ValueError("El hash del modelo base declarado no coincide con el quick recibido.")

    semillas_prohibidas = {164_000, 166_000}
    semillas_prohibidas.update(_semillas_json(quick_cfg))
    semillas_prohibidas.update(_semillas_json(ext_cfg))
    if quick_holdout_result is not None and Path(quick_holdout_result).is_file():
        quick_holdout = _leer_objeto_json(
            quick_holdout_result, "el resultado del holdout quick"
        )
        semillas_prohibidas.update(_semillas_json(quick_holdout))

    semillas = [int(seed) for seed in semillas_holdout]
    if len(semillas) != len(set(semillas)):
        raise ValueError("El holdout contiene semillas repetidas.")
    interseccion = sorted(set(semillas).intersection(semillas_prohibidas))
    if interseccion:
        raise ValueError(
            "Las semillas del holdout se superponen con fases previas: "
            + ", ".join(str(seed) for seed in interseccion)
        )
    if semillas and min(semillas) < SEED_HOLDOUT_FORMAL_MINIMO:
        raise ValueError("Se detectó una semilla holdout anterior a 272000.")

    return {
        "quick_config": quick_cfg,
        "quick_selection": quick_sel,
        "extension_config": ext_cfg,
        "extension_selection": ext_sel,
        "extension_summary": ext_summary,
        "hashes_modelos": hashes,
        "semillas_prohibidas_declaradas": sorted(semillas_prohibidas),
        "semillas_holdout_validadas": semillas,
        "modelo_promovido": False,
    }


def _normalizar_json(valor: Any) -> Any:
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, dict):
        return {
            str(clave): _normalizar_json(contenido)
            for clave, contenido in valor.items()
        }
    if isinstance(valor, (list, tuple)):
        return [_normalizar_json(item) for item in valor]
    return valor


def _escribir_csv(ruta: Path, filas: Sequence[Mapping[str, Any]]) -> None:
    if not filas:
        ruta.write_text("", encoding="utf-8")
        return
    columnas = list(filas[0].keys())
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def escribir_resultados(
    directorio: str | Path,
    *,
    metadatos: Mapping[str, Any],
    registros: Sequence[RegistroHoldoutExtension],
    resumen_global: Sequence[ResumenModoExtension],
    resumen_estratos: Sequence[ResumenModoExtension],
    resumen_segmentos: Sequence[ResumenModoExtension],
    casos: Sequence[Mapping[str, Any]],
    clasicos: Mapping[str, Any],
    veredicto: Mapping[str, Any],
    semillas: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    destino = Path(directorio)
    destino.mkdir(parents=True, exist_ok=True)

    rutas = {
        "evaluacion_json": destino / "balanced_policy_holdout.json",
        "corridas_csv": destino / "balanced_policy_holdout_runs.csv",
        "resumen_csv": destino / "balanced_policy_holdout_summary.csv",
        "estratos_csv": destino / "balanced_policy_holdout_strata.csv",
        "segmentos_csv": destino / "balanced_policy_holdout_segments.csv",
        "casos_csv": destino / "balanced_policy_holdout_cases.csv",
        "semillas_csv": destino / "balanced_policy_holdout_seeds.csv",
    }

    contenido = {
        "version_evaluacion": VERSION_EVALUACION,
        "metadatos": dict(metadatos),
        "veredicto": dict(veredicto),
        "analisis_clasicos": dict(clasicos),
        "resumen_global": [asdict(item) for item in resumen_global],
        "resumen_estratos": [asdict(item) for item in resumen_estratos],
        "resumen_segmentos": [asdict(item) for item in resumen_segmentos],
        "casos": [dict(item) for item in casos],
        "semillas": [dict(item) for item in semillas],
        "registros": [asdict(item) for item in registros],
    }
    rutas["evaluacion_json"].write_text(
        dumps(_normalizar_json(contenido), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _escribir_csv(rutas["corridas_csv"], [asdict(item) for item in registros])
    _escribir_csv(rutas["resumen_csv"], [asdict(item) for item in resumen_global])
    _escribir_csv(rutas["estratos_csv"], [asdict(item) for item in resumen_estratos])
    _escribir_csv(rutas["segmentos_csv"], [asdict(item) for item in resumen_segmentos])
    _escribir_csv(rutas["casos_csv"], [dict(item) for item in casos])
    _escribir_csv(rutas["semillas_csv"], [dict(item) for item in semillas])
    return rutas
