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
from planner.core.schema import PlanTurno
from planner.domain.validator import validar_plan
from planner.evaluation.balanced_policy_holdout import (
    CasoHoldoutExtension,
    ESTRATOS_FASE_16D7,
    comparar_lexicografico as _comparar_registros_compatibles,
    crear_casos_clasicos,
    crear_casos_sinteticos,
    firma_plan,
    hash_archivo,
    percentil,
    secuencia_generacion,
    validar_metadatos_fase16d7,
)
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import ProveedorViaje, construir_matriz_viaje


VERSION_EVALUACION = "rl-temporal-v4-full-holdout-v1"
MODO_RL_HISTORICO = "RL_HISTORICO"
MODO_RL_TEMPORAL_V4_QUICK = "RL_TEMPORAL_V4_QUICK"
MODO_RL_TEMPORAL_V4_EXTENSION = "RL_TEMPORAL_V4_EXTENSION_9_12"
MODO_RL_TEMPORAL_V4_FULL = "RL_TEMPORAL_V4_FULL_11_12"
MODO_GREEDY = "GREEDY"
ORDEN_MODOS = (
    MODO_RL_HISTORICO,
    MODO_RL_TEMPORAL_V4_QUICK,
    MODO_RL_TEMPORAL_V4_EXTENSION,
    MODO_RL_TEMPORAL_V4_FULL,
    MODO_GREEDY,
)

SEED_HOLDOUT_FINAL_MINIMO = 274_000
TOLERANCIA = 1e-9
UMBRAL_COSTO_EXTREMO_PCT = 500.0
TOLERANCIA_PRESERVACION_PP = 2.5
TOLERANCIA_COSTO_CLASICOS_PCT = 10.0


@dataclass(frozen=True)
class RegistroHoldoutFull:
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
    comparacion_vs_full: str = "NO_DISPONIBLE"
    comparacion_vs_greedy: str = "NO_DISPONIBLE"
    gap_costo_vs_historico_pct: float | None = None
    gap_costo_vs_quick_pct: float | None = None
    gap_costo_vs_extension_pct: float | None = None
    gap_costo_vs_greedy_pct: float | None = None


@dataclass(frozen=True)
class ResumenModoFull:
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
    comparables_vs_extension: int
    victorias_vs_extension: int
    empates_vs_extension: int
    derrotas_vs_extension: int
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


def _gap_porcentual(candidato: float | None, referencia: float | None) -> float | None:
    if candidato is None or referencia is None:
        return None
    if abs(referencia) <= TOLERANCIA:
        return 0.0 if abs(candidato) <= TOLERANCIA else None
    return (candidato - referencia) / referencia * 100.0


def comparar_lexicografico(
    candidato: RegistroHoldoutFull,
    referencia: RegistroHoldoutFull,
) -> str:
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
    if tardios_candidato != tardios_referencia:
        return "MEJOR" if tardios_candidato < tardios_referencia else "PEOR"

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


def _registro_error(
    caso: CasoHoldoutExtension,
    modo: str,
    exc: Exception,
) -> RegistroHoldoutFull:
    return RegistroHoldoutFull(
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
) -> list[RegistroHoldoutFull]:
    configuracion_efectiva = configuracion or ConfiguracionPlanificacion()
    matriz = construir_matriz_viaje(
        caso.instancia,
        configuracion_efectiva,
        proveedor=proveedor_viaje,
    )
    registros: list[RegistroHoldoutFull] = []

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
                RegistroHoldoutFull(
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
        except Exception as exc:  # noqa: BLE001
            registros.append(_registro_error(caso, modo, exc))

    por_modo = {registro.modo: registro for registro in registros}
    salida: list[RegistroHoldoutFull] = []
    for registro in registros:
        historico = por_modo.get(MODO_RL_HISTORICO)
        quick = por_modo.get(MODO_RL_TEMPORAL_V4_QUICK)
        extension = por_modo.get(MODO_RL_TEMPORAL_V4_EXTENSION)
        full = por_modo.get(MODO_RL_TEMPORAL_V4_FULL)
        greedy = por_modo.get(MODO_GREEDY)
        salida.append(
            replace(
                registro,
                comparacion_vs_historico=(
                    comparar_lexicografico(registro, historico)
                    if historico is not None else "NO_DISPONIBLE"
                ),
                comparacion_vs_quick=(
                    comparar_lexicografico(registro, quick)
                    if quick is not None else "NO_DISPONIBLE"
                ),
                comparacion_vs_extension=(
                    comparar_lexicografico(registro, extension)
                    if extension is not None else "NO_DISPONIBLE"
                ),
                comparacion_vs_full=(
                    comparar_lexicografico(registro, full)
                    if full is not None else "NO_DISPONIBLE"
                ),
                comparacion_vs_greedy=(
                    comparar_lexicografico(registro, greedy)
                    if greedy is not None else "NO_DISPONIBLE"
                ),
                gap_costo_vs_historico_pct=(
                    _gap_porcentual(registro.costo_recalculado, historico.costo_recalculado)
                    if historico is not None else None
                ),
                gap_costo_vs_quick_pct=(
                    _gap_porcentual(registro.costo_recalculado, quick.costo_recalculado)
                    if quick is not None else None
                ),
                gap_costo_vs_extension_pct=(
                    _gap_porcentual(registro.costo_recalculado, extension.costo_recalculado)
                    if extension is not None else None
                ),
                gap_costo_vs_greedy_pct=(
                    _gap_porcentual(registro.costo_recalculado, greedy.costo_recalculado)
                    if greedy is not None else None
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
) -> list[RegistroHoldoutFull]:
    salida: list[RegistroHoldoutFull] = []
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


def crear_casos_sinteticos_finales(
    *,
    casos_por_estrato: int = 30,
    seed_inicio: int = SEED_HOLDOUT_FINAL_MINIMO,
) -> list[CasoHoldoutExtension]:
    if seed_inicio < SEED_HOLDOUT_FINAL_MINIMO:
        raise ValueError(
            f"La Fase 16D.9 exige seed_inicio >= {SEED_HOLDOUT_FINAL_MINIMO}."
        )
    return crear_casos_sinteticos(
        casos_por_estrato=casos_por_estrato,
        seed_inicio=seed_inicio,
    )


def _resumir_seleccion(
    registros: Sequence[RegistroHoldoutFull],
    *,
    grupo: str,
    alcance: str,
    modo: str,
    filtro: Callable[[RegistroHoldoutFull], bool],
) -> ResumenModoFull:
    seleccion = [
        registro for registro in registros
        if registro.grupo == grupo and registro.modo == modo and filtro(registro)
    ]
    ok = [registro for registro in seleccion if registro.estado == "OK"]
    tardios = [
        float(r.pedidos_tardios_estimados)
        for r in ok if r.pedidos_tardios_estimados is not None
    ]
    tardanzas = [
        float(r.tardanza_estimada_min)
        for r in ok if r.tardanza_estimada_min is not None
    ]
    costos = [
        float(r.costo_recalculado)
        for r in ok if r.costo_recalculado is not None
    ]
    gaps_greedy = [
        float(r.gap_costo_vs_greedy_pct)
        for r in ok if r.gap_costo_vs_greedy_pct is not None
    ]

    def comps(nombre: str) -> list[str]:
        return [
            str(getattr(r, nombre))
            for r in ok
            if getattr(r, nombre) != "NO_DISPONIBLE"
        ]

    hist = comps("comparacion_vs_historico")
    quick = comps("comparacion_vs_quick")
    extension = comps("comparacion_vs_extension")
    greedy = comps("comparacion_vs_greedy")
    sin_riesgo = sum(1 for r in ok if r.sin_riesgo_temporal_estimado is True)

    return ResumenModoFull(
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
        comparables_vs_historico=len(hist),
        victorias_vs_historico=hist.count("MEJOR"),
        empates_vs_historico=hist.count("EMPATE"),
        derrotas_vs_historico=hist.count("PEOR"),
        comparables_vs_quick=len(quick),
        victorias_vs_quick=quick.count("MEJOR"),
        empates_vs_quick=quick.count("EMPATE"),
        derrotas_vs_quick=quick.count("PEOR"),
        comparables_vs_extension=len(extension),
        victorias_vs_extension=extension.count("MEJOR"),
        empates_vs_extension=extension.count("EMPATE"),
        derrotas_vs_extension=extension.count("PEOR"),
        comparables_vs_greedy=len(greedy),
        victorias_vs_greedy=greedy.count("MEJOR"),
        empates_vs_greedy=greedy.count("EMPATE"),
        derrotas_vs_greedy=greedy.count("PEOR"),
        gap_costo_mediano_vs_greedy_pct=(
            float(median(gaps_greedy)) if gaps_greedy else None
        ),
        gap_costo_p95_vs_greedy_pct=percentil(gaps_greedy, 95.0),
        costos_extremos_vs_greedy=sum(
            1 for gap in gaps_greedy if gap > UMBRAL_COSTO_EXTREMO_PCT
        ),
    )


def resumir_registros(
    registros: Sequence[RegistroHoldoutFull],
) -> tuple[list[ResumenModoFull], list[ResumenModoFull], list[ResumenModoFull]]:
    globales: list[ResumenModoFull] = []
    estratos: list[ResumenModoFull] = []
    segmentos: list[ResumenModoFull] = []

    for grupo in sorted({r.grupo for r in registros}):
        for modo in ORDEN_MODOS:
            globales.append(
                _resumir_seleccion(
                    registros, grupo=grupo, alcance="TODOS", modo=modo,
                    filtro=lambda _r: True,
                )
            )

    nombres_estrato = sorted({
        r.estrato for r in registros
        if r.grupo == "HOLDOUT_SINTETICO" and r.estrato
    })
    for nombre in nombres_estrato:
        for modo in ORDEN_MODOS:
            estratos.append(
                _resumir_seleccion(
                    registros,
                    grupo="HOLDOUT_SINTETICO",
                    alcance=nombre,
                    modo=modo,
                    filtro=lambda r, n=nombre: r.estrato == n,
                )
            )

    definiciones: tuple[tuple[str, Callable[[RegistroHoldoutFull], bool]], ...] = (
        ("PEDIDOS_3_8", lambda r: 3 <= r.cantidad_pedidos <= 8),
        ("PEDIDOS_9_12", lambda r: 9 <= r.cantidad_pedidos <= 12),
        ("PEDIDOS_9_10", lambda r: 9 <= r.cantidad_pedidos <= 10),
        ("PEDIDOS_11_12", lambda r: 11 <= r.cantidad_pedidos <= 12),
        ("PEDIDOS_11", lambda r: r.cantidad_pedidos == 11),
        ("PEDIDOS_12", lambda r: r.cantidad_pedidos == 12),
    )
    for alcance, filtro in definiciones:
        for modo in ORDEN_MODOS:
            segmentos.append(
                _resumir_seleccion(
                    registros,
                    grupo="HOLDOUT_SINTETICO",
                    alcance=alcance,
                    modo=modo,
                    filtro=filtro,
                )
            )
    return globales, estratos, segmentos


def _buscar_resumen(
    resumenes: Sequence[ResumenModoFull],
    alcance: str,
    modo: str,
) -> ResumenModoFull | None:
    return next(
        (
            r for r in resumenes
            if r.grupo == "HOLDOUT_SINTETICO"
            and r.alcance == alcance
            and r.modo == modo
        ),
        None,
    )


def _buscar_registro(
    registros: Sequence[RegistroHoldoutFull],
    caso_id: str,
    modo: str,
) -> RegistroHoldoutFull | None:
    return next(
        (r for r in registros if r.caso_id == caso_id and r.modo == modo),
        None,
    )


def _no_regresion_clasica(
    candidato: RegistroHoldoutFull | None,
    referencia: RegistroHoldoutFull | None,
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
    limite = referencia.costo_recalculado * (
        1.0 + TOLERANCIA_COSTO_CLASICOS_PCT / 100.0
    )
    return candidato.costo_recalculado <= limite + TOLERANCIA


def analizar_clasicos(
    registros: Sequence[RegistroHoldoutFull],
) -> dict[str, Any]:
    salida: dict[str, Any] = {}
    for caso_id in ("B04_VENTANAS", "B05_VOLCADOR", "B06_SPLIT"):
        modos: dict[str, Any] = {}
        for modo in ORDEN_MODOS:
            r = _buscar_registro(registros, caso_id, modo)
            modos[modo] = (
                {
                    "estado": r.estado,
                    "error": r.error,
                    "pedidos_tardios": r.pedidos_tardios_estimados,
                    "tardanza_total_min": r.tardanza_estimada_min,
                    "costo_recalculado": r.costo_recalculado,
                    "firma_plan": r.firma_plan,
                    "secuencia_generacion": r.secuencia_generacion,
                }
                if r is not None else {"estado": "FALTANTE"}
            )
        full = _buscar_registro(registros, caso_id, MODO_RL_TEMPORAL_V4_FULL)
        salida[caso_id] = {
            "modos": modos,
            "full_vs_extension": (
                full.comparacion_vs_extension if full else "NO_DISPONIBLE"
            ),
            "full_vs_quick": (
                full.comparacion_vs_quick if full else "NO_DISPONIBLE"
            ),
            "full_vs_historico": (
                full.comparacion_vs_historico if full else "NO_DISPONIBLE"
            ),
        }
    return salida


def construir_veredicto(
    registros: Sequence[RegistroHoldoutFull],
    globales: Sequence[ResumenModoFull],
    segmentos: Sequence[ResumenModoFull],
) -> dict[str, Any]:
    full_global = _buscar_resumen(globales, "TODOS", MODO_RL_TEMPORAL_V4_FULL)
    ext_global = _buscar_resumen(globales, "TODOS", MODO_RL_TEMPORAL_V4_EXTENSION)

    def seg(nombre: str, modo: str) -> ResumenModoFull | None:
        return _buscar_resumen(segmentos, nombre, modo)

    full_3_8, ext_3_8 = seg("PEDIDOS_3_8", MODO_RL_TEMPORAL_V4_FULL), seg("PEDIDOS_3_8", MODO_RL_TEMPORAL_V4_EXTENSION)
    full_9_10, ext_9_10 = seg("PEDIDOS_9_10", MODO_RL_TEMPORAL_V4_FULL), seg("PEDIDOS_9_10", MODO_RL_TEMPORAL_V4_EXTENSION)
    full_11_12, ext_11_12 = seg("PEDIDOS_11_12", MODO_RL_TEMPORAL_V4_FULL), seg("PEDIDOS_11_12", MODO_RL_TEMPORAL_V4_EXTENSION)
    full_12, ext_12 = seg("PEDIDOS_12", MODO_RL_TEMPORAL_V4_FULL), seg("PEDIDOS_12", MODO_RL_TEMPORAL_V4_EXTENSION)

    b04 = _buscar_registro(registros, "B04_VENTANAS", MODO_RL_TEMPORAL_V4_FULL)
    b04_ok = bool(
        b04 is not None
        and b04.estado == "OK"
        and b04.pedidos_tardios_estimados == 0
        and b04.tardanza_estimada_min is not None
        and b04.tardanza_estimada_min <= TOLERANCIA
    )
    clasicos = {}
    for caso_id in ("B05_VOLCADOR", "B06_SPLIT"):
        clasicos[caso_id] = _no_regresion_clasica(
            _buscar_registro(registros, caso_id, MODO_RL_TEMPORAL_V4_FULL),
            _buscar_registro(registros, caso_id, MODO_RL_TEMPORAL_V4_EXTENSION),
        )

    sin_errores = bool(
        registros
        and all(r.estado == "OK" for r in registros)
        and full_global is not None
        and full_global.casos_error == 0
    )

    def preserva(candidato: ResumenModoFull | None, referencia: ResumenModoFull | None) -> bool:
        return bool(
            candidato is not None
            and referencia is not None
            and candidato.tasa_sin_riesgo_pct is not None
            and referencia.tasa_sin_riesgo_pct is not None
            and candidato.tasa_sin_riesgo_pct
            >= referencia.tasa_sin_riesgo_pct - TOLERANCIA_PRESERVACION_PP
        )

    def mejora(candidato: ResumenModoFull | None, referencia: ResumenModoFull | None) -> bool:
        if (
            candidato is None
            or referencia is None
            or candidato.tasa_sin_riesgo_pct is None
            or referencia.tasa_sin_riesgo_pct is None
        ):
            return False
        tasa_mejor = candidato.tasa_sin_riesgo_pct > referencia.tasa_sin_riesgo_pct + TOLERANCIA
        balance_positivo = candidato.victorias_vs_extension > candidato.derrotas_vs_extension
        return tasa_mejor and balance_positivo

    balance_global = bool(
        full_global is not None
        and full_global.victorias_vs_extension > full_global.derrotas_vs_extension
    )
    costos_extremos = bool(
        full_global is not None
        and ext_global is not None
        and full_global.costos_extremos_vs_greedy
        <= ext_global.costos_extremos_vs_greedy
    )
    tasa_global_no_regresiva = bool(
        full_global is not None
        and ext_global is not None
        and full_global.tasa_sin_riesgo_pct is not None
        and ext_global.tasa_sin_riesgo_pct is not None
        and full_global.tasa_sin_riesgo_pct
        >= ext_global.tasa_sin_riesgo_pct - TOLERANCIA_PRESERVACION_PP
    )

    criterios = {
        "b04_tardanza_cero": b04_ok,
        "b05_sin_regresion_vs_extension": clasicos["B05_VOLCADOR"],
        "b06_sin_regresion_vs_extension": clasicos["B06_SPLIT"],
        "sin_errores": sin_errores,
        "preserva_3_8_vs_extension": preserva(full_3_8, ext_3_8),
        "preserva_9_10_vs_extension": preserva(full_9_10, ext_9_10),
        "mejora_11_12_vs_extension": mejora(full_11_12, ext_11_12),
        "mejora_12_vs_extension": mejora(full_12, ext_12),
        "balance_global_positivo_vs_extension": balance_global,
        "tasa_global_no_regresiva": tasa_global_no_regresiva,
        "costos_extremos_no_empeoran": costos_extremos,
    }

    duros = (
        b04_ok
        and clasicos["B05_VOLCADOR"]
        and clasicos["B06_SPLIT"]
        and sin_errores
        and preserva(full_3_8, ext_3_8)
        and preserva(full_9_10, ext_9_10)
        and tasa_global_no_regresiva
        and costos_extremos
    )
    if duros and criterios["mejora_11_12_vs_extension"] and criterios["mejora_12_vs_extension"] and balance_global:
        estado = "CANDIDATO_PROMOCION_MANUAL"
    elif duros and balance_global and (
        criterios["mejora_11_12_vs_extension"]
        or criterios["mejora_12_vs_extension"]
    ):
        estado = "PROMETEDOR_NO_PROMOVER"
    else:
        estado = "NO_RECOMENDADO_PARA_PROMOCION"

    return {
        "estado": estado,
        "criterios": criterios,
        "observaciones": {
            "full_global": asdict(full_global) if full_global else None,
            "extension_global": asdict(ext_global) if ext_global else None,
            "full_3_8": asdict(full_3_8) if full_3_8 else None,
            "extension_3_8": asdict(ext_3_8) if ext_3_8 else None,
            "full_9_10": asdict(full_9_10) if full_9_10 else None,
            "extension_9_10": asdict(ext_9_10) if ext_9_10 else None,
            "full_11_12": asdict(full_11_12) if full_11_12 else None,
            "extension_11_12": asdict(ext_11_12) if ext_11_12 else None,
            "full_12": asdict(full_12) if full_12 else None,
            "extension_12": asdict(ext_12) if ext_12 else None,
        },
        "umbrales": {
            "preservacion_max_caida_pp": TOLERANCIA_PRESERVACION_PP,
            "costo_extremo_desde_pct": UMBRAL_COSTO_EXTREMO_PCT,
            "tolerancia_costo_clasicos_pct": TOLERANCIA_COSTO_CLASICOS_PCT,
        },
        "modelo_promovido": False,
        "nota": (
            "El veredicto no copia ni promueve modelos. "
            "CANDIDATO_PROMOCION_MANUAL exige una decisión expresa posterior."
        ),
    }


def resumir_casos(
    registros: Sequence[RegistroHoldoutFull],
) -> list[dict[str, Any]]:
    salida: list[dict[str, Any]] = []
    slugs = {
        MODO_RL_HISTORICO: "historico",
        MODO_RL_TEMPORAL_V4_QUICK: "quick",
        MODO_RL_TEMPORAL_V4_EXTENSION: "extension",
        MODO_RL_TEMPORAL_V4_FULL: "full",
        MODO_GREEDY: "greedy",
    }
    for grupo, caso_id in sorted({(r.grupo, r.caso_id) for r in registros}):
        seleccion = [r for r in registros if r.grupo == grupo and r.caso_id == caso_id]
        if not seleccion:
            continue
        primero = seleccion[0]
        por_modo = {r.modo: r for r in seleccion}
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
            r = por_modo.get(modo)
            slug = slugs[modo]
            fila[f"estado_{slug}"] = r.estado if r else "FALTANTE"
            fila[f"error_{slug}"] = r.error if r else ""
            fila[f"pedidos_tardios_{slug}"] = (
                r.pedidos_tardios_estimados if r else None
            )
            fila[f"tardanza_{slug}_min"] = r.tardanza_estimada_min if r else None
            fila[f"costo_{slug}"] = r.costo_recalculado if r else None
            fila[f"firma_{slug}"] = r.firma_plan if r else ""
        full = por_modo.get(MODO_RL_TEMPORAL_V4_FULL)
        fila["comparacion_full_vs_historico"] = (
            full.comparacion_vs_historico if full else "NO_DISPONIBLE"
        )
        fila["comparacion_full_vs_quick"] = (
            full.comparacion_vs_quick if full else "NO_DISPONIBLE"
        )
        fila["comparacion_full_vs_extension"] = (
            full.comparacion_vs_extension if full else "NO_DISPONIBLE"
        )
        fila["comparacion_full_vs_greedy"] = (
            full.comparacion_vs_greedy if full else "NO_DISPONIBLE"
        )
        fila["gap_costo_full_vs_extension_pct"] = (
            full.gap_costo_vs_extension_pct if full else None
        )
        fila["gap_costo_full_vs_greedy_pct"] = (
            full.gap_costo_vs_greedy_pct if full else None
        )
        salida.append(fila)
    return salida


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
        elif isinstance(valor, list) and (
            "semilla" in clave.lower() or "seed" in clave.lower()
        ):
            for item in valor:
                if isinstance(item, int) and not isinstance(item, bool):
                    salida.add(int(item))
        elif isinstance(valor, int) and not isinstance(valor, bool) and (
            "semilla" in clave.lower() or clave.lower() == "seed"
        ):
            salida.add(int(valor))

    recorrer(contenido)
    return salida


def validar_metadatos_fase16d9(
    *,
    historical_model: str | Path,
    quick_model: str | Path,
    quick_config: str | Path,
    quick_selection: str | Path,
    extension_model: str | Path,
    extension_config: str | Path,
    extension_selection: str | Path,
    extension_summary: str | Path,
    quick_holdout_result: str | Path | None,
    full_model: str | Path,
    full_config: str | Path,
    full_selection: str | Path,
    full_summary: str | Path,
    full_audit: str | Path,
    holdout_16d7_result: str | Path,
    seed_inicio: int,
    semillas_holdout: Sequence[int],
) -> dict[str, Any]:
    if seed_inicio < SEED_HOLDOUT_FINAL_MINIMO:
        raise ValueError(
            f"La Fase 16D.9 exige seed_inicio >= {SEED_HOLDOUT_FINAL_MINIMO}."
        )

    base = validar_metadatos_fase16d7(
        quick_model=quick_model,
        quick_config=quick_config,
        quick_selection=quick_selection,
        extension_model=extension_model,
        extension_config=extension_config,
        extension_selection=extension_selection,
        extension_summary=extension_summary,
        historical_model=historical_model,
        seed_inicio=seed_inicio,
        semillas_holdout=(),
        quick_holdout_result=quick_holdout_result,
    )

    full_path = Path(full_model)
    if not full_path.is_file():
        raise FileNotFoundError(f"No existe el modelo full: {full_path}")
    full_cfg = _leer_objeto_json(full_config, "la configuración full 16D.8")
    full_sel = _leer_objeto_json(full_selection, "la selección full 16D.8")
    full_res = _leer_objeto_json(full_summary, "el resumen final full 16D.8")
    audit = _leer_objeto_json(full_audit, "la auditoría full 16D.8")
    holdout_16d7 = _leer_objeto_json(
        holdout_16d7_result, "el holdout independiente 16D.7"
    )

    if full_cfg.get("fase") != "16D.8":
        raise ValueError("La configuración full no corresponde a la Fase 16D.8.")
    if full_cfg.get("version_run") != "pedemonte-rl-temporal-v4-full-11-12-v1":
        raise ValueError("version_run full inesperada.")
    temporal = full_cfg.get("temporal")
    if not isinstance(temporal, dict) or temporal.get("usar_mascara_temporal_dura") is not False:
        raise ValueError("La máscara temporal dura del full debe estar desactivada.")
    if full_sel.get("criterio") != "VALIDACION_EXTERNA_FULL_11_12_LEXICOGRAFICA_V4":
        raise ValueError("Criterio de selección full inesperado.")
    if int(full_sel.get("timestep_final_seleccionado", -1)) != 158_288:
        raise ValueError("El checkpoint full esperado debe ser 158288.")
    for clave in (
        "modelo_promovido",
        "modelo_historico_sobrescrito",
        "modelo_v4_quick_sobrescrito",
        "modelo_extension_9_12_sobrescrito",
    ):
        if full_sel.get(clave) is not False:
            raise ValueError(f"La selección full no preserva {clave}=false.")

    if int(full_res.get("timestep", -1)) != 158_288:
        raise ValueError("El resumen full no corresponde al checkpoint 158288.")
    if int(full_res.get("guard_3_8_sin_riesgo", -1)) != 24:
        raise ValueError("La selección full no preserva guard 3-8 24/24.")
    if int(full_res.get("guard_9_10_sin_riesgo", -1)) != 12:
        raise ValueError("La selección full no preserva guard 9-10 12/12.")
    if int(full_res.get("objetivo_12_sin_riesgo", -1)) != 12:
        raise ValueError("La selección full no alcanza exactos 12 en 12/12.")
    if int(full_res.get("objetivo_general_11_12_sin_riesgo", -1)) != 12:
        raise ValueError("La selección full no alcanza general 11-12 en 12/12.")

    if audit.get("estado") != "SALIDAS_16D_8_OK":
        raise ValueError("La auditoría 16D.8 no está aprobada.")
    if int(audit.get("timestep_final_seleccionado", -1)) != 158_288:
        raise ValueError("La auditoría 16D.8 no corresponde al checkpoint 158288.")
    if audit.get("modelo_promovido") is not False:
        raise ValueError("El modelo full aparece promovido en la auditoría.")

    if holdout_16d7.get("veredicto", {}).get("estado") != "CANDIDATO_ENTRENAMIENTO_COMPLETO":
        raise ValueError("El holdout 16D.7 no contiene el veredicto esperado.")

    hash_full = hash_archivo(full_path)
    if full_sel.get("sha256_modelo_final") != hash_full:
        raise ValueError("El hash del modelo full no coincide con su selección.")
    if audit.get("sha256_modelo_final") != hash_full:
        raise ValueError("El hash del modelo full no coincide con la auditoría.")

    hashes = dict(base["hashes_modelos"])
    hashes["full"] = hash_full
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("Los modelos histórico, quick, extensión y full deben ser distintos.")

    semillas = [int(s) for s in semillas_holdout]
    if len(semillas) != len(set(semillas)):
        raise ValueError("El holdout final contiene semillas repetidas.")
    prohibidas = set(base["semillas_prohibidas_declaradas"])
    prohibidas.update(_semillas_json(full_cfg))
    prohibidas.update(_semillas_json(full_sel))
    prohibidas.update(_semillas_json(full_res))
    prohibidas.update(_semillas_json(audit))
    prohibidas.update(_semillas_json(holdout_16d7))
    prohibidas.update(range(273_000, 273_062))
    interseccion = sorted(set(semillas).intersection(prohibidas))
    if interseccion:
        raise ValueError(
            "Las semillas del holdout final se superponen con fases previas: "
            + ", ".join(str(s) for s in interseccion)
        )
    if semillas and min(semillas) < SEED_HOLDOUT_FINAL_MINIMO:
        raise ValueError("Se detectó una semilla final anterior a 274000.")

    return {
        "base_16d7": base,
        "full_config": full_cfg,
        "full_selection": full_sel,
        "full_summary": full_res,
        "full_audit": audit,
        "hashes_modelos": hashes,
        "semillas_prohibidas_declaradas": sorted(prohibidas),
        "semillas_holdout_validadas": semillas,
        "modelo_promovido": False,
    }


def _normalizar_json(valor: Any) -> Any:
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, dict):
        return {str(k): _normalizar_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_normalizar_json(v) for v in valor]
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
    registros: Sequence[RegistroHoldoutFull],
    resumen_global: Sequence[ResumenModoFull],
    resumen_estratos: Sequence[ResumenModoFull],
    resumen_segmentos: Sequence[ResumenModoFull],
    casos: Sequence[Mapping[str, Any]],
    clasicos: Mapping[str, Any],
    veredicto: Mapping[str, Any],
    semillas: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    destino = Path(directorio)
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {
        "evaluacion_json": destino / "high_demand_policy_holdout.json",
        "corridas_csv": destino / "high_demand_policy_holdout_runs.csv",
        "resumen_csv": destino / "high_demand_policy_holdout_summary.csv",
        "estratos_csv": destino / "high_demand_policy_holdout_strata.csv",
        "segmentos_csv": destino / "high_demand_policy_holdout_segments.csv",
        "casos_csv": destino / "high_demand_policy_holdout_cases.csv",
        "semillas_csv": destino / "high_demand_policy_holdout_seeds.csv",
    }
    contenido = {
        "version_evaluacion": VERSION_EVALUACION,
        "metadatos": dict(metadatos),
        "veredicto": dict(veredicto),
        "analisis_clasicos": dict(clasicos),
        "resumen_global": [asdict(r) for r in resumen_global],
        "resumen_estratos": [asdict(r) for r in resumen_estratos],
        "resumen_segmentos": [asdict(r) for r in resumen_segmentos],
        "casos": [dict(c) for c in casos],
        "semillas": [dict(s) for s in semillas],
        "registros": [asdict(r) for r in registros],
    }
    rutas["evaluacion_json"].write_text(
        dumps(_normalizar_json(contenido), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _escribir_csv(rutas["corridas_csv"], [asdict(r) for r in registros])
    _escribir_csv(rutas["resumen_csv"], [asdict(r) for r in resumen_global])
    _escribir_csv(rutas["estratos_csv"], [asdict(r) for r in resumen_estratos])
    _escribir_csv(rutas["segmentos_csv"], [asdict(r) for r in resumen_segmentos])
    _escribir_csv(rutas["casos_csv"], [dict(c) for c in casos])
    _escribir_csv(rutas["semillas_csv"], [dict(s) for s in semillas])
    return rutas
