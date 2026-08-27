from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from json import dumps
from math import isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from planner.core.base import PlanificadorTurno
from planner.core.config import ConfiguracionPlanificacion
from planner.domain.validator import validar_plan
from planner.evaluation.rl_temporal_v4_extension_holdout import (
    CasoHoldoutExtension,
    firma_plan,
)
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import ProveedorViaje, construir_matriz_viaje


VERSION_EVALUACION = "rl-temporal-v4-operational-holdout-v1"
MODO_OPERACIONAL = "RL_TEMPORAL_V4_OPERACIONAL"
MODO_EXTENSION = "RL_TEMPORAL_V4_EXTENSION"
MODO_FULL = "RL_TEMPORAL_V4_FULL"
MODO_GREEDY = "GREEDY"
ORDEN_MODOS = (MODO_OPERACIONAL, MODO_EXTENSION, MODO_FULL, MODO_GREEDY)
TOLERANCIA = 1e-9


@dataclass(frozen=True)
class RegistroHoldoutOperacional:
    grupo: str
    caso_id: str
    categoria: str
    instancia_id: str
    seed_escenario: int
    cantidad_pedidos: int
    estrato: str
    modo: str
    estado: str
    error: str
    fuente_operacional: str
    pedidos_tardios: int | None
    tardanza_total_min: float | None
    costo_total: float | None
    tiempo_plan_ms: float | None
    firma_plan: str


@dataclass(frozen=True)
class ResumenHoldoutOperacional:
    alcance: str
    modo: str
    casos: int
    ok: int
    errores: int
    sin_riesgo: int
    tasa_sin_riesgo_pct: float | None
    pedidos_tardios_total: int | None
    tardanza_media_min: float | None
    tardanza_mediana_min: float | None
    tardanza_max_min: float | None
    costo_mediano: float | None


def _pedidos_tardios(estimacion: Any) -> int:
    valor = getattr(estimacion, "pedidos_tardios", None)
    if valor is not None:
        return int(valor)
    return 0 if float(estimacion.tardanza_total_min) <= TOLERANCIA else 1


def clave_lexicografica(registro: RegistroHoldoutOperacional) -> tuple[int, float, float]:
    if (
        registro.estado != "OK"
        or registro.pedidos_tardios is None
        or registro.tardanza_total_min is None
        or registro.costo_total is None
    ):
        return (10**9, float("inf"), float("inf"))
    return (
        int(registro.pedidos_tardios),
        float(registro.tardanza_total_min),
        float(registro.costo_total),
    )


def comparar_lexicografico(
    candidato: RegistroHoldoutOperacional,
    referencia: RegistroHoldoutOperacional,
) -> str:
    clave_c = clave_lexicografica(candidato)
    clave_r = clave_lexicografica(referencia)
    if not all(isfinite(float(v)) for v in (*clave_c[1:], *clave_r[1:])):
        return "NO_DISPONIBLE"
    if clave_c < clave_r:
        return "MEJOR"
    if clave_c > clave_r:
        return "PEOR"
    return "EMPATE"


def evaluar_casos_operacionales(
    casos: Sequence[CasoHoldoutExtension],
    planificadores: Mapping[str, PlanificadorTurno],
    *,
    configuracion: ConfiguracionPlanificacion | None = None,
    proveedor_viaje: ProveedorViaje | None = None,
) -> list[RegistroHoldoutOperacional]:
    cfg = configuracion or ConfiguracionPlanificacion()
    salida: list[RegistroHoldoutOperacional] = []

    for caso in casos:
        matriz = construir_matriz_viaje(
            caso.instancia,
            cfg,
            proveedor_viaje,
        )
        for modo, planificador in planificadores.items():
            try:
                plan = planificador.generar_plan(caso.instancia)
                validacion = validar_plan(caso.instancia, plan)
                if not validacion.valido:
                    raise RuntimeError(" | ".join(validacion.errores))
                estimacion = evaluar_plan_estimado(
                    caso.instancia,
                    plan,
                    matriz,
                    cfg,
                )
                fuente = ""
                decision = getattr(planificador, "ultima_decision", None)
                if decision is not None:
                    fuente = str(
                        getattr(decision, "fuente_seleccionada", "")
                    )
                salida.append(
                    RegistroHoldoutOperacional(
                        grupo=caso.grupo,
                        caso_id=caso.caso_id,
                        categoria=caso.categoria,
                        instancia_id=caso.instancia.instancia_id,
                        seed_escenario=caso.instancia.seed_escenario,
                        cantidad_pedidos=len(caso.instancia.pedidos),
                        estrato=caso.estrato,
                        modo=modo,
                        estado="OK",
                        error="",
                        fuente_operacional=fuente,
                        pedidos_tardios=_pedidos_tardios(estimacion),
                        tardanza_total_min=float(estimacion.tardanza_total_min),
                        costo_total=float(estimacion.costo_total),
                        tiempo_plan_ms=float(plan.tiempo_computo_ms),
                        firma_plan=firma_plan(plan),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - evidencia por corrida
                salida.append(
                    RegistroHoldoutOperacional(
                        grupo=caso.grupo,
                        caso_id=caso.caso_id,
                        categoria=caso.categoria,
                        instancia_id=caso.instancia.instancia_id,
                        seed_escenario=caso.instancia.seed_escenario,
                        cantidad_pedidos=len(caso.instancia.pedidos),
                        estrato=caso.estrato,
                        modo=modo,
                        estado="ERROR",
                        error=f"{type(exc).__name__}: {exc}",
                        fuente_operacional="",
                        pedidos_tardios=None,
                        tardanza_total_min=None,
                        costo_total=None,
                        tiempo_plan_ms=None,
                        firma_plan="",
                    )
                )
    return salida


def _segmento(registro: RegistroHoldoutOperacional) -> str:
    n = registro.cantidad_pedidos
    if 3 <= n <= 8:
        return "PEDIDOS_3_8"
    if 9 <= n <= 10:
        return "PEDIDOS_9_10"
    if 11 <= n <= 12:
        return "PEDIDOS_11_12"
    return "OTROS"


def resumir(
    registros: Sequence[RegistroHoldoutOperacional],
) -> list[ResumenHoldoutOperacional]:
    salida: list[ResumenHoldoutOperacional] = []
    alcances = ["TODOS", "PEDIDOS_3_8", "PEDIDOS_9_10", "PEDIDOS_11_12", "PEDIDOS_12"]
    for alcance in alcances:
        for modo in ORDEN_MODOS:
            seleccion = [
                r for r in registros
                if r.grupo == "HOLDOUT_SINTETICO"
                and r.modo == modo
                and (
                    alcance == "TODOS"
                    or (alcance == "PEDIDOS_12" and r.cantidad_pedidos == 12)
                    or (alcance != "PEDIDOS_12" and _segmento(r) == alcance)
                )
            ]
            ok = [r for r in seleccion if r.estado == "OK"]
            tardios = [int(r.pedidos_tardios or 0) for r in ok]
            tardanzas = [float(r.tardanza_total_min or 0.0) for r in ok]
            costos = [float(r.costo_total or 0.0) for r in ok]
            salida.append(
                ResumenHoldoutOperacional(
                    alcance=alcance,
                    modo=modo,
                    casos=len(seleccion),
                    ok=len(ok),
                    errores=len(seleccion) - len(ok),
                    sin_riesgo=sum(1 for valor in tardios if valor == 0),
                    tasa_sin_riesgo_pct=(
                        100.0 * sum(1 for valor in tardios if valor == 0) / len(ok)
                        if ok else None
                    ),
                    pedidos_tardios_total=sum(tardios) if ok else None,
                    tardanza_media_min=mean(tardanzas) if tardanzas else None,
                    tardanza_mediana_min=median(tardanzas) if tardanzas else None,
                    tardanza_max_min=max(tardanzas) if tardanzas else None,
                    costo_mediano=median(costos) if costos else None,
                )
            )
    return salida


def construir_veredicto(
    registros: Sequence[RegistroHoldoutOperacional],
    resumenes: Sequence[ResumenHoldoutOperacional],
) -> dict[str, Any]:
    por_caso: dict[str, dict[str, RegistroHoldoutOperacional]] = {}
    for registro in registros:
        por_caso.setdefault(registro.caso_id, {})[registro.modo] = registro

    dominancias: dict[str, bool] = {}
    for referencia in (MODO_EXTENSION, MODO_FULL, MODO_GREEDY):
        dominancias[referencia] = all(
            comparar_lexicografico(
                modos[MODO_OPERACIONAL],
                modos[referencia],
            ) in {"MEJOR", "EMPATE"}
            for modos in por_caso.values()
            if MODO_OPERACIONAL in modos and referencia in modos
        )

    def resumen(alcance: str, modo: str) -> ResumenHoldoutOperacional | None:
        return next(
            (r for r in resumenes if r.alcance == alcance and r.modo == modo),
            None,
        )

    op_3_8 = resumen("PEDIDOS_3_8", MODO_OPERACIONAL)
    op_9_10 = resumen("PEDIDOS_9_10", MODO_OPERACIONAL)
    ext_9_10 = resumen("PEDIDOS_9_10", MODO_EXTENSION)
    op_11_12 = resumen("PEDIDOS_11_12", MODO_OPERACIONAL)
    full_11_12 = resumen("PEDIDOS_11_12", MODO_FULL)
    op_12 = resumen("PEDIDOS_12", MODO_OPERACIONAL)
    full_12 = resumen("PEDIDOS_12", MODO_FULL)

    clasicos_ok = True
    for caso_id in ("B04_VENTANAS", "B05_VOLCADOR", "B06_SPLIT"):
        registro = por_caso.get(caso_id, {}).get(MODO_OPERACIONAL)
        clasicos_ok = clasicos_ok and bool(
            registro is not None
            and registro.estado == "OK"
            and registro.pedidos_tardios == 0
        )

    criterios = {
        "sin_errores_operacionales": all(
            r.estado == "OK" for r in registros if r.modo == MODO_OPERACIONAL
        ),
        "b04_b05_b06_sin_tardanza": clasicos_ok,
        "domina_extension_caso_a_caso": dominancias[MODO_EXTENSION],
        "domina_full_caso_a_caso": dominancias[MODO_FULL],
        "domina_greedy_caso_a_caso": dominancias[MODO_GREEDY],
        "preserva_3_8_100_pct": bool(
            op_3_8 and op_3_8.tasa_sin_riesgo_pct == 100.0
        ),
        "9_10_no_peor_que_extension": bool(
            op_9_10 and ext_9_10
            and op_9_10.sin_riesgo >= ext_9_10.sin_riesgo
        ),
        "11_12_no_peor_que_full": bool(
            op_11_12 and full_11_12
            and op_11_12.sin_riesgo >= full_11_12.sin_riesgo
        ),
        "12_no_peor_que_full": bool(
            op_12 and full_12 and op_12.sin_riesgo >= full_12.sin_riesgo
        ),
    }
    return {
        "estado": (
            "APTO_PARA_INTEGRACION_ANYLOGIC"
            if all(criterios.values())
            else "NO_APTO_PARA_INTEGRACION_ANYLOGIC"
        ),
        "criterios": criterios,
    }


def escribir_resultados(
    output_dir: str | Path,
    *,
    metadatos: Mapping[str, Any],
    registros: Sequence[RegistroHoldoutOperacional],
    resumenes: Sequence[ResumenHoldoutOperacional],
    veredicto: Mapping[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "rl_temporal_v4_operational_holdout.json"
    runs_path = output / "rl_temporal_v4_operational_holdout_runs.csv"
    summary_path = output / "rl_temporal_v4_operational_holdout_summary.csv"

    json_path.write_text(
        dumps(
            {
                "metadatos": dict(metadatos),
                "veredicto": dict(veredicto),
                "resumenes": [asdict(r) for r in resumenes],
                "registros": [asdict(r) for r in registros],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for path, filas in (
        (runs_path, [asdict(r) for r in registros]),
        (summary_path, [asdict(r) for r in resumenes]),
    ):
        with path.open("w", encoding="utf-8-sig", newline="") as archivo:
            if filas:
                writer = csv.DictWriter(archivo, fieldnames=list(filas[0]))
                writer.writeheader()
                writer.writerows(filas)

    return {"json": json_path, "runs": runs_path, "summary": summary_path}
