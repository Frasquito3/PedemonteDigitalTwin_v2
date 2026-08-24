from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Mapping, Protocol

from planner.algorithms.hybrid_rl_ga_greedy import (
    HybridRLGAGreedyPlanner,
)
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.evaluation.classic_benchmark import firma_plan
from planner.evaluation.classic_instances import CasoBenchmarkClasico
from planner.routing.objective import (
    VERSION_AUDITORIA_COSTO,
    evaluar_plan_estimado,
)
from planner.routing.travel import (
    ProveedorViaje,
    construir_matriz_viaje,
)


VERSION_BENCHMARK_HIBRIDO_ROBUSTO = "benchmark-hibrido-robusto-v1"


class PlanificadorCompatible(Protocol):
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        ...


@dataclass(frozen=True)
class ResultadoCasoHibridoRobusto:
    caso_id: str
    categoria: str
    modelo_alias: str
    estado: str
    fuente_seleccionada: str
    motivo: str
    costo_seleccionado: float
    costo_greedy: float
    costo_ga: float | None
    costo_rl: float | None
    mejora_vs_greedy_pct: float
    mejora_vs_ga_pct: float | None
    cumple_garantia_greedy: bool
    cumple_garantia_ga: bool | None
    tiempo_total_ms: float
    tiempo_greedy_ms: float
    tiempo_ga_ms: float
    tiempo_rl_ms: float
    seed_ga: int
    firma_ruta: str
    errores_ga: str
    errores_rl: str
    fuente_viaje: str
    version_viaje: str
    fallbacks_matriz: int


@dataclass(frozen=True)
class ResumenModeloHibridoRobusto:
    modelo_alias: str
    casos: int
    fuente_greedy: int
    fuente_ga: int
    fuente_rl: int
    casos_con_error_ga: int
    casos_con_error_rl: int
    violaciones_garantia_greedy: int
    violaciones_garantia_ga: int
    mejora_media_vs_greedy_pct: float
    mejora_mediana_vs_greedy_pct: float
    mejora_media_vs_ga_pct: float | None
    mejora_mediana_vs_ga_pct: float | None
    tiempo_medio_ms: float
    tiempo_p90_ms: float
    tiempo_maximo_ms: float


@dataclass(frozen=True)
class ResultadoBenchmarkHibridoRobusto:
    version_benchmark: str
    version_objetivo: str
    generado_utc: str
    fuente_viaje: str
    version_viaje: str
    cantidad_casos: int
    cantidad_modelos: int
    cantidad_filas: int
    corridas: tuple[ResultadoCasoHibridoRobusto, ...]
    resumenes: tuple[ResumenModeloHibridoRobusto, ...]


def _porcentaje_mejora(costo: float, referencia: float) -> float:
    if abs(referencia) <= 1e-12:
        return 0.0 if abs(costo) <= 1e-12 else float("-inf")
    return (referencia - costo) / referencia * 100.0


def _percentil_lineal(valores: list[float], proporcion: float) -> float:
    if not valores:
        raise ValueError("No se puede calcular un percentil vacío.")
    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return ordenados[0]
    posicion = (len(ordenados) - 1) * proporcion
    inferior = int(posicion)
    superior = min(inferior + 1, len(ordenados) - 1)
    fraccion = posicion - inferior
    return (
        ordenados[inferior] * (1.0 - fraccion)
        + ordenados[superior] * fraccion
    )


def _resumir_modelo(
    alias: str,
    filas: list[ResultadoCasoHibridoRobusto],
) -> ResumenModeloHibridoRobusto:
    mejoras_greedy = [fila.mejora_vs_greedy_pct for fila in filas]
    mejoras_ga = [
        fila.mejora_vs_ga_pct
        for fila in filas
        if fila.mejora_vs_ga_pct is not None
    ]
    tiempos = [fila.tiempo_total_ms for fila in filas]

    return ResumenModeloHibridoRobusto(
        modelo_alias=alias,
        casos=len(filas),
        fuente_greedy=sum(
            fila.fuente_seleccionada == "GREEDY" for fila in filas
        ),
        fuente_ga=sum(fila.fuente_seleccionada == "GA" for fila in filas),
        fuente_rl=sum(fila.fuente_seleccionada == "RL" for fila in filas),
        casos_con_error_ga=sum(bool(fila.errores_ga) for fila in filas),
        casos_con_error_rl=sum(bool(fila.errores_rl) for fila in filas),
        violaciones_garantia_greedy=sum(
            not fila.cumple_garantia_greedy for fila in filas
        ),
        violaciones_garantia_ga=sum(
            fila.cumple_garantia_ga is False for fila in filas
        ),
        mejora_media_vs_greedy_pct=fmean(mejoras_greedy),
        mejora_mediana_vs_greedy_pct=median(mejoras_greedy),
        mejora_media_vs_ga_pct=(
            fmean(mejoras_ga) if mejoras_ga else None
        ),
        mejora_mediana_vs_ga_pct=(
            median(mejoras_ga) if mejoras_ga else None
        ),
        tiempo_medio_ms=fmean(tiempos),
        tiempo_p90_ms=_percentil_lineal(tiempos, 0.90),
        tiempo_maximo_ms=max(tiempos),
    )


def ejecutar_benchmark_hibrido_robusto(
    casos: tuple[CasoBenchmarkClasico, ...],
    *,
    proveedor_viaje: ProveedorViaje,
    planners_rl: Mapping[str, PlanificadorCompatible],
    configuracion_planificacion: ConfiguracionPlanificacion | None = None,
    exigir_sin_fallback: bool = True,
    tolerancia_costo: float = 1e-6,
) -> ResultadoBenchmarkHibridoRobusto:
    if not casos:
        raise ValueError("El benchmark requiere al menos un caso.")
    if not planners_rl:
        raise ValueError("El benchmark requiere al menos un modelo RL.")
    if tolerancia_costo < 0.0:
        raise ValueError("tolerancia_costo no puede ser negativa.")

    configuracion = (
        configuracion_planificacion
        if configuracion_planificacion is not None
        else ConfiguracionPlanificacion()
    )

    corridas: list[ResultadoCasoHibridoRobusto] = []

    for alias in sorted(planners_rl):
        planner_hibrido = HybridRLGAGreedyPlanner(
            planner_rl=planners_rl[alias],
            configuracion_planificacion=configuracion,
            proveedor_viaje=proveedor_viaje,
        )

        for caso in casos:
            plan = planner_hibrido.generar_plan(caso.instancia)
            decision = planner_hibrido.ultima_decision
            if decision is None:
                raise RuntimeError(
                    f"No se registró decisión para {caso.caso_id}/{alias}."
                )

            validacion = validar_plan(caso.instancia, plan)
            if not validacion.valido:
                raise RuntimeError(
                    f"Plan inválido en {caso.caso_id}/{alias}: "
                    + " | ".join(validacion.errores)
                )

            matriz = construir_matriz_viaje(
                caso.instancia,
                configuracion,
                proveedor=proveedor_viaje,
            )
            if exigir_sin_fallback and matriz.usa_fallback:
                raise RuntimeError(
                    f"{caso.caso_id}/{alias} utilizó "
                    f"{matriz.cantidad_fallbacks} fallback(s) viales."
                )

            estimacion = evaluar_plan_estimado(
                caso.instancia,
                plan,
                matriz,
                configuracion,
            )
            if abs(plan.costo_estimado - estimacion.costo_total) > tolerancia_costo:
                raise RuntimeError(
                    f"Costo inconsistente en {caso.caso_id}/{alias}: "
                    f"plan={plan.costo_estimado}, "
                    f"auditoria={estimacion.costo_total}."
                )

            corridas.append(
                ResultadoCasoHibridoRobusto(
                    caso_id=caso.caso_id,
                    categoria=caso.categoria,
                    modelo_alias=alias,
                    estado="OK",
                    fuente_seleccionada=decision.fuente_seleccionada.value,
                    motivo=decision.motivo.value,
                    costo_seleccionado=estimacion.costo_total,
                    costo_greedy=decision.costo_greedy,
                    costo_ga=decision.costo_ga,
                    costo_rl=decision.costo_rl,
                    mejora_vs_greedy_pct=_porcentaje_mejora(
                        estimacion.costo_total,
                        decision.costo_greedy,
                    ),
                    mejora_vs_ga_pct=(
                        None
                        if decision.costo_ga is None
                        else _porcentaje_mejora(
                            estimacion.costo_total,
                            decision.costo_ga,
                        )
                    ),
                    cumple_garantia_greedy=(
                        estimacion.costo_total
                        <= decision.costo_greedy + tolerancia_costo
                    ),
                    cumple_garantia_ga=(
                        None
                        if decision.costo_ga is None
                        else estimacion.costo_total
                        <= decision.costo_ga + tolerancia_costo
                    ),
                    tiempo_total_ms=decision.tiempo_total_ms,
                    tiempo_greedy_ms=decision.tiempo_greedy_ms,
                    tiempo_ga_ms=decision.tiempo_ga_ms,
                    tiempo_rl_ms=decision.tiempo_rl_ms,
                    seed_ga=decision.seed_ga,
                    firma_ruta=firma_plan(plan),
                    errores_ga=" | ".join(decision.errores_ga),
                    errores_rl=" | ".join(decision.errores_rl),
                    fuente_viaje=matriz.fuente.value,
                    version_viaje=matriz.version_fuente,
                    fallbacks_matriz=matriz.cantidad_fallbacks,
                )
            )

    resumenes = tuple(
        _resumir_modelo(
            alias,
            [fila for fila in corridas if fila.modelo_alias == alias],
        )
        for alias in sorted(planners_rl)
    )

    primera_matriz = construir_matriz_viaje(
        casos[0].instancia,
        configuracion,
        proveedor=proveedor_viaje,
    )

    return ResultadoBenchmarkHibridoRobusto(
        version_benchmark=VERSION_BENCHMARK_HIBRIDO_ROBUSTO,
        version_objetivo=VERSION_AUDITORIA_COSTO,
        generado_utc=datetime.now(timezone.utc).isoformat(),
        fuente_viaje=primera_matriz.fuente.value,
        version_viaje=primera_matriz.version_fuente,
        cantidad_casos=len(casos),
        cantidad_modelos=len(planners_rl),
        cantidad_filas=len(corridas),
        corridas=tuple(corridas),
        resumenes=resumenes,
    )


def escribir_resultados_benchmark_hibrido_robusto(
    resultado: ResultadoBenchmarkHibridoRobusto,
    directorio_salida: str | Path,
) -> dict[str, str]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ruta_corridas = salida / "corridas.csv"
    ruta_resumen = salida / "resumen.csv"
    ruta_json = salida / "benchmark.json"

    filas_corridas = [asdict(fila) for fila in resultado.corridas]
    filas_resumen = [asdict(fila) for fila in resultado.resumenes]

    with ruta_corridas.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(filas_corridas[0]),
        )
        escritor.writeheader()
        escritor.writerows(filas_corridas)

    with ruta_resumen.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(filas_resumen[0]),
        )
        escritor.writeheader()
        escritor.writerows(filas_resumen)

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(
            asdict(resultado),
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "corridas_csv": str(ruta_corridas),
        "resumen_csv": str(ruta_resumen),
        "benchmark_json": str(ruta_json),
    }
