from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from planner.algorithms.ga import (
    ConfiguracionGA,
    GeneticAlgorithmPlanner,
)
from planner.algorithms.greedy import GreedyFeasiblePlanner
from planner.algorithms.hybrid_rl_greedy import HybridRLGreedyPlanner
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


VERSION_BENCHMARK_RL_CONTROLADO = "benchmark-rl-controlado-v1"


class PlanificadorCompatible(Protocol):
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        ...


@dataclass(frozen=True)
class MetadatosModeloRL:
    alias: str
    ruta_modelo: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.alias.strip():
            raise ValueError("alias no puede estar vacío.")
        if not self.ruta_modelo.strip():
            raise ValueError("ruta_modelo no puede estar vacía.")
        if not self.sha256.strip():
            raise ValueError("sha256 no puede estar vacío.")


@dataclass(frozen=True)
class ConfiguracionBenchmarkRLControlado:
    configuracion_ga: ConfiguracionGA = field(
        default_factory=ConfiguracionGA
    )
    seed_ga: int = 101
    tolerancia_costo: float = 1e-6
    exigir_sin_fallback: bool = True
    verificar_hibrido_no_peor_greedy: bool = True

    def __post_init__(self) -> None:
        if self.tolerancia_costo < 0.0:
            raise ValueError(
                "tolerancia_costo no puede ser negativa."
            )


@dataclass(frozen=True)
class ResultadoCorridaRLControlada:
    caso_id: str
    categoria: str
    instancia_id: str
    algoritmo: str
    modelo_alias: str
    modelo_sha256: str
    estado: str
    error: str
    plan_valido: bool
    costo_estimado: float | None
    distancia_total_km: float | None
    duracion_operacion_min: float | None
    tardanza_total_min: float | None
    exceso_tolerancia_min: float | None
    viajes_totales: int | None
    desbalance_fin_min: float | None
    tiempo_computo_ms: float | None
    firma_ruta: str
    advertencias: str
    diferencia_vs_greedy: float | None
    diferencia_vs_greedy_pct: float | None
    diferencia_vs_ga: float | None
    diferencia_vs_ga_pct: float | None
    fuente_hibrida: str
    motivo_hibrido: str
    costo_rl_crudo_hibrido: float | None
    costo_greedy_hibrido: float | None
    fuente_viaje: str
    version_viaje: str
    fallbacks_matriz: int


@dataclass(frozen=True)
class ResumenCasoModeloRL:
    caso_id: str
    categoria: str
    modelo_alias: str
    costo_greedy: float
    costo_ga: float
    costo_rl: float | None
    costo_hibrido: float | None
    rl_estado: str
    hibrido_estado: str
    fuente_hibrida: str
    mejora_rl_vs_greedy_pct: float | None
    mejora_hibrido_vs_greedy_pct: float | None
    mejora_rl_vs_ga_pct: float | None
    mejora_hibrido_vs_ga_pct: float | None
    hibrido_cumple_garantia: bool


@dataclass(frozen=True)
class ResultadoBenchmarkRLControlado:
    version_benchmark: str
    version_objetivo: str
    generado_utc: str
    fuente_viaje: str
    version_viaje: str
    casos: tuple[str, ...]
    modelos: tuple[MetadatosModeloRL, ...]
    configuracion_ga: dict[str, Any]
    seed_ga: int
    corridas: tuple[ResultadoCorridaRLControlada, ...]
    resumenes: tuple[ResumenCasoModeloRL, ...]


def calcular_sha256_archivo(ruta: str | Path) -> str:
    ruta_resuelta = Path(ruta).expanduser().resolve()
    if not ruta_resuelta.is_file():
        raise FileNotFoundError(
            f"No existe el archivo para calcular SHA-256: {ruta_resuelta}"
        )

    digest = hashlib.sha256()
    with ruta_resuelta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _porcentaje(diferencia: float, referencia: float) -> float:
    if abs(referencia) <= 1e-12:
        return 0.0 if abs(diferencia) <= 1e-12 else float("inf")
    return diferencia / referencia * 100.0


def _auditar_plan(
    *,
    caso: CasoBenchmarkClasico,
    algoritmo: str,
    modelo_alias: str,
    modelo_sha256: str,
    plan: PlanTurno,
    proveedor_viaje: ProveedorViaje,
    configuracion: ConfiguracionPlanificacion,
    costo_greedy: float,
    costo_ga: float,
    tolerancia_costo: float,
    exigir_sin_fallback: bool,
    estado: str = "OK",
    fuente_hibrida: str = "",
    motivo_hibrido: str = "",
    costo_rl_crudo_hibrido: float | None = None,
    costo_greedy_hibrido: float | None = None,
) -> ResultadoCorridaRLControlada:
    validacion = validar_plan(caso.instancia, plan)
    if not validacion.valido:
        raise RuntimeError(
            f"{algoritmo} produjo un plan inválido para "
            f"{caso.caso_id}: " + " | ".join(validacion.errores)
        )

    matriz = construir_matriz_viaje(
        caso.instancia,
        configuracion,
        proveedor=proveedor_viaje,
    )
    if exigir_sin_fallback and matriz.usa_fallback:
        raise RuntimeError(
            f"{caso.caso_id}/{algoritmo} utilizó "
            f"{matriz.cantidad_fallbacks} fallback(s) viales."
        )

    estimacion = evaluar_plan_estimado(
        caso.instancia,
        plan,
        matriz,
        configuracion,
    )
    diferencia_costo = abs(plan.costo_estimado - estimacion.costo_total)
    if diferencia_costo > tolerancia_costo:
        raise RuntimeError(
            f"Costo inconsistente en {caso.caso_id}/{algoritmo}: "
            f"plan={plan.costo_estimado}, "
            f"auditoria={estimacion.costo_total}, "
            f"diferencia={diferencia_costo}."
        )

    diferencia_greedy = estimacion.costo_total - costo_greedy
    diferencia_ga = estimacion.costo_total - costo_ga

    return ResultadoCorridaRLControlada(
        caso_id=caso.caso_id,
        categoria=caso.categoria,
        instancia_id=caso.instancia.instancia_id,
        algoritmo=algoritmo,
        modelo_alias=modelo_alias,
        modelo_sha256=modelo_sha256,
        estado=estado,
        error="",
        plan_valido=True,
        costo_estimado=estimacion.costo_total,
        distancia_total_km=estimacion.distancia_total_km,
        duracion_operacion_min=estimacion.duracion_operacion_min,
        tardanza_total_min=estimacion.tardanza_total_min,
        exceso_tolerancia_min=estimacion.exceso_tolerancia_min,
        viajes_totales=estimacion.viajes_totales,
        desbalance_fin_min=estimacion.diferencia_fin_camiones_min,
        tiempo_computo_ms=plan.tiempo_computo_ms,
        firma_ruta=firma_plan(plan),
        advertencias=" | ".join(plan.warnings),
        diferencia_vs_greedy=diferencia_greedy,
        diferencia_vs_greedy_pct=_porcentaje(
            diferencia_greedy,
            costo_greedy,
        ),
        diferencia_vs_ga=diferencia_ga,
        diferencia_vs_ga_pct=_porcentaje(diferencia_ga, costo_ga),
        fuente_hibrida=fuente_hibrida,
        motivo_hibrido=motivo_hibrido,
        costo_rl_crudo_hibrido=costo_rl_crudo_hibrido,
        costo_greedy_hibrido=costo_greedy_hibrido,
        fuente_viaje=matriz.fuente.value,
        version_viaje=matriz.version_fuente,
        fallbacks_matriz=matriz.cantidad_fallbacks,
    )


def _resultado_error(
    *,
    caso: CasoBenchmarkClasico,
    algoritmo: str,
    modelo: MetadatosModeloRL,
    exc: Exception,
    fuente_viaje: str,
    version_viaje: str,
) -> ResultadoCorridaRLControlada:
    return ResultadoCorridaRLControlada(
        caso_id=caso.caso_id,
        categoria=caso.categoria,
        instancia_id=caso.instancia.instancia_id,
        algoritmo=algoritmo,
        modelo_alias=modelo.alias,
        modelo_sha256=modelo.sha256,
        estado="ERROR",
        error=f"{type(exc).__name__}: {exc}",
        plan_valido=False,
        costo_estimado=None,
        distancia_total_km=None,
        duracion_operacion_min=None,
        tardanza_total_min=None,
        exceso_tolerancia_min=None,
        viajes_totales=None,
        desbalance_fin_min=None,
        tiempo_computo_ms=None,
        firma_ruta="",
        advertencias="",
        diferencia_vs_greedy=None,
        diferencia_vs_greedy_pct=None,
        diferencia_vs_ga=None,
        diferencia_vs_ga_pct=None,
        fuente_hibrida="",
        motivo_hibrido="",
        costo_rl_crudo_hibrido=None,
        costo_greedy_hibrido=None,
        fuente_viaje=fuente_viaje,
        version_viaje=version_viaje,
        fallbacks_matriz=0,
    )


def ejecutar_benchmark_rl_controlado(
    casos: tuple[CasoBenchmarkClasico, ...],
    *,
    proveedor_viaje: ProveedorViaje,
    planners_rl: Mapping[str, PlanificadorCompatible],
    metadatos_modelos: Mapping[str, MetadatosModeloRL],
    configuracion_planificacion: ConfiguracionPlanificacion | None = None,
    configuracion_benchmark: ConfiguracionBenchmarkRLControlado | None = None,
) -> ResultadoBenchmarkRLControlado:
    if not casos:
        raise ValueError("casos no puede estar vacío.")
    if not planners_rl:
        raise ValueError("planners_rl no puede estar vacío.")
    if set(planners_rl) != set(metadatos_modelos):
        raise ValueError(
            "planners_rl y metadatos_modelos deben tener los mismos alias."
        )

    configuracion = (
        configuracion_planificacion
        if configuracion_planificacion is not None
        else ConfiguracionPlanificacion()
    )
    configuracion_eval = (
        configuracion_benchmark
        if configuracion_benchmark is not None
        else ConfiguracionBenchmarkRLControlado()
    )

    ids = [caso.caso_id for caso in casos]
    if len(set(ids)) != len(ids):
        raise ValueError("Los caso_id no pueden repetirse.")

    matriz_referencia = construir_matriz_viaje(
        casos[0].instancia,
        configuracion,
        proveedor=proveedor_viaje,
    )
    if configuracion_eval.exigir_sin_fallback and matriz_referencia.usa_fallback:
        raise RuntimeError("La matriz de referencia utilizó fallback vial.")

    corridas: list[ResultadoCorridaRLControlada] = []
    resumenes: list[ResumenCasoModeloRL] = []

    for caso in casos:
        greedy = GreedyFeasiblePlanner(
            configuracion=configuracion,
            proveedor_viaje=proveedor_viaje,
        ).generar_plan(caso.instancia)

        ga_planner = GeneticAlgorithmPlanner(
            configuracion=configuracion,
            configuracion_ga=configuracion_eval.configuracion_ga,
            seed=configuracion_eval.seed_ga,
            proveedor_viaje=proveedor_viaje,
        )
        ga = ga_planner.generar_plan(caso.instancia)

        # Primero auditamos referencias con su propio costo como base.
        ref_greedy = _auditar_plan(
            caso=caso,
            algoritmo="GREEDY",
            modelo_alias="REFERENCIA",
            modelo_sha256="",
            plan=greedy,
            proveedor_viaje=proveedor_viaje,
            configuracion=configuracion,
            costo_greedy=greedy.costo_estimado,
            costo_ga=ga.costo_estimado,
            tolerancia_costo=configuracion_eval.tolerancia_costo,
            exigir_sin_fallback=configuracion_eval.exigir_sin_fallback,
        )
        ref_ga = _auditar_plan(
            caso=caso,
            algoritmo="GA",
            modelo_alias="REFERENCIA",
            modelo_sha256="",
            plan=ga,
            proveedor_viaje=proveedor_viaje,
            configuracion=configuracion,
            costo_greedy=ref_greedy.costo_estimado or 0.0,
            costo_ga=ga.costo_estimado,
            tolerancia_costo=configuracion_eval.tolerancia_costo,
            exigir_sin_fallback=configuracion_eval.exigir_sin_fallback,
        )
        corridas.extend((ref_greedy, ref_ga))

        costo_greedy = float(ref_greedy.costo_estimado)
        costo_ga = float(ref_ga.costo_estimado)

        for alias in sorted(planners_rl):
            planner_rl = planners_rl[alias]
            metadatos = metadatos_modelos[alias]

            try:
                plan_rl = planner_rl.generar_plan(caso.instancia)
                corrida_rl = _auditar_plan(
                    caso=caso,
                    algoritmo="RL",
                    modelo_alias=metadatos.alias,
                    modelo_sha256=metadatos.sha256,
                    plan=plan_rl,
                    proveedor_viaje=proveedor_viaje,
                    configuracion=configuracion,
                    costo_greedy=costo_greedy,
                    costo_ga=costo_ga,
                    tolerancia_costo=configuracion_eval.tolerancia_costo,
                    exigir_sin_fallback=configuracion_eval.exigir_sin_fallback,
                )
            except Exception as exc:  # diagnóstico: no abortar todo
                corrida_rl = _resultado_error(
                    caso=caso,
                    algoritmo="RL",
                    modelo=metadatos,
                    exc=exc,
                    fuente_viaje=matriz_referencia.fuente.value,
                    version_viaje=matriz_referencia.version_fuente,
                )
            corridas.append(corrida_rl)

            planner_hibrido = HybridRLGreedyPlanner(
                planner_rl=planner_rl,
                configuracion_planificacion=configuracion,
                proveedor_viaje=proveedor_viaje,
            )

            try:
                plan_hibrido = planner_hibrido.generar_plan(caso.instancia)
                decision = planner_hibrido.ultima_decision
                fuente = decision.fuente_seleccionada.value if decision else ""
                motivo = decision.motivo.value if decision else ""
                costo_rl_crudo = decision.costo_rl if decision else None
                costo_greedy_hibrido = (
                    decision.costo_greedy if decision else None
                )
                estado_hibrido = (
                    "FALLBACK_GREEDY"
                    if motivo in {"RL_INVALIDO", "RL_EXCEPCION"}
                    else "OK"
                )

                corrida_hibrida = _auditar_plan(
                    caso=caso,
                    algoritmo="HIBRIDO",
                    modelo_alias=metadatos.alias,
                    modelo_sha256=metadatos.sha256,
                    plan=plan_hibrido,
                    proveedor_viaje=proveedor_viaje,
                    configuracion=configuracion,
                    costo_greedy=costo_greedy,
                    costo_ga=costo_ga,
                    tolerancia_costo=configuracion_eval.tolerancia_costo,
                    exigir_sin_fallback=configuracion_eval.exigir_sin_fallback,
                    estado=estado_hibrido,
                    fuente_hibrida=fuente,
                    motivo_hibrido=motivo,
                    costo_rl_crudo_hibrido=costo_rl_crudo,
                    costo_greedy_hibrido=costo_greedy_hibrido,
                )

                if (
                    configuracion_eval.verificar_hibrido_no_peor_greedy
                    and corrida_hibrida.costo_estimado is not None
                    and corrida_hibrida.costo_estimado
                    > costo_greedy + configuracion_eval.tolerancia_costo
                ):
                    raise RuntimeError(
                        f"El híbrido superó a Greedy en {caso.caso_id}/"
                        f"{alias}: hibrido={corrida_hibrida.costo_estimado}, "
                        f"greedy={costo_greedy}."
                    )
            except Exception as exc:
                corrida_hibrida = _resultado_error(
                    caso=caso,
                    algoritmo="HIBRIDO",
                    modelo=metadatos,
                    exc=exc,
                    fuente_viaje=matriz_referencia.fuente.value,
                    version_viaje=matriz_referencia.version_fuente,
                )
            corridas.append(corrida_hibrida)

            costo_rl = corrida_rl.costo_estimado
            costo_hibrido = corrida_hibrida.costo_estimado
            hibrido_cumple = (
                costo_hibrido is not None
                and costo_hibrido
                <= costo_greedy + configuracion_eval.tolerancia_costo
            )

            resumenes.append(
                ResumenCasoModeloRL(
                    caso_id=caso.caso_id,
                    categoria=caso.categoria,
                    modelo_alias=alias,
                    costo_greedy=costo_greedy,
                    costo_ga=costo_ga,
                    costo_rl=costo_rl,
                    costo_hibrido=costo_hibrido,
                    rl_estado=corrida_rl.estado,
                    hibrido_estado=corrida_hibrida.estado,
                    fuente_hibrida=corrida_hibrida.fuente_hibrida,
                    mejora_rl_vs_greedy_pct=(
                        -corrida_rl.diferencia_vs_greedy_pct
                        if corrida_rl.diferencia_vs_greedy_pct is not None
                        else None
                    ),
                    mejora_hibrido_vs_greedy_pct=(
                        -corrida_hibrida.diferencia_vs_greedy_pct
                        if corrida_hibrida.diferencia_vs_greedy_pct is not None
                        else None
                    ),
                    mejora_rl_vs_ga_pct=(
                        -corrida_rl.diferencia_vs_ga_pct
                        if corrida_rl.diferencia_vs_ga_pct is not None
                        else None
                    ),
                    mejora_hibrido_vs_ga_pct=(
                        -corrida_hibrida.diferencia_vs_ga_pct
                        if corrida_hibrida.diferencia_vs_ga_pct is not None
                        else None
                    ),
                    hibrido_cumple_garantia=hibrido_cumple,
                )
            )

    modelos_ordenados = tuple(
        metadatos_modelos[alias]
        for alias in sorted(metadatos_modelos)
    )

    return ResultadoBenchmarkRLControlado(
        version_benchmark=VERSION_BENCHMARK_RL_CONTROLADO,
        version_objetivo=VERSION_AUDITORIA_COSTO,
        generado_utc=datetime.now(timezone.utc).isoformat(),
        fuente_viaje=matriz_referencia.fuente.value,
        version_viaje=matriz_referencia.version_fuente,
        casos=tuple(ids),
        modelos=modelos_ordenados,
        configuracion_ga=asdict(configuracion_eval.configuracion_ga),
        seed_ga=configuracion_eval.seed_ga,
        corridas=tuple(corridas),
        resumenes=tuple(resumenes),
    )


def escribir_resultados_benchmark_rl_controlado(
    resultado: ResultadoBenchmarkRLControlado,
    directorio_salida: str | Path,
) -> dict[str, Path]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ruta_corridas = salida / "corridas.csv"
    ruta_resumen = salida / "resumen.csv"
    ruta_json = salida / "benchmark.json"

    _escribir_csv_dataclasses(ruta_corridas, resultado.corridas)
    _escribir_csv_dataclasses(ruta_resumen, resultado.resumenes)

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(asdict(resultado), archivo, ensure_ascii=False, indent=2)

    return {
        "corridas_csv": ruta_corridas,
        "resumen_csv": ruta_resumen,
        "benchmark_json": ruta_json,
    }


def _escribir_csv_dataclasses(
    ruta: Path,
    filas: tuple[Any, ...],
) -> None:
    if not filas:
        raise ValueError("No hay filas para escribir en CSV.")

    datos = [asdict(fila) for fila in filas]
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(datos[0]))
        escritor.writeheader()
        escritor.writerows(datos)
