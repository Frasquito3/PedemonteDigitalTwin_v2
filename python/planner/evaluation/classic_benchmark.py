from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from planner.algorithms.ga import (
    ConfiguracionGA,
    GeneticAlgorithmPlanner,
)
from planner.algorithms.greedy import GreedyFeasiblePlanner
from planner.algorithms.random_feasible import RandomFeasiblePlanner
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import PlanTurno
from planner.domain.validator import validar_plan
from planner.evaluation.classic_instances import CasoBenchmarkClasico
from planner.routing.objective import (
    VERSION_AUDITORIA_COSTO,
    evaluar_plan_estimado,
)
from planner.routing.travel import (
    ProveedorViaje,
    construir_matriz_viaje,
)


VERSION_BENCHMARK_CLASICO = "benchmark-clasico-v1"


@dataclass(frozen=True)
class ConfiguracionBenchmarkClasico:
    seeds_estocasticas: tuple[int, ...] = (
        101,
        211,
        307,
        401,
        503,
    )
    configuracion_ga: ConfiguracionGA = field(
        default_factory=ConfiguracionGA
    )
    tolerancia_costo: float = 1e-6
    exigir_sin_fallback: bool = True
    verificar_ga_no_peor_greedy: bool = True

    def __post_init__(self) -> None:
        if not self.seeds_estocasticas:
            raise ValueError(
                "seeds_estocasticas no puede estar vacía."
            )

        if len(set(self.seeds_estocasticas)) != len(
            self.seeds_estocasticas
        ):
            raise ValueError(
                "seeds_estocasticas no puede contener duplicados."
            )

        if self.tolerancia_costo < 0.0:
            raise ValueError(
                "tolerancia_costo no puede ser negativa."
            )


@dataclass(frozen=True)
class ResultadoCorridaClasica:
    caso_id: str
    categoria: str
    instancia_id: str
    algoritmo: str
    repeticion: int
    seed_algoritmo: int | None
    plan_valido: bool
    costo_estimado: float
    distancia_total_km: float
    duracion_operacion_min: float
    tardanza_total_min: float
    exceso_tolerancia_min: float
    viajes_totales: int
    desbalance_fin_min: float
    tiempo_computo_ms: float
    firma_ruta: str
    advertencias: str
    generaciones_ga: int | None
    fuente_viaje: str
    version_viaje: str
    fallbacks_matriz: int


@dataclass(frozen=True)
class ResumenAlgoritmoCaso:
    caso_id: str
    categoria: str
    algoritmo: str
    corridas: int
    planes_validos: int
    firmas_distintas: int
    costo_promedio: float
    costo_desviacion: float
    costo_minimo: float
    costo_maximo: float
    distancia_promedio_km: float
    duracion_promedio_min: float
    tardanza_promedio_min: float
    viajes_promedio: float
    desbalance_promedio_min: float
    tiempo_computo_promedio_ms: float
    mejor_seed: int | None
    mejor_firma_ruta: str
    diferencia_promedio_vs_greedy: float
    diferencia_promedio_vs_greedy_pct: float
    diferencia_mejor_vs_greedy: float
    diferencia_mejor_vs_greedy_pct: float
    cumple_ga_no_peor_greedy: bool | None


@dataclass(frozen=True)
class ResultadoBenchmarkClasico:
    version_benchmark: str
    version_objetivo: str
    generado_utc: str
    fuente_viaje: str
    version_viaje: str
    casos: tuple[str, ...]
    seeds_estocasticas: tuple[int, ...]
    configuracion_ga: dict[str, Any]
    corridas: tuple[ResultadoCorridaClasica, ...]
    resumenes: tuple[ResumenAlgoritmoCaso, ...]


def firma_plan(plan: PlanTurno) -> str:
    partes_camion: list[str] = []

    for camion in sorted(
        plan.camiones,
        key=lambda actual: actual.camion_id,
    ):
        partes_viaje: list[str] = []

        for viaje in sorted(
            camion.viajes,
            key=lambda actual: actual.numero_viaje,
        ):
            pedidos = ">".join(viaje.pedido_ids)
            partes_viaje.append(
                f"v{viaje.numero_viaje}[{pedidos}]"
            )

        contenido = (
            "/".join(partes_viaje)
            if partes_viaje
            else "SIN_VIAJES"
        )
        partes_camion.append(
            f"c{camion.camion_id}:{contenido}"
        )

    return "||".join(partes_camion)


def _evaluar_corrida(
    *,
    caso: CasoBenchmarkClasico,
    algoritmo: str,
    repeticion: int,
    seed_algoritmo: int | None,
    plan: PlanTurno,
    proveedor_viaje: ProveedorViaje,
    configuracion: ConfiguracionPlanificacion,
    generaciones_ga: int | None,
    tolerancia_costo: float,
    exigir_sin_fallback: bool,
) -> ResultadoCorridaClasica:
    validacion = validar_plan(
        caso.instancia,
        plan,
    )

    if not validacion.valido:
        raise RuntimeError(
            f"{algoritmo} produjo un plan inválido para "
            f"{caso.caso_id}: "
            + " | ".join(validacion.errores)
        )

    matriz = construir_matriz_viaje(
        caso.instancia,
        configuracion,
        proveedor=proveedor_viaje,
    )

    if exigir_sin_fallback and matriz.usa_fallback:
        raise RuntimeError(
            f"{caso.caso_id} utilizó "
            f"{matriz.cantidad_fallbacks} fallback(s) viales."
        )

    estimacion = evaluar_plan_estimado(
        caso.instancia,
        plan,
        matriz,
        configuracion,
    )

    diferencia_costo = abs(
        plan.costo_estimado
        - estimacion.costo_total
    )

    if diferencia_costo > tolerancia_costo:
        raise RuntimeError(
            f"Costo inconsistente en {caso.caso_id}/{algoritmo}: "
            f"plan={plan.costo_estimado}, "
            f"auditoria={estimacion.costo_total}, "
            f"diferencia={diferencia_costo}."
        )

    return ResultadoCorridaClasica(
        caso_id=caso.caso_id,
        categoria=caso.categoria,
        instancia_id=caso.instancia.instancia_id,
        algoritmo=algoritmo,
        repeticion=repeticion,
        seed_algoritmo=seed_algoritmo,
        plan_valido=True,
        costo_estimado=estimacion.costo_total,
        distancia_total_km=estimacion.distancia_total_km,
        duracion_operacion_min=(
            estimacion.duracion_operacion_min
        ),
        tardanza_total_min=estimacion.tardanza_total_min,
        exceso_tolerancia_min=(
            estimacion.exceso_tolerancia_min
        ),
        viajes_totales=estimacion.viajes_totales,
        desbalance_fin_min=(
            estimacion.diferencia_fin_camiones_min
        ),
        tiempo_computo_ms=plan.tiempo_computo_ms,
        firma_ruta=firma_plan(plan),
        advertencias=" | ".join(plan.warnings),
        generaciones_ga=generaciones_ga,
        fuente_viaje=matriz.fuente.value,
        version_viaje=matriz.version_fuente,
        fallbacks_matriz=matriz.cantidad_fallbacks,
    )


def _porcentaje(
    diferencia: float,
    referencia: float,
) -> float:
    if abs(referencia) <= 1e-12:
        return 0.0 if abs(diferencia) <= 1e-12 else float("inf")

    return diferencia / referencia * 100.0


def _crear_resumenes(
    corridas: Iterable[ResultadoCorridaClasica],
    *,
    tolerancia_costo: float,
) -> tuple[ResumenAlgoritmoCaso, ...]:
    corridas_lista = list(corridas)

    greedy_por_caso = {
        corrida.caso_id: corrida.costo_estimado
        for corrida in corridas_lista
        if corrida.algoritmo == "GREEDY"
    }

    grupos: dict[
        tuple[str, str, str],
        list[ResultadoCorridaClasica],
    ] = {}

    for corrida in corridas_lista:
        clave = (
            corrida.caso_id,
            corrida.categoria,
            corrida.algoritmo,
        )
        grupos.setdefault(clave, []).append(corrida)

    resumenes: list[ResumenAlgoritmoCaso] = []

    for clave in sorted(grupos):
        caso_id, categoria, algoritmo = clave
        grupo = grupos[clave]
        costos = [corrida.costo_estimado for corrida in grupo]
        costo_promedio = fmean(costos)
        costo_minimo = min(costos)
        costo_maximo = max(costos)
        costo_greedy = greedy_por_caso[caso_id]
        mejor = min(
            grupo,
            key=lambda corrida: (
                corrida.costo_estimado,
                corrida.tiempo_computo_ms,
                corrida.seed_algoritmo
                if corrida.seed_algoritmo is not None
                else -1,
            ),
        )

        diferencia_promedio = costo_promedio - costo_greedy
        diferencia_mejor = costo_minimo - costo_greedy

        cumple_ga: bool | None = None
        if algoritmo == "GA":
            cumple_ga = all(
                corrida.costo_estimado
                <= costo_greedy + tolerancia_costo
                for corrida in grupo
            )

        resumenes.append(
            ResumenAlgoritmoCaso(
                caso_id=caso_id,
                categoria=categoria,
                algoritmo=algoritmo,
                corridas=len(grupo),
                planes_validos=sum(
                    1 for corrida in grupo if corrida.plan_valido
                ),
                firmas_distintas=len(
                    {corrida.firma_ruta for corrida in grupo}
                ),
                costo_promedio=costo_promedio,
                costo_desviacion=(
                    pstdev(costos) if len(costos) > 1 else 0.0
                ),
                costo_minimo=costo_minimo,
                costo_maximo=costo_maximo,
                distancia_promedio_km=fmean(
                    corrida.distancia_total_km
                    for corrida in grupo
                ),
                duracion_promedio_min=fmean(
                    corrida.duracion_operacion_min
                    for corrida in grupo
                ),
                tardanza_promedio_min=fmean(
                    corrida.tardanza_total_min
                    for corrida in grupo
                ),
                viajes_promedio=fmean(
                    corrida.viajes_totales
                    for corrida in grupo
                ),
                desbalance_promedio_min=fmean(
                    corrida.desbalance_fin_min
                    for corrida in grupo
                ),
                tiempo_computo_promedio_ms=fmean(
                    corrida.tiempo_computo_ms
                    for corrida in grupo
                ),
                mejor_seed=mejor.seed_algoritmo,
                mejor_firma_ruta=mejor.firma_ruta,
                diferencia_promedio_vs_greedy=(
                    diferencia_promedio
                ),
                diferencia_promedio_vs_greedy_pct=_porcentaje(
                    diferencia_promedio,
                    costo_greedy,
                ),
                diferencia_mejor_vs_greedy=diferencia_mejor,
                diferencia_mejor_vs_greedy_pct=_porcentaje(
                    diferencia_mejor,
                    costo_greedy,
                ),
                cumple_ga_no_peor_greedy=cumple_ga,
            )
        )

    return tuple(resumenes)


def ejecutar_benchmark_clasico(
    casos: Iterable[CasoBenchmarkClasico],
    *,
    proveedor_viaje: ProveedorViaje,
    configuracion_planificacion:
        ConfiguracionPlanificacion | None = None,
    configuracion_benchmark:
        ConfiguracionBenchmarkClasico | None = None,
) -> ResultadoBenchmarkClasico:
    configuracion = (
        configuracion_planificacion
        if configuracion_planificacion is not None
        else ConfiguracionPlanificacion()
    )
    benchmark = (
        configuracion_benchmark
        if configuracion_benchmark is not None
        else ConfiguracionBenchmarkClasico()
    )
    casos_lista = list(casos)

    if not casos_lista:
        raise ValueError(
            "El benchmark requiere al menos un caso."
        )

    ids = [caso.caso_id for caso in casos_lista]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Los caso_id del benchmark deben ser únicos."
        )

    corridas: list[ResultadoCorridaClasica] = []

    for caso in casos_lista:
        # Preflight estricto: detecta rutas faltantes antes de ejecutar
        # cualquiera de los algoritmos sobre el caso.
        matriz_preflight = construir_matriz_viaje(
            caso.instancia,
            configuracion,
            proveedor=proveedor_viaje,
        )

        if (
            benchmark.exigir_sin_fallback
            and matriz_preflight.usa_fallback
        ):
            raise RuntimeError(
                f"El preflight de {caso.caso_id} utilizó "
                f"{matriz_preflight.cantidad_fallbacks} fallback(s)."
            )

        greedy = GreedyFeasiblePlanner(
            configuracion=configuracion,
            proveedor_viaje=proveedor_viaje,
        )
        plan_greedy = greedy.generar_plan(caso.instancia)
        corrida_greedy = _evaluar_corrida(
            caso=caso,
            algoritmo="GREEDY",
            repeticion=1,
            seed_algoritmo=None,
            plan=plan_greedy,
            proveedor_viaje=proveedor_viaje,
            configuracion=configuracion,
            generaciones_ga=None,
            tolerancia_costo=benchmark.tolerancia_costo,
            exigir_sin_fallback=benchmark.exigir_sin_fallback,
        )
        corridas.append(corrida_greedy)

        for repeticion, seed in enumerate(
            benchmark.seeds_estocasticas,
            start=1,
        ):
            random_planner = RandomFeasiblePlanner(
                configuracion=configuracion,
                seed=seed,
                proveedor_viaje=proveedor_viaje,
            )
            plan_random = random_planner.generar_plan(
                caso.instancia
            )
            corridas.append(
                _evaluar_corrida(
                    caso=caso,
                    algoritmo="RANDOM",
                    repeticion=repeticion,
                    seed_algoritmo=seed,
                    plan=plan_random,
                    proveedor_viaje=proveedor_viaje,
                    configuracion=configuracion,
                    generaciones_ga=None,
                    tolerancia_costo=benchmark.tolerancia_costo,
                    exigir_sin_fallback=(
                        benchmark.exigir_sin_fallback
                    ),
                )
            )

            ga_planner = GeneticAlgorithmPlanner(
                configuracion=configuracion,
                configuracion_ga=benchmark.configuracion_ga,
                seed=seed,
                proveedor_viaje=proveedor_viaje,
            )
            plan_ga = ga_planner.generar_plan(caso.instancia)
            corrida_ga = _evaluar_corrida(
                caso=caso,
                algoritmo="GA",
                repeticion=repeticion,
                seed_algoritmo=seed,
                plan=plan_ga,
                proveedor_viaje=proveedor_viaje,
                configuracion=configuracion,
                generaciones_ga=ga_planner.generaciones_ejecutadas,
                tolerancia_costo=benchmark.tolerancia_costo,
                exigir_sin_fallback=benchmark.exigir_sin_fallback,
            )

            if (
                benchmark.verificar_ga_no_peor_greedy
                and corrida_ga.costo_estimado
                > corrida_greedy.costo_estimado
                + benchmark.tolerancia_costo
            ):
                raise RuntimeError(
                    "La garantía GA <= Greedy no se cumplió en "
                    f"{caso.caso_id}, seed={seed}: "
                    f"GA={corrida_ga.costo_estimado}, "
                    f"Greedy={corrida_greedy.costo_estimado}."
                )

            corridas.append(corrida_ga)

    resumenes = _crear_resumenes(
        corridas,
        tolerancia_costo=benchmark.tolerancia_costo,
    )

    primera_matriz = construir_matriz_viaje(
        casos_lista[0].instancia,
        configuracion,
        proveedor=proveedor_viaje,
    )

    return ResultadoBenchmarkClasico(
        version_benchmark=VERSION_BENCHMARK_CLASICO,
        version_objetivo=VERSION_AUDITORIA_COSTO,
        generado_utc=datetime.now(timezone.utc).isoformat(),
        fuente_viaje=primera_matriz.fuente.value,
        version_viaje=primera_matriz.version_fuente,
        casos=tuple(ids),
        seeds_estocasticas=benchmark.seeds_estocasticas,
        configuracion_ga=asdict(benchmark.configuracion_ga),
        corridas=tuple(corridas),
        resumenes=resumenes,
    )


def _normalizar_valor_csv(valor: Any) -> Any:
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if valor is None:
        return ""
    return valor


def _escribir_csv_dataclasses(
    ruta: Path,
    filas: Iterable[Any],
) -> None:
    filas_dict = [asdict(fila) for fila in filas]

    if not filas_dict:
        raise ValueError(
            f"No hay filas para escribir en {ruta}."
        )

    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(filas_dict[0].keys()),
        )
        escritor.writeheader()

        for fila in filas_dict:
            escritor.writerow(
                {
                    clave: _normalizar_valor_csv(valor)
                    for clave, valor in fila.items()
                }
            )


def escribir_resultados_benchmark(
    resultado: ResultadoBenchmarkClasico,
    directorio_salida: str | Path,
) -> dict[str, Path]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ruta_corridas = salida / "corridas.csv"
    ruta_resumen = salida / "resumen.csv"
    ruta_json = salida / "benchmark.json"

    _escribir_csv_dataclasses(
        ruta_corridas,
        resultado.corridas,
    )
    _escribir_csv_dataclasses(
        ruta_resumen,
        resultado.resumenes,
    )

    with ruta_json.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            asdict(resultado),
            archivo,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        archivo.write("\n")

    return {
        "corridas_csv": ruta_corridas,
        "resumen_csv": ruta_resumen,
        "benchmark_json": ruta_json,
    }
