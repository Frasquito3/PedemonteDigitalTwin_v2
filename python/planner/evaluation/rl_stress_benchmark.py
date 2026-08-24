from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Mapping

from planner.core.config import ConfiguracionPlanificacion
from planner.evaluation.classic_instances import CasoBenchmarkClasico
from planner.evaluation.rl_controlled_benchmark import (
    ConfiguracionBenchmarkRLControlado,
    MetadatosModeloRL,
    PlanificadorCompatible,
    ResultadoBenchmarkRLControlado,
    ejecutar_benchmark_rl_controlado,
)
from planner.routing.travel import ProveedorViaje


VERSION_BENCHMARK_RL_STRESS = "benchmark-rl-stress-v1"


@dataclass(frozen=True)
class ResumenModeloStress:
    modelo_alias: str
    casos: int
    rl_ok: int
    rl_error: int
    rl_gana_greedy: int
    rl_empata_greedy: int
    rl_pierde_greedy: int
    rl_gana_ga: int
    rl_empata_ga: int
    rl_pierde_ga: int
    hibrido_fuente_rl: int
    hibrido_fuente_greedy: int
    hibrido_fallback: int
    violaciones_garantia_hibrida: int
    gap_rl_vs_greedy_promedio_pct: float | None
    gap_rl_vs_greedy_mediana_pct: float | None
    gap_rl_vs_greedy_p90_pct: float | None
    gap_rl_vs_greedy_peor_pct: float | None
    gap_rl_vs_ga_promedio_pct: float | None
    gap_rl_vs_ga_mediana_pct: float | None
    gap_rl_vs_ga_peor_pct: float | None
    gap_hibrido_vs_greedy_promedio_pct: float | None
    gap_hibrido_vs_greedy_peor_pct: float | None


@dataclass(frozen=True)
class ResumenEstratoModeloStress:
    categoria: str
    modelo_alias: str
    casos: int
    rl_gana_greedy: int
    rl_empata_greedy: int
    rl_pierde_greedy: int
    rl_gana_ga: int
    rl_empata_ga: int
    rl_pierde_ga: int
    hibrido_fuente_rl: int
    gap_rl_vs_greedy_promedio_pct: float | None
    gap_rl_vs_greedy_peor_pct: float | None
    gap_rl_vs_ga_promedio_pct: float | None
    gap_rl_vs_ga_peor_pct: float | None


@dataclass(frozen=True)
class ResultadoBenchmarkRLStress:
    version_benchmark: str
    cantidad_casos: int
    cantidad_modelos: int
    cantidad_filas: int
    controlado: ResultadoBenchmarkRLControlado
    resumen_modelos: tuple[ResumenModeloStress, ...]
    resumen_estratos: tuple[ResumenEstratoModeloStress, ...]


def _gap_pct(costo: float, referencia: float) -> float:
    if abs(referencia) <= 1e-12:
        return 0.0 if abs(costo) <= 1e-12 else float("inf")
    return (costo - referencia) / referencia * 100.0


def _percentil_lineal(valores: list[float], proporcion: float) -> float:
    if not valores:
        raise ValueError("No se puede calcular un percentil vacío.")
    if not 0.0 <= proporcion <= 1.0:
        raise ValueError("proporcion debe estar entre 0 y 1.")
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


def _clasificar(
    costo: float,
    referencia: float,
    tolerancia: float,
) -> str:
    diferencia = costo - referencia
    if diferencia < -tolerancia:
        return "GANA"
    if diferencia > tolerancia:
        return "PIERDE"
    return "EMPATA"


def _resumir_grupo(
    filas,
    *,
    modelo_alias: str,
    tolerancia: float,
) -> ResumenModeloStress:
    gaps_greedy: list[float] = []
    gaps_ga: list[float] = []
    gaps_hibrido: list[float] = []
    estados_greedy = {"GANA": 0, "EMPATA": 0, "PIERDE": 0}
    estados_ga = {"GANA": 0, "EMPATA": 0, "PIERDE": 0}
    fuente_rl = 0
    fuente_greedy = 0
    fallback = 0
    violaciones = 0
    rl_ok = 0
    rl_error = 0

    for fila in filas:
        if fila.rl_estado == "OK" and fila.costo_rl is not None:
            rl_ok += 1
            estado_g = _clasificar(
                fila.costo_rl,
                fila.costo_greedy,
                tolerancia,
            )
            estado_ga = _clasificar(
                fila.costo_rl,
                fila.costo_ga,
                tolerancia,
            )
            estados_greedy[estado_g] += 1
            estados_ga[estado_ga] += 1
            gaps_greedy.append(
                _gap_pct(fila.costo_rl, fila.costo_greedy)
            )
            gaps_ga.append(_gap_pct(fila.costo_rl, fila.costo_ga))
        else:
            rl_error += 1

        if fila.fuente_hibrida == "RL":
            fuente_rl += 1
        elif fila.fuente_hibrida == "GREEDY":
            fuente_greedy += 1

        if fila.hibrido_estado == "FALLBACK_GREEDY":
            fallback += 1
        if not fila.hibrido_cumple_garantia:
            violaciones += 1
        if fila.costo_hibrido is not None:
            gaps_hibrido.append(
                _gap_pct(fila.costo_hibrido, fila.costo_greedy)
            )

    return ResumenModeloStress(
        modelo_alias=modelo_alias,
        casos=len(filas),
        rl_ok=rl_ok,
        rl_error=rl_error,
        rl_gana_greedy=estados_greedy["GANA"],
        rl_empata_greedy=estados_greedy["EMPATA"],
        rl_pierde_greedy=estados_greedy["PIERDE"],
        rl_gana_ga=estados_ga["GANA"],
        rl_empata_ga=estados_ga["EMPATA"],
        rl_pierde_ga=estados_ga["PIERDE"],
        hibrido_fuente_rl=fuente_rl,
        hibrido_fuente_greedy=fuente_greedy,
        hibrido_fallback=fallback,
        violaciones_garantia_hibrida=violaciones,
        gap_rl_vs_greedy_promedio_pct=(
            fmean(gaps_greedy) if gaps_greedy else None
        ),
        gap_rl_vs_greedy_mediana_pct=(
            median(gaps_greedy) if gaps_greedy else None
        ),
        gap_rl_vs_greedy_p90_pct=(
            _percentil_lineal(gaps_greedy, 0.90)
            if gaps_greedy
            else None
        ),
        gap_rl_vs_greedy_peor_pct=(
            max(gaps_greedy) if gaps_greedy else None
        ),
        gap_rl_vs_ga_promedio_pct=(
            fmean(gaps_ga) if gaps_ga else None
        ),
        gap_rl_vs_ga_mediana_pct=(
            median(gaps_ga) if gaps_ga else None
        ),
        gap_rl_vs_ga_peor_pct=max(gaps_ga) if gaps_ga else None,
        gap_hibrido_vs_greedy_promedio_pct=(
            fmean(gaps_hibrido) if gaps_hibrido else None
        ),
        gap_hibrido_vs_greedy_peor_pct=(
            max(gaps_hibrido) if gaps_hibrido else None
        ),
    )


def _resumir_estrato(
    filas,
    *,
    categoria: str,
    modelo_alias: str,
    tolerancia: float,
) -> ResumenEstratoModeloStress:
    estados_greedy = {"GANA": 0, "EMPATA": 0, "PIERDE": 0}
    estados_ga = {"GANA": 0, "EMPATA": 0, "PIERDE": 0}
    gaps_greedy: list[float] = []
    gaps_ga: list[float] = []
    fuente_rl = 0

    for fila in filas:
        if fila.costo_rl is None:
            continue
        estado_g = _clasificar(
            fila.costo_rl,
            fila.costo_greedy,
            tolerancia,
        )
        estado_ga = _clasificar(
            fila.costo_rl,
            fila.costo_ga,
            tolerancia,
        )
        estados_greedy[estado_g] += 1
        estados_ga[estado_ga] += 1
        gaps_greedy.append(_gap_pct(fila.costo_rl, fila.costo_greedy))
        gaps_ga.append(_gap_pct(fila.costo_rl, fila.costo_ga))
        if fila.fuente_hibrida == "RL":
            fuente_rl += 1

    return ResumenEstratoModeloStress(
        categoria=categoria,
        modelo_alias=modelo_alias,
        casos=len(filas),
        rl_gana_greedy=estados_greedy["GANA"],
        rl_empata_greedy=estados_greedy["EMPATA"],
        rl_pierde_greedy=estados_greedy["PIERDE"],
        rl_gana_ga=estados_ga["GANA"],
        rl_empata_ga=estados_ga["EMPATA"],
        rl_pierde_ga=estados_ga["PIERDE"],
        hibrido_fuente_rl=fuente_rl,
        gap_rl_vs_greedy_promedio_pct=(
            fmean(gaps_greedy) if gaps_greedy else None
        ),
        gap_rl_vs_greedy_peor_pct=(
            max(gaps_greedy) if gaps_greedy else None
        ),
        gap_rl_vs_ga_promedio_pct=(
            fmean(gaps_ga) if gaps_ga else None
        ),
        gap_rl_vs_ga_peor_pct=max(gaps_ga) if gaps_ga else None,
    )


def ejecutar_benchmark_rl_stress(
    casos: tuple[CasoBenchmarkClasico, ...],
    *,
    proveedor_viaje: ProveedorViaje,
    planners_rl: Mapping[str, PlanificadorCompatible],
    metadatos_modelos: Mapping[str, MetadatosModeloRL],
    configuracion_planificacion: ConfiguracionPlanificacion | None = None,
    configuracion_benchmark: ConfiguracionBenchmarkRLControlado | None = None,
) -> ResultadoBenchmarkRLStress:
    configuracion_eval = (
        configuracion_benchmark
        if configuracion_benchmark is not None
        else ConfiguracionBenchmarkRLControlado()
    )
    controlado = ejecutar_benchmark_rl_controlado(
        casos,
        proveedor_viaje=proveedor_viaje,
        planners_rl=planners_rl,
        metadatos_modelos=metadatos_modelos,
        configuracion_planificacion=configuracion_planificacion,
        configuracion_benchmark=configuracion_eval,
    )

    resumen_modelos = []
    resumen_estratos = []
    for alias in sorted(planners_rl):
        filas_modelo = [
            fila
            for fila in controlado.resumenes
            if fila.modelo_alias == alias
        ]
        resumen_modelos.append(
            _resumir_grupo(
                filas_modelo,
                modelo_alias=alias,
                tolerancia=configuracion_eval.tolerancia_costo,
            )
        )
        categorias = sorted({fila.categoria for fila in filas_modelo})
        for categoria in categorias:
            filas_categoria = [
                fila
                for fila in filas_modelo
                if fila.categoria == categoria
            ]
            resumen_estratos.append(
                _resumir_estrato(
                    filas_categoria,
                    categoria=categoria,
                    modelo_alias=alias,
                    tolerancia=configuracion_eval.tolerancia_costo,
                )
            )

    return ResultadoBenchmarkRLStress(
        version_benchmark=VERSION_BENCHMARK_RL_STRESS,
        cantidad_casos=len(casos),
        cantidad_modelos=len(planners_rl),
        cantidad_filas=len(controlado.corridas),
        controlado=controlado,
        resumen_modelos=tuple(resumen_modelos),
        resumen_estratos=tuple(resumen_estratos),
    )


def _escribir_csv(ruta: Path, filas) -> None:
    datos = [asdict(fila) for fila in filas]
    if not datos:
        raise ValueError(f"No hay filas para escribir en {ruta.name}.")
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(datos[0]))
        escritor.writeheader()
        escritor.writerows(datos)


def escribir_resultados_benchmark_rl_stress(
    resultado: ResultadoBenchmarkRLStress,
    directorio_salida: str | Path,
) -> dict[str, str]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ruta_corridas = salida / "corridas.csv"
    ruta_casos = salida / "resumen_casos.csv"
    ruta_modelos = salida / "resumen_modelos.csv"
    ruta_estratos = salida / "resumen_estratos.csv"
    ruta_json = salida / "benchmark.json"

    _escribir_csv(ruta_corridas, resultado.controlado.corridas)
    _escribir_csv(ruta_casos, resultado.controlado.resumenes)
    _escribir_csv(ruta_modelos, resultado.resumen_modelos)
    _escribir_csv(ruta_estratos, resultado.resumen_estratos)

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(
            asdict(resultado),
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "corridas_csv": str(ruta_corridas),
        "resumen_casos_csv": str(ruta_casos),
        "resumen_modelos_csv": str(ruta_modelos),
        "resumen_estratos_csv": str(ruta_estratos),
        "benchmark_json": str(ruta_json),
    }
