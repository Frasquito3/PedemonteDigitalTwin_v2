from __future__ import annotations

import csv
import json

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from planner.evaluation.comparison_execution import (
    ORDEN_MODOS_ESPERADO,
    ConfiguracionEjecucionComparacion,
    EjecutorVectoresAnyLogic,
    RegistroEjecucionComparacion,
    ResultadoEjecucionComparacion,
    ejecutar_contrato_comparacion,
)


VERSION_SUITE_COMPARACION = "comparacion-anylogic-suite-v1"
NIVEL_EVIDENCIA_SUITE = "una-seed-ejecucion-por-caso-v1"


@dataclass(frozen=True)
class CasoContratoComparacion:
    caso_id: str
    categoria: str
    descripcion: str
    contrato: Mapping[str, Any]


@dataclass(frozen=True)
class ConfiguracionSuiteComparacion:
    continuar_ante_error: bool = True
    exigir_seis_casos: bool = True
    exigir_orden_rl_primero: bool = True
    exigir_cinco_planes_ok: bool = True
    tolerancia_empate: float = 1e-9

    def __post_init__(self) -> None:
        if self.tolerancia_empate < 0.0:
            raise ValueError(
                "tolerancia_empate no puede ser negativa."
            )


@dataclass(frozen=True)
class RegistroSuiteComparacion:
    caso_id: str
    categoria: str
    descripcion: str
    instancia_id: str

    orden_modo: int
    modo_solicitado: str
    algoritmo_resultante: str
    fuente_seleccionada: str
    firma_ruta: str

    seed_escenario: int
    seed_planificacion: int | None
    seed_ejecucion: int

    estado_ejecucion: str
    error_ejecucion: str

    costo_estimado: float | None
    costo_real: float | None
    diferencia_costo_real_estimado: float | None
    error_relativo_estimacion_pct: float | None

    tiempo_plan_ms: float | None
    tiempo_selector_ms: float | None
    tiempo_simulado_min: float | None

    tareas_entregadas: int | None
    tareas_no_entregadas: int | None
    viajes_totales: int | None

    ranking_caso: int | None
    diferencia_costo_vs_rl: float | None
    mejora_vs_rl_pct: float | None
    comparacion_vs_rl: str
    diferencia_costo_vs_greedy: float | None
    mejora_vs_greedy_pct: float | None
    comparacion_vs_greedy: str

    estado_final_motor: str
    stop_condition: bool | None
    mensaje_anylogic: str


@dataclass(frozen=True)
class ResultadoCasoSuiteComparacion:
    caso_id: str
    categoria: str
    descripcion: str
    instancia_id: str
    seed_escenario: int
    seed_ejecucion: int
    cantidad_pedidos: int

    contratos_ok: int
    contratos_error: int
    ejecuciones_ok: int
    ejecuciones_error: int
    error_caso: str

    mejor_costo_real: float | None
    modos_mejor_costo: tuple[str, ...]

    costo_real_rl: float | None
    ranking_rl: int | None
    brecha_rl_vs_mejor_pct: float | None

    costo_real_ga: float | None
    costo_real_greedy: float | None
    costo_real_random: float | None
    costo_real_hibrido: float | None


@dataclass(frozen=True)
class ResumenAlgoritmoSuiteComparacion:
    modo_solicitado: str
    casos_totales: int
    casos_ok: int
    casos_error: int

    primeros_puestos: int
    ranking_promedio: float | None

    victorias_vs_rl: int
    empates_vs_rl: int
    derrotas_vs_rl: int
    mejora_media_vs_rl_pct: float | None
    mejora_mediana_vs_rl_pct: float | None

    victorias_vs_greedy: int
    empates_vs_greedy: int
    derrotas_vs_greedy: int
    mejora_media_vs_greedy_pct: float | None
    mejora_mediana_vs_greedy_pct: float | None

    error_estimacion_abs_medio_pct: float | None
    tiempo_plan_promedio_ms: float | None
    tiempo_selector_promedio_ms: float | None
    tiempo_simulado_promedio_min: float | None

    fuentes_seleccionadas: str


@dataclass(frozen=True)
class ResultadoSuiteComparacion:
    version_suite: str
    generado_utc: str
    nivel_evidencia: str

    cantidad_casos: int
    orden_modos: tuple[str, ...]
    corridas_esperadas: int
    corridas_ok: int
    corridas_error: int

    common_random_numbers_por_caso: bool
    proceso_nuevo_por_plan: bool

    fuente_viaje: str
    version_viaje: str
    version_objetivo: str
    modelo: str
    java: str

    casos: tuple[ResultadoCasoSuiteComparacion, ...]
    corridas: tuple[RegistroSuiteComparacion, ...]
    resumen_algoritmos: tuple[ResumenAlgoritmoSuiteComparacion, ...]


def ejecutar_suite_comparacion(
    casos: Sequence[CasoContratoComparacion],
    *,
    ejecutor: EjecutorVectoresAnyLogic,
    configuracion: ConfiguracionSuiteComparacion | None = None,
) -> ResultadoSuiteComparacion:
    config = configuracion or ConfiguracionSuiteComparacion()
    casos_normalizados = _validar_casos(casos, config=config)

    resultados_brutos: list[
        tuple[
            CasoContratoComparacion,
            ResultadoEjecucionComparacion | None,
            str,
        ]
    ] = []

    modelo_global = ""
    java_global = ""

    for caso in casos_normalizados:
        try:
            resultado = ejecutar_contrato_comparacion(
                caso.contrato,
                ejecutor=_EjecutorConPrefijo(
                    ejecutor,
                    prefijo=caso.caso_id,
                ),
                configuracion=ConfiguracionEjecucionComparacion(
                    continuar_ante_error=config.continuar_ante_error,
                    exigir_cinco_planes_ok=(
                        config.exigir_cinco_planes_ok
                    ),
                    exigir_orden_rl_primero=(
                        config.exigir_orden_rl_primero
                    ),
                ),
            )
            error_caso = ""
            modelo_global = resultado.modelo or modelo_global
            java_global = resultado.java or java_global
        except Exception as exc:
            if not config.continuar_ante_error:
                raise
            resultado = None
            error_caso = f"{type(exc).__name__}: {exc}"

        resultados_brutos.append((caso, resultado, error_caso))

    corridas = _construir_corridas(
        resultados_brutos,
        tolerancia_empate=config.tolerancia_empate,
    )
    resultados_casos = _construir_resultados_casos(
        resultados_brutos,
        corridas,
        tolerancia_empate=config.tolerancia_empate,
    )
    resumen_algoritmos = _construir_resumen_algoritmos(corridas)

    fuente_viaje = _valor_comun_contrato(
        casos_normalizados,
        "fuente_viaje",
    )
    version_viaje = _valor_comun_contrato(
        casos_normalizados,
        "version_viaje",
    )
    version_objetivo = _valor_comun_contrato(
        casos_normalizados,
        "version_objetivo",
    )

    corridas_ok = sum(
        1
        for corrida in corridas
        if corrida.estado_ejecucion == "OK"
    )
    corridas_esperadas = (
        len(casos_normalizados) * len(ORDEN_MODOS_ESPERADO)
    )

    return ResultadoSuiteComparacion(
        version_suite=VERSION_SUITE_COMPARACION,
        generado_utc=datetime.now(timezone.utc).isoformat(),
        nivel_evidencia=NIVEL_EVIDENCIA_SUITE,
        cantidad_casos=len(casos_normalizados),
        orden_modos=ORDEN_MODOS_ESPERADO,
        corridas_esperadas=corridas_esperadas,
        corridas_ok=corridas_ok,
        corridas_error=corridas_esperadas - corridas_ok,
        common_random_numbers_por_caso=True,
        proceso_nuevo_por_plan=True,
        fuente_viaje=fuente_viaje,
        version_viaje=version_viaje,
        version_objetivo=version_objetivo,
        modelo=modelo_global,
        java=java_global,
        casos=tuple(resultados_casos),
        corridas=tuple(corridas),
        resumen_algoritmos=tuple(resumen_algoritmos),
    )


def escribir_resultado_suite_comparacion(
    resultado: ResultadoSuiteComparacion,
    directorio_salida: str | Path,
) -> dict[str, Path]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ruta_json = salida / "comparison_suite.json"
    ruta_corridas = salida / "comparison_runs.csv"
    ruta_casos = salida / "comparison_case_summary.csv"
    ruta_algoritmos = salida / "comparison_algorithm_summary.csv"

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(
            asdict(resultado),
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    _escribir_csv_dataclasses(ruta_corridas, resultado.corridas)
    _escribir_csv_dataclasses(ruta_casos, resultado.casos)
    _escribir_csv_dataclasses(
        ruta_algoritmos,
        resultado.resumen_algoritmos,
    )

    return {
        "suite_json": ruta_json,
        "corridas_csv": ruta_corridas,
        "casos_csv": ruta_casos,
        "algoritmos_csv": ruta_algoritmos,
    }


class _EjecutorConPrefijo:
    def __init__(
        self,
        ejecutor: EjecutorVectoresAnyLogic,
        *,
        prefijo: str,
    ) -> None:
        self._ejecutor = ejecutor
        self._prefijo = _normalizar_identificador(prefijo)

    def ejecutar_vectores(self, **kwargs: Any) -> Any:
        identificador = str(kwargs["identificador_corrida"])
        kwargs["identificador_corrida"] = (
            f"{self._prefijo}_{identificador}"
        )
        return self._ejecutor.ejecutar_vectores(**kwargs)


def _validar_casos(
    casos: Sequence[CasoContratoComparacion],
    *,
    config: ConfiguracionSuiteComparacion,
) -> tuple[CasoContratoComparacion, ...]:
    if not casos:
        raise ValueError("La suite requiere al menos un caso.")

    normalizados: list[CasoContratoComparacion] = []
    ids: list[str] = []

    for caso in casos:
        caso_id = str(caso.caso_id).strip().upper()
        if not caso_id:
            raise ValueError("caso_id no puede estar vacío.")
        if not isinstance(caso.contrato, Mapping):
            raise ValueError(
                f"{caso_id}.contrato debe ser un mapping."
            )

        normalizados.append(
            CasoContratoComparacion(
                caso_id=caso_id,
                categoria=str(caso.categoria).strip().upper(),
                descripcion=str(caso.descripcion).strip(),
                contrato=caso.contrato,
            )
        )
        ids.append(caso_id)

    duplicados = sorted(
        caso_id
        for caso_id, cantidad in Counter(ids).items()
        if cantidad > 1
    )
    if duplicados:
        raise ValueError(
            "Los casos no pueden repetirse: "
            + ", ".join(duplicados)
        )

    if config.exigir_seis_casos and len(normalizados) != 6:
        raise ValueError(
            "La suite formal 16C exige 6 casos; "
            f"se recibieron {len(normalizados)}."
        )

    return tuple(normalizados)


def _construir_corridas(
    resultados_brutos: Sequence[
        tuple[
            CasoContratoComparacion,
            ResultadoEjecucionComparacion | None,
            str,
        ]
    ],
    *,
    tolerancia_empate: float,
) -> list[RegistroSuiteComparacion]:
    corridas: list[RegistroSuiteComparacion] = []

    for caso, resultado, error_caso in resultados_brutos:
        if resultado is None:
            corridas.extend(
                _corridas_error_caso(caso, error_caso)
            )
            continue

        exitosas = [
            registro
            for registro in resultado.registros
            if (
                registro.estado_ejecucion == "OK"
                and registro.costo_real is not None
            )
        ]
        rankings = _calcular_rankings(
            exitosas,
            tolerancia_empate=tolerancia_empate,
        )
        rl = _costo_modo(exitosas, "RL")
        greedy = _costo_modo(exitosas, "GREEDY")

        for registro in resultado.registros:
            costo_real = registro.costo_real
            diferencia_rl, mejora_rl, comparacion_rl = _comparar_costos(
                costo_actual=costo_real,
                costo_referencia=rl,
                tolerancia=tolerancia_empate,
            )
            diferencia_greedy, mejora_greedy, comparacion_greedy = (
                _comparar_costos(
                    costo_actual=costo_real,
                    costo_referencia=greedy,
                    tolerancia=tolerancia_empate,
                )
            )

            corridas.append(
                RegistroSuiteComparacion(
                    caso_id=caso.caso_id,
                    categoria=caso.categoria,
                    descripcion=caso.descripcion,
                    instancia_id=resultado.instancia_id,
                    orden_modo=registro.orden,
                    modo_solicitado=registro.modo_solicitado,
                    algoritmo_resultante=(
                        registro.algoritmo_resultante
                    ),
                    fuente_seleccionada=(
                        registro.fuente_seleccionada
                    ),
                    firma_ruta=registro.firma_ruta,
                    seed_escenario=registro.seed_escenario,
                    seed_planificacion=(
                        registro.seed_planificacion
                    ),
                    seed_ejecucion=registro.seed_ejecucion,
                    estado_ejecucion=registro.estado_ejecucion,
                    error_ejecucion=registro.error_ejecucion,
                    costo_estimado=registro.costo_estimado,
                    costo_real=costo_real,
                    diferencia_costo_real_estimado=(
                        registro.diferencia_costo_real_estimado
                    ),
                    error_relativo_estimacion_pct=(
                        registro.error_relativo_estimacion_pct
                    ),
                    tiempo_plan_ms=registro.tiempo_plan_ms,
                    tiempo_selector_ms=registro.tiempo_selector_ms,
                    tiempo_simulado_min=registro.tiempo_simulado_min,
                    tareas_entregadas=registro.tareas_entregadas,
                    tareas_no_entregadas=(
                        registro.tareas_no_entregadas
                    ),
                    viajes_totales=registro.viajes_totales,
                    ranking_caso=rankings.get(
                        registro.modo_solicitado
                    ),
                    diferencia_costo_vs_rl=diferencia_rl,
                    mejora_vs_rl_pct=mejora_rl,
                    comparacion_vs_rl=comparacion_rl,
                    diferencia_costo_vs_greedy=(
                        diferencia_greedy
                    ),
                    mejora_vs_greedy_pct=mejora_greedy,
                    comparacion_vs_greedy=comparacion_greedy,
                    estado_final_motor=registro.estado_final_motor,
                    stop_condition=registro.stop_condition,
                    mensaje_anylogic=registro.mensaje_anylogic,
                )
            )

    return corridas


def _corridas_error_caso(
    caso: CasoContratoComparacion,
    error_caso: str,
) -> list[RegistroSuiteComparacion]:
    contrato = caso.contrato
    seed_escenario = _entero_seguro(
        contrato.get("seed_escenario"),
        0,
    )
    seed_ejecucion = _entero_seguro(
        contrato.get("seed_ejecucion"),
        0,
    )
    instancia_id = str(
        contrato.get("instancia_id", caso.caso_id)
    ).strip()

    return [
        RegistroSuiteComparacion(
            caso_id=caso.caso_id,
            categoria=caso.categoria,
            descripcion=caso.descripcion,
            instancia_id=instancia_id,
            orden_modo=orden,
            modo_solicitado=modo,
            algoritmo_resultante="",
            fuente_seleccionada="",
            firma_ruta="",
            seed_escenario=seed_escenario,
            seed_planificacion=None,
            seed_ejecucion=seed_ejecucion,
            estado_ejecucion="ERROR",
            error_ejecucion=error_caso,
            costo_estimado=None,
            costo_real=None,
            diferencia_costo_real_estimado=None,
            error_relativo_estimacion_pct=None,
            tiempo_plan_ms=None,
            tiempo_selector_ms=None,
            tiempo_simulado_min=None,
            tareas_entregadas=None,
            tareas_no_entregadas=None,
            viajes_totales=None,
            ranking_caso=None,
            diferencia_costo_vs_rl=None,
            mejora_vs_rl_pct=None,
            comparacion_vs_rl="NO_DISPONIBLE",
            diferencia_costo_vs_greedy=None,
            mejora_vs_greedy_pct=None,
            comparacion_vs_greedy="NO_DISPONIBLE",
            estado_final_motor="",
            stop_condition=None,
            mensaje_anylogic="",
        )
        for orden, modo in enumerate(ORDEN_MODOS_ESPERADO, start=1)
    ]


def _construir_resultados_casos(
    resultados_brutos: Sequence[
        tuple[
            CasoContratoComparacion,
            ResultadoEjecucionComparacion | None,
            str,
        ]
    ],
    corridas: Sequence[RegistroSuiteComparacion],
    *,
    tolerancia_empate: float,
) -> list[ResultadoCasoSuiteComparacion]:
    resultados: list[ResultadoCasoSuiteComparacion] = []

    for caso, ejecucion, error_caso in resultados_brutos:
        contrato = caso.contrato
        corridas_caso = [
            corrida
            for corrida in corridas
            if corrida.caso_id == caso.caso_id
        ]
        exitosas = [
            corrida
            for corrida in corridas_caso
            if (
                corrida.estado_ejecucion == "OK"
                and corrida.costo_real is not None
            )
        ]

        mejor = (
            min(corrida.costo_real for corrida in exitosas)
            if exitosas
            else None
        )
        modos_mejor = tuple(
            corrida.modo_solicitado
            for corrida in exitosas
            if (
                mejor is not None
                and abs(corrida.costo_real - mejor)
                <= tolerancia_empate
            )
        )

        rl = _corrida_modo(corridas_caso, "RL")
        brecha_rl = None
        if (
            rl is not None
            and rl.costo_real is not None
            and mejor is not None
            and mejor > 0.0
        ):
            brecha_rl = (rl.costo_real - mejor) / mejor * 100.0

        resultados.append(
            ResultadoCasoSuiteComparacion(
                caso_id=caso.caso_id,
                categoria=caso.categoria,
                descripcion=caso.descripcion,
                instancia_id=str(
                    contrato.get("instancia_id", "")
                ).strip(),
                seed_escenario=_entero_seguro(
                    contrato.get("seed_escenario"),
                    0,
                ),
                seed_ejecucion=_entero_seguro(
                    contrato.get("seed_ejecucion"),
                    0,
                ),
                cantidad_pedidos=_entero_seguro(
                    contrato.get("cantidad_pedidos"),
                    0,
                ),
                contratos_ok=_entero_seguro(
                    contrato.get("planes_ok"),
                    0,
                ),
                contratos_error=_entero_seguro(
                    contrato.get("planes_error"),
                    0,
                ),
                ejecuciones_ok=(
                    ejecucion.ejecuciones_ok
                    if ejecucion is not None
                    else 0
                ),
                ejecuciones_error=(
                    ejecucion.ejecuciones_error
                    if ejecucion is not None
                    else len(ORDEN_MODOS_ESPERADO)
                ),
                error_caso=error_caso,
                mejor_costo_real=mejor,
                modos_mejor_costo=modos_mejor,
                costo_real_rl=_costo_corrida_modo(corridas_caso, "RL"),
                ranking_rl=(rl.ranking_caso if rl is not None else None),
                brecha_rl_vs_mejor_pct=brecha_rl,
                costo_real_ga=_costo_corrida_modo(corridas_caso, "GA"),
                costo_real_greedy=_costo_corrida_modo(
                    corridas_caso,
                    "GREEDY",
                ),
                costo_real_random=_costo_corrida_modo(
                    corridas_caso,
                    "RANDOM",
                ),
                costo_real_hibrido=_costo_corrida_modo(
                    corridas_caso,
                    "HIBRIDO",
                ),
            )
        )

    return resultados


def _construir_resumen_algoritmos(
    corridas: Sequence[RegistroSuiteComparacion],
) -> list[ResumenAlgoritmoSuiteComparacion]:
    cantidad_casos = len({corrida.caso_id for corrida in corridas})
    resultados: list[ResumenAlgoritmoSuiteComparacion] = []

    for modo in ORDEN_MODOS_ESPERADO:
        grupo = [
            corrida
            for corrida in corridas
            if corrida.modo_solicitado == modo
        ]
        exitosas = [
            corrida
            for corrida in grupo
            if corrida.estado_ejecucion == "OK"
        ]

        comparaciones_rl = Counter(
            corrida.comparacion_vs_rl
            for corrida in exitosas
        )
        comparaciones_greedy = Counter(
            corrida.comparacion_vs_greedy
            for corrida in exitosas
        )
        fuentes = Counter(
            corrida.fuente_seleccionada or "SIN_FUENTE"
            for corrida in exitosas
        )

        resultados.append(
            ResumenAlgoritmoSuiteComparacion(
                modo_solicitado=modo,
                casos_totales=cantidad_casos,
                casos_ok=len(exitosas),
                casos_error=len(grupo) - len(exitosas),
                primeros_puestos=sum(
                    1
                    for corrida in exitosas
                    if corrida.ranking_caso == 1
                ),
                ranking_promedio=_promedio_opcional(
                    corrida.ranking_caso
                    for corrida in exitosas
                ),
                victorias_vs_rl=comparaciones_rl["MEJOR"],
                empates_vs_rl=comparaciones_rl["EMPATE"],
                derrotas_vs_rl=comparaciones_rl["PEOR"],
                mejora_media_vs_rl_pct=_promedio_opcional(
                    corrida.mejora_vs_rl_pct
                    for corrida in exitosas
                ),
                mejora_mediana_vs_rl_pct=_mediana_opcional(
                    corrida.mejora_vs_rl_pct
                    for corrida in exitosas
                ),
                victorias_vs_greedy=(
                    comparaciones_greedy["MEJOR"]
                ),
                empates_vs_greedy=(
                    comparaciones_greedy["EMPATE"]
                ),
                derrotas_vs_greedy=(
                    comparaciones_greedy["PEOR"]
                ),
                mejora_media_vs_greedy_pct=_promedio_opcional(
                    corrida.mejora_vs_greedy_pct
                    for corrida in exitosas
                ),
                mejora_mediana_vs_greedy_pct=_mediana_opcional(
                    corrida.mejora_vs_greedy_pct
                    for corrida in exitosas
                ),
                error_estimacion_abs_medio_pct=_promedio_opcional(
                    abs(corrida.error_relativo_estimacion_pct)
                    if corrida.error_relativo_estimacion_pct is not None
                    else None
                    for corrida in exitosas
                ),
                tiempo_plan_promedio_ms=_promedio_opcional(
                    corrida.tiempo_plan_ms
                    for corrida in exitosas
                ),
                tiempo_selector_promedio_ms=_promedio_opcional(
                    corrida.tiempo_selector_ms
                    for corrida in exitosas
                ),
                tiempo_simulado_promedio_min=_promedio_opcional(
                    corrida.tiempo_simulado_min
                    for corrida in exitosas
                ),
                fuentes_seleccionadas="|".join(
                    f"{fuente}={cantidad}"
                    for fuente, cantidad in sorted(fuentes.items())
                ),
            )
        )

    return resultados


def _calcular_rankings(
    registros: Sequence[RegistroEjecucionComparacion],
    *,
    tolerancia_empate: float,
) -> dict[str, int]:
    ordenados = sorted(
        registros,
        key=lambda registro: float(registro.costo_real),
    )
    rankings: dict[str, int] = {}
    costo_grupo: float | None = None
    ranking_actual = 0

    for registro in ordenados:
        costo = float(registro.costo_real)
        if (
            costo_grupo is None
            or abs(costo - costo_grupo) > tolerancia_empate
        ):
            ranking_actual += 1
            costo_grupo = costo
        rankings[registro.modo_solicitado] = ranking_actual

    return rankings


def _comparar_costos(
    *,
    costo_actual: float | None,
    costo_referencia: float | None,
    tolerancia: float,
) -> tuple[float | None, float | None, str]:
    if costo_actual is None or costo_referencia is None:
        return None, None, "NO_DISPONIBLE"

    diferencia = costo_actual - costo_referencia
    mejora = None
    if costo_referencia > 0.0:
        mejora = (
            costo_referencia - costo_actual
        ) / costo_referencia * 100.0

    if abs(diferencia) <= tolerancia:
        comparacion = "EMPATE"
    elif diferencia < 0.0:
        comparacion = "MEJOR"
    else:
        comparacion = "PEOR"

    return diferencia, mejora, comparacion


def _costo_modo(
    registros: Sequence[RegistroEjecucionComparacion],
    modo: str,
) -> float | None:
    for registro in registros:
        if registro.modo_solicitado == modo:
            return registro.costo_real
    return None


def _corrida_modo(
    corridas: Sequence[RegistroSuiteComparacion],
    modo: str,
) -> RegistroSuiteComparacion | None:
    for corrida in corridas:
        if corrida.modo_solicitado == modo:
            return corrida
    return None


def _costo_corrida_modo(
    corridas: Sequence[RegistroSuiteComparacion],
    modo: str,
) -> float | None:
    corrida = _corrida_modo(corridas, modo)
    return corrida.costo_real if corrida is not None else None


def _valor_comun_contrato(
    casos: Sequence[CasoContratoComparacion],
    campo: str,
) -> str:
    valores = {
        str(caso.contrato.get(campo, "")).strip()
        for caso in casos
        if str(caso.contrato.get(campo, "")).strip()
    }
    if not valores:
        return ""
    if len(valores) > 1:
        raise ValueError(
            f"Los contratos usan valores distintos para {campo}: "
            f"{sorted(valores)}"
        )
    return next(iter(valores))


def _promedio_opcional(valores: Any) -> float | None:
    normalizados = _normalizar_numeros(valores)
    return mean(normalizados) if normalizados else None


def _mediana_opcional(valores: Any) -> float | None:
    normalizados = _normalizar_numeros(valores)
    return median(normalizados) if normalizados else None


def _normalizar_numeros(valores: Any) -> list[float]:
    resultado: list[float] = []
    for valor in valores:
        if valor is None:
            continue
        numero = float(valor)
        if isfinite(numero):
            resultado.append(numero)
    return resultado


def _escribir_csv_dataclasses(
    ruta: Path,
    registros: Sequence[Any],
) -> None:
    if not registros:
        raise ValueError(
            f"No hay registros para escribir en {ruta.name}."
        )

    filas: list[dict[str, Any]] = []
    for registro in registros:
        fila = asdict(registro)
        for clave, valor in tuple(fila.items()):
            if isinstance(valor, (list, tuple)):
                fila[clave] = "|".join(str(elemento) for elemento in valor)
        filas.append(fila)

    with ruta.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(filas[0]),
        )
        escritor.writeheader()
        escritor.writerows(filas)


def _normalizar_identificador(valor: str) -> str:
    texto = str(valor).strip().upper()
    seguro = "".join(
        caracter if caracter.isalnum() else "_"
        for caracter in texto
    ).strip("_")
    if not seguro:
        raise ValueError("El identificador del caso no es válido.")
    return seguro[:80]


def _entero_seguro(valor: Any, predeterminado: int) -> int:
    try:
        return int(round(float(valor)))
    except (TypeError, ValueError):
        return predeterminado
