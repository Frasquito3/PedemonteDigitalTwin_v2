from __future__ import annotations

import csv
import json

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    AlgoritmoPlanificacion,
    PlanCamion,
    PlanTurno,
    ViajePlan,
)
from planner.evaluation.classic_instances import (
    CasoBenchmarkClasico,
    crear_casos_benchmark_clasico,
)
from planner.routing.operational import (
    estimar_espera_cliente,
    simular_plan_operativo_estimado,
    tiempo_descarga_estimado_min,
)
from planner.routing.travel import (
    ProveedorViaje,
    construir_matriz_viaje,
    tiempo_viaje_esperado_min,
)


VERSION_AUDITORIA_SERVICIO = "comparacion-anylogic-servicio-v1"
VERSION_AUDITORIA_VENTANAS = "auditoria-ventanas-estimada-v1"
ESTADOS_SERVICIO = ("COMPLETA", "INCOMPLETA", "ERROR")


@dataclass(frozen=True)
class RegistroServicioComparacion:
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
    estado_servicio: str
    elegible_ranking: bool
    error_ejecucion: str

    cantidad_pedidos: int
    tareas_entregadas: int | None
    tareas_no_entregadas: int | None
    pedidos_pendientes: int | None
    nivel_servicio_pct: float | None
    viajes_totales: int | None

    costo_estimado: float | None
    costo_real: float | None
    diferencia_costo_real_estimado: float | None
    error_relativo_estimacion_pct: float | None

    tiempo_plan_ms: float | None
    tiempo_selector_ms: float | None
    tiempo_simulado_min: float | None

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
class ResumenCasoServicio:
    caso_id: str
    categoria: str
    descripcion: str
    instancia_id: str
    seed_escenario: int
    seed_ejecucion: int
    cantidad_pedidos: int

    ejecuciones_tecnicas_ok: int
    ejecuciones_tecnicas_error: int
    servicios_completos: int
    servicios_incompletos: int
    servicios_error: int

    mejor_costo_real_completo: float | None
    modos_mejor_costo_completo: tuple[str, ...]

    estado_servicio_rl: str
    nivel_servicio_rl_pct: float | None
    costo_real_rl: float | None
    ranking_rl: int | None
    brecha_rl_vs_mejor_completo_pct: float | None

    costo_real_ga: float | None
    costo_real_greedy: float | None
    costo_real_random: float | None
    costo_real_hibrido: float | None


@dataclass(frozen=True)
class ResumenAlgoritmoServicio:
    modo_solicitado: str
    casos_totales: int
    ejecuciones_tecnicas_ok: int
    ejecuciones_tecnicas_error: int
    servicios_completos: int
    servicios_incompletos: int
    tasa_completitud_pct: float
    nivel_servicio_promedio_pct: float | None

    primeros_puestos: int
    ranking_promedio: float | None

    comparables_vs_rl: int
    victorias_vs_rl: int
    empates_vs_rl: int
    derrotas_vs_rl: int
    mejora_media_vs_rl_pct: float | None
    mejora_mediana_vs_rl_pct: float | None

    comparables_vs_greedy: int
    victorias_vs_greedy: int
    empates_vs_greedy: int
    derrotas_vs_greedy: int
    mejora_media_vs_greedy_pct: float | None
    mejora_mediana_vs_greedy_pct: float | None

    error_estimacion_abs_medio_pct_completas: float | None
    tiempo_plan_promedio_ms: float | None
    tiempo_selector_promedio_ms: float | None
    tiempo_simulado_promedio_min: float | None
    fuentes_seleccionadas: str


@dataclass(frozen=True)
class RegistroAuditoriaVentana:
    caso_id: str
    modo_solicitado: str
    algoritmo_resultante: str
    estado_servicio_real: str
    nivel_servicio_real_pct: float | None

    camion_id: int
    numero_viaje: int
    orden_visita: int
    secuencia_viaje: str
    pedido_id: str

    ventana_desde_min: int
    ventana_hasta_min: int
    ventana_desde_hhmm: str
    ventana_hasta_hhmm: str

    minuto_salida_viaje: float
    llegada_estimada_min: float
    llegada_estimada_hhmm: str
    espera_apertura_estimada_min: float
    espera_respuesta_estimada_min: float
    inicio_descarga_estimado_min: float
    fin_descarga_estimado_min: float
    tardanza_estimada_min: float

    clasificacion_temporal: str
    riesgo_rechazo: bool
    auditoria: str


@dataclass(frozen=True)
class ResumenPlanVentanas:
    caso_id: str
    modo_solicitado: str
    algoritmo_resultante: str
    estado_servicio_real: str
    secuencia_plan: str
    entregas_auditadas: int
    llegadas_tardias_estimadas: int
    tardanza_total_estimada_min: float
    tardanza_max_estimada_min: float
    pedidos_riesgo_rechazo: tuple[str, ...]


@dataclass(frozen=True)
class ResultadoAuditoriaServicio:
    version_auditoria: str
    version_suite_origen: str
    generado_utc: str
    suite_origen: str

    cantidad_casos: int
    corridas_esperadas: int
    ejecuciones_tecnicas_ok: int
    ejecuciones_tecnicas_error: int
    servicios_completos: int
    servicios_incompletos: int
    servicios_error: int
    tasa_completitud_global_pct: float

    orden_modos: tuple[str, ...]
    common_random_numbers_por_caso: bool
    proceso_nuevo_por_plan: bool
    version_viaje: str
    version_objetivo: str

    casos: tuple[ResumenCasoServicio, ...]
    corridas: tuple[RegistroServicioComparacion, ...]
    resumen_algoritmos: tuple[ResumenAlgoritmoServicio, ...]

    version_auditoria_ventanas: str
    auditoria_ventanas: tuple[RegistroAuditoriaVentana, ...]
    resumen_planes_ventanas: tuple[ResumenPlanVentanas, ...]


@dataclass(frozen=True)
class ConfiguracionAuditoriaServicio:
    tolerancia_empate: float = 1e-9

    def __post_init__(self) -> None:
        if self.tolerancia_empate < 0.0:
            raise ValueError("tolerancia_empate no puede ser negativa.")


def cargar_suite_cruda(ruta: str | Path) -> dict[str, Any]:
    archivo = Path(ruta).expanduser().resolve()
    if not archivo.is_file():
        raise FileNotFoundError(f"No existe la suite cruda: {archivo}")

    with archivo.open("r", encoding="utf-8") as entrada:
        datos = json.load(entrada)

    if not isinstance(datos, dict):
        raise ValueError("La suite cruda debe ser un objeto JSON.")
    return datos


def auditar_suite_servicio(
    suite_cruda: Mapping[str, Any],
    *,
    suite_origen: str = "",
    contratos_dir: str | Path | None = None,
    proveedor_viaje: ProveedorViaje | None = None,
    configuracion_planificacion: ConfiguracionPlanificacion | None = None,
    configuracion: ConfiguracionAuditoriaServicio | None = None,
) -> ResultadoAuditoriaServicio:
    config = configuracion or ConfiguracionAuditoriaServicio()
    orden_modos = _tupla_textos(suite_cruda.get("orden_modos"))
    casos_raw = _lista_mappings(suite_cruda.get("casos"), "casos")
    corridas_raw = _lista_mappings(suite_cruda.get("corridas"), "corridas")

    cantidad_por_caso = {
        _texto(caso.get("caso_id")): _entero(caso.get("cantidad_pedidos"), 0)
        for caso in casos_raw
    }
    metadatos_por_caso = {
        _texto(caso.get("caso_id")): caso
        for caso in casos_raw
    }

    corridas_base = [
        _clasificar_corrida(corrida, cantidad_por_caso)
        for corrida in corridas_raw
    ]
    corridas = _aplicar_rankings_y_comparaciones(
        corridas_base,
        tolerancia=config.tolerancia_empate,
    )

    casos = _resumir_casos(
        corridas,
        metadatos_por_caso,
        tolerancia=config.tolerancia_empate,
    )
    resumen_algoritmos = _resumir_algoritmos(corridas, orden_modos)

    auditoria_ventanas: tuple[RegistroAuditoriaVentana, ...] = ()
    resumen_ventanas: tuple[ResumenPlanVentanas, ...] = ()
    if contratos_dir is not None:
        if proveedor_viaje is None:
            raise ValueError(
                "proveedor_viaje es obligatorio cuando se auditan ventanas."
            )
        auditoria_ventanas, resumen_ventanas = auditar_ventanas_contratos(
            contratos_dir=contratos_dir,
            corridas=corridas,
            proveedor_viaje=proveedor_viaje,
            configuracion=(
                configuracion_planificacion
                or ConfiguracionPlanificacion()
            ),
        )

    tecnicas_ok = sum(
        1 for corrida in corridas if corrida.estado_ejecucion == "OK"
    )
    tecnicas_error = len(corridas) - tecnicas_ok
    completas = sum(
        1 for corrida in corridas if corrida.estado_servicio == "COMPLETA"
    )
    incompletas = sum(
        1 for corrida in corridas if corrida.estado_servicio == "INCOMPLETA"
    )
    errores_servicio = sum(
        1 for corrida in corridas if corrida.estado_servicio == "ERROR"
    )
    tasa_global = (
        completas / len(corridas) * 100.0
        if corridas
        else 0.0
    )

    return ResultadoAuditoriaServicio(
        version_auditoria=VERSION_AUDITORIA_SERVICIO,
        version_suite_origen=_texto(suite_cruda.get("version_suite")),
        generado_utc=datetime.now(timezone.utc).isoformat(),
        suite_origen=suite_origen,
        cantidad_casos=len(casos),
        corridas_esperadas=_entero(
            suite_cruda.get("corridas_esperadas"),
            len(corridas),
        ),
        ejecuciones_tecnicas_ok=tecnicas_ok,
        ejecuciones_tecnicas_error=tecnicas_error,
        servicios_completos=completas,
        servicios_incompletos=incompletas,
        servicios_error=errores_servicio,
        tasa_completitud_global_pct=tasa_global,
        orden_modos=orden_modos,
        common_random_numbers_por_caso=bool(
            suite_cruda.get("common_random_numbers_por_caso", False)
        ),
        proceso_nuevo_por_plan=bool(
            suite_cruda.get("proceso_nuevo_por_plan", False)
        ),
        version_viaje=_texto(suite_cruda.get("version_viaje")),
        version_objetivo=_texto(suite_cruda.get("version_objetivo")),
        casos=tuple(casos),
        corridas=tuple(corridas),
        resumen_algoritmos=tuple(resumen_algoritmos),
        version_auditoria_ventanas=VERSION_AUDITORIA_VENTANAS,
        auditoria_ventanas=auditoria_ventanas,
        resumen_planes_ventanas=resumen_ventanas,
    )


def escribir_auditoria_servicio(
    resultado: ResultadoAuditoriaServicio,
    directorio_salida: str | Path,
) -> dict[str, Path]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    rutas = {
        "auditoria_json": salida / "comparison_suite_service.json",
        "corridas_csv": salida / "comparison_runs_service.csv",
        "casos_csv": salida / "comparison_case_summary_service.csv",
        "algoritmos_csv": salida / "comparison_algorithm_summary_service.csv",
        "ventanas_json": salida / "window_audit.json",
        "ventanas_csv": salida / "window_audit.csv",
        "ventanas_planes_csv": salida / "window_plan_summary.csv",
    }

    with rutas["auditoria_json"].open("w", encoding="utf-8") as archivo:
        json.dump(asdict(resultado), archivo, ensure_ascii=False, indent=2)

    with rutas["ventanas_json"].open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "version": resultado.version_auditoria_ventanas,
                "generado_utc": resultado.generado_utc,
                "registros": [
                    asdict(registro)
                    for registro in resultado.auditoria_ventanas
                ],
                "resumen_planes": [
                    asdict(resumen)
                    for resumen in resultado.resumen_planes_ventanas
                ],
            },
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    _escribir_csv(rutas["corridas_csv"], resultado.corridas)
    _escribir_csv(rutas["casos_csv"], resultado.casos)
    _escribir_csv(rutas["algoritmos_csv"], resultado.resumen_algoritmos)
    _escribir_csv(rutas["ventanas_csv"], resultado.auditoria_ventanas)
    _escribir_csv(
        rutas["ventanas_planes_csv"],
        resultado.resumen_planes_ventanas,
    )
    return rutas


def auditar_ventanas_contratos(
    *,
    contratos_dir: str | Path,
    corridas: Sequence[RegistroServicioComparacion],
    proveedor_viaje: ProveedorViaje,
    configuracion: ConfiguracionPlanificacion,
) -> tuple[
    tuple[RegistroAuditoriaVentana, ...],
    tuple[ResumenPlanVentanas, ...],
]:
    raiz = Path(contratos_dir).expanduser().resolve()
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de contratos: {raiz}"
        )

    casos_clasicos = {
        caso.caso_id: caso
        for caso in crear_casos_benchmark_clasico()
    }
    corridas_por_clave = {
        (corrida.caso_id, corrida.modo_solicitado): corrida
        for corrida in corridas
    }

    registros: list[RegistroAuditoriaVentana] = []
    resumenes: list[ResumenPlanVentanas] = []

    for caso_id, caso in casos_clasicos.items():
        if not any(
            pedido.tiene_ventana_especifica
            for pedido in caso.instancia.pedidos
        ):
            continue

        ruta_contrato = raiz / caso_id / "comparison_contract.json"
        contrato = _cargar_mapping_json(ruta_contrato)
        planes = _lista_mappings(contrato.get("planes"), "planes")
        matriz = construir_matriz_viaje(
            caso.instancia,
            configuracion,
            proveedor=proveedor_viaje,
        )

        for plan_raw in planes:
            if _texto(plan_raw.get("estado")) != "OK":
                continue

            modo = _texto(plan_raw.get("modo_solicitado"))
            corrida = corridas_por_clave.get((caso_id, modo))
            estado_servicio = (
                corrida.estado_servicio if corrida is not None else "ERROR"
            )
            nivel_servicio = (
                corrida.nivel_servicio_pct if corrida is not None else None
            )
            plan = _plan_desde_contrato(caso, plan_raw)
            operacion = simular_plan_operativo_estimado(
                caso.instancia,
                plan,
                matriz,
                configuracion,
            )
            salida_por_viaje = {
                (carga.camion_id, carga.numero_viaje): carga.minuto_fin
                for carga in operacion.cargas
            }
            pedidos = {
                pedido.pedido_id: pedido
                for pedido in caso.instancia.pedidos
            }

            filas_plan: list[RegistroAuditoriaVentana] = []
            secuencias_plan: list[str] = []

            for camion in plan.camiones:
                for viaje in camion.viajes:
                    secuencia = ">".join(viaje.pedido_ids)
                    secuencias_plan.append(
                        f"c{camion.camion_id}:v{viaje.numero_viaje}[{secuencia}]"
                    )
                    minuto_actual = salida_por_viaje[
                        (camion.camion_id, viaje.numero_viaje)
                    ]
                    nodo_actual = configuracion.id_nodo_corralon

                    for orden_visita, pedido_id in enumerate(
                        viaje.pedido_ids,
                        start=1,
                    ):
                        pedido = pedidos[pedido_id]
                        viaje_min = tiempo_viaje_esperado_min(
                            matriz,
                            nodo_actual,
                            pedido_id,
                            minuto_actual,
                            configuracion,
                        )
                        llegada = minuto_actual + viaje_min
                        espera = estimar_espera_cliente(
                            pedido,
                            llegada,
                            configuracion,
                        )
                        tardanza = max(0.0, llegada - pedido.hora_hasta_min)
                        descarga = tiempo_descarga_estimado_min(
                            pedido,
                            configuracion,
                        )
                        fin_descarga = espera.minuto_inicio_descarga + descarga

                        if llegada > pedido.hora_hasta_min:
                            clasificacion = "TARDIA_RIESGO_RECHAZO"
                            riesgo = True
                        elif llegada < pedido.hora_desde_min:
                            clasificacion = "ESPERA_APERTURA"
                            riesgo = False
                        else:
                            clasificacion = "EN_VENTANA"
                            riesgo = False

                        if pedido.tiene_ventana_especifica:
                            fila = RegistroAuditoriaVentana(
                                caso_id=caso_id,
                                modo_solicitado=modo,
                                algoritmo_resultante=_texto(
                                    plan_raw.get("algoritmo_resultante")
                                ),
                                estado_servicio_real=estado_servicio,
                                nivel_servicio_real_pct=nivel_servicio,
                                camion_id=camion.camion_id,
                                numero_viaje=viaje.numero_viaje,
                                orden_visita=orden_visita,
                                secuencia_viaje=secuencia,
                                pedido_id=pedido_id,
                                ventana_desde_min=pedido.hora_desde_min,
                                ventana_hasta_min=pedido.hora_hasta_min,
                                ventana_desde_hhmm=_formato_hhmm(
                                    pedido.hora_desde_min
                                ),
                                ventana_hasta_hhmm=_formato_hhmm(
                                    pedido.hora_hasta_min
                                ),
                                minuto_salida_viaje=salida_por_viaje[
                                    (camion.camion_id, viaje.numero_viaje)
                                ],
                                llegada_estimada_min=llegada,
                                llegada_estimada_hhmm=_formato_hhmm(llegada),
                                espera_apertura_estimada_min=(
                                    espera.tiempo_espera_ventana_min
                                ),
                                espera_respuesta_estimada_min=(
                                    espera.tiempo_espera_respuesta_min
                                ),
                                inicio_descarga_estimado_min=(
                                    espera.minuto_inicio_descarga
                                ),
                                fin_descarga_estimado_min=fin_descarga,
                                tardanza_estimada_min=tardanza,
                                clasificacion_temporal=clasificacion,
                                riesgo_rechazo=riesgo,
                                auditoria=VERSION_AUDITORIA_VENTANAS,
                            )
                            filas_plan.append(fila)
                            registros.append(fila)

                        minuto_actual = fin_descarga
                        nodo_actual = pedido_id

            riesgos = tuple(
                fila.pedido_id
                for fila in filas_plan
                if fila.riesgo_rechazo
            )
            tardanzas = [fila.tardanza_estimada_min for fila in filas_plan]
            resumenes.append(
                ResumenPlanVentanas(
                    caso_id=caso_id,
                    modo_solicitado=modo,
                    algoritmo_resultante=_texto(
                        plan_raw.get("algoritmo_resultante")
                    ),
                    estado_servicio_real=estado_servicio,
                    secuencia_plan="||".join(secuencias_plan),
                    entregas_auditadas=len(filas_plan),
                    llegadas_tardias_estimadas=len(riesgos),
                    tardanza_total_estimada_min=sum(tardanzas),
                    tardanza_max_estimada_min=max(tardanzas, default=0.0),
                    pedidos_riesgo_rechazo=riesgos,
                )
            )

    return tuple(registros), tuple(resumenes)


def _clasificar_corrida(
    corrida: Mapping[str, Any],
    cantidad_por_caso: Mapping[str, int],
) -> RegistroServicioComparacion:
    caso_id = _texto(corrida.get("caso_id"))
    cantidad = cantidad_por_caso.get(caso_id, 0)
    estado_ejecucion = _texto(corrida.get("estado_ejecucion")).upper()
    entregadas = _entero_opcional(corrida.get("tareas_entregadas"))
    no_entregadas = _entero_opcional(corrida.get("tareas_no_entregadas"))

    if estado_ejecucion != "OK":
        estado_servicio = "ERROR"
        pendientes = None
        nivel_servicio = None
    else:
        entregadas_seguras = entregadas if entregadas is not None else 0
        no_entregadas_seguras = (
            no_entregadas if no_entregadas is not None else 0
        )
        pendientes = max(
            0,
            cantidad - entregadas_seguras,
            no_entregadas_seguras,
        )
        nivel_servicio = (
            min(100.0, entregadas_seguras / cantidad * 100.0)
            if cantidad > 0
            else None
        )
        estado_servicio = (
            "COMPLETA" if pendientes == 0 else "INCOMPLETA"
        )

    costo_real = _flotante_opcional(corrida.get("costo_real"))
    elegible = (
        estado_servicio == "COMPLETA"
        and costo_real is not None
        and isfinite(costo_real)
    )

    return RegistroServicioComparacion(
        caso_id=caso_id,
        categoria=_texto(corrida.get("categoria")),
        descripcion=_texto(corrida.get("descripcion")),
        instancia_id=_texto(corrida.get("instancia_id")),
        orden_modo=_entero(corrida.get("orden_modo"), 0),
        modo_solicitado=_texto(corrida.get("modo_solicitado")),
        algoritmo_resultante=_texto(corrida.get("algoritmo_resultante")),
        fuente_seleccionada=_texto(corrida.get("fuente_seleccionada")),
        firma_ruta=_texto(corrida.get("firma_ruta")),
        seed_escenario=_entero(corrida.get("seed_escenario"), 0),
        seed_planificacion=_entero_opcional(
            corrida.get("seed_planificacion")
        ),
        seed_ejecucion=_entero(corrida.get("seed_ejecucion"), 0),
        estado_ejecucion=estado_ejecucion,
        estado_servicio=estado_servicio,
        elegible_ranking=elegible,
        error_ejecucion=_texto(corrida.get("error_ejecucion")),
        cantidad_pedidos=cantidad,
        tareas_entregadas=entregadas,
        tareas_no_entregadas=no_entregadas,
        pedidos_pendientes=pendientes,
        nivel_servicio_pct=nivel_servicio,
        viajes_totales=_entero_opcional(corrida.get("viajes_totales")),
        costo_estimado=_flotante_opcional(corrida.get("costo_estimado")),
        costo_real=costo_real,
        diferencia_costo_real_estimado=_flotante_opcional(
            corrida.get("diferencia_costo_real_estimado")
        ),
        error_relativo_estimacion_pct=_flotante_opcional(
            corrida.get("error_relativo_estimacion_pct")
        ),
        tiempo_plan_ms=_flotante_opcional(corrida.get("tiempo_plan_ms")),
        tiempo_selector_ms=_flotante_opcional(
            corrida.get("tiempo_selector_ms")
        ),
        tiempo_simulado_min=_flotante_opcional(
            corrida.get("tiempo_simulado_min")
        ),
        ranking_caso=None,
        diferencia_costo_vs_rl=None,
        mejora_vs_rl_pct=None,
        comparacion_vs_rl="NO_DISPONIBLE",
        diferencia_costo_vs_greedy=None,
        mejora_vs_greedy_pct=None,
        comparacion_vs_greedy="NO_DISPONIBLE",
        estado_final_motor=_texto(corrida.get("estado_final_motor")),
        stop_condition=_booleano_opcional(corrida.get("stop_condition")),
        mensaje_anylogic=_texto(corrida.get("mensaje_anylogic")),
    )


def _aplicar_rankings_y_comparaciones(
    corridas: Sequence[RegistroServicioComparacion],
    *,
    tolerancia: float,
) -> list[RegistroServicioComparacion]:
    resultado: list[RegistroServicioComparacion] = []
    casos = _agrupar_por(corridas, lambda corrida: corrida.caso_id)

    for corridas_caso in casos.values():
        elegibles = [corrida for corrida in corridas_caso if corrida.elegible_ranking]
        rankings = _rankings(elegibles, tolerancia=tolerancia)
        rl = _costo_referencia(elegibles, "RL")
        greedy = _costo_referencia(elegibles, "GREEDY")

        for corrida in corridas_caso:
            diferencia_rl, mejora_rl, comparacion_rl = _comparar(
                corrida.costo_real if corrida.elegible_ranking else None,
                rl,
                tolerancia,
            )
            diferencia_g, mejora_g, comparacion_g = _comparar(
                corrida.costo_real if corrida.elegible_ranking else None,
                greedy,
                tolerancia,
            )
            resultado.append(
                replace(
                    corrida,
                    ranking_caso=rankings.get(corrida.modo_solicitado),
                    diferencia_costo_vs_rl=diferencia_rl,
                    mejora_vs_rl_pct=mejora_rl,
                    comparacion_vs_rl=comparacion_rl,
                    diferencia_costo_vs_greedy=diferencia_g,
                    mejora_vs_greedy_pct=mejora_g,
                    comparacion_vs_greedy=comparacion_g,
                )
            )

    return sorted(resultado, key=lambda corrida: (corrida.caso_id, corrida.orden_modo))


def _resumir_casos(
    corridas: Sequence[RegistroServicioComparacion],
    metadatos: Mapping[str, Mapping[str, Any]],
    *,
    tolerancia: float,
) -> list[ResumenCasoServicio]:
    resultados: list[ResumenCasoServicio] = []
    por_caso = _agrupar_por(corridas, lambda corrida: corrida.caso_id)

    for caso_id, grupo in por_caso.items():
        elegibles = [corrida for corrida in grupo if corrida.elegible_ranking]
        mejor = min(
            (corrida.costo_real for corrida in elegibles if corrida.costo_real is not None),
            default=None,
        )
        mejores = tuple(
            corrida.modo_solicitado
            for corrida in elegibles
            if (
                mejor is not None
                and corrida.costo_real is not None
                and abs(corrida.costo_real - mejor) <= tolerancia
            )
        )
        rl = _buscar_modo(grupo, "RL")
        brecha = None
        if (
            rl is not None
            and rl.elegible_ranking
            and rl.costo_real is not None
            and mejor is not None
            and mejor > 0.0
        ):
            brecha = (rl.costo_real - mejor) / mejor * 100.0

        meta = metadatos.get(caso_id, {})
        primero = grupo[0]
        resultados.append(
            ResumenCasoServicio(
                caso_id=caso_id,
                categoria=primero.categoria,
                descripcion=primero.descripcion,
                instancia_id=primero.instancia_id,
                seed_escenario=primero.seed_escenario,
                seed_ejecucion=primero.seed_ejecucion,
                cantidad_pedidos=_entero(meta.get("cantidad_pedidos"), primero.cantidad_pedidos),
                ejecuciones_tecnicas_ok=sum(
                    1 for corrida in grupo if corrida.estado_ejecucion == "OK"
                ),
                ejecuciones_tecnicas_error=sum(
                    1 for corrida in grupo if corrida.estado_ejecucion != "OK"
                ),
                servicios_completos=sum(
                    1 for corrida in grupo if corrida.estado_servicio == "COMPLETA"
                ),
                servicios_incompletos=sum(
                    1 for corrida in grupo if corrida.estado_servicio == "INCOMPLETA"
                ),
                servicios_error=sum(
                    1 for corrida in grupo if corrida.estado_servicio == "ERROR"
                ),
                mejor_costo_real_completo=mejor,
                modos_mejor_costo_completo=mejores,
                estado_servicio_rl=(rl.estado_servicio if rl is not None else "ERROR"),
                nivel_servicio_rl_pct=(rl.nivel_servicio_pct if rl is not None else None),
                costo_real_rl=(rl.costo_real if rl is not None else None),
                ranking_rl=(rl.ranking_caso if rl is not None else None),
                brecha_rl_vs_mejor_completo_pct=brecha,
                costo_real_ga=_costo_modo(grupo, "GA"),
                costo_real_greedy=_costo_modo(grupo, "GREEDY"),
                costo_real_random=_costo_modo(grupo, "RANDOM"),
                costo_real_hibrido=_costo_modo(grupo, "HIBRIDO"),
            )
        )

    return sorted(resultados, key=lambda caso: caso.caso_id)


def _resumir_algoritmos(
    corridas: Sequence[RegistroServicioComparacion],
    orden_modos: Sequence[str],
) -> list[ResumenAlgoritmoServicio]:
    resultados: list[ResumenAlgoritmoServicio] = []
    cantidad_casos = len({corrida.caso_id for corrida in corridas})

    for modo in orden_modos:
        grupo = [corrida for corrida in corridas if corrida.modo_solicitado == modo]
        tecnicas_ok = [corrida for corrida in grupo if corrida.estado_ejecucion == "OK"]
        completas = [corrida for corrida in grupo if corrida.estado_servicio == "COMPLETA"]
        comparables_rl = [
            corrida for corrida in completas
            if corrida.comparacion_vs_rl != "NO_DISPONIBLE"
        ]
        comparables_g = [
            corrida for corrida in completas
            if corrida.comparacion_vs_greedy != "NO_DISPONIBLE"
        ]
        conteo_rl = Counter(corrida.comparacion_vs_rl for corrida in comparables_rl)
        conteo_g = Counter(corrida.comparacion_vs_greedy for corrida in comparables_g)
        fuentes = Counter(
            corrida.fuente_seleccionada or "SIN_FUENTE"
            for corrida in tecnicas_ok
        )

        resultados.append(
            ResumenAlgoritmoServicio(
                modo_solicitado=modo,
                casos_totales=cantidad_casos,
                ejecuciones_tecnicas_ok=len(tecnicas_ok),
                ejecuciones_tecnicas_error=len(grupo) - len(tecnicas_ok),
                servicios_completos=len(completas),
                servicios_incompletos=sum(
                    1 for corrida in grupo if corrida.estado_servicio == "INCOMPLETA"
                ),
                tasa_completitud_pct=(
                    len(completas) / cantidad_casos * 100.0
                    if cantidad_casos
                    else 0.0
                ),
                nivel_servicio_promedio_pct=_promedio(
                    corrida.nivel_servicio_pct for corrida in tecnicas_ok
                ),
                primeros_puestos=sum(
                    1 for corrida in completas if corrida.ranking_caso == 1
                ),
                ranking_promedio=_promedio(
                    corrida.ranking_caso for corrida in completas
                ),
                comparables_vs_rl=len(comparables_rl),
                victorias_vs_rl=conteo_rl["MEJOR"],
                empates_vs_rl=conteo_rl["EMPATE"],
                derrotas_vs_rl=conteo_rl["PEOR"],
                mejora_media_vs_rl_pct=_promedio(
                    corrida.mejora_vs_rl_pct for corrida in comparables_rl
                ),
                mejora_mediana_vs_rl_pct=_mediana(
                    corrida.mejora_vs_rl_pct for corrida in comparables_rl
                ),
                comparables_vs_greedy=len(comparables_g),
                victorias_vs_greedy=conteo_g["MEJOR"],
                empates_vs_greedy=conteo_g["EMPATE"],
                derrotas_vs_greedy=conteo_g["PEOR"],
                mejora_media_vs_greedy_pct=_promedio(
                    corrida.mejora_vs_greedy_pct for corrida in comparables_g
                ),
                mejora_mediana_vs_greedy_pct=_mediana(
                    corrida.mejora_vs_greedy_pct for corrida in comparables_g
                ),
                error_estimacion_abs_medio_pct_completas=_promedio(
                    abs(corrida.error_relativo_estimacion_pct)
                    if corrida.error_relativo_estimacion_pct is not None
                    else None
                    for corrida in completas
                ),
                tiempo_plan_promedio_ms=_promedio(
                    corrida.tiempo_plan_ms for corrida in tecnicas_ok
                ),
                tiempo_selector_promedio_ms=_promedio(
                    corrida.tiempo_selector_ms for corrida in tecnicas_ok
                ),
                tiempo_simulado_promedio_min=_promedio(
                    corrida.tiempo_simulado_min for corrida in tecnicas_ok
                ),
                fuentes_seleccionadas="|".join(
                    f"{fuente}={cantidad}"
                    for fuente, cantidad in sorted(fuentes.items())
                ),
            )
        )

    return resultados


def _plan_desde_contrato(
    caso: CasoBenchmarkClasico,
    plan_raw: Mapping[str, Any],
) -> PlanTurno:
    algoritmo_texto = _texto(plan_raw.get("algoritmo_resultante"))
    try:
        algoritmo = AlgoritmoPlanificacion(algoritmo_texto)
    except ValueError as exc:
        raise ValueError(
            f"Algoritmo no soportado en contrato: {algoritmo_texto}."
        ) from exc

    camiones: list[PlanCamion] = []
    for camion_raw in _lista_mappings(plan_raw.get("camiones"), "camiones"):
        viajes: list[ViajePlan] = []
        for viaje_raw in _lista_mappings(camion_raw.get("viajes"), "viajes"):
            viajes.append(
                ViajePlan(
                    numero_viaje=_entero(viaje_raw.get("numero_viaje"), 0),
                    pedido_ids=list(_tupla_textos(viaje_raw.get("pedido_ids"))),
                )
            )
        camiones.append(
            PlanCamion(
                camion_id=_entero(camion_raw.get("camion_id"), 0),
                viajes=viajes,
            )
        )

    return PlanTurno(
        instancia_id=caso.instancia.instancia_id,
        algoritmo=algoritmo,
        camiones=camiones,
        costo_estimado=_flotante_opcional(plan_raw.get("costo_estimado")) or 0.0,
        tiempo_computo_ms=_flotante_opcional(plan_raw.get("tiempo_plan_ms")) or 0.0,
    )


def _rankings(
    corridas: Sequence[RegistroServicioComparacion],
    *,
    tolerancia: float,
) -> dict[str, int]:
    ordenadas = sorted(
        corridas,
        key=_costo_requerido,
    )
    resultado: dict[str, int] = {}
    costo_grupo: float | None = None
    ranking = 0
    for corrida in ordenadas:
        costo = _costo_requerido(corrida)
        if costo_grupo is None or abs(costo - costo_grupo) > tolerancia:
            ranking += 1
            costo_grupo = costo
        resultado[corrida.modo_solicitado] = ranking
    return resultado



def _costo_requerido(corrida: RegistroServicioComparacion) -> float:
    costo = corrida.costo_real
    if costo is None or not isfinite(costo):
        raise ValueError(
            "Una corrida elegible para ranking debe tener costo real finito."
        )
    return costo

def _comparar(
    costo_actual: float | None,
    costo_referencia: float | None,
    tolerancia: float,
) -> tuple[float | None, float | None, str]:
    if costo_actual is None or costo_referencia is None:
        return None, None, "NO_DISPONIBLE"
    diferencia = costo_actual - costo_referencia
    mejora = (
        (costo_referencia - costo_actual) / costo_referencia * 100.0
        if costo_referencia > 0.0
        else None
    )
    if abs(diferencia) <= tolerancia:
        comparacion = "EMPATE"
    elif diferencia < 0.0:
        comparacion = "MEJOR"
    else:
        comparacion = "PEOR"
    return diferencia, mejora, comparacion


def _costo_referencia(
    corridas: Sequence[RegistroServicioComparacion],
    modo: str,
) -> float | None:
    corrida = _buscar_modo(corridas, modo)
    return corrida.costo_real if corrida is not None else None


def _costo_modo(
    corridas: Sequence[RegistroServicioComparacion],
    modo: str,
) -> float | None:
    corrida = _buscar_modo(corridas, modo)
    return corrida.costo_real if corrida is not None else None


def _buscar_modo(
    corridas: Sequence[RegistroServicioComparacion],
    modo: str,
) -> RegistroServicioComparacion | None:
    for corrida in corridas:
        if corrida.modo_solicitado == modo:
            return corrida
    return None


def _agrupar_por(
    valores: Sequence[RegistroServicioComparacion],
    clave: Any,
) -> dict[str, list[RegistroServicioComparacion]]:
    resultado: dict[str, list[RegistroServicioComparacion]] = {}
    for valor in valores:
        resultado.setdefault(str(clave(valor)), []).append(valor)
    return resultado


def _promedio(valores: Iterable[float | int | None]) -> float | None:
    numeros = [float(valor) for valor in valores if valor is not None]
    return mean(numeros) if numeros else None


def _mediana(valores: Iterable[float | int | None]) -> float | None:
    numeros = [float(valor) for valor in valores if valor is not None]
    return median(numeros) if numeros else None


def _cargar_mapping_json(ruta: Path) -> dict[str, Any]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el contrato: {ruta}")
    with ruta.open("r", encoding="utf-8") as entrada:
        datos = json.load(entrada)
    if not isinstance(datos, dict):
        raise ValueError(f"El contrato debe ser un objeto JSON: {ruta}")
    return datos


def _lista_mappings(valor: Any, campo: str) -> list[Mapping[str, Any]]:
    if not isinstance(valor, (list, tuple)):
        raise ValueError(f"{campo} debe ser una lista o tupla.")
    resultado: list[Mapping[str, Any]] = []
    for indice, elemento in enumerate(valor):
        if not isinstance(elemento, Mapping):
            raise ValueError(f"{campo}[{indice}] debe ser un mapping.")
        resultado.append(elemento)
    return resultado


def _tupla_textos(valor: Any) -> tuple[str, ...]:
    if not isinstance(valor, (list, tuple)):
        return ()
    return tuple(_texto(elemento) for elemento in valor)


def _texto(valor: Any) -> str:
    return "" if valor is None else str(valor).strip()


def _entero(valor: Any, predeterminado: int) -> int:
    try:
        return int(round(float(valor)))
    except (TypeError, ValueError):
        return predeterminado


def _entero_opcional(valor: Any) -> int | None:
    if valor is None or valor == "":
        return None
    try:
        return int(round(float(valor)))
    except (TypeError, ValueError):
        return None


def _flotante_opcional(valor: Any) -> float | None:
    if valor is None or valor == "":
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if isfinite(numero) else None


def _booleano_opcional(valor: Any) -> bool | None:
    if valor is None or valor == "":
        return None
    if isinstance(valor, bool):
        return valor
    texto = str(valor).strip().lower()
    if texto in {"true", "1", "si", "sí"}:
        return True
    if texto in {"false", "0", "no"}:
        return False
    return None


def _formato_hhmm(minuto: float | int) -> str:
    total = int(round(float(minuto)))
    horas = (total // 60) % 24
    minutos = total % 60
    return f"{horas:02d}:{minutos:02d}"


def _escribir_csv(ruta: Path, registros: Sequence[Any]) -> None:
    if not registros:
        ruta.write_text("", encoding="utf-8-sig")
        return

    filas: list[dict[str, Any]] = []
    for registro in registros:
        fila = asdict(registro)
        for clave, valor in tuple(fila.items()):
            if isinstance(valor, (list, tuple)):
                fila[clave] = "|".join(str(elemento) for elemento in valor)
        filas.append(fila)

    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
