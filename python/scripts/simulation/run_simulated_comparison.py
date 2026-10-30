"""Script de experimentación y comparación de algoritmos en simulación AnyLogic.

Ejecuta los cinco métodos de planificación (`RL`, `HIBRIDO`, `GREEDY`, `RANDOM`, `GA`)
a lo largo de múltiples réplicas y turnos (matutinos o vespertinos), evaluando tanto
el costo y factibilidad estimados como las variables operativas en el motor de simulación
(demoras reales, tiempos de espera de clientes y desbalance entre camiones).

Consolida los resultados y las semillas en carpetas con formato `seed_<seed_inicial>_<timestamp>`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from planner.core.schema import InstanciaTurno, PedidoInput, Turno
from planner.domain.preprocess import preprocesar_pedidos
from planner.integration import selector_bridge
from planner.integration.alpyne_codec import codificar_instancia_alpyne
from planner.integration.order_excel_import import (
    ErrorImportacionExcel,
    importar_pedidos_excel,
)
from planner.integration.simulated_comparison import (
    METODOS_COMPARACION_SIMULADA,
    ComparacionSimulada,
    RunnerSimulado,
    ejecutar_comparacion_simulada,
)
from planner.integration.simulated_execution import (
    codificar_identificadores_pedidos,
    ejecutar_plan_en_modelo_exportado,
)

from scripts.data_processing.generar_pedidos_dia import (
    DEFAULT_HISTORICO_CSV,
    NOMBRE_METADATA_GENERACION,
    ejecutar_generacion,
    generar_semilla_aleatoria,
    parsear_rango_pedidos,
)

NOMBRE_METADATA_EXPERIMENTO = "metadata_experimento.json"
NOMBRE_RESUMEN_CSV = "resumen_resultados.csv"
NOMBRE_RESUMEN_JSON = "resumen_resultados.json"

HORARIOS_OPERACION_TURNO: dict[Turno, tuple[int, int, int]] = {
    Turno.MANANA: (450, 720, 735),
    Turno.TARDE: (840, 1020, 1035),
}

COLUMNAS_RESUMEN_CSV: list[str] = [
    "seed_inicial",
    "dia_numero",
    "instancia_id",
    "archivo_excel",
    "turno",
    "cantidad_tareas",
    "repeticion",
    "seed_escenario",
    "seed_ejecucion",
    "metodo",
    "seed_algoritmo",
    "algoritmo_estimado",
    "factible_estimado",
    "costo_estimado",
    "distancia_estimada_km",
    "duracion_estimada_min",
    "viajes_estimados",
    "pedidos_tardios_estimados",
    "tardanza_estimada_min",
    "desbalance_estimado_min",
    "tiempo_plan_ms",
    "estado_simulado",
    "algoritmo_simulado",
    "costo_simulado",
    "distancia_simulada_km",
    "duracion_simulada_min",
    "viajes_simulados",
    "tardanza_simulada_min",
    "desbalance_simulado_min",
    "tiempo_motor_s",
    "detalle_estimado",
    "error_estimado",
    "error_simulado",
]

__all__ = [
    "COLUMNAS_RESUMEN_CSV",
    "HORARIOS_OPERACION_TURNO",
    "NOMBRE_METADATA_EXPERIMENTO",
    "NOMBRE_RESUMEN_CSV",
    "NOMBRE_RESUMEN_JSON",
    "TurnoEntrada",
    "calcular_semillas_por_metodo",
    "crear_instancia_desde_excel",
    "ejecutar_experimento_comparacion",
    "main",
]


@dataclass(frozen=True)
class TurnoEntrada:
    """Descriptor de un turno importable para el experimento comparativo.

    Attributes:
        dia_numero: Número correlativo del día/turno en el lote.
        archivo_excel: Ruta al archivo Excel con los pedidos correspondientes.
        seed_escenario: Semilla fija asignada al escenario para replicabilidad.
        turno: Turno horario declarado ("AUTO", "MANANA", "TARDE", "MIXTO").
    """

    dia_numero: int
    archivo_excel: Path
    seed_escenario: int
    turno: str = "AUTO"


def calcular_semillas_por_metodo(
    seed_escenario: int,
    seed_ejecucion: int,
) -> dict[str, int]:
    """Devuelve el mapa de semillas deterministas para cada uno de los 5 algoritmos.

    Args:
        seed_escenario: Semilla del escenario generador.
        seed_ejecucion: Semilla de la réplica de ejecución estocástica.

    Returns:
        Diccionario con semillas individuales para RL, HIBRIDO, GREEDY, RANDOM y GA.
    """
    seed_base = seed_ejecucion if seed_ejecucion > 0 else seed_escenario
    return {
        "RL": seed_base,
        "HIBRIDO": seed_base + 9001,
        "GREEDY": seed_base,
        "RANDOM": seed_base + 7001,
        "GA": seed_base + 8001,
    }


def _resolver_turnos_entrada(
    *,
    archivo_excel: Path | None,
    directorio_turnos: Path | None,
    seed_escenario_default: int,
    turno_forzado: str,
    rng: random.Random,
) -> tuple[list[TurnoEntrada], int | None]:
    """Resuelve la lista de turnos a evaluar desde un directorio o desde un único Excel.

    Args:
        archivo_excel: Archivo Excel individual (opcional).
        directorio_turnos: Directorio con archivos de turnos (opcional).
        seed_escenario_default: Semilla fallback para archivos individuales.
        turno_forzado: Forzado de franja ("AUTO", "MANANA", etc.).
        rng: Generador pseudoaleatorio para sortear semillas faltantes.

    Returns:
        Tupla con lista de TurnoEntrada y semilla inicial detectada de metadata (o None).

    Raises:
        NotADirectoryError: Si directorio_turnos no es un directorio válido.
        FileNotFoundError: Si los archivos Excel especificados no existen.
        ValueError: Si no se indicó ni archivo Excel ni directorio.
    """
    if directorio_turnos is not None:
        carpeta = directorio_turnos.expanduser().resolve()
        if not carpeta.is_dir():
            raise NotADirectoryError(f"No existe el directorio de turnos: {carpeta}")

        ruta_meta = carpeta / NOMBRE_METADATA_GENERACION
        if ruta_meta.is_file():
            datos_meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
            seed_gen = datos_meta.get("seed_inicial")
            turnos_meta = datos_meta.get("turnos", [])
            entradas: list[TurnoEntrada] = []

            for idx, item in enumerate(turnos_meta, start=1):
                nombre_xlsx = item.get("archivo_xlsx")
                if not nombre_xlsx:
                    continue
                ruta_xlsx = carpeta / str(nombre_xlsx)
                if not ruta_xlsx.is_file():
                    raise FileNotFoundError(
                        f"No se encontró el archivo Excel declarado en metadata: {ruta_xlsx}"
                    )
                turno_item = (
                    str(item.get("turno", "MANANA")) if turno_forzado == "AUTO" else turno_forzado
                )
                entradas.append(
                    TurnoEntrada(
                        dia_numero=int(item.get("dia_numero", idx)),
                        archivo_excel=ruta_xlsx,
                        seed_escenario=int(
                            item.get("seed_escenario", rng.randint(100_000, 999_999_999))
                        ),
                        turno=turno_item,
                    )
                )

            if entradas:
                return entradas, int(seed_gen) if seed_gen is not None else None

        archivos_xlsx = sorted(
            p for p in carpeta.glob("*.xlsx") if not p.stem.endswith("_consolidado")
        )
        if not archivos_xlsx:
            raise FileNotFoundError(f"No se encontraron archivos .xlsx de turnos en: {carpeta}")

        return (
            [
                TurnoEntrada(
                    dia_numero=idx,
                    archivo_excel=ruta_xlsx,
                    seed_escenario=rng.randint(100_000, 999_999_999),
                    turno=turno_forzado,
                )
                for idx, ruta_xlsx in enumerate(archivos_xlsx, start=1)
            ],
            None,
        )

    if archivo_excel is None:
        raise ValueError("Se debe indicar --archivo-excel, --directorio-turnos o --generar-dias.")

    ruta_unica = archivo_excel.expanduser().resolve()
    if not ruta_unica.is_file():
        raise FileNotFoundError(f"No existe el archivo Excel: {ruta_unica}")

    return (
        [
            TurnoEntrada(
                dia_numero=1,
                archivo_excel=ruta_unica,
                seed_escenario=int(seed_escenario_default),
                turno=turno_forzado,
            )
        ],
        None,
    )


def crear_instancia_desde_excel(
    *,
    archivo_excel: Path,
    seed_escenario: int,
    seed_ejecucion: int,
    turno: str = "AUTO",
    dia_numero: int = 1,
    fecha_operacion: str = "2026-08-30",
) -> InstanciaTurno:
    """Importa un Excel de pedidos y construye la InstanciaTurno preprocesada.

    Args:
        archivo_excel: Archivo de pedidos a cargar.
        seed_escenario: Semilla fija asignada al lote de pedidos.
        seed_ejecucion: Semilla de la corrida de simulación o algoritmo.
        turno: Franja solicitada ("AUTO", "MANANA", "TARDE").
        dia_numero: Índice de día dentro de la serie de experimentos.
        fecha_operacion: Fecha ISO de la jornada analizada.

    Returns:
        InstanciaTurno preprocesada y verificada.

    Raises:
        RuntimeError: Si la importación de pedidos falla en ambos turnos.
    """
    candidatos_turno = (
        ["MANANA", "TARDE"] if turno.upper() in ("AUTO", "MIXTO") else [turno.upper()]
    )

    importacion = None
    ultimo_error: Exception | None = None
    for candidato in candidatos_turno:
        try:
            importacion = importar_pedidos_excel(
                archivo_excel,
                turno=candidato,
                capacidad_camion=8,
                max_tareas=30,
            )
            break
        except ErrorImportacionExcel as exc:
            ultimo_error = exc

    if importacion is None:
        if ultimo_error is not None:
            raise ultimo_error
        raise RuntimeError(f"No se pudo importar el archivo {archivo_excel}")

    turno_enum = Turno.TARDE if importacion.turno.upper() == "TARDE" else Turno.MANANA
    inicio_min, fin_obj_min, fin_tol_min = HORARIOS_OPERACION_TURNO[turno_enum]

    unidades_volcador_mapa = {
        pedido.pedido_id: pedido.unidades_volcador for pedido in importacion.pedidos
    }

    originales = [
        PedidoInput(
            pedido_id=pedido.pedido_id,
            pedido_original_id=pedido.pedido_id,
            numero_parte=1,
            total_partes=1,
            turno=turno_enum,
            latitud=pedido.latitud,
            longitud=pedido.longitud,
            unidades_capacidad=pedido.unidades,
            requiere_volcador=pedido.requiere_volcador,
            tiene_ventana_especifica=pedido.tiene_ventana,
            hora_desde_min=pedido.hora_desde_min,
            hora_hasta_min=pedido.hora_hasta_min,
            cliente=pedido.cliente,
            direccion=pedido.direccion,
            barrio=pedido.barrio,
            observaciones=pedido.observaciones,
        )
        for pedido in importacion.pedidos
    ]

    tareas = preprocesar_pedidos(
        originales,
        capacidad_camion=8,
        unidades_volcador_por_pedido=unidades_volcador_mapa,
    )

    return InstanciaTurno(
        instancia_id=f"UI-{fecha_operacion}-D{dia_numero:02d}-{seed_escenario}",
        fecha_operacion=fecha_operacion,
        turno=turno_enum,
        pedidos=tareas,
        lat_corralon=-32.8495006,
        lon_corralon=-60.722653,
        capacidad_camion=8,
        cantidad_camiones=2,
        hora_inicio_turno_min=inicio_min,
        hora_fin_objetivo_min=fin_obj_min,
        hora_fin_tolerancia_min=fin_tol_min,
        seed_escenario=int(seed_escenario),
        seed_ejecucion=int(seed_ejecucion),
    )


def _extraer_resultados_estimados(
    semillas_metodos: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Extrae las métricas estimadas del puente global para los 5 métodos."""
    comparacion_est = selector_bridge._ultima_comparacion_estimada
    salida: dict[str, dict[str, Any]] = {}
    if comparacion_est is None:
        return salida

    for res in comparacion_est.resultados:
        modo = res.modo_solicitado.value
        salida[modo] = {
            "metodo_solicitado": modo,
            "seed_algoritmo": semillas_metodos.get(modo, -1),
            "factible": res.factible,
            "algoritmo_resultante": (
                None if res.algoritmo_resultante is None else res.algoritmo_resultante.value
            ),
            "costo_estimado": res.costo_estimado,
            "distancia_total_km": res.distancia_total_km,
            "duracion_operacion_min": res.duracion_operacion_min,
            "viajes_totales": res.viajes_totales,
            "pedidos_tardios": res.pedidos_tardios,
            "tardanza_total_min": res.tardanza_total_min,
            "diferencia_fin_camiones_min": res.diferencia_fin_camiones_min,
            "tiempo_plan_ms": res.tiempo_plan_ms,
            "tiempo_selector_ms": res.tiempo_selector_ms,
            "detalle": res.detalle,
            "error": res.error,
        }

    return salida


def ejecutar_experimento_comparacion(
    *,
    raiz_python: Path = PYTHON_ROOT,
    modelo_exportado: Path | None = None,
    archivo_excel: Path | None = None,
    directorio_turnos: Path | None = None,
    generar_dias: int | None = None,
    rango_pedidos: tuple[int, int] | None = (8, 12),
    turno: str = "AUTO",
    repeticiones: int = 3,
    seed: int | None = None,
    seed_escenario_default: int = 6001,
    directorio_salida: Path | None = None,
    salida_archivo_unico: Path | None = None,
    solo_estimado: bool = False,
    sin_proveedores: bool = False,
    permitir_fallback_vial: bool = True,
    timeout: int = 240,
    horizonte_min: float = 600.0,
    runner_simulado: RunnerSimulado = ejecutar_plan_en_modelo_exportado,
) -> dict[str, Any]:
    """Ejecuta los 5 algoritmos repeticiones veces sobre cada turno en simulación.

    Args:
        raiz_python: Ruta raíz del paquete Python.
        modelo_exportado: Ruta al archivo zip del modelo AnyLogic.
        archivo_excel: Excel de pedidos único (opcional).
        directorio_turnos: Directorio con lote de Excel de turnos (opcional).
        generar_dias: Días aleatorios a generar en el acto si se indica N > 0.
        rango_pedidos: Rango `(min, max)` de pedidos para generación estocástica.
        turno: Franja horaria ("AUTO", "MANANA", "TARDE", "MIXTO").
        repeticiones: Veces que se evalúan los 5 métodos por turno (default: 3).
        seed: Semilla inicial del experimento (None para aleatoria).
        seed_escenario_default: Semilla para archivo individual.
        directorio_salida: Directorio base donde alojar la carpeta del experimento.
        salida_archivo_unico: Archivo JSON adicional para última corrida.
        solo_estimado: Si es True, omite invocar el motor Java AnyLogic.
        sin_proveedores: Si es True, desactiva viajes a canteras externas.
        permitir_fallback_vial: Si es True, permite Haversine ajustado ante fallos de caché.
        timeout: Segundos máximos por simulación de método.
        horizonte_min: Horizonte de tiempo simulado máximo en minutos.
        runner_simulado: Función que ejecuta el plan en el motor exportado.

    Returns:
        Diccionario serializable con la metadata y resultados globales del experimento.

    Raises:
        ValueError: Si los parámetros numéricos de control son inválidos.
    """
    if repeticiones <= 0:
        raise ValueError("repeticiones debe ser >= 1.")

    raiz = raiz_python.expanduser().resolve()
    seed_inicial = int(seed) if seed is not None else generar_semilla_aleatoria()
    if seed_inicial <= 0:
        raise ValueError("La semilla inicial debe ser > 0.")

    rng = random.Random(seed_inicial)
    inicio_dt = datetime.now()
    marca_tiempo = inicio_dt.strftime("%Y%m%d_%H%M%S")

    base_resultados = (
        directorio_salida.expanduser().resolve()
        if directorio_salida is not None
        else (raiz / "data" / "results")
    )
    carpeta_experimento = base_resultados / f"seed_{seed_inicial}_{marca_tiempo}"
    carpeta_corridas = carpeta_experimento / "corridas"
    carpeta_corridas.mkdir(parents=True, exist_ok=True)

    if generar_dias is not None:
        if generar_dias <= 0:
            raise ValueError("generar_dias debe ser >= 1.")
        carpeta_turnos_gen = carpeta_experimento / "turnos"
        turno_gen = "MANANA" if turno.upper() == "AUTO" else turno.upper()
        ejecutar_generacion(
            dias_n=generar_dias,
            rango_pedidos=rango_pedidos,
            ruta_historico=DEFAULT_HISTORICO_CSV,
            directorio_salida=carpeta_turnos_gen,
            formato="ambos",
            turno=turno_gen,
            prefijo="pedidos_dia",
            crear_consolidado=True,
            seed=seed_inicial,
            usar_timestamp=False,
            max_tareas=12,
        )
        directorio_turnos = carpeta_turnos_gen

    turnos_entrada, seed_generacion_origen = _resolver_turnos_entrada(
        archivo_excel=archivo_excel,
        directorio_turnos=directorio_turnos,
        seed_escenario_default=seed_escenario_default,
        turno_forzado=turno,
        rng=rng,
    )

    estado_selector = selector_bridge.inicializar(
        model_path=str(raiz / "models" / "rl" / "rl_policies.json"),
        max_pedidos=30,
        deterministic=True,
        cache_vial_path=str(raiz / "data" / "routing" / "cache_vial.csv"),
        version_cache_vial="pedemonte-vial-v1",
        permitir_fallback_vial=permitir_fallback_vial,
    )
    print(f"SELECTOR | {estado_selector}")

    modelo_zip = (
        modelo_exportado.expanduser().resolve()
        if modelo_exportado is not None
        else (raiz / "anylogic_export" / "simulated_comparison" / "PedemonteDigitalTwin_v2.zip")
    )

    filas_resumen: list[dict[str, Any]] = []
    metadata_turnos_evaluados: list[dict[str, Any]] = []
    todas_completas = True
    ultimo_payload_corrida: dict[str, Any] | None = None

    for entrada in turnos_entrada:
        repeticiones_meta: list[dict[str, Any]] = []

        for rep in range(1, repeticiones + 1):
            if (
                len(turnos_entrada) == 1
                and repeticiones == 1
                and seed is None
                and salida_archivo_unico is not None
            ):
                seed_ejecucion = 1_000_000 + entrada.seed_escenario
            else:
                seed_ejecucion = rng.randint(1_000_000, 999_999_999)

            semillas_metodos = calcular_semillas_por_metodo(
                entrada.seed_escenario,
                seed_ejecucion,
            )

            instancia = crear_instancia_desde_excel(
                archivo_excel=entrada.archivo_excel,
                seed_escenario=entrada.seed_escenario,
                seed_ejecucion=seed_ejecucion,
                turno=entrada.turno,
                dia_numero=entrada.dia_numero,
            )
            instancia_vector = codificar_instancia_alpyne(instancia)

            selector_bridge.comparar_estimado_vector(
                instancia_vector,
                instancia.seed_escenario,
                instancia.seed_ejecucion,
            )

            estimados_por_metodo = _extraer_resultados_estimados(semillas_metodos)

            planes: dict[str, list[float]] = {}
            for metodo in METODOS_COMPARACION_SIMULADA:
                info_est = estimados_por_metodo.get(metodo, {})
                if info_est.get("factible", False):
                    planes[metodo] = selector_bridge.obtener_plan_comparacion_vector(metodo)
                    print(
                        f"PLAN | dia={entrada.dia_numero:02d} | rep={rep:02d} | "
                        f"metodo={metodo} | seed_algo={semillas_metodos[metodo]} | "
                        f"plan_vector={len(planes[metodo])}"
                    )
                else:
                    todas_completas = False
                    print(
                        f"PLAN_ERROR | dia={entrada.dia_numero:02d} | rep={rep:02d} | "
                        f"metodo={metodo} | error={info_est.get('error', '')}"
                    )

            comparacion_sim: ComparacionSimulada | None = None
            if not solo_estimado:
                comparacion_sim = ejecutar_comparacion_simulada(
                    modelo_exportado=modelo_zip,
                    raiz_python=raiz,
                    instancia_vector=instancia_vector,
                    planes_por_metodo=planes,
                    identificadores_pedidos=codificar_identificadores_pedidos(instancia.pedidos),
                    instancia_id=instancia.instancia_id,
                    fecha_operacion=instancia.fecha_operacion,
                    seed_escenario=instancia.seed_escenario,
                    seed_ejecucion=instancia.seed_ejecucion,
                    proveedores_habilitados=not sin_proveedores,
                    timeout_segundos_por_metodo=timeout,
                    horizonte_simulacion_min=horizonte_min,
                    continuar_ante_error=True,
                    runner=runner_simulado,
                )
                if not comparacion_sim.completa:
                    todas_completas = False

                for resultado_metodo in comparacion_sim.resultados:
                    if resultado_metodo.resultado is None:
                        print(
                            f"METODO | dia={entrada.dia_numero:02d} | rep={rep:02d} | "
                            f"solicitado={resultado_metodo.metodo_solicitado} | "
                            f"estado={resultado_metodo.estado} | "
                            f"error={resultado_metodo.error}"
                        )
                        continue

                    res_sim = resultado_metodo.resultado
                    print(
                        f"METODO | dia={entrada.dia_numero:02d} | rep={rep:02d} | "
                        f"solicitado={resultado_metodo.metodo_solicitado} | "
                        f"estado={resultado_metodo.estado} | "
                        f"aplicado={res_sim.algoritmo_aplicado} | "
                        f"seed_algo={semillas_metodos[resultado_metodo.metodo_solicitado]} | "
                        f"costo={res_sim.costo_total:.6f} | "
                        f"distancia_km={res_sim.distancia_total_km:.6f} | "
                        f"duracion_min={res_sim.duracion_simulada_min:.6f} | "
                        f"viajes={res_sim.viajes_totales} | "
                        f"tardanza_min={res_sim.tardanza_total_min:.6f} | "
                        f"desbalance_min={res_sim.diferencia_fin_camiones_min:.6f}"
                    )

            payload_corrida: dict[str, Any] = (
                comparacion_sim.como_dict()
                if comparacion_sim is not None
                else {
                    "version": 1,
                    "instancia_id": instancia.instancia_id,
                    "fecha_operacion": instancia.fecha_operacion,
                    "firma_instancia": (selector_bridge.obtener_firma_comparacion_estimada()),
                    "seed_escenario": instancia.seed_escenario,
                    "seed_ejecucion": instancia.seed_ejecucion,
                    "proveedores_habilitados": not sin_proveedores,
                    "completa": len(planes) == len(METODOS_COMPARACION_SIMULADA),
                    "cantidad_finalizados": len(planes),
                    "cantidad_metodos": len(METODOS_COMPARACION_SIMULADA),
                    "tiempo_total_segundos": 0.0,
                    "resultados": [],
                }
            )
            payload_corrida["seed_inicial_experimento"] = seed_inicial
            payload_corrida["dia_numero"] = entrada.dia_numero
            payload_corrida["repeticion"] = rep
            payload_corrida["turno"] = instancia.turno.value
            payload_corrida["archivo_excel"] = entrada.archivo_excel.name
            payload_corrida["cantidad_tareas"] = len(instancia.pedidos)
            payload_corrida["semillas_por_metodo"] = semillas_metodos
            payload_corrida["resultados_estimados"] = estimados_por_metodo

            nombre_corrida = (
                f"turno_{entrada.dia_numero:02d}_rep_{rep:02d}_seed_{seed_ejecucion}.json"
            )
            ruta_corrida = carpeta_corridas / nombre_corrida
            ruta_corrida.write_text(
                json.dumps(
                    payload_corrida,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            ultimo_payload_corrida = payload_corrida

            for metodo in METODOS_COMPARACION_SIMULADA:
                est = estimados_por_metodo.get(metodo, {})
                sim_metodo = (
                    comparacion_sim.obtener_resultado(metodo)
                    if comparacion_sim is not None
                    else None
                )
                sim_res = sim_metodo.resultado if sim_metodo is not None else None

                filas_resumen.append(
                    {
                        "seed_inicial": seed_inicial,
                        "dia_numero": entrada.dia_numero,
                        "instancia_id": instancia.instancia_id,
                        "archivo_excel": entrada.archivo_excel.name,
                        "turno": instancia.turno.value,
                        "cantidad_tareas": len(instancia.pedidos),
                        "repeticion": rep,
                        "seed_escenario": instancia.seed_escenario,
                        "seed_ejecucion": instancia.seed_ejecucion,
                        "metodo": metodo,
                        "seed_algoritmo": semillas_metodos[metodo],
                        "algoritmo_estimado": est.get("algoritmo_resultante"),
                        "factible_estimado": est.get("factible", False),
                        "costo_estimado": est.get("costo_estimado"),
                        "distancia_estimada_km": est.get("distancia_total_km"),
                        "duracion_estimada_min": est.get("duracion_operacion_min"),
                        "viajes_estimados": est.get("viajes_totales"),
                        "pedidos_tardios_estimados": est.get("pedidos_tardios"),
                        "tardanza_estimada_min": est.get("tardanza_total_min"),
                        "desbalance_estimado_min": est.get("diferencia_fin_camiones_min"),
                        "tiempo_plan_ms": est.get("tiempo_plan_ms"),
                        "estado_simulado": (
                            "SOLO_ESTIMADO" if sim_metodo is None else sim_metodo.estado
                        ),
                        "algoritmo_simulado": (
                            None if sim_res is None else sim_res.algoritmo_aplicado
                        ),
                        "costo_simulado": None if sim_res is None else sim_res.costo_total,
                        "distancia_simulada_km": (
                            None if sim_res is None else sim_res.distancia_total_km
                        ),
                        "duracion_simulada_min": (
                            None if sim_res is None else sim_res.duracion_simulada_min
                        ),
                        "viajes_simulados": (None if sim_res is None else sim_res.viajes_totales),
                        "tardanza_simulada_min": (
                            None if sim_res is None else sim_res.tardanza_total_min
                        ),
                        "desbalance_simulado_min": (
                            None if sim_res is None else sim_res.diferencia_fin_camiones_min
                        ),
                        "tiempo_motor_s": (
                            None if sim_metodo is None else sim_metodo.tiempo_motor_segundos
                        ),
                        "detalle_estimado": est.get("detalle", ""),
                        "error_estimado": est.get("error", ""),
                        "error_simulado": "" if sim_metodo is None else sim_metodo.error,
                    }
                )

            repeticiones_meta.append(
                {
                    "repeticion": rep,
                    "seed_ejecucion": seed_ejecucion,
                    "semillas_por_metodo": semillas_metodos,
                    "archivo_resultado": f"corridas/{nombre_corrida}",
                }
            )

        metadata_turnos_evaluados.append(
            {
                "dia_numero": entrada.dia_numero,
                "archivo_excel": str(entrada.archivo_excel),
                "turno": entrada.turno,
                "seed_escenario": entrada.seed_escenario,
                "repeticiones": repeticiones_meta,
            }
        )

    ruta_resumen_csv = carpeta_experimento / NOMBRE_RESUMEN_CSV
    with open(ruta_resumen_csv, "w", encoding="utf-8-sig", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=COLUMNAS_RESUMEN_CSV)
        writer.writeheader()
        for fila in filas_resumen:
            writer.writerow(fila)

    ruta_resumen_json = carpeta_experimento / NOMBRE_RESUMEN_JSON
    ruta_resumen_json.write_text(
        json.dumps(filas_resumen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fin_dt = datetime.now()
    metadata_experimento = {
        "version": 1,
        "seed_inicial": seed_inicial,
        "seed_generacion_origen": seed_generacion_origen,
        "fecha_inicio": inicio_dt.isoformat(timespec="seconds"),
        "fecha_fin": fin_dt.isoformat(timespec="seconds"),
        "marca_tiempo": marca_tiempo,
        "carpeta_experimento": str(carpeta_experimento),
        "parametros": {
            "cantidad_turnos": len(turnos_entrada),
            "repeticiones_por_turno": repeticiones,
            "metodos": list(METODOS_COMPARACION_SIMULADA),
            "solo_estimado": solo_estimado,
            "proveedores_habilitados": not sin_proveedores,
            "timeout_segundos_por_metodo": timeout,
            "horizonte_simulacion_min": horizonte_min,
        },
        "archivos_consolidados": {
            "resumen_csv": NOMBRE_RESUMEN_CSV,
            "resumen_json": NOMBRE_RESUMEN_JSON,
        },
        "turnos_evaluados": metadata_turnos_evaluados,
        "completa": todas_completas,
    }

    ruta_metadata_exp = carpeta_experimento / NOMBRE_METADATA_EXPERIMENTO
    ruta_metadata_exp.write_text(
        json.dumps(metadata_experimento, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if salida_archivo_unico is not None and ultimo_payload_corrida is not None:
        salida_resuelta = salida_archivo_unico.expanduser().resolve()
        salida_resuelta.parent.mkdir(parents=True, exist_ok=True)
        salida_resuelta.write_text(
            json.dumps(
                ultimo_payload_corrida,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"COMPARACION_JSON | {salida_resuelta}")

    print(f"CARPETA_EXPERIMENTO | {carpeta_experimento}")
    print(f"METADATA_EXPERIMENTO | {ruta_metadata_exp}")
    print(f"RESUMEN_CSV | {ruta_resumen_csv}")

    return metadata_experimento


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta RL, HÍBRIDO, GREEDY, RANDOM y GA múltiples veces por turno "
            "guardando semillas y resultados en carpetas seed_<semilla>_<fecha>."
        )
    )
    parser.add_argument(
        "--modelo-exportado",
        default=str(
            PYTHON_ROOT / "anylogic_export" / "simulated_comparison" / "PedemonteDigitalTwin_v2.zip"
        ),
    )
    parser.add_argument("--raiz-python", default=str(PYTHON_ROOT))
    parser.add_argument(
        "--archivo-excel",
        default=str(PYTHON_ROOT / "templates" / "ejemplo_pedidos_importacion.xlsx"),
        help="Archivo Excel individual a evaluar si no se usa --directorio-turnos ni --generar-dias.",
    )
    parser.add_argument(
        "--directorio-turnos",
        default=None,
        help="Carpeta generada por generar_pedidos_dia.py con turnos (.xlsx y metadata_generacion.json).",
    )
    parser.add_argument(
        "--generar-dias",
        type=int,
        default=None,
        help="Si se indica N > 0, genera N turnos aleatorios nuevos y corre los 5 algoritmos sobre ellos.",
    )
    parser.add_argument(
        "--pedidos-por-dia",
        type=str,
        default="8-12",
        help="Rango de pedidos por día al usar --generar-dias (por defecto: 8-12).",
    )
    parser.add_argument(
        "--turno",
        choices=["AUTO", "MANANA", "TARDE", "MIXTO"],
        default="AUTO",
        help="Turno operativo: AUTO detecta desde el archivo o metadata (por defecto: AUTO).",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=3,
        help="Cantidad de veces que se ejecutan los 5 algoritmos sobre cada día/turno (por defecto: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla inicial del experimento (si se omite, se sortea aleatoriamente y queda grabada).",
    )
    parser.add_argument(
        "--seed-escenario",
        type=int,
        default=6001,
        help="Semilla de escenario cuando se evalúa un único --archivo-excel manual (por defecto: 6001).",
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--horizonte-min",
        type=float,
        default=600.0,
        help="Horizonte máximo de tiempo simulado, en minutos, para cada motor AnyLogic.",
    )
    parser.add_argument("--sin-proveedores", action="store_true")
    parser.add_argument(
        "--solo-estimado",
        action="store_true",
        help="Ejecuta los 5 algoritmos y registra métricas estimadas sin invocar el motor Java AnyLogic.",
    )
    parser.add_argument(
        "--directorio-salida",
        default=str(PYTHON_ROOT / "data" / "results"),
        help="Directorio base donde se creará la carpeta seed_<semilla>_<fecha>.",
    )
    parser.add_argument(
        "--salida",
        default=None,
        help="Ruta opcional a un archivo JSON adicional para compatibilidad con corridas individuales.",
    )
    return parser


def main() -> int:
    """Punto de entrada CLI para la ejecución del experimento comparativo."""
    args = _parser().parse_args()

    rango = parsear_rango_pedidos(args.pedidos_por_dia)
    salida_unica = Path(args.salida) if args.salida else None
    dir_turnos = Path(args.directorio_turnos) if args.directorio_turnos else None
    archivo_excel = Path(args.archivo_excel) if args.archivo_excel else None

    metadata = ejecutar_experimento_comparacion(
        raiz_python=Path(args.raiz_python),
        modelo_exportado=Path(args.modelo_exportado),
        archivo_excel=archivo_excel,
        directorio_turnos=dir_turnos,
        generar_dias=args.generar_dias,
        rango_pedidos=rango,
        turno=args.turno,
        repeticiones=args.repeticiones,
        seed=args.seed,
        seed_escenario_default=args.seed_escenario,
        directorio_salida=Path(args.directorio_salida),
        salida_archivo_unico=salida_unica,
        solo_estimado=args.solo_estimado,
        sin_proveedores=args.sin_proveedores,
        timeout=args.timeout,
        horizonte_min=args.horizonte_min,
    )

    return 0 if bool(metadata.get("completa", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
