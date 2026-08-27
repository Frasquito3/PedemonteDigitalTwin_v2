from __future__ import annotations

import csv
import json

from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TypeGuard


VERSION_EJECUCION_COMPARACION = "comparacion-anylogic-ejecucion-v1"
VERSION_CONTRATO_ESPERADA = "comparacion-anylogic-v1"
ORDEN_MODOS_ESPERADO = (
    "RL",
    "GA",
    "GREEDY",
    "RANDOM",
    "HIBRIDO",
)


class EjecutorVectoresAnyLogic(Protocol):
    def ejecutar_vectores(
        self,
        *,
        instancia_vector: Sequence[float],
        plan_vector: Sequence[float],
        seed_ejecucion: int,
        cantidad_pedidos: int,
        cantidad_viajes: int,
        identificador_corrida: str,
    ) -> Any:
        ...


@dataclass(frozen=True)
class ConfiguracionEjecucionComparacion:
    continuar_ante_error: bool = True
    exigir_cinco_planes_ok: bool = True
    exigir_orden_rl_primero: bool = True


@dataclass(frozen=True)
class RegistroEjecucionComparacion:
    orden: int
    modo_solicitado: str
    algoritmo_resultante: str
    fuente_seleccionada: str
    firma_ruta: str

    seed_escenario: int
    seed_planificacion: int | None
    seed_ejecucion: int

    costo_estimado: float | None
    tiempo_plan_ms: float | None
    tiempo_selector_ms: float | None

    estado_ejecucion: str
    error_ejecucion: str

    costo_real: float | None
    diferencia_costo_real_estimado: float | None
    error_relativo_estimacion_pct: float | None

    tareas_entregadas: int | None
    tareas_no_entregadas: int | None
    viajes_totales: int | None
    tiempo_simulado_min: float | None

    modelo: str
    java: str
    estado_final_motor: str
    stop_condition: bool | None
    mensaje_anylogic: str


@dataclass(frozen=True)
class ResultadoEjecucionComparacion:
    version_ejecucion: str
    version_contrato: str
    generado_utc: str

    instancia_id: str
    seed_escenario: int
    seed_ejecucion: int
    cantidad_pedidos: int

    orden_modos: tuple[str, ...]
    common_random_numbers: bool
    proceso_nuevo_por_plan: bool

    ejecuciones_ok: int
    ejecuciones_error: int

    modelo: str
    java: str

    registros: tuple[RegistroEjecucionComparacion, ...]


def cargar_contrato_comparacion(
    ruta: str | Path,
) -> dict[str, Any]:
    archivo = Path(ruta).expanduser().resolve()
    if not archivo.is_file():
        raise FileNotFoundError(
            f"No existe el contrato de comparación: {archivo}"
        )

    with archivo.open("r", encoding="utf-8") as entrada:
        datos = json.load(entrada)

    if not isinstance(datos, dict):
        raise ValueError(
            "El contrato de comparación debe ser un objeto JSON."
        )

    return datos


def ejecutar_contrato_comparacion(
    contrato: Mapping[str, Any],
    *,
    ejecutor: EjecutorVectoresAnyLogic,
    configuracion: ConfiguracionEjecucionComparacion | None = None,
) -> ResultadoEjecucionComparacion:
    config = configuracion or ConfiguracionEjecucionComparacion()
    normalizado = _validar_y_normalizar_contrato(
        contrato,
        config=config,
    )

    registros: list[RegistroEjecucionComparacion] = []
    modelo_global = ""
    java_global = ""

    for plan in normalizado["planes"]:
        try:
            resultado = ejecutor.ejecutar_vectores(
                instancia_vector=normalizado["instancia_vector"],
                plan_vector=plan["plan_vector"],
                seed_ejecucion=normalizado["seed_ejecucion"],
                cantidad_pedidos=normalizado["cantidad_pedidos"],
                cantidad_viajes=_contar_viajes(plan),
                identificador_corrida=(
                    f"{plan['orden']:02d}_{plan['modo_solicitado']}"
                ),
            )

            registro = _crear_registro_exitoso(
                plan=plan,
                resultado=resultado,
            )
            modelo_global = registro.modelo or modelo_global
            java_global = registro.java or java_global

        except Exception as exc:
            if not config.continuar_ante_error:
                raise

            registro = _crear_registro_error(
                plan=plan,
                exc=exc,
            )

        registros.append(registro)

    ejecuciones_ok = sum(
        1
        for registro in registros
        if registro.estado_ejecucion == "OK"
    )

    return ResultadoEjecucionComparacion(
        version_ejecucion=VERSION_EJECUCION_COMPARACION,
        version_contrato=normalizado["version_contrato"],
        generado_utc=datetime.now(timezone.utc).isoformat(),
        instancia_id=normalizado["instancia_id"],
        seed_escenario=normalizado["seed_escenario"],
        seed_ejecucion=normalizado["seed_ejecucion"],
        cantidad_pedidos=normalizado["cantidad_pedidos"],
        orden_modos=tuple(normalizado["orden_modos"]),
        common_random_numbers=True,
        proceso_nuevo_por_plan=True,
        ejecuciones_ok=ejecuciones_ok,
        ejecuciones_error=len(registros) - ejecuciones_ok,
        modelo=modelo_global,
        java=java_global,
        registros=tuple(registros),
    )


def escribir_resultado_ejecucion_comparacion(
    resultado: ResultadoEjecucionComparacion,
    directorio_salida: str | Path,
) -> dict[str, Path]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ruta_json = salida / "comparison_execution.json"
    ruta_csv = salida / "comparison_execution.csv"

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(
            asdict(resultado),
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    filas = [
        _registro_csv(registro)
        for registro in resultado.registros
    ]

    with ruta_csv.open(
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

    return {
        "ejecucion_json": ruta_json,
        "ejecucion_csv": ruta_csv,
    }


def _validar_y_normalizar_contrato(
    contrato: Mapping[str, Any],
    *,
    config: ConfiguracionEjecucionComparacion,
) -> dict[str, Any]:
    version = str(contrato.get("version_contrato", "")).strip()
    if version != VERSION_CONTRATO_ESPERADA:
        raise ValueError(
            "Versión de contrato no soportada: "
            f"{version or 'VACÍA'}."
        )

    instancia_id = str(contrato.get("instancia_id", "")).strip()
    if not instancia_id:
        raise ValueError("instancia_id no puede estar vacío.")

    cantidad_pedidos = _entero_positivo(
        contrato.get("cantidad_pedidos"),
        "cantidad_pedidos",
    )
    seed_escenario = _entero_no_negativo(
        contrato.get("seed_escenario"),
        "seed_escenario",
    )
    seed_ejecucion = _entero_no_negativo(
        contrato.get("seed_ejecucion"),
        "seed_ejecucion",
    )

    instancia_vector = _vector_finito(
        contrato.get("instancia_vector"),
        "instancia_vector",
    )

    orden_modos_raw = contrato.get("orden_modos")
    if not _es_secuencia_no_texto(orden_modos_raw):
        raise ValueError("orden_modos debe ser una secuencia no vacía.")

    orden_modos = tuple(
        str(modo).strip().upper()
        for modo in orden_modos_raw
    )

    if config.exigir_orden_rl_primero:
        if orden_modos != ORDEN_MODOS_ESPERADO:
            raise ValueError(
                "El orden de comparación debe ser "
                f"{ORDEN_MODOS_ESPERADO}, recibido={orden_modos}."
            )

    planes_raw = contrato.get("planes")
    if not _es_secuencia_no_texto(planes_raw) or not planes_raw:
        raise ValueError("planes debe ser una secuencia no vacía.")

    planes = [
        _normalizar_plan(
            plan,
            seed_ejecucion=seed_ejecucion,
            cantidad_pedidos=cantidad_pedidos,
        )
        for plan in planes_raw
    ]

    if config.exigir_cinco_planes_ok:
        if len(planes) != 5:
            raise ValueError(
                f"Se esperaban 5 planes, se recibieron {len(planes)}."
            )

        no_ok = [
            plan["modo_solicitado"]
            for plan in planes
            if plan["estado"] != "OK"
        ]
        if no_ok:
            raise ValueError(
                "El contrato contiene planes no ejecutables: "
                + ", ".join(no_ok)
            )

    modos_planes = tuple(
        plan["modo_solicitado"]
        for plan in planes
    )
    if modos_planes != orden_modos:
        raise ValueError(
            "El orden de planes no coincide con orden_modos. "
            f"Planes={modos_planes}, orden_modos={orden_modos}."
        )

    ordenes = tuple(plan["orden"] for plan in planes)
    if ordenes != tuple(range(1, len(planes) + 1)):
        raise ValueError(
            f"Los planes deben tener orden consecutivo: {ordenes}."
        )

    return {
        "version_contrato": version,
        "instancia_id": instancia_id,
        "cantidad_pedidos": cantidad_pedidos,
        "seed_escenario": seed_escenario,
        "seed_ejecucion": seed_ejecucion,
        "instancia_vector": instancia_vector,
        "orden_modos": orden_modos,
        "planes": planes,
    }


def _normalizar_plan(
    plan: Any,
    *,
    seed_ejecucion: int,
    cantidad_pedidos: int,
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Cada plan debe ser un objeto JSON.")

    orden = _entero_positivo(plan.get("orden"), "plan.orden")
    modo = str(plan.get("modo_solicitado", "")).strip().upper()
    algoritmo = str(
        plan.get("algoritmo_resultante", "")
    ).strip().upper()
    estado = str(plan.get("estado", "")).strip().upper()

    if not modo:
        raise ValueError("modo_solicitado no puede estar vacío.")
    if estado == "OK" and not algoritmo:
        raise ValueError(
            f"{modo} está OK pero algoritmo_resultante está vacío."
        )

    seed_plan = plan.get("seed_planificacion")
    if seed_plan is not None:
        seed_plan = _entero_no_negativo(
            seed_plan,
            f"{modo}.seed_planificacion",
        )

    seed_plan_ejecucion = _entero_no_negativo(
        plan.get("seed_ejecucion"),
        f"{modo}.seed_ejecucion",
    )
    if seed_plan_ejecucion != seed_ejecucion:
        raise ValueError(
            f"{modo} usa seed_ejecucion={seed_plan_ejecucion}, "
            f"pero el contrato declara {seed_ejecucion}."
        )

    plan_vector = _vector_finito(
        plan.get("plan_vector"),
        f"{modo}.plan_vector",
    )

    asignaciones = _entero_positivo(
        plan_vector[1] if len(plan_vector) > 1 else None,
        f"{modo}.plan_vector.cantidadAsignaciones",
    )
    if asignaciones != cantidad_pedidos:
        raise ValueError(
            f"{modo} declara {asignaciones} asignaciones, "
            f"esperadas={cantidad_pedidos}."
        )

    camiones = plan.get("camiones")
    if not _es_secuencia_no_texto(camiones):
        raise ValueError(f"{modo}.camiones debe ser una secuencia.")

    return {
        "orden": orden,
        "modo_solicitado": modo,
        "algoritmo_resultante": algoritmo,
        "estado": estado,
        "fuente_seleccionada": str(
            plan.get("fuente_seleccionada", "")
        ).strip().upper(),
        "firma_ruta": str(plan.get("firma_ruta", "")).strip(),
        "seed_escenario": _entero_no_negativo(
            plan.get("seed_escenario"),
            f"{modo}.seed_escenario",
        ),
        "seed_planificacion": seed_plan,
        "seed_ejecucion": seed_plan_ejecucion,
        "costo_estimado": _flotante_opcional(
            plan.get("costo_estimado"),
            f"{modo}.costo_estimado",
        ),
        "tiempo_plan_ms": _flotante_opcional(
            plan.get("tiempo_plan_ms"),
            f"{modo}.tiempo_plan_ms",
        ),
        "tiempo_selector_ms": _flotante_opcional(
            plan.get("tiempo_selector_ms"),
            f"{modo}.tiempo_selector_ms",
        ),
        "camiones": list(camiones),
        "plan_vector": plan_vector,
    }


def _crear_registro_exitoso(
    *,
    plan: Mapping[str, Any],
    resultado: Any,
) -> RegistroEjecucionComparacion:
    observacion = dict(resultado.observacion_final)

    costo_real = float(observacion["costoTotal"])
    costo_estimado = plan["costo_estimado"]

    diferencia: float | None = None
    error_pct: float | None = None

    if costo_estimado is not None:
        diferencia = costo_real - costo_estimado
        if costo_estimado > 0.0:
            error_pct = diferencia / costo_estimado * 100.0

    return RegistroEjecucionComparacion(
        orden=plan["orden"],
        modo_solicitado=plan["modo_solicitado"],
        algoritmo_resultante=plan["algoritmo_resultante"],
        fuente_seleccionada=plan["fuente_seleccionada"],
        firma_ruta=plan["firma_ruta"],
        seed_escenario=plan["seed_escenario"],
        seed_planificacion=plan["seed_planificacion"],
        seed_ejecucion=plan["seed_ejecucion"],
        costo_estimado=costo_estimado,
        tiempo_plan_ms=plan["tiempo_plan_ms"],
        tiempo_selector_ms=plan["tiempo_selector_ms"],
        estado_ejecucion="OK",
        error_ejecucion="",
        costo_real=costo_real,
        diferencia_costo_real_estimado=diferencia,
        error_relativo_estimacion_pct=error_pct,
        tareas_entregadas=int(observacion["tareasEntregadas"]),
        tareas_no_entregadas=int(observacion["tareasNoEntregadas"]),
        viajes_totales=int(observacion["viajesTotales"]),
        tiempo_simulado_min=float(observacion["tiempoSimuladoMin"]),
        modelo=str(resultado.modelo),
        java=str(resultado.java),
        estado_final_motor=str(resultado.estado_final),
        stop_condition=bool(resultado.stop_condition),
        mensaje_anylogic=str(observacion.get("mensaje", "")),
    )


def _crear_registro_error(
    *,
    plan: Mapping[str, Any],
    exc: Exception,
) -> RegistroEjecucionComparacion:
    return RegistroEjecucionComparacion(
        orden=plan["orden"],
        modo_solicitado=plan["modo_solicitado"],
        algoritmo_resultante=plan["algoritmo_resultante"],
        fuente_seleccionada=plan["fuente_seleccionada"],
        firma_ruta=plan["firma_ruta"],
        seed_escenario=plan["seed_escenario"],
        seed_planificacion=plan["seed_planificacion"],
        seed_ejecucion=plan["seed_ejecucion"],
        costo_estimado=plan["costo_estimado"],
        tiempo_plan_ms=plan["tiempo_plan_ms"],
        tiempo_selector_ms=plan["tiempo_selector_ms"],
        estado_ejecucion="ERROR",
        error_ejecucion=(
            f"{exc.__class__.__name__}: {exc}"
        ),
        costo_real=None,
        diferencia_costo_real_estimado=None,
        error_relativo_estimacion_pct=None,
        tareas_entregadas=None,
        tareas_no_entregadas=None,
        viajes_totales=None,
        tiempo_simulado_min=None,
        modelo="",
        java="",
        estado_final_motor="",
        stop_condition=None,
        mensaje_anylogic="",
    )


def _contar_viajes(plan: Mapping[str, Any]) -> int:
    total = 0
    for camion in plan["camiones"]:
        if not isinstance(camion, dict):
            raise ValueError(
                f"{plan['modo_solicitado']}.camiones contiene un valor inválido."
            )
        viajes = camion.get("viajes")
        if not _es_secuencia_no_texto(viajes):
            raise ValueError(
                f"{plan['modo_solicitado']}.viajes debe ser una secuencia."
            )
        total += len(viajes)
    return total


def _registro_csv(
    registro: RegistroEjecucionComparacion,
) -> dict[str, Any]:
    return {
        "orden": registro.orden,
        "modo_solicitado": registro.modo_solicitado,
        "algoritmo_resultante": registro.algoritmo_resultante,
        "fuente_seleccionada": registro.fuente_seleccionada,
        "estado_ejecucion": registro.estado_ejecucion,
        "error_ejecucion": registro.error_ejecucion,
        "seed_escenario": registro.seed_escenario,
        "seed_planificacion": registro.seed_planificacion,
        "seed_ejecucion": registro.seed_ejecucion,
        "costo_estimado": registro.costo_estimado,
        "costo_real": registro.costo_real,
        "diferencia_costo_real_estimado": (
            registro.diferencia_costo_real_estimado
        ),
        "error_relativo_estimacion_pct": (
            registro.error_relativo_estimacion_pct
        ),
        "tiempo_plan_ms": registro.tiempo_plan_ms,
        "tiempo_selector_ms": registro.tiempo_selector_ms,
        "tiempo_simulado_min": registro.tiempo_simulado_min,
        "tareas_entregadas": registro.tareas_entregadas,
        "tareas_no_entregadas": registro.tareas_no_entregadas,
        "viajes_totales": registro.viajes_totales,
        "firma_ruta": registro.firma_ruta,
        "mensaje_anylogic": registro.mensaje_anylogic,
        "estado_final_motor": registro.estado_final_motor,
        "stop_condition": registro.stop_condition,
    }


def _entero_positivo(valor: Any, nombre: str) -> int:
    entero = _entero_no_negativo(valor, nombre)
    if entero <= 0:
        raise ValueError(f"{nombre} debe ser > 0.")
    return entero


def _entero_no_negativo(valor: Any, nombre: str) -> int:
    if isinstance(valor, bool) or valor is None:
        raise ValueError(f"{nombre} debe ser un entero no negativo.")

    flotante = float(valor)
    entero = int(round(flotante))

    if not isfinite(flotante) or abs(flotante - entero) > 1e-9:
        raise ValueError(f"{nombre} debe ser entero: {valor}.")

    if entero < 0:
        raise ValueError(f"{nombre} no puede ser negativo.")

    return entero


def _flotante_opcional(valor: Any, nombre: str) -> float | None:
    if valor is None:
        return None

    resultado = float(valor)
    if not isfinite(resultado):
        raise ValueError(f"{nombre} debe ser finito: {valor}.")
    return resultado



def _es_secuencia_no_texto(
    valor: object,
) -> TypeGuard[Sequence[Any]]:
    return isinstance(valor, SequenceABC) and not isinstance(
        valor,
        (str, bytes, bytearray),
    )

def _vector_finito(valor: Any, nombre: str) -> list[float]:
    if not _es_secuencia_no_texto(valor) or not valor:
        raise ValueError(f"{nombre} debe ser una secuencia no vacía.")

    resultado = [float(elemento) for elemento in valor]
    for indice, elemento in enumerate(resultado):
        if not isfinite(elemento):
            raise ValueError(
                f"{nombre}[{indice}] no es finito: {elemento}."
            )
    return resultado
