from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from typing import Any, Iterable

from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.evaluation.classic_benchmark import firma_plan
from planner.integration.alpyne_codec import (
    codificar_instancia_alpyne,
    codificar_plan_alpyne,
)
from planner.integration.planner_selector import (
    DecisionSelector,
    ModoPlanificacion,
    SelectorPlanificadores,
)
from planner.routing.objective import (
    VERSION_AUDITORIA_COSTO,
    evaluar_plan_estimado,
)
from planner.routing.travel import (
    ProveedorViaje,
    construir_matriz_viaje,
)


VERSION_CONTRATO_COMPARACION = "comparacion-anylogic-v1"
ESTRATEGIA_SEMILLAS = "derivadas-de-seed-escenario-v1"

# RL se presenta primero porque es el algoritmo central del proyecto.
# El orden no cambia la lógica productiva ni el criterio del híbrido.
MODOS_COMPARACION: tuple[ModoPlanificacion, ...] = (
    ModoPlanificacion.RL,
    ModoPlanificacion.GA,
    ModoPlanificacion.GREEDY,
    ModoPlanificacion.RANDOM,
    ModoPlanificacion.HIBRIDO,
)


@dataclass(frozen=True)
class ConfiguracionContratoComparacion:
    tolerancia_costo: float = 1e-6
    exigir_sin_fallback: bool = True
    continuar_ante_error: bool = True

    def __post_init__(self) -> None:
        if self.tolerancia_costo < 0.0:
            raise ValueError(
                "tolerancia_costo no puede ser negativa."
            )


@dataclass(frozen=True)
class SemillaComponente:
    componente: str
    valor: int


@dataclass(frozen=True)
class ViajeContratoComparacion:
    numero_viaje: int
    pedido_ids: tuple[str, ...]


@dataclass(frozen=True)
class CamionContratoComparacion:
    camion_id: int
    viajes: tuple[ViajeContratoComparacion, ...]


@dataclass(frozen=True)
class RegistroPlanComparacion:
    orden: int
    modo_solicitado: str
    algoritmo_resultante: str
    estado: str
    error: str

    instancia_id: str
    seed_escenario: int
    seed_planificacion: int | None
    seed_ejecucion: int
    semillas_componentes: tuple[SemillaComponente, ...]

    plan_valido: bool
    costo_estimado: float | None
    tiempo_plan_ms: float | None
    tiempo_selector_ms: float | None
    firma_ruta: str
    advertencias: tuple[str, ...]

    fuente_seleccionada: str
    motivo_seleccion: str
    detalle_decision: str

    camiones: tuple[CamionContratoComparacion, ...]
    plan_vector: tuple[float, ...]


@dataclass(frozen=True)
class ContratoComparacionAnyLogic:
    version_contrato: str
    version_objetivo: str
    generado_utc: str
    estrategia_semillas: str

    instancia_id: str
    fecha_operacion: str
    turno: str
    cantidad_pedidos: int
    cantidad_camiones: int
    capacidad_camion: int

    seed_escenario: int
    seed_ejecucion: int

    fuente_viaje: str
    version_viaje: str
    fallbacks_matriz: int
    advertencias_matriz: tuple[str, ...]

    orden_modos: tuple[str, ...]
    planes_ok: int
    planes_error: int

    instancia_vector: tuple[float, ...]
    planes: tuple[RegistroPlanComparacion, ...]


def preparar_contrato_comparacion(
    instancia: InstanciaTurno,
    *,
    selector: SelectorPlanificadores,
    proveedor_viaje: ProveedorViaje,
    configuracion:
        ConfiguracionContratoComparacion
        | None = None,
    modos: Iterable[ModoPlanificacion | str] = MODOS_COMPARACION,
) -> ContratoComparacionAnyLogic:
    """
    Genera los planes comparables de una misma instancia.

    Todos los registros conservan exactamente la misma seed de escenario
    y la misma seed de ejecución. Las semillas estocásticas de cada
    planificador se derivan de seed_escenario con la convención productiva.

    El contrato contiene tanto la representación estructurada del plan como
    su vector Alpyne, de modo que la Fase 16B pueda ejecutar cada alternativa
    en AnyLogic sin reconstruir decisiones.
    """
    config = (
        configuracion
        if configuracion is not None
        else ConfiguracionContratoComparacion()
    )

    modos_normalizados = tuple(
        _normalizar_modo_comparacion(modo)
        for modo in modos
    )
    _validar_modos(modos_normalizados)

    matriz = construir_matriz_viaje(
        instancia,
        selector.configuracion,
        proveedor=proveedor_viaje,
    )

    if config.exigir_sin_fallback and matriz.usa_fallback:
        raise RuntimeError(
            "El contrato de comparación exige una matriz vial sin "
            f"fallbacks, pero se detectaron {matriz.cantidad_fallbacks}."
        )

    registros: list[RegistroPlanComparacion] = []

    for orden, modo in enumerate(modos_normalizados, start=1):
        try:
            plan = selector.generar_plan(instancia, modo)
            decision = selector.ultima_decision
            registro = _crear_registro_exitoso(
                orden=orden,
                modo=modo,
                instancia=instancia,
                plan=plan,
                decision=decision,
                proveedor_viaje=proveedor_viaje,
                selector=selector,
                tolerancia_costo=config.tolerancia_costo,
                exigir_sin_fallback=config.exigir_sin_fallback,
            )
        except Exception as exc:
            if not config.continuar_ante_error:
                raise

            registro = _crear_registro_error(
                orden=orden,
                modo=modo,
                instancia=instancia,
                exc=exc,
            )

        registros.append(registro)

    planes_ok = sum(
        1 for registro in registros
        if registro.estado == "OK"
    )

    return ContratoComparacionAnyLogic(
        version_contrato=VERSION_CONTRATO_COMPARACION,
        version_objetivo=VERSION_AUDITORIA_COSTO,
        generado_utc=datetime.now(timezone.utc).isoformat(),
        estrategia_semillas=ESTRATEGIA_SEMILLAS,
        instancia_id=instancia.instancia_id,
        fecha_operacion=instancia.fecha_operacion,
        turno=instancia.turno.value,
        cantidad_pedidos=len(instancia.pedidos),
        cantidad_camiones=instancia.cantidad_camiones,
        capacidad_camion=instancia.capacidad_camion,
        seed_escenario=instancia.seed_escenario,
        seed_ejecucion=instancia.seed_ejecucion,
        fuente_viaje=matriz.fuente.value,
        version_viaje=matriz.version_fuente,
        fallbacks_matriz=matriz.cantidad_fallbacks,
        advertencias_matriz=matriz.advertencias,
        orden_modos=tuple(modo.value for modo in modos_normalizados),
        planes_ok=planes_ok,
        planes_error=len(registros) - planes_ok,
        instancia_vector=tuple(codificar_instancia_alpyne(instancia)),
        planes=tuple(registros),
    )


def escribir_contrato_comparacion(
    contrato: ContratoComparacionAnyLogic,
    directorio_salida: str | Path,
) -> dict[str, Path]:
    salida = Path(directorio_salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ruta_json = salida / "comparison_contract.json"
    ruta_csv = salida / "comparison_plans.csv"

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(
            asdict(contrato),
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    filas = [_registro_resumen_csv(registro) for registro in contrato.planes]
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
        "contrato_json": ruta_json,
        "planes_csv": ruta_csv,
    }


def _crear_registro_exitoso(
    *,
    orden: int,
    modo: ModoPlanificacion,
    instancia: InstanciaTurno,
    plan: PlanTurno,
    decision: DecisionSelector | None,
    proveedor_viaje: ProveedorViaje,
    selector: SelectorPlanificadores,
    tolerancia_costo: float,
    exigir_sin_fallback: bool,
) -> RegistroPlanComparacion:
    validacion = validar_plan(instancia, plan)
    if not validacion.valido:
        raise RuntimeError(
            f"{modo.value} produjo un plan inválido: "
            + " | ".join(validacion.errores)
        )

    matriz = construir_matriz_viaje(
        instancia,
        selector.configuracion,
        proveedor=proveedor_viaje,
    )
    if exigir_sin_fallback and matriz.usa_fallback:
        raise RuntimeError(
            f"{modo.value} utilizó {matriz.cantidad_fallbacks} "
            "fallback(s) viales."
        )

    estimacion = evaluar_plan_estimado(
        instancia,
        plan,
        matriz,
        selector.configuracion,
    )

    if not isclose(
        plan.costo_estimado,
        estimacion.costo_total,
        rel_tol=1e-10,
        abs_tol=tolerancia_costo,
    ):
        raise RuntimeError(
            f"Costo inconsistente para {modo.value}: "
            f"plan={plan.costo_estimado}, "
            f"auditoria={estimacion.costo_total}."
        )

    fuente, motivo = _extraer_fuente_y_motivo(
        modo=modo,
        plan=plan,
        decision=decision,
    )

    return RegistroPlanComparacion(
        orden=orden,
        modo_solicitado=modo.value,
        algoritmo_resultante=plan.algoritmo.value,
        estado="OK",
        error="",
        instancia_id=instancia.instancia_id,
        seed_escenario=instancia.seed_escenario,
        seed_planificacion=_seed_principal(instancia, modo),
        seed_ejecucion=instancia.seed_ejecucion,
        semillas_componentes=_semillas_componentes(instancia, modo),
        plan_valido=True,
        costo_estimado=estimacion.costo_total,
        tiempo_plan_ms=plan.tiempo_computo_ms,
        tiempo_selector_ms=(
            decision.tiempo_selector_ms
            if decision is not None
            else None
        ),
        firma_ruta=firma_plan(plan),
        advertencias=tuple(plan.warnings),
        fuente_seleccionada=fuente,
        motivo_seleccion=motivo,
        detalle_decision=(
            decision.detalle
            if decision is not None
            else ""
        ),
        camiones=_serializar_camiones(plan),
        plan_vector=tuple(codificar_plan_alpyne(instancia, plan)),
    )


def _crear_registro_error(
    *,
    orden: int,
    modo: ModoPlanificacion,
    instancia: InstanciaTurno,
    exc: Exception,
) -> RegistroPlanComparacion:
    return RegistroPlanComparacion(
        orden=orden,
        modo_solicitado=modo.value,
        algoritmo_resultante="",
        estado="ERROR",
        error=f"{type(exc).__name__}: {exc}",
        instancia_id=instancia.instancia_id,
        seed_escenario=instancia.seed_escenario,
        seed_planificacion=_seed_principal(instancia, modo),
        seed_ejecucion=instancia.seed_ejecucion,
        semillas_componentes=_semillas_componentes(instancia, modo),
        plan_valido=False,
        costo_estimado=None,
        tiempo_plan_ms=None,
        tiempo_selector_ms=None,
        firma_ruta="",
        advertencias=(),
        fuente_seleccionada="",
        motivo_seleccion="",
        detalle_decision="",
        camiones=(),
        plan_vector=(),
    )


def _serializar_camiones(
    plan: PlanTurno,
) -> tuple[CamionContratoComparacion, ...]:
    return tuple(
        CamionContratoComparacion(
            camion_id=camion.camion_id,
            viajes=tuple(
                ViajeContratoComparacion(
                    numero_viaje=viaje.numero_viaje,
                    pedido_ids=tuple(viaje.pedido_ids),
                )
                for viaje in sorted(
                    camion.viajes,
                    key=lambda actual: actual.numero_viaje,
                )
            ),
        )
        for camion in sorted(
            plan.camiones,
            key=lambda actual: actual.camion_id,
        )
    )


def _extraer_fuente_y_motivo(
    *,
    modo: ModoPlanificacion,
    plan: PlanTurno,
    decision: DecisionSelector | None,
) -> tuple[str, str]:
    if modo != ModoPlanificacion.HIBRIDO:
        return plan.algoritmo.value, "MODO_DIRECTO"

    detalle = decision.detalle if decision is not None else ""
    campos = _campos_detalle(detalle)

    # En el híbrido actual, ``resultado`` identifica qué plan terminó
    # ejecutándose: la semilla RL o el refinamiento GA. ``fuente_rl`` indica
    # solamente qué checkpoint originó la semilla y queda preservado dentro
    # de detalle_decision.
    resultado = campos.get("resultado", "").strip().upper()
    if resultado == "REFINADO_GA":
        fuente = "GA"
    elif resultado == "SEMILLA_RL":
        fuente = "RL"
    else:
        # Compatibilidad de lectura con contratos históricos y defensa ante
        # decisiones incompletas.
        fuente = campos.get("fuente", plan.algoritmo.value)

    return fuente, campos.get("motivo", "")


def _campos_detalle(detalle: str) -> dict[str, str]:
    campos: dict[str, str] = {}
    for parte in detalle.split("|"):
        if "=" not in parte:
            continue
        clave, valor = parte.split("=", 1)
        clave = clave.strip()
        if clave:
            campos[clave] = valor.strip()
    return campos


def _seed_principal(
    instancia: InstanciaTurno,
    modo: ModoPlanificacion,
) -> int | None:
    if modo == ModoPlanificacion.GREEDY:
        return None
    if modo == ModoPlanificacion.RANDOM:
        return instancia.seed_escenario + 7001
    if modo == ModoPlanificacion.GA:
        return instancia.seed_escenario + 8001

    # RL usa seed_escenario al reiniciar el entorno. El híbrido queda
    # anclado a la misma seed y deriva internamente la seed de GA.
    return instancia.seed_escenario


def _semillas_componentes(
    instancia: InstanciaTurno,
    modo: ModoPlanificacion,
) -> tuple[SemillaComponente, ...]:
    if modo == ModoPlanificacion.RL:
        return (
            SemillaComponente("RL", instancia.seed_escenario),
        )
    if modo == ModoPlanificacion.GA:
        return (
            SemillaComponente("GA", instancia.seed_escenario + 8001),
        )
    if modo == ModoPlanificacion.RANDOM:
        return (
            SemillaComponente(
                "RANDOM",
                instancia.seed_escenario + 7001,
            ),
        )
    if modo == ModoPlanificacion.HIBRIDO:
        return (
            SemillaComponente("RL", instancia.seed_escenario),
            SemillaComponente("GA", instancia.seed_escenario + 8001),
        )
    return ()


def _normalizar_modo_comparacion(
    modo: ModoPlanificacion | str,
) -> ModoPlanificacion:
    if isinstance(modo, ModoPlanificacion):
        return modo

    texto = str(modo).strip().upper()
    try:
        return ModoPlanificacion(texto)
    except ValueError as exc:
        raise ValueError(
            f"Modo de comparación no soportado: {modo!r}."
        ) from exc


def _validar_modos(
    modos: tuple[ModoPlanificacion, ...],
) -> None:
    if not modos:
        raise ValueError(
            "El contrato requiere al menos un modo de planificación."
        )
    if len(modos) != len(set(modos)):
        raise ValueError(
            "Los modos del contrato no pueden repetirse."
        )


def _registro_resumen_csv(
    registro: RegistroPlanComparacion,
) -> dict[str, Any]:
    return {
        "orden": registro.orden,
        "modo_solicitado": registro.modo_solicitado,
        "algoritmo_resultante": registro.algoritmo_resultante,
        "estado": registro.estado,
        "error": registro.error,
        "instancia_id": registro.instancia_id,
        "seed_escenario": registro.seed_escenario,
        "seed_planificacion": registro.seed_planificacion,
        "seed_ejecucion": registro.seed_ejecucion,
        "semillas_componentes": "|".join(
            f"{semilla.componente}={semilla.valor}"
            for semilla in registro.semillas_componentes
        ),
        "plan_valido": registro.plan_valido,
        "costo_estimado": registro.costo_estimado,
        "tiempo_plan_ms": registro.tiempo_plan_ms,
        "tiempo_selector_ms": registro.tiempo_selector_ms,
        "firma_ruta": registro.firma_ruta,
        "advertencias": " | ".join(registro.advertencias),
        "fuente_seleccionada": registro.fuente_seleccionada,
        "motivo_seleccion": registro.motivo_seleccion,
        "detalle_decision": registro.detalle_decision,
    }
