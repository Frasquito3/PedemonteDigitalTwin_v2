"""Estimación de consecuencias de segundo orden y arrepentimiento local para la política RL."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, tanh

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PedidoInput
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.schedule_estimator import (
    RegistroTemporalPedido,
    ResumenTemporalPrefijo,
    analizar_prefijo_temporal,
    proyectar_acciones_pendientes,
)
from planner.routing.travel import MatrizViaje

__all__ = [
    "ConsecuenciaAccionRL",
    "ResultadoArrepentimientoLocal",
    "ResultadoTerminalRL",
    "completar_prefijo_temporal",
    "proyectar_consecuencias_segundo_orden",
    "clave_lexicografica_consecuencia",
    "seleccionar_mejor_consecuencia",
    "calcular_arrepentimiento_local",
    "calcular_reward_terminal",
]


@dataclass(frozen=True)
class ConsecuenciaAccionRL:
    """Consecuencia proyectada de elegir una acción y completar los pedidos pendientes."""

    pedido_id: str
    registro_inmediato: RegistroTemporalPedido
    secuencia_completada: tuple[str, ...]
    resumen_final: ResumenTemporalPrefijo

    pedidos_tardios_finales: int
    tardanza_total_final_min: float
    pedidos_nuevos_en_riesgo: int
    perdida_holgura_total_min: float
    holgura_minima_final_min: float
    espera_apertura_total_final_min: float
    duracion_operacion_final_min: float

    @property
    def sin_riesgo_final(self) -> bool:
        return self.pedidos_tardios_finales == 0


@dataclass(frozen=True)
class ResultadoArrepentimientoLocal:
    """Diferencial de desempeño entre la acción ejecutada y la mejor alternativa local."""

    pedido_elegido_id: str
    mejor_pedido_id: str
    arrepentimiento_normalizado: float
    reward_local: float
    es_mejor_accion: bool


@dataclass(frozen=True)
class ResultadoTerminalRL:
    """Componentes desglosados de la recompensa escalar terminal de un episodio."""

    factible_temporalmente: bool
    pedidos_tardios: int
    tardanza_total_min: float
    componente_factibilidad: float
    componente_costo_acotado: float
    reward_terminal_total: float


def _pedidos_por_id(instancia: InstanciaTurno) -> dict[str, PedidoInput]:
    return {pedido.pedido_id: pedido for pedido in instancia.pedidos}


def completar_prefijo_temporal(
    instancia: InstanciaTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    prefijo: Iterable[str],
    pendientes: Iterable[str],
) -> tuple[tuple[str, ...], ResumenTemporalPrefijo]:
    """Completa el prefijo con una regla EDD (Earliest Due Date) reproducible.

    Ordena los pendientes por cierre y apertura de ventana y simula la secuencia completa.
    Permite evaluar el impacto a largo plazo de la acción inmediata.
    """
    secuencia = list(prefijo)
    restantes = list(pendientes)
    pedidos = _pedidos_por_id(instancia)

    if len(restantes) != len(set(restantes)):
        raise ValueError("pendientes contiene pedidos repetidos.")

    restantes_ordenados = sorted(
        restantes,
        key=lambda p_id: (
            pedidos[p_id].hora_hasta_min,
            pedidos[p_id].hora_desde_min,
            not pedidos[p_id].tiene_ventana_especifica,
            p_id,
        ),
    )
    secuencia.extend(restantes_ordenados)
    resumen_final = analizar_prefijo_temporal(
        instancia,
        matriz,
        configuracion,
        secuencia,
    )
    return tuple(secuencia), resumen_final


def proyectar_consecuencias_segundo_orden(
    instancia: InstanciaTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    prefijo: Iterable[str],
) -> dict[str, ConsecuenciaAccionRL]:
    """Proyecta cada acción pendiente más allá de su llegada inmediata.

    Para cada candidato se simula esa decisión seguida del completado heurístico EDD,
    visibilizando si una espera aparentemente inocua generará retrasos en pedidos posteriores.
    """
    prefijo_tuple = tuple(prefijo)
    seleccionados = set(prefijo_tuple)
    pendientes = [
        pedido.pedido_id for pedido in instancia.pedidos if pedido.pedido_id not in seleccionados
    ]

    if not pendientes:
        return {}

    proyecciones_base = proyectar_acciones_pendientes(
        instancia,
        matriz,
        configuracion,
        prefijo_tuple,
    )
    consecuencias: dict[str, ConsecuenciaAccionRL] = {}

    for candidato_id in pendientes:
        resumen_inmediato = analizar_prefijo_temporal(
            instancia,
            matriz,
            configuracion,
            (*prefijo_tuple, candidato_id),
        )
        registro_inmediato = resumen_inmediato.registro_de(candidato_id)
        if registro_inmediato is None:
            raise RuntimeError(f"No se pudo recuperar la proyección inmediata de {candidato_id}.")

        restantes = [p_id for p_id in pendientes if p_id != candidato_id]
        secuencia_final, resumen_final = completar_prefijo_temporal(
            instancia,
            matriz,
            configuracion,
            (*prefijo_tuple, candidato_id),
            restantes,
        )

        nuevos_en_riesgo = 0
        perdida_holgura = 0.0
        holguras_finales: list[float] = []

        for pedido_id in pendientes:
            registro_final = resumen_final.registro_de(pedido_id)
            if registro_final is None:
                raise RuntimeError(f"La secuencia completada no contiene el pedido {pedido_id}.")

            holguras_finales.append(registro_final.holgura_llegada_min)
            if pedido_id == candidato_id:
                continue

            proyeccion_base = proyecciones_base[pedido_id]
            registro_base = proyeccion_base.registro

            if not registro_base.llegada_tardia and registro_final.llegada_tardia:
                nuevos_en_riesgo += 1

            perdida_holgura += max(
                0.0,
                registro_base.holgura_llegada_min - registro_final.holgura_llegada_min,
            )

        duracion = max(
            0.0,
            resumen_final.minuto_referencia - instancia.hora_inicio_turno_min,
        )
        holgura_minima = min(holguras_finales, default=0.0)

        valores = (
            resumen_final.tardanza_total_min,
            perdida_holgura,
            holgura_minima,
            resumen_final.espera_apertura_total_min,
            duracion,
        )
        if not all(isfinite(valor) for valor in valores):
            raise RuntimeError("La proyección de la política RL produjo valores no finitos.")

        consecuencias[candidato_id] = ConsecuenciaAccionRL(
            pedido_id=candidato_id,
            registro_inmediato=registro_inmediato,
            secuencia_completada=secuencia_final,
            resumen_final=resumen_final,
            pedidos_tardios_finales=resumen_final.pedidos_tardios,
            tardanza_total_final_min=resumen_final.tardanza_total_min,
            pedidos_nuevos_en_riesgo=nuevos_en_riesgo,
            perdida_holgura_total_min=perdida_holgura,
            holgura_minima_final_min=holgura_minima,
            espera_apertura_total_final_min=resumen_final.espera_apertura_total_min,
            duracion_operacion_final_min=duracion,
        )

    return consecuencias


def clave_lexicografica_consecuencia(
    consecuencia: ConsecuenciaAccionRL,
) -> tuple[float | str, ...]:
    """Clave lexicográfica para comparar la calidad operativa de dos consecuencias."""
    return (
        float(consecuencia.pedidos_tardios_finales),
        float(consecuencia.tardanza_total_final_min),
        float(consecuencia.pedidos_nuevos_en_riesgo),
        float(consecuencia.perdida_holgura_total_min),
        float(consecuencia.espera_apertura_total_final_min),
        float(consecuencia.duracion_operacion_final_min),
        consecuencia.pedido_id,
    )


def seleccionar_mejor_consecuencia(
    consecuencias: dict[str, ConsecuenciaAccionRL],
) -> ConsecuenciaAccionRL:
    """Selecciona la mejor consecuencia disponible mediante ordenamiento lexicográfico."""
    if not consecuencias:
        raise ValueError("No hay consecuencias candidatas.")
    return min(consecuencias.values(), key=clave_lexicografica_consecuencia)


def _diferencia_normalizada(elegido: float, mejor: float, escala: float) -> float:
    return min(1.0, max(0.0, elegido - mejor) / escala)


def calcular_arrepentimiento_local(
    consecuencias: dict[str, ConsecuenciaAccionRL],
    pedido_elegido_id: str,
    configuracion: ConfiguracionPoliticaRL,
) -> ResultadoArrepentimientoLocal:
    """Calcula la penalización por arrepentimiento local frente a la mejor alternativa."""
    try:
        elegido = consecuencias[pedido_elegido_id]
    except KeyError as exc:
        raise ValueError(
            f"La acción elegida no pertenece a las consecuencias: {pedido_elegido_id}."
        ) from exc

    mejor = seleccionar_mejor_consecuencia(consecuencias)
    cantidad = max(1, len(consecuencias))

    diferencia_tardios = min(
        1.0,
        max(0, elegido.pedidos_tardios_finales - mejor.pedidos_tardios_finales) / cantidad,
    )
    diferencia_tardanza = _diferencia_normalizada(
        elegido.tardanza_total_final_min,
        mejor.tardanza_total_final_min,
        configuracion.escala_tardanza_min,
    )
    diferencia_nuevos_riesgos = min(
        1.0,
        max(0, elegido.pedidos_nuevos_en_riesgo - mejor.pedidos_nuevos_en_riesgo) / cantidad,
    )
    diferencia_holgura = _diferencia_normalizada(
        elegido.perdida_holgura_total_min,
        mejor.perdida_holgura_total_min,
        configuracion.escala_perdida_holgura_min,
    )
    diferencia_espera = _diferencia_normalizada(
        elegido.espera_apertura_total_final_min,
        mejor.espera_apertura_total_final_min,
        configuracion.escala_espera_min,
    )

    arrepentimiento = min(
        1.0,
        configuracion.peso_arrepentimiento_pedidos_tardios * diferencia_tardios
        + configuracion.peso_arrepentimiento_tardanza * diferencia_tardanza
        + configuracion.peso_arrepentimiento_nuevos_riesgos * diferencia_nuevos_riesgos
        + configuracion.peso_arrepentimiento_holgura * diferencia_holgura
        + configuracion.peso_arrepentimiento_espera * diferencia_espera,
    )

    es_mejor = arrepentimiento <= configuracion.epsilon_mejor_accion
    reward_local = (
        configuracion.bonificacion_mejor_accion_local
        if es_mejor
        else -configuracion.penalizacion_arrepentimiento_max * arrepentimiento
    )

    return ResultadoArrepentimientoLocal(
        pedido_elegido_id=pedido_elegido_id,
        mejor_pedido_id=mejor.pedido_id,
        arrepentimiento_normalizado=arrepentimiento,
        reward_local=reward_local,
        es_mejor_accion=es_mejor,
    )


def calcular_reward_terminal(
    resumen_final: ResumenTemporalPrefijo,
    reward_costo_base: float,
    configuracion: ConfiguracionPoliticaRL,
) -> ResultadoTerminalRL:
    """Convierte la tupla temporal-costo en bandas escalares separadas."""
    componente_costo = configuracion.peso_terminal_costo_acotado * tanh(float(reward_costo_base))

    if resumen_final.pedidos_tardios == 0:
        componente_factibilidad = configuracion.bonificacion_terminal_factible
        total = componente_factibilidad + componente_costo
        factible = True
    else:
        penalizacion_tardanza = configuracion.penalizacion_terminal_tardanza_max * tanh(
            resumen_final.tardanza_total_min / configuracion.escala_tardanza_min
        )
        componente_factibilidad = -(
            configuracion.penalizacion_terminal_no_factible
            + configuracion.penalizacion_terminal_por_pedido_tardio * resumen_final.pedidos_tardios
            + penalizacion_tardanza
        )
        total = componente_factibilidad + componente_costo
        factible = False

    if not isfinite(total):
        raise RuntimeError("El reward terminal no es finito.")

    return ResultadoTerminalRL(
        factible_temporalmente=factible,
        pedidos_tardios=resumen_final.pedidos_tardios,
        tardanza_total_min=resumen_final.tardanza_total_min,
        componente_factibilidad=componente_factibilidad,
        componente_costo_acotado=componente_costo,
        reward_terminal_total=total,
    )
