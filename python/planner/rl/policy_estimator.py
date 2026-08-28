from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh
from typing import Iterable

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PedidoInput
from planner.rl.policy_config import ConfiguracionTemporalV4RL
from planner.rl.temporal_estimator import (
    ProyeccionTemporalAccion,
    RegistroTemporalPedido,
    ResumenTemporalPrefijo,
    analizar_prefijo_temporal,
    proyectar_acciones_pendientes,
)
from planner.routing.travel import MatrizViaje


@dataclass(frozen=True)
class ConsecuenciaTemporalAccionV4:
    """Consecuencia de elegir una acción y completar los pendientes."""

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
class ResultadoArrepentimientoLocalV4:
    pedido_elegido_id: str
    mejor_pedido_id: str
    arrepentimiento_normalizado: float
    reward_local: float
    es_mejor_accion: bool


@dataclass(frozen=True)
class ResultadoTerminalV4:
    factible_temporalmente: bool
    pedidos_tardios: int
    tardanza_total_min: float
    componente_factibilidad: float
    componente_costo_acotado: float
    reward_terminal_total: float


def _pedidos_por_id(
    instancia: InstanciaTurno,
) -> dict[str, PedidoInput]:
    return {
        pedido.pedido_id: pedido
        for pedido in instancia.pedidos
    }


def completar_prefijo_temporal_v4(
    instancia: InstanciaTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    prefijo: Iterable[str],
    pendientes: Iterable[str],
) -> tuple[tuple[str, ...], ResumenTemporalPrefijo]:
    """
    Completa el prefijo con una regla EDD reproducible.

    Todos los pendientes se ordenan por cierre y apertura de ventana.
    Luego se simula cronológicamente la secuencia completa una sola vez.
    Esta aproximación mantiene visible la consecuencia futura de cada
    candidato sin introducir una búsqueda combinatoria dentro de cada
    paso de entrenamiento.
    """

    secuencia = list(prefijo)
    restantes = list(pendientes)
    pedidos = _pedidos_por_id(instancia)

    if len(restantes) != len(set(restantes)):
        raise ValueError("pendientes contiene pedidos repetidos.")

    restantes_ordenados = sorted(
        restantes,
        key=lambda pedido_id: (
            pedidos[pedido_id].hora_hasta_min,
            pedidos[pedido_id].hora_desde_min,
            not pedidos[pedido_id].tiene_ventana_especifica,
            pedido_id,
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


def proyectar_consecuencias_segundo_orden_v4(
    instancia: InstanciaTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    prefijo: Iterable[str],
) -> dict[str, ConsecuenciaTemporalAccionV4]:
    """
    Proyecta cada acción pendiente más allá de su llegada inmediata.

    Para cada candidato se fija primero esa decisión y luego se completan
    todos los pedidos restantes con la misma heurística temporal. Así se
    vuelve observable si una espera aparentemente inocua deja pedidos
    posteriores fuera de ventana, como ocurre en B04 al elegir la ventana
    tardía antes de la ventana media.
    """

    prefijo_tuple = tuple(prefijo)
    seleccionados = set(prefijo_tuple)
    pendientes = [
        pedido.pedido_id
        for pedido in instancia.pedidos
        if pedido.pedido_id not in seleccionados
    ]

    if not pendientes:
        return {}

    proyecciones_base = proyectar_acciones_pendientes(
        instancia,
        matriz,
        configuracion,
        prefijo_tuple,
    )
    consecuencias: dict[str, ConsecuenciaTemporalAccionV4] = {}

    for candidato_id in pendientes:
        resumen_inmediato = analizar_prefijo_temporal(
            instancia,
            matriz,
            configuracion,
            (*prefijo_tuple, candidato_id),
        )
        registro_inmediato = resumen_inmediato.registro_de(candidato_id)

        if registro_inmediato is None:
            raise RuntimeError(
                "No se pudo recuperar la proyección inmediata de "
                f"{candidato_id}."
            )

        restantes = [
            pedido_id
            for pedido_id in pendientes
            if pedido_id != candidato_id
        ]
        secuencia_final, resumen_final = completar_prefijo_temporal_v4(
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
                raise RuntimeError(
                    "La secuencia completada no contiene el pedido "
                    f"{pedido_id}."
                )

            holguras_finales.append(registro_final.holgura_llegada_min)

            if pedido_id == candidato_id:
                continue

            proyeccion_base: ProyeccionTemporalAccion = (
                proyecciones_base[pedido_id]
            )
            registro_base = proyeccion_base.registro

            if (
                not registro_base.llegada_tardia
                and registro_final.llegada_tardia
            ):
                nuevos_en_riesgo += 1

            perdida_holgura += max(
                0.0,
                registro_base.holgura_llegada_min
                - registro_final.holgura_llegada_min,
            )

        duracion = max(
            0.0,
            resumen_final.minuto_referencia
            - instancia.hora_inicio_turno_min,
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
            raise RuntimeError(
                "La proyección temporal v4 produjo valores no finitos."
            )

        consecuencias[candidato_id] = ConsecuenciaTemporalAccionV4(
            pedido_id=candidato_id,
            registro_inmediato=registro_inmediato,
            secuencia_completada=secuencia_final,
            resumen_final=resumen_final,
            pedidos_tardios_finales=resumen_final.pedidos_tardios,
            tardanza_total_final_min=resumen_final.tardanza_total_min,
            pedidos_nuevos_en_riesgo=nuevos_en_riesgo,
            perdida_holgura_total_min=perdida_holgura,
            holgura_minima_final_min=holgura_minima,
            espera_apertura_total_final_min=(
                resumen_final.espera_apertura_total_min
            ),
            duracion_operacion_final_min=duracion,
        )

    return consecuencias


def clave_lexicografica_consecuencia_v4(
    consecuencia: ConsecuenciaTemporalAccionV4,
) -> tuple[float | str, ...]:
    return (
        float(consecuencia.pedidos_tardios_finales),
        float(consecuencia.tardanza_total_final_min),
        float(consecuencia.pedidos_nuevos_en_riesgo),
        float(consecuencia.perdida_holgura_total_min),
        float(consecuencia.espera_apertura_total_final_min),
        float(consecuencia.duracion_operacion_final_min),
        consecuencia.pedido_id,
    )


def seleccionar_mejor_consecuencia_v4(
    consecuencias: dict[str, ConsecuenciaTemporalAccionV4],
) -> ConsecuenciaTemporalAccionV4:
    if not consecuencias:
        raise ValueError("No hay consecuencias candidatas.")

    return min(
        consecuencias.values(),
        key=clave_lexicografica_consecuencia_v4,
    )


def _diferencia_normalizada(
    elegido: float,
    mejor: float,
    escala: float,
) -> float:
    return min(1.0, max(0.0, elegido - mejor) / escala)


def calcular_arrepentimiento_local_v4(
    consecuencias: dict[str, ConsecuenciaTemporalAccionV4],
    pedido_elegido_id: str,
    configuracion: ConfiguracionTemporalV4RL,
) -> ResultadoArrepentimientoLocalV4:
    try:
        elegido = consecuencias[pedido_elegido_id]
    except KeyError as exc:
        raise ValueError(
            "La acción elegida no pertenece a las consecuencias: "
            f"{pedido_elegido_id}."
        ) from exc

    mejor = seleccionar_mejor_consecuencia_v4(consecuencias)
    cantidad = max(1, len(consecuencias))

    diferencia_tardios = min(
        1.0,
        max(
            0,
            elegido.pedidos_tardios_finales
            - mejor.pedidos_tardios_finales,
        )
        / cantidad,
    )
    diferencia_tardanza = _diferencia_normalizada(
        elegido.tardanza_total_final_min,
        mejor.tardanza_total_final_min,
        configuracion.escala_tardanza_min,
    )
    diferencia_nuevos_riesgos = min(
        1.0,
        max(
            0,
            elegido.pedidos_nuevos_en_riesgo
            - mejor.pedidos_nuevos_en_riesgo,
        )
        / cantidad,
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
        configuracion.peso_arrepentimiento_pedidos_tardios
        * diferencia_tardios
        + configuracion.peso_arrepentimiento_tardanza
        * diferencia_tardanza
        + configuracion.peso_arrepentimiento_nuevos_riesgos
        * diferencia_nuevos_riesgos
        + configuracion.peso_arrepentimiento_holgura
        * diferencia_holgura
        + configuracion.peso_arrepentimiento_espera
        * diferencia_espera,
    )

    # Si todas las métricas empatan, no se penaliza una acción sólo por
    # el desempate alfabético usado para hacer reproducible la selección.
    es_mejor = (
        arrepentimiento <= configuracion.epsilon_mejor_accion
    )
    reward_local = (
        configuracion.bonificacion_mejor_accion_local
        if es_mejor
        else -configuracion.penalizacion_arrepentimiento_max
        * arrepentimiento
    )

    return ResultadoArrepentimientoLocalV4(
        pedido_elegido_id=pedido_elegido_id,
        mejor_pedido_id=mejor.pedido_id,
        arrepentimiento_normalizado=arrepentimiento,
        reward_local=reward_local,
        es_mejor_accion=es_mejor,
    )


def calcular_reward_terminal_v4(
    resumen_final: ResumenTemporalPrefijo,
    reward_costo_base: float,
    configuracion: ConfiguracionTemporalV4RL,
) -> ResultadoTerminalV4:
    """Convierte la tupla temporal-costo en bandas escalares separadas."""

    componente_costo = (
        configuracion.peso_terminal_costo_acotado
        * tanh(float(reward_costo_base))
    )

    if resumen_final.pedidos_tardios == 0:
        componente_factibilidad = (
            configuracion.bonificacion_terminal_factible
        )
        total = componente_factibilidad + componente_costo
        factible = True
    else:
        penalizacion_tardanza = (
            configuracion.penalizacion_terminal_tardanza_max
            * tanh(
                resumen_final.tardanza_total_min
                / configuracion.escala_tardanza_min
            )
        )
        componente_factibilidad = -(
            configuracion.penalizacion_terminal_no_factible
            + configuracion.penalizacion_terminal_por_pedido_tardio
            * resumen_final.pedidos_tardios
            + penalizacion_tardanza
        )
        total = componente_factibilidad + componente_costo
        factible = False

    if not isfinite(total):
        raise RuntimeError("El reward terminal v4 no es finito.")

    return ResultadoTerminalV4(
        factible_temporalmente=factible,
        pedidos_tardios=resumen_final.pedidos_tardios,
        tardanza_total_min=resumen_final.tardanza_total_min,
        componente_factibilidad=componente_factibilidad,
        componente_costo_acotado=componente_costo,
        reward_terminal_total=total,
    )
