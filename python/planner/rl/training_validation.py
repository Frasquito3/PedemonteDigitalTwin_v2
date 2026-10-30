"""Validación externa y selección de checkpoints para políticas de RL.

Este módulo provee utilidades de auditoría y evaluación fuera de distribución
para modelos de RL en entrenamiento, ejecutando checkpoints contra la batería de
casos canónicos (B04) e instancias sintéticas complejas para seleccionar el mejor
modelo minimizando penalizaciones por tardanza y costos operativos.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import numpy as np

from planner.core.schema import InstanciaTurno
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.policy_env import EntornoPlanificacionRL
from planner.rl.reward import ConfiguracionRewardRL, ModoRewardRL
from planner.routing.travel import ProveedorViaje

__all__ = [
    "ResultadoCasoValidacionRL",
    "ResumenValidacionExternaRL",
    "clave_seleccion_externa",
    "ejecutar_validacion_instancia",
    "es_mejor_validacion_externa",
    "evaluar_modelo_externamente",
]


@dataclass(frozen=True)
class ResultadoCasoValidacionRL:
    """Resultado detallado de evaluación de un modelo en una instancia individual."""

    caso_id: str
    pedidos_tardios: int
    tardanza_total_min: float
    costo_estimado: float
    costo_greedy_referencia: float
    gap_costo_vs_greedy: float
    permutacion: tuple[str, ...]

    @property
    def sin_riesgo(self) -> bool:
        """Indica si el plan ejecutado no presentó pedidos tardíos."""
        return self.pedidos_tardios == 0


@dataclass(frozen=True)
class ResumenValidacionExternaRL:
    """Consolidado de métricas de validación externa sobre una batería de evaluación."""

    timestep: int
    b04_pedidos_tardios: int
    b04_tardanza_min: float
    b04_costo_estimado: float
    sinteticos_totales: int
    sinteticos_sin_riesgo: int
    tasa_sintetica_sin_riesgo_pct: float
    tardanza_sintetica_total_min: float
    tardanza_sintetica_mediana_min: float
    gap_costo_mediano_vs_greedy_pct: float
    casos: tuple[ResultadoCasoValidacionRL, ...]

    def como_dict(self) -> dict[str, Any]:
        """Serializa el resumen a un diccionario incluyendo la clave de selección."""
        contenido = asdict(self)
        contenido["clave_seleccion"] = list(clave_seleccion_externa(self))
        return contenido


def ejecutar_validacion_instancia(
    model: Any,
    instancia: InstanciaTurno,
    *,
    proveedor_viaje: ProveedorViaje | None,
    configuracion_temporal: ConfiguracionPoliticaRL,
) -> ResultadoCasoValidacionRL:
    """Evalúa un modelo en modo determinista sobre una única instancia dada."""
    reward = ConfiguracionRewardRL(
        modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
        denominador_relativo_minimo=1.0,
    )
    env = EntornoPlanificacionRL(
        instancia=instancia,
        proveedor_viaje=proveedor_viaje,
        configuracion_reward=reward,
        configuracion_temporal=configuracion_temporal,
        max_pedidos=30,
    )

    try:
        observacion, _ = env.reset(seed=instancia.seed_escenario)
        terminado = False
        info_final: dict[str, Any] = {}

        while not terminado:
            mascara = env.action_masks()
            accion, _ = model.predict(
                observacion,
                action_masks=mascara,
                deterministic=True,
            )
            accion_entera = int(np.asarray(accion).item())
            observacion, _, terminado, truncado, info = env.step(accion_entera)
            info_final = dict(info)

            if truncado:
                raise RuntimeError("La validación externa produjo un episodio truncado.")

        plan = env.ultimo_plan
        if plan is None:
            raise RuntimeError("La validación externa finalizó sin plan.")

        costo_greedy = float(info_final.get("costo_greedy_referencia", 0.0))
        if costo_greedy <= 0.0:
            raise RuntimeError("La validación externa no recibió costo Greedy de referencia.")

        costo = float(plan.costo_estimado)
        gap = 100.0 * (costo - costo_greedy) / costo_greedy

        return ResultadoCasoValidacionRL(
            caso_id=instancia.instancia_id,
            pedidos_tardios=int(info_final.get("pedidos_tardios_prefijo", -1)),
            tardanza_total_min=float(info_final.get("tardanza_prefijo_min", -1.0)),
            costo_estimado=costo,
            costo_greedy_referencia=costo_greedy,
            gap_costo_vs_greedy=gap,
            permutacion=tuple(env.permutacion_actual),
        )
    finally:
        env.close()


def evaluar_modelo_externamente(
    model: Any,
    *,
    timestep: int,
    instancia_b04: InstanciaTurno,
    proveedor_b04: ProveedorViaje,
    instancias_sinteticas: Iterable[InstanciaTurno],
    configuracion_temporal: ConfiguracionPoliticaRL,
) -> ResumenValidacionExternaRL:
    """Evalúa un modelo en B04 y un lote sintético para consolidar métricas de selección."""
    b04 = ejecutar_validacion_instancia(
        model,
        instancia_b04,
        proveedor_viaje=proveedor_b04,
        configuracion_temporal=configuracion_temporal,
    )
    sinteticos = tuple(
        ejecutar_validacion_instancia(
            model,
            instancia,
            proveedor_viaje=None,
            configuracion_temporal=configuracion_temporal,
        )
        for instancia in instancias_sinteticas
    )

    total = len(sinteticos)
    if total <= 0:
        raise ValueError("La validación externa requiere instancias sintéticas.")

    sin_riesgo = sum(1 for caso in sinteticos if caso.sin_riesgo)
    tardanzas = [caso.tardanza_total_min for caso in sinteticos]
    gaps = [caso.gap_costo_vs_greedy for caso in sinteticos]

    return ResumenValidacionExternaRL(
        timestep=int(timestep),
        b04_pedidos_tardios=b04.pedidos_tardios,
        b04_tardanza_min=b04.tardanza_total_min,
        b04_costo_estimado=b04.costo_estimado,
        sinteticos_totales=total,
        sinteticos_sin_riesgo=sin_riesgo,
        tasa_sintetica_sin_riesgo_pct=100.0 * sin_riesgo / total,
        tardanza_sintetica_total_min=sum(tardanzas),
        tardanza_sintetica_mediana_min=float(median(tardanzas)),
        gap_costo_mediano_vs_greedy_pct=float(median(gaps)),
        casos=(b04, *sinteticos),
    )


def clave_seleccion_externa(resumen: ResumenValidacionExternaRL) -> tuple[float, ...]:
    """Genera una tupla lexicográfica de criterios a minimizar para comparar checkpoints.

    Prioridad:
    1. Pedidos tardíos en B04 (criterio crítico de ventana horaria).
    2. Tardanza acumulada en minutos en B04.
    3. Tasa negativa de casos sintéticos sin riesgo (más casos sin riesgo es mejor).
    4. Tardanza total acumulada en casos sintéticos.
    5. Gap relativo mediano de costo respecto a la heurística voraz.
    """
    return (
        float(resumen.b04_pedidos_tardios),
        float(resumen.b04_tardanza_min),
        float(-resumen.sinteticos_sin_riesgo),
        float(resumen.tardanza_sintetica_total_min),
        float(resumen.gap_costo_mediano_vs_greedy_pct),
    )


def es_mejor_validacion_externa(
    candidata: ResumenValidacionExternaRL,
    actual: ResumenValidacionExternaRL | None,
) -> bool:
    """Determina si un checkpoint candidato supera al mejor checkpoint histórico registrado."""
    if actual is None:
        return True
    return clave_seleccion_externa(candidata) < clave_seleccion_externa(actual)
