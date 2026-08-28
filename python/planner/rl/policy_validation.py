from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable

import numpy as np

from planner.core.schema import InstanciaTurno
from planner.rl.rl_reward import ConfiguracionRewardRL, ModoRewardRL
from planner.rl.policy_config import ConfiguracionTemporalV4RL
from planner.rl.policy_env import PedemonteTemporalV4PlanEnv
from planner.routing.travel import ProveedorViaje


@dataclass(frozen=True)
class ResultadoCasoValidacionV4:
    caso_id: str
    pedidos_tardios: int
    tardanza_total_min: float
    costo_estimado: float
    costo_greedy_referencia: float
    gap_costo_vs_greedy: float
    permutacion: tuple[str, ...]

    @property
    def sin_riesgo(self) -> bool:
        return self.pedidos_tardios == 0


@dataclass(frozen=True)
class ResumenValidacionExternaV4:
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
    casos: tuple[ResultadoCasoValidacionV4, ...]

    def como_dict(self) -> dict[str, Any]:
        contenido = asdict(self)
        contenido["clave_seleccion"] = list(
            clave_seleccion_externa_v4(self)
        )
        return contenido


def ejecutar_validacion_instancia_v4(
    model: Any,
    instancia: InstanciaTurno,
    *,
    proveedor_viaje: ProveedorViaje | None,
    configuracion_temporal: ConfiguracionTemporalV4RL,
) -> ResultadoCasoValidacionV4:
    reward = ConfiguracionRewardRL(
        modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
        denominador_relativo_minimo=1.0,
    )
    env = PedemonteTemporalV4PlanEnv(
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
            (
                observacion,
                _,
                terminado,
                truncado,
                info,
            ) = env.step(accion_entera)
            info_final = dict(info)

            if truncado:
                raise RuntimeError(
                    "La validación externa v4 produjo un episodio truncado."
                )

        plan = env.ultimo_plan
        if plan is None:
            raise RuntimeError(
                "La validación externa v4 finalizó sin plan."
            )

        costo_greedy = float(
            info_final.get("costo_greedy_referencia", 0.0)
        )
        if costo_greedy <= 0.0:
            raise RuntimeError(
                "La validación externa v4 no recibió costo Greedy."
            )

        costo = float(plan.costo_estimado)
        gap = 100.0 * (costo - costo_greedy) / costo_greedy

        return ResultadoCasoValidacionV4(
            caso_id=instancia.instancia_id,
            pedidos_tardios=int(
                info_final.get("pedidos_tardios_prefijo", -1)
            ),
            tardanza_total_min=float(
                info_final.get("tardanza_prefijo_min", -1.0)
            ),
            costo_estimado=costo,
            costo_greedy_referencia=costo_greedy,
            gap_costo_vs_greedy=gap,
            permutacion=tuple(env.permutacion_actual),
        )
    finally:
        env.close()


def evaluar_modelo_externamente_v4(
    model: Any,
    *,
    timestep: int,
    instancia_b04: InstanciaTurno,
    proveedor_b04: ProveedorViaje,
    instancias_sinteticas: Iterable[InstanciaTurno],
    configuracion_temporal: ConfiguracionTemporalV4RL,
) -> ResumenValidacionExternaV4:
    b04 = ejecutar_validacion_instancia_v4(
        model,
        instancia_b04,
        proveedor_viaje=proveedor_b04,
        configuracion_temporal=configuracion_temporal,
    )
    sinteticos = tuple(
        ejecutar_validacion_instancia_v4(
            model,
            instancia,
            proveedor_viaje=None,
            configuracion_temporal=configuracion_temporal,
        )
        for instancia in instancias_sinteticas
    )

    total = len(sinteticos)
    if total <= 0:
        raise ValueError(
            "La validación externa requiere instancias sintéticas."
        )

    sin_riesgo = sum(1 for caso in sinteticos if caso.sin_riesgo)
    tardanzas = [caso.tardanza_total_min for caso in sinteticos]
    gaps = [caso.gap_costo_vs_greedy for caso in sinteticos]

    return ResumenValidacionExternaV4(
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


def clave_seleccion_externa_v4(
    resumen: ResumenValidacionExternaV4,
) -> tuple[float, ...]:
    """Clave minimizable: B04, factibilidad general, tardanza y costo."""

    return (
        float(resumen.b04_pedidos_tardios),
        float(resumen.b04_tardanza_min),
        float(-resumen.sinteticos_sin_riesgo),
        float(resumen.tardanza_sintetica_total_min),
        float(resumen.gap_costo_mediano_vs_greedy_pct),
    )


def es_mejor_validacion_externa_v4(
    candidata: ResumenValidacionExternaV4,
    actual: ResumenValidacionExternaV4 | None,
) -> bool:
    if actual is None:
        return True
    return clave_seleccion_externa_v4(candidata) < (
        clave_seleccion_externa_v4(actual)
    )
