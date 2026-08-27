from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from json import loads
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np

from planner.core.schema import InstanciaTurno
from planner.evaluation.classic_instances import crear_casos_benchmark_clasico
from planner.rl.instance_generator import (
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)
from planner.rl.rl_reward import ConfiguracionRewardRL, ModoRewardRL
from planner.rl.rl_temporal_v4_config import ConfiguracionTemporalV4RL
from planner.rl.rl_temporal_v4_env import PedemonteTemporalV4PlanEnv
from planner.rl.temporal_v4_instance_generator import (
    ConfiguracionGeneradorTemporalV4,
    GeneradorInstanciasTemporalV4RL,
)
from planner.routing.travel import ProveedorViaje


UMBRAL_REGRESION_CLASICA_COSTO_PCT = 10.0
UMBRAL_COSTO_EXTREMO_PCT = 500.0

ESTRATOS_VALIDACION_EXTENSION_V4 = (
    ("GUARD_CONFLICTIVO_3_5", 3, 5, True, False),
    ("GUARD_GENERAL_3_5", 3, 5, False, False),
    ("GUARD_CONFLICTIVO_6_8", 6, 8, True, False),
    ("GUARD_GENERAL_6_8", 6, 8, False, False),
    ("OBJETIVO_CONFLICTIVO_9_10", 9, 10, True, True),
    ("OBJETIVO_GENERAL_9_10", 9, 10, False, True),
    ("OBJETIVO_CONFLICTIVO_11_12", 11, 12, True, True),
    ("OBJETIVO_GENERAL_11_12", 11, 12, False, True),
)


@dataclass(frozen=True)
class CasoValidacionExtensionV4:
    caso_id: str
    estrato: str
    instancia: InstanciaTurno
    usar_cache_vial: bool
    es_objetivo_9_12: bool


@dataclass(frozen=True)
class ResultadoCasoExtensionV4:
    caso_id: str
    estrato: str
    cantidad_pedidos: int
    pedidos_tardios: int
    tardanza_total_min: float
    costo_estimado: float
    costo_greedy_referencia: float
    gap_costo_vs_greedy_pct: float
    costo_extremo: bool
    permutacion: tuple[str, ...]

    @property
    def sin_riesgo(self) -> bool:
        return self.pedidos_tardios == 0


@dataclass(frozen=True)
class ResumenEstratoExtensionV4:
    estrato: str
    casos_totales: int
    casos_sin_riesgo: int
    tasa_sin_riesgo_pct: float
    tardanza_total_min: float
    gap_costo_mediano_vs_greedy_pct: float
    costos_extremos: int


@dataclass(frozen=True)
class ResumenValidacionExtensionV4:
    timestep: int
    clasicos_pedidos_tardios: int
    clasicos_tardanza_total_min: float
    clasicos_regresiones_costo: int
    objetivo_9_12_totales: int
    objetivo_9_12_sin_riesgo: int
    objetivo_9_12_tardanza_total_min: float
    objetivo_9_12_costos_extremos: int
    guard_3_8_totales: int
    guard_3_8_sin_riesgo: int
    guard_3_8_tardanza_total_min: float
    guard_3_8_costos_extremos: int
    gap_costo_mediano_vs_greedy_pct: float
    resumenes_estrato: tuple[ResumenEstratoExtensionV4, ...]
    casos: tuple[ResultadoCasoExtensionV4, ...]

    def como_dict(self) -> dict[str, Any]:
        contenido = asdict(self)
        contenido["clave_seleccion"] = list(
            clave_seleccion_extension_v4(self)
        )
        return contenido


@dataclass(frozen=True)
class BateriaValidacionExtensionV4:
    casos: tuple[CasoValidacionExtensionV4, ...]
    semillas_sinteticas: tuple[int, ...]
    cantidad_por_estrato: int


def _tiene_patron_v4(instancia: InstanciaTurno) -> bool:
    return any(
        "PATRON_TEMPORAL_CONFLICTIVO_V4" in pedido.observaciones
        for pedido in instancia.pedidos
    )


def _crear_generador_estrato(
    min_pedidos: int,
    max_pedidos: int,
    conflictivo: bool,
) -> GeneradorInstanciasTemporalV4RL:
    base = GeneradorInstanciasRL(
        ConfiguracionGeneradorInstancias(
            min_pedidos_finales=min_pedidos,
            max_pedidos_finales=max_pedidos,
            probabilidad_volcador=0.15,
            probabilidad_ventana_especifica=0.90,
            probabilidad_pedido_mayor_capacidad=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        )
    )
    return GeneradorInstanciasTemporalV4RL(
        base,
        ConfiguracionGeneradorTemporalV4(
            probabilidad_patron_ventanas_conflictivas=(
                1.0 if conflictivo else 0.0
            )
        ),
    )


def crear_bateria_validacion_extension_v4(
    *,
    cantidad_por_estrato: int = 4,
    seed_inicio: int = 271_000,
) -> BateriaValidacionExtensionV4:
    if cantidad_por_estrato <= 0:
        raise ValueError("cantidad_por_estrato debe ser > 0.")
    if seed_inicio < 270_000:
        raise ValueError(
            "La validación de extensión debe usar semillas >= 270000."
        )

    clasicos_por_id = {
        caso.caso_id: caso
        for caso in crear_casos_benchmark_clasico()
    }
    casos: list[CasoValidacionExtensionV4] = []
    for caso_id in ("B04_VENTANAS", "B05_VOLCADOR", "B06_SPLIT"):
        caso = clasicos_por_id.get(caso_id)
        if caso is None:
            raise RuntimeError(f"No se encontró el caso clásico {caso_id}.")
        casos.append(
            CasoValidacionExtensionV4(
                caso_id=caso_id,
                estrato="CLASICO_GUARD",
                instancia=caso.instancia,
                usar_cache_vial=True,
                es_objetivo_9_12=False,
            )
        )

    semillas_usadas: list[int] = []
    seed = int(seed_inicio)
    for nombre, minimo, maximo, conflictivo, objetivo in (
        ESTRATOS_VALIDACION_EXTENSION_V4
    ):
        generador = _crear_generador_estrato(
            minimo,
            maximo,
            conflictivo,
        )
        aceptados = 0
        intentos = 0
        limite = max(2_000, cantidad_por_estrato * 200)

        while aceptados < cantidad_por_estrato:
            if intentos >= limite:
                raise RuntimeError(
                    f"No fue posible completar el estrato {nombre}."
                )
            seed_actual = seed
            seed += 1
            intentos += 1
            instancia = generador.generar(seed_actual)

            if _tiene_patron_v4(instancia) != conflictivo:
                continue
            cantidad = len(instancia.pedidos)
            if not minimo <= cantidad <= maximo:
                continue

            casos.append(
                CasoValidacionExtensionV4(
                    caso_id=f"EXT-{nombre}-{seed_actual}",
                    estrato=nombre,
                    instancia=instancia,
                    usar_cache_vial=False,
                    es_objetivo_9_12=objetivo,
                )
            )
            semillas_usadas.append(seed_actual)
            aceptados += 1

    return BateriaValidacionExtensionV4(
        casos=tuple(casos),
        semillas_sinteticas=tuple(semillas_usadas),
        cantidad_por_estrato=cantidad_por_estrato,
    )


def ejecutar_caso_validacion_extension_v4(
    model: Any,
    caso: CasoValidacionExtensionV4,
    *,
    proveedor_clasicos: ProveedorViaje,
    configuracion_temporal: ConfiguracionTemporalV4RL,
) -> ResultadoCasoExtensionV4:
    reward = ConfiguracionRewardRL(
        modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
        denominador_relativo_minimo=1.0,
    )
    proveedor = proveedor_clasicos if caso.usar_cache_vial else None
    env = PedemonteTemporalV4PlanEnv(
        instancia=caso.instancia,
        proveedor_viaje=proveedor,
        configuracion_reward=reward,
        configuracion_temporal=configuracion_temporal,
        max_pedidos=30,
    )

    try:
        observacion, _ = env.reset(seed=caso.instancia.seed_escenario)
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
                    "La validación de extensión produjo un episodio "
                    "truncado."
                )

        plan = env.ultimo_plan
        if plan is None:
            raise RuntimeError(
                "La validación de extensión finalizó sin plan."
            )

        costo_greedy = float(
            info_final.get("costo_greedy_referencia", 0.0)
        )
        if costo_greedy <= 0.0:
            raise RuntimeError(
                "La validación de extensión no recibió costo Greedy."
            )
        costo = float(plan.costo_estimado)
        gap = 100.0 * (costo - costo_greedy) / costo_greedy

        return ResultadoCasoExtensionV4(
            caso_id=caso.caso_id,
            estrato=caso.estrato,
            cantidad_pedidos=len(caso.instancia.pedidos),
            pedidos_tardios=int(
                info_final.get("pedidos_tardios_prefijo", -1)
            ),
            tardanza_total_min=float(
                info_final.get("tardanza_prefijo_min", -1.0)
            ),
            costo_estimado=costo,
            costo_greedy_referencia=costo_greedy,
            gap_costo_vs_greedy_pct=gap,
            costo_extremo=(gap > UMBRAL_COSTO_EXTREMO_PCT),
            permutacion=tuple(env.permutacion_actual),
        )
    finally:
        env.close()


def _resumir_estratos(
    resultados: Sequence[ResultadoCasoExtensionV4],
) -> tuple[ResumenEstratoExtensionV4, ...]:
    salida: list[ResumenEstratoExtensionV4] = []
    for estrato in sorted({item.estrato for item in resultados}):
        seleccion = [item for item in resultados if item.estrato == estrato]
        gaps = [item.gap_costo_vs_greedy_pct for item in seleccion]
        sin_riesgo = sum(1 for item in seleccion if item.sin_riesgo)
        salida.append(
            ResumenEstratoExtensionV4(
                estrato=estrato,
                casos_totales=len(seleccion),
                casos_sin_riesgo=sin_riesgo,
                tasa_sin_riesgo_pct=(
                    100.0 * sin_riesgo / len(seleccion)
                ),
                tardanza_total_min=sum(
                    item.tardanza_total_min for item in seleccion
                ),
                gap_costo_mediano_vs_greedy_pct=float(median(gaps)),
                costos_extremos=sum(
                    1 for item in seleccion if item.costo_extremo
                ),
            )
        )
    return tuple(salida)


def evaluar_modelo_externamente_extension_v4(
    model: Any,
    *,
    timestep: int,
    bateria: BateriaValidacionExtensionV4,
    proveedor_clasicos: ProveedorViaje,
    configuracion_temporal: ConfiguracionTemporalV4RL,
) -> ResumenValidacionExtensionV4:
    resultados = tuple(
        ejecutar_caso_validacion_extension_v4(
            model,
            caso,
            proveedor_clasicos=proveedor_clasicos,
            configuracion_temporal=configuracion_temporal,
        )
        for caso in bateria.casos
    )

    clasicos = [
        item for item in resultados if item.estrato == "CLASICO_GUARD"
    ]
    objetivos = [
        item for item in resultados if item.estrato.startswith("OBJETIVO_")
    ]
    guardas = [
        item for item in resultados if item.estrato.startswith("GUARD_")
    ]
    if len(clasicos) != 3:
        raise RuntimeError("La batería debe incluir B04, B05 y B06.")
    if not objetivos or not guardas:
        raise RuntimeError(
            "La batería debe incluir objetivos 9-12 y guardas 3-8."
        )

    gaps = [item.gap_costo_vs_greedy_pct for item in resultados]
    return ResumenValidacionExtensionV4(
        timestep=int(timestep),
        clasicos_pedidos_tardios=sum(
            item.pedidos_tardios for item in clasicos
        ),
        clasicos_tardanza_total_min=sum(
            item.tardanza_total_min for item in clasicos
        ),
        clasicos_regresiones_costo=sum(
            1
            for item in clasicos
            if item.gap_costo_vs_greedy_pct
            > UMBRAL_REGRESION_CLASICA_COSTO_PCT
        ),
        objetivo_9_12_totales=len(objetivos),
        objetivo_9_12_sin_riesgo=sum(
            1 for item in objetivos if item.sin_riesgo
        ),
        objetivo_9_12_tardanza_total_min=sum(
            item.tardanza_total_min for item in objetivos
        ),
        objetivo_9_12_costos_extremos=sum(
            1 for item in objetivos if item.costo_extremo
        ),
        guard_3_8_totales=len(guardas),
        guard_3_8_sin_riesgo=sum(
            1 for item in guardas if item.sin_riesgo
        ),
        guard_3_8_tardanza_total_min=sum(
            item.tardanza_total_min for item in guardas
        ),
        guard_3_8_costos_extremos=sum(
            1 for item in guardas if item.costo_extremo
        ),
        gap_costo_mediano_vs_greedy_pct=float(median(gaps)),
        resumenes_estrato=_resumir_estratos(resultados),
        casos=resultados,
    )


def clave_seleccion_extension_v4(
    resumen: ResumenValidacionExtensionV4,
) -> tuple[float, ...]:
    """
    Clave minimizable.

    Primero protege B04/B05/B06. Después prioriza la factibilidad 9-12,
    luego preserva 3-8 y finalmente usa el costo como desempate.
    """

    return (
        float(resumen.clasicos_pedidos_tardios),
        float(resumen.clasicos_tardanza_total_min),
        float(resumen.clasicos_regresiones_costo),
        float(-resumen.objetivo_9_12_sin_riesgo),
        float(resumen.objetivo_9_12_tardanza_total_min),
        float(resumen.objetivo_9_12_costos_extremos),
        float(-resumen.guard_3_8_sin_riesgo),
        float(resumen.guard_3_8_tardanza_total_min),
        float(resumen.guard_3_8_costos_extremos),
        float(resumen.gap_costo_mediano_vs_greedy_pct),
    )


def es_mejor_validacion_extension_v4(
    candidata: ResumenValidacionExtensionV4,
    actual: ResumenValidacionExtensionV4 | None,
) -> bool:
    if actual is None:
        return True
    return clave_seleccion_extension_v4(candidata) < (
        clave_seleccion_extension_v4(actual)
    )


def validar_origen_v4_quick(
    ruta_config: str | Path,
    ruta_seleccion: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(ruta_config)
    selection_path = Path(ruta_seleccion)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No existe la configuración v4 quick: {config_path}"
        )
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"No existe la selección v4 quick: {selection_path}"
        )

    config = loads(config_path.read_text(encoding="utf-8"))
    seleccion = loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(seleccion, dict):
        raise ValueError("Los metadatos v4 deben ser objetos JSON.")
    if config.get("version_entorno") != "pedemonte-rl-temporal-v4":
        raise ValueError("version_entorno v4 inválida.")
    if config.get("quick") is not True:
        raise ValueError("La extensión 16D.6 parte del modelo quick v4.")
    if config.get("continuacion_entre_etapas") != "EXTERNAL_BEST":
        raise ValueError("El origen v4 no fue continuado desde EXTERNAL_BEST.")
    if config.get("modelo_historico_sobrescrito") is not False:
        raise ValueError("La configuración no preservó el modelo histórico.")
    if config.get("modelo_v3_sobrescrito") is not False:
        raise ValueError("La configuración no preservó el modelo v3.")

    temporal = config.get("temporal")
    if not isinstance(temporal, dict):
        raise ValueError("Falta la configuración temporal v4.")
    if temporal.get("usar_mascara_temporal_dura") is not False:
        raise ValueError("La máscara temporal dura debe seguir desactivada.")

    if seleccion.get("criterio") != "VALIDACION_EXTERNA_LEXICOGRAFICA_V4":
        raise ValueError("El modelo base no fue elegido externamente.")
    if seleccion.get("modelo_promovido") is not False:
        raise ValueError("El modelo quick base no debe estar promovido.")

    return config, seleccion


def hash_archivo(ruta: str | Path) -> str:
    digest = sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def buscar_resultado(
    resumen: ResumenValidacionExtensionV4,
    caso_id: str,
) -> ResultadoCasoExtensionV4:
    for resultado in resumen.casos:
        if resultado.caso_id == caso_id:
            return resultado
    raise KeyError(f"No existe el caso {caso_id} en el resumen.")


def semillas_por_estrato(
    bateria: BateriaValidacionExtensionV4,
) -> dict[str, tuple[int, ...]]:
    salida: dict[str, list[int]] = {}
    for caso in bateria.casos:
        if caso.usar_cache_vial:
            continue
        salida.setdefault(caso.estrato, []).append(
            caso.instancia.seed_escenario
        )
    return {
        estrato: tuple(semillas)
        for estrato, semillas in salida.items()
    }
