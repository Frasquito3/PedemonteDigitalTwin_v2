from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from json import loads
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from planner.core.schema import InstanciaTurno
from planner.evaluation.classic_instances import crear_casos_benchmark_clasico
from planner.rl.instance_generator import (
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)
from planner.rl.rl_reward import ConfiguracionRewardRL, ModoRewardRL
from planner.rl.policy_config import ConfiguracionTemporalV4RL
from planner.rl.policy_instance_generator import (
    ConfiguracionGeneradorTemporalV4,
    GeneradorInstanciasTemporalV4RL,
)
from planner.routing.travel import ProveedorViaje


UMBRAL_REGRESION_CLASICA_COSTO_PCT = 10.0
UMBRAL_COSTO_EXTREMO_PCT = 500.0
SEED_VALIDACION_COMPLETA_MINIMO = 273_000
SEED_VALIDACION_COMPLETA_LIMITE = 274_000
TIMESTEPS_BASE_ESPERADOS = 68_288


@dataclass(frozen=True)
class DefinicionEstratoValidacionCompletaV4:
    nombre: str
    grupo: str
    min_pedidos: int
    max_pedidos: int
    conflictivo: bool


ESTRATOS_VALIDACION_COMPLETA_V4 = (
    DefinicionEstratoValidacionCompletaV4(
        "GUARD_CONFLICTIVO_3_5", "GUARD_3_8", 3, 5, True
    ),
    DefinicionEstratoValidacionCompletaV4(
        "GUARD_GENERAL_3_5", "GUARD_3_8", 3, 5, False
    ),
    DefinicionEstratoValidacionCompletaV4(
        "GUARD_CONFLICTIVO_6_8", "GUARD_3_8", 6, 8, True
    ),
    DefinicionEstratoValidacionCompletaV4(
        "GUARD_GENERAL_6_8", "GUARD_3_8", 6, 8, False
    ),
    DefinicionEstratoValidacionCompletaV4(
        "GUARD_CONFLICTIVO_9_10", "GUARD_9_10", 9, 10, True
    ),
    DefinicionEstratoValidacionCompletaV4(
        "GUARD_GENERAL_9_10", "GUARD_9_10", 9, 10, False
    ),
    DefinicionEstratoValidacionCompletaV4(
        "OBJETIVO_CONFLICTIVO_11", "OBJETIVO_11", 11, 11, True
    ),
    DefinicionEstratoValidacionCompletaV4(
        "OBJETIVO_GENERAL_11", "OBJETIVO_11", 11, 11, False
    ),
    DefinicionEstratoValidacionCompletaV4(
        "OBJETIVO_CONFLICTIVO_12", "OBJETIVO_12", 12, 12, True
    ),
    DefinicionEstratoValidacionCompletaV4(
        "OBJETIVO_GENERAL_12", "OBJETIVO_12", 12, 12, False
    ),
)


@dataclass(frozen=True)
class CasoValidacionCompletaV4:
    caso_id: str
    estrato: str
    grupo: str
    instancia: InstanciaTurno
    usar_cache_vial: bool
    conflictivo: bool


@dataclass(frozen=True)
class ResultadoCasoValidacionCompletaV4:
    caso_id: str
    estrato: str
    grupo: str
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
class ResumenGrupoValidacionCompletaV4:
    grupo: str
    casos_totales: int
    casos_sin_riesgo: int
    pedidos_tardios_total: int
    tardanza_total_min: float
    costos_extremos: int
    gap_costo_mediano_vs_greedy_pct: float


@dataclass(frozen=True)
class ResumenEstratoValidacionCompletaV4:
    estrato: str
    casos_totales: int
    casos_sin_riesgo: int
    pedidos_tardios_total: int
    tardanza_total_min: float
    costos_extremos: int
    gap_costo_mediano_vs_greedy_pct: float


@dataclass(frozen=True)
class ResumenValidacionCompletaV4:
    timestep: int
    clasicos_pedidos_tardios: int
    clasicos_tardanza_total_min: float
    clasicos_regresiones_costo: int
    guard_3_8_totales: int
    guard_3_8_sin_riesgo: int
    guard_3_8_pedidos_tardios: int
    guard_3_8_tardanza_total_min: float
    guard_3_8_costos_extremos: int
    guard_9_10_totales: int
    guard_9_10_sin_riesgo: int
    guard_9_10_pedidos_tardios: int
    guard_9_10_tardanza_total_min: float
    guard_9_10_costos_extremos: int
    objetivo_11_totales: int
    objetivo_11_sin_riesgo: int
    objetivo_11_pedidos_tardios: int
    objetivo_11_tardanza_total_min: float
    objetivo_12_totales: int
    objetivo_12_sin_riesgo: int
    objetivo_12_pedidos_tardios: int
    objetivo_12_tardanza_total_min: float
    objetivo_general_11_12_totales: int
    objetivo_general_11_12_sin_riesgo: int
    objetivo_general_11_12_pedidos_tardios: int
    objetivo_general_11_12_tardanza_total_min: float
    objetivo_11_12_totales: int
    objetivo_11_12_sin_riesgo: int
    objetivo_11_12_pedidos_tardios: int
    objetivo_11_12_tardanza_total_min: float
    objetivo_11_12_costos_extremos: int
    gap_costo_mediano_vs_greedy_pct: float
    resumenes_grupo: tuple[ResumenGrupoValidacionCompletaV4, ...]
    resumenes_estrato: tuple[ResumenEstratoValidacionCompletaV4, ...]
    casos: tuple[ResultadoCasoValidacionCompletaV4, ...]

    def como_dict(self) -> dict[str, Any]:
        contenido = asdict(self)
        contenido["clave_seleccion"] = list(
            clave_seleccion_validacion_completa_v4(self)
        )
        return contenido


@dataclass(frozen=True)
class BateriaValidacionCompletaV4:
    casos: tuple[CasoValidacionCompletaV4, ...]
    semillas_sinteticas: tuple[int, ...]
    cantidad_por_estrato: int


def _tiene_patron_v4(instancia: InstanciaTurno) -> bool:
    return any(
        "PATRON_TEMPORAL_CONFLICTIVO_V4" in pedido.observaciones
        for pedido in instancia.pedidos
    )


def _crear_generador_exactitud(
    cantidad: int,
    conflictivo: bool,
) -> GeneradorInstanciasTemporalV4RL:
    base = GeneradorInstanciasRL(
        ConfiguracionGeneradorInstancias(
            min_pedidos_finales=cantidad,
            max_pedidos_finales=cantidad,
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


def crear_bateria_validacion_completa_v4(
    *,
    cantidad_por_estrato: int = 6,
    seed_inicio: int = SEED_VALIDACION_COMPLETA_MINIMO,
) -> BateriaValidacionCompletaV4:
    if cantidad_por_estrato <= 0:
        raise ValueError("cantidad_por_estrato debe ser > 0.")
    if seed_inicio < SEED_VALIDACION_COMPLETA_MINIMO:
        raise ValueError(
            "La selección de la Fase 16D.8 exige semillas >= 273000."
        )
    if seed_inicio >= SEED_VALIDACION_COMPLETA_LIMITE:
        raise ValueError(
            "La selección debe permanecer en el rango reservado 273000-273999."
        )

    clasicos_por_id = {
        caso.caso_id: caso for caso in crear_casos_benchmark_clasico()
    }
    casos: list[CasoValidacionCompletaV4] = []
    for caso_id in ("B04_VENTANAS", "B05_VOLCADOR", "B06_SPLIT"):
        caso = clasicos_por_id.get(caso_id)
        if caso is None:
            raise RuntimeError(f"No se encontró el caso clásico {caso_id}.")
        casos.append(
            CasoValidacionCompletaV4(
                caso_id=caso_id,
                estrato="CLASICO_GUARD",
                grupo="CLASICO_GUARD",
                instancia=caso.instancia,
                usar_cache_vial=True,
                conflictivo=(caso_id == "B04_VENTANAS"),
            )
        )

    semillas_usadas: list[int] = []
    seed = int(seed_inicio)
    for definicion in ESTRATOS_VALIDACION_COMPLETA_V4:
        generadores = {
            cantidad: _crear_generador_exactitud(
                cantidad,
                definicion.conflictivo,
            )
            for cantidad in range(
                definicion.min_pedidos,
                definicion.max_pedidos + 1,
            )
        }
        aceptados = 0
        intentos = 0
        limite = max(5_000, cantidad_por_estrato * 500)

        while aceptados < cantidad_por_estrato:
            if intentos >= limite:
                raise RuntimeError(
                    f"No fue posible completar el estrato {definicion.nombre}."
                )
            if seed >= SEED_VALIDACION_COMPLETA_LIMITE:
                raise RuntimeError(
                    "La batería agotó el rango reservado 273000-273999."
                )

            cantidad_objetivo = definicion.min_pedidos + (
                aceptados
                % (
                    definicion.max_pedidos
                    - definicion.min_pedidos
                    + 1
                )
            )
            seed_actual = seed
            seed += 1
            intentos += 1
            instancia = generadores[cantidad_objetivo].generar(seed_actual)

            if _tiene_patron_v4(instancia) != definicion.conflictivo:
                continue
            if len(instancia.pedidos) != cantidad_objetivo:
                continue

            casos.append(
                CasoValidacionCompletaV4(
                    caso_id=(
                        f"FULL16D8-{definicion.nombre}-{seed_actual}"
                    ),
                    estrato=definicion.nombre,
                    grupo=definicion.grupo,
                    instancia=instancia,
                    usar_cache_vial=False,
                    conflictivo=definicion.conflictivo,
                )
            )
            semillas_usadas.append(seed_actual)
            aceptados += 1

    return BateriaValidacionCompletaV4(
        casos=tuple(casos),
        semillas_sinteticas=tuple(semillas_usadas),
        cantidad_por_estrato=cantidad_por_estrato,
    )


def ejecutar_caso_validacion_completa_v4(
    model: Any,
    caso: CasoValidacionCompletaV4,
    *,
    proveedor_clasicos: ProveedorViaje,
    configuracion_temporal: ConfiguracionTemporalV4RL,
) -> ResultadoCasoValidacionCompletaV4:
    # Importaciones locales: permiten probar la lógica de selección y la
    # generación de la batería sin exigir Gymnasium/SB3.
    import numpy as np

    from planner.rl.policy_env import PedemonteTemporalV4PlanEnv

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
            observacion, _, terminado, truncado, info = env.step(
                accion_entera
            )
            info_final = dict(info)
            if truncado:
                raise RuntimeError(
                    "La validación completa produjo un episodio truncado."
                )

        plan = env.ultimo_plan
        if plan is None:
            raise RuntimeError(
                "La validación completa finalizó sin plan."
            )

        costo_greedy = float(
            info_final.get("costo_greedy_referencia", 0.0)
        )
        if costo_greedy <= 0.0:
            raise RuntimeError(
                "La validación completa no recibió costo Greedy."
            )
        costo = float(plan.costo_estimado)
        gap = 100.0 * (costo - costo_greedy) / costo_greedy

        return ResultadoCasoValidacionCompletaV4(
            caso_id=caso.caso_id,
            estrato=caso.estrato,
            grupo=caso.grupo,
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


def _resumir_seleccion(
    resultados: Sequence[ResultadoCasoValidacionCompletaV4],
    *,
    atributo: str,
) -> tuple[ResumenGrupoValidacionCompletaV4, ...] | tuple[
    ResumenEstratoValidacionCompletaV4, ...
]:
    nombres = sorted({str(getattr(item, atributo)) for item in resultados})
    salida: list[Any] = []
    clase = (
        ResumenGrupoValidacionCompletaV4
        if atributo == "grupo"
        else ResumenEstratoValidacionCompletaV4
    )
    clave_nombre = "grupo" if atributo == "grupo" else "estrato"

    for nombre in nombres:
        seleccion = [
            item for item in resultados if str(getattr(item, atributo)) == nombre
        ]
        gaps = [item.gap_costo_vs_greedy_pct for item in seleccion]
        salida.append(
            clase(
                **{
                    clave_nombre: nombre,
                    "casos_totales": len(seleccion),
                    "casos_sin_riesgo": sum(
                        1 for item in seleccion if item.sin_riesgo
                    ),
                    "pedidos_tardios_total": sum(
                        item.pedidos_tardios for item in seleccion
                    ),
                    "tardanza_total_min": sum(
                        item.tardanza_total_min for item in seleccion
                    ),
                    "costos_extremos": sum(
                        1 for item in seleccion if item.costo_extremo
                    ),
                    "gap_costo_mediano_vs_greedy_pct": float(
                        median(gaps)
                    ),
                }
            )
        )
    return tuple(salida)


def _seleccionar_grupo(
    resultados: Sequence[ResultadoCasoValidacionCompletaV4],
    grupo: str,
) -> list[ResultadoCasoValidacionCompletaV4]:
    return [item for item in resultados if item.grupo == grupo]


def _seleccionar_general_11_12(
    resultados: Sequence[ResultadoCasoValidacionCompletaV4],
) -> list[ResultadoCasoValidacionCompletaV4]:
    return [
        item
        for item in resultados
        if item.grupo in {"OBJETIVO_11", "OBJETIVO_12"}
        and "GENERAL" in item.estrato
    ]


def _metricas(
    seleccion: Sequence[ResultadoCasoValidacionCompletaV4],
) -> tuple[int, int, int, float, int]:
    return (
        len(seleccion),
        sum(1 for item in seleccion if item.sin_riesgo),
        sum(item.pedidos_tardios for item in seleccion),
        sum(item.tardanza_total_min for item in seleccion),
        sum(1 for item in seleccion if item.costo_extremo),
    )


def evaluar_modelo_externamente_completo_v4(
    model: Any,
    *,
    timestep: int,
    bateria: BateriaValidacionCompletaV4,
    proveedor_clasicos: ProveedorViaje,
    configuracion_temporal: ConfiguracionTemporalV4RL,
) -> ResumenValidacionCompletaV4:
    resultados = tuple(
        ejecutar_caso_validacion_completa_v4(
            model,
            caso,
            proveedor_clasicos=proveedor_clasicos,
            configuracion_temporal=configuracion_temporal,
        )
        for caso in bateria.casos
    )

    clasicos = _seleccionar_grupo(resultados, "CLASICO_GUARD")
    guard_3_8 = _seleccionar_grupo(resultados, "GUARD_3_8")
    guard_9_10 = _seleccionar_grupo(resultados, "GUARD_9_10")
    objetivo_11 = _seleccionar_grupo(resultados, "OBJETIVO_11")
    objetivo_12 = _seleccionar_grupo(resultados, "OBJETIVO_12")
    general_11_12 = _seleccionar_general_11_12(resultados)
    objetivo_11_12 = objetivo_11 + objetivo_12

    if len(clasicos) != 3:
        raise RuntimeError("La batería debe incluir B04, B05 y B06.")
    if not guard_3_8 or not guard_9_10:
        raise RuntimeError("La batería debe incluir guardas 3-8 y 9-10.")
    if not objetivo_11 or not objetivo_12 or not general_11_12:
        raise RuntimeError(
            "La batería debe incluir objetivos 11, 12 y generales 11-12."
        )

    g38 = _metricas(guard_3_8)
    g910 = _metricas(guard_9_10)
    o11 = _metricas(objetivo_11)
    o12 = _metricas(objetivo_12)
    ogeneral = _metricas(general_11_12)
    o1112 = _metricas(objetivo_11_12)
    gaps = [item.gap_costo_vs_greedy_pct for item in resultados]

    return ResumenValidacionCompletaV4(
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
        guard_3_8_totales=g38[0],
        guard_3_8_sin_riesgo=g38[1],
        guard_3_8_pedidos_tardios=g38[2],
        guard_3_8_tardanza_total_min=g38[3],
        guard_3_8_costos_extremos=g38[4],
        guard_9_10_totales=g910[0],
        guard_9_10_sin_riesgo=g910[1],
        guard_9_10_pedidos_tardios=g910[2],
        guard_9_10_tardanza_total_min=g910[3],
        guard_9_10_costos_extremos=g910[4],
        objetivo_11_totales=o11[0],
        objetivo_11_sin_riesgo=o11[1],
        objetivo_11_pedidos_tardios=o11[2],
        objetivo_11_tardanza_total_min=o11[3],
        objetivo_12_totales=o12[0],
        objetivo_12_sin_riesgo=o12[1],
        objetivo_12_pedidos_tardios=o12[2],
        objetivo_12_tardanza_total_min=o12[3],
        objetivo_general_11_12_totales=ogeneral[0],
        objetivo_general_11_12_sin_riesgo=ogeneral[1],
        objetivo_general_11_12_pedidos_tardios=ogeneral[2],
        objetivo_general_11_12_tardanza_total_min=ogeneral[3],
        objetivo_11_12_totales=o1112[0],
        objetivo_11_12_sin_riesgo=o1112[1],
        objetivo_11_12_pedidos_tardios=o1112[2],
        objetivo_11_12_tardanza_total_min=o1112[3],
        objetivo_11_12_costos_extremos=o1112[4],
        gap_costo_mediano_vs_greedy_pct=float(median(gaps)),
        resumenes_grupo=_resumir_seleccion(
            resultados,
            atributo="grupo",
        ),
        resumenes_estrato=_resumir_seleccion(
            resultados,
            atributo="estrato",
        ),
        casos=resultados,
    )


def clave_seleccion_validacion_completa_v4(
    resumen: ResumenValidacionCompletaV4,
) -> tuple[float, ...]:
    """
    Clave minimizable de selección externa para la Fase 16D.8.

    Se preservan primero B04-B06 y las bandas ya dominadas. Después se
    prioriza exactamente 12 pedidos, los escenarios generales de 11-12 y el
    conjunto completo 11-12. El costo solo actúa después de la factibilidad.
    """

    return (
        float(resumen.clasicos_pedidos_tardios),
        float(resumen.clasicos_tardanza_total_min),
        float(resumen.clasicos_regresiones_costo),
        float(resumen.guard_3_8_pedidos_tardios),
        float(resumen.guard_3_8_tardanza_total_min),
        float(resumen.guard_9_10_pedidos_tardios),
        float(resumen.guard_9_10_tardanza_total_min),
        float(-resumen.objetivo_12_sin_riesgo),
        float(resumen.objetivo_12_pedidos_tardios),
        float(resumen.objetivo_12_tardanza_total_min),
        float(-resumen.objetivo_general_11_12_sin_riesgo),
        float(resumen.objetivo_general_11_12_pedidos_tardios),
        float(resumen.objetivo_general_11_12_tardanza_total_min),
        float(-resumen.objetivo_11_12_sin_riesgo),
        float(resumen.objetivo_11_12_pedidos_tardios),
        float(resumen.objetivo_11_12_tardanza_total_min),
        float(resumen.objetivo_11_12_costos_extremos),
        float(resumen.guard_3_8_costos_extremos),
        float(resumen.guard_9_10_costos_extremos),
        float(resumen.gap_costo_mediano_vs_greedy_pct),
    )


def es_mejor_validacion_completa_v4(
    candidata: ResumenValidacionCompletaV4,
    actual: ResumenValidacionCompletaV4 | None,
) -> bool:
    if actual is None:
        return True
    return clave_seleccion_validacion_completa_v4(candidata) < (
        clave_seleccion_validacion_completa_v4(actual)
    )


def validar_origen_extension_v4(
    ruta_config: str | Path,
    ruta_seleccion: str | Path,
    ruta_resumen: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = Path(ruta_config)
    selection_path = Path(ruta_seleccion)
    summary_path = Path(ruta_resumen)
    for nombre, ruta in {
        "configuración": config_path,
        "selección": selection_path,
        "resumen": summary_path,
    }.items():
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe la {nombre} base: {ruta}")

    config = loads(config_path.read_text(encoding="utf-8"))
    seleccion = loads(selection_path.read_text(encoding="utf-8"))
    resumen = loads(summary_path.read_text(encoding="utf-8"))
    if not all(isinstance(item, dict) for item in (config, seleccion, resumen)):
        raise ValueError("Los metadatos base deben ser objetos JSON.")

    if config.get("version_run") != (
        "pedemonte-rl-temporal-v4-extension-9-12-v1"
    ):
        raise ValueError("version_run de la extensión v4 inválida.")
    if config.get("observacion") != 702 or config.get("acciones") != 30:
        raise ValueError("La extensión base no usa espacios v4 de 702/30.")
    if config.get("reward_modificado") is not False:
        raise ValueError("El reward de la extensión base fue modificado.")
    if config.get("observacion_modificada") is not False:
        raise ValueError("La observación de la extensión base fue modificada.")
    if config.get("mascara_temporal_dura") is not False:
        raise ValueError("La máscara temporal dura debe seguir desactivada.")
    if config.get("continuacion_entre_etapas") != "EXTERNAL_BEST_9_12":
        raise ValueError("La extensión base no continuó desde external best.")
    if config.get("modelo_promovido") is not False:
        raise ValueError("La extensión base no debe estar promovida.")

    if seleccion.get("criterio") != (
        "VALIDACION_EXTERNA_9_12_LEXICOGRAFICA_V4_EXTENSION"
    ):
        raise ValueError("La extensión base no fue seleccionada externamente.")
    if seleccion.get("modelo_promovido") is not False:
        raise ValueError("La selección base figura como promovida.")

    if int(resumen.get("timestep", -1)) != TIMESTEPS_BASE_ESPERADOS:
        raise ValueError(
            "El resumen base no corresponde al checkpoint de 68.288 pasos."
        )
    if int(resumen.get("clasicos_pedidos_tardios", -1)) != 0:
        raise ValueError("El checkpoint base no preserva los clásicos.")
    if int(resumen.get("guard_3_8_sin_riesgo", -1)) != int(
        resumen.get("guard_3_8_totales", -2)
    ):
        raise ValueError("El checkpoint base no preserva completamente 3-8.")

    return config, seleccion, resumen


def validar_evidencia_holdout_16d7(ruta: str | Path) -> dict[str, Any]:
    path = Path(ruta)
    if not path.is_file():
        raise FileNotFoundError(
            "No existe la evidencia de holdout de la Fase 16D.7: "
            f"{path}"
        )
    datos = loads(path.read_text(encoding="utf-8"))
    if not isinstance(datos, dict):
        raise ValueError("La evidencia de holdout debe ser un objeto JSON.")

    metadatos = datos.get("metadatos")
    veredicto = datos.get("veredicto")
    if not isinstance(metadatos, dict) or not isinstance(veredicto, dict):
        raise ValueError("La evidencia de holdout está incompleta.")
    if veredicto.get("estado") != "CANDIDATO_ENTRENAMIENTO_COMPLETO":
        raise ValueError(
            "La Fase 16D.8 requiere veredicto "
            "CANDIDATO_ENTRENAMIENTO_COMPLETO."
        )
    criterios = veredicto.get("criterios")
    if not isinstance(criterios, dict) or not criterios:
        raise ValueError("La evidencia no contiene criterios de veredicto.")
    if not all(valor is True for valor in criterios.values()):
        raise ValueError("No todos los criterios del holdout fueron aprobados.")
    if metadatos.get("casos_sinteticos") != 160:
        raise ValueError("La evidencia no contiene los 160 casos formales.")
    if metadatos.get("seed_min_usada") != 272_000:
        raise ValueError("La evidencia no comienza en la semilla 272000.")
    if metadatos.get("modelo_promovido") is not False:
        raise ValueError("La evidencia indica una promoción no autorizada.")
    if metadatos.get("modo_solo_clasicos") is not False:
        raise ValueError("La evidencia corresponde solo a casos clásicos.")
    return datos


def hash_archivo(ruta: str | Path) -> str:
    digest = sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def semillas_por_estrato(
    bateria: BateriaValidacionCompletaV4,
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
