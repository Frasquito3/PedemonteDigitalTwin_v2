from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import loads
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol

from planner.algorithms.greedy import generar_plan_greedy
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.routing.objective import EstimacionPlan, evaluar_plan_estimado
from planner.routing.travel import ProveedorViaje, construir_matriz_viaje


VERSION_OPERACIONAL = "pedemonte-rl-temporal-v4-operational-v1"
TOLERANCIA = 1e-9


class PlanificadorCompatible(Protocol):
    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        ...


@dataclass(frozen=True)
class ConfiguracionOperacionTemporalV4:
    version: str
    modelo_extension: Path
    modelo_full: Path
    sha256_extension: str
    sha256_full: str
    max_pedidos_rl_validado: int
    usar_guardia_greedy: bool
    preferencia_empate_hasta_10: str
    preferencia_empate_desde_11: str


@dataclass(frozen=True)
class MetricasCandidatoOperacional:
    fuente: str
    pedidos_tardios: int
    tardanza_total_min: float
    costo_total: float
    prioridad_empate: int

    @property
    def clave(self) -> tuple[int, float, float, int]:
        return (
            self.pedidos_tardios,
            self.tardanza_total_min,
            self.costo_total,
            self.prioridad_empate,
        )


@dataclass(frozen=True)
class DecisionOperacionTemporalV4:
    cantidad_pedidos: int
    fuente_seleccionada: str
    metricas_seleccionadas: MetricasCandidatoOperacional
    metricas_por_fuente: dict[str, MetricasCandidatoOperacional]
    errores_por_fuente: dict[str, str]
    tiempo_total_ms: float

    def serializar(self) -> str:
        candidatos = ";".join(
            (
                f"{fuente}:tardios={metricas.pedidos_tardios},"
                f"tardanza={metricas.tardanza_total_min:.6f},"
                f"costo={metricas.costo_total:.6f}"
            )
            for fuente, metricas in sorted(self.metricas_por_fuente.items())
        )
        errores = (
            ";".join(
                f"{fuente}:{mensaje}"
                for fuente, mensaje in sorted(self.errores_por_fuente.items())
            )
            if self.errores_por_fuente
            else "NINGUNO"
        )
        return (
            f"version={VERSION_OPERACIONAL}"
            f"|pedidos={self.cantidad_pedidos}"
            f"|fuente={self.fuente_seleccionada}"
            f"|tardios={self.metricas_seleccionadas.pedidos_tardios}"
            f"|tardanza={self.metricas_seleccionadas.tardanza_total_min:.6f}"
            f"|costo={self.metricas_seleccionadas.costo_total:.6f}"
            f"|candidatos={candidatos}"
            f"|errores={errores}"
            f"|tiempo_total_ms={self.tiempo_total_ms:.3f}"
        )


@dataclass
class _Candidato:
    fuente: str
    plan: PlanTurno
    estimacion: EstimacionPlan
    metricas: MetricasCandidatoOperacional


def _hash_archivo(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _resolver_ruta(base: Path, valor: str) -> Path:
    ruta = Path(valor).expanduser()
    if not ruta.is_absolute():
        ruta = base / ruta
    return ruta.resolve()


def cargar_configuracion_operacional(
    manifest_path: str | Path,
) -> ConfiguracionOperacionTemporalV4:
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No existe el manifiesto RL temporal v4 operacional: {manifest}"
        )

    datos = loads(manifest.read_text(encoding="utf-8"))
    version = str(datos.get("version", ""))
    if version != VERSION_OPERACIONAL:
        raise ValueError(
            "Versión de manifiesto operacional no soportada: "
            f"{version!r}. Esperada: {VERSION_OPERACIONAL!r}."
        )

    modelo_extension = _resolver_ruta(
        manifest.parent,
        str(datos["modelo_extension"]),
    )
    modelo_full = _resolver_ruta(
        manifest.parent,
        str(datos["modelo_full"]),
    )

    for nombre, ruta in (
        ("modelo_extension", modelo_extension),
        ("modelo_full", modelo_full),
    ):
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe {nombre}: {ruta}")

    hash_extension = str(datos.get("sha256_extension", "")).lower()
    hash_full = str(datos.get("sha256_full", "")).lower()
    if hash_extension and _hash_archivo(modelo_extension) != hash_extension:
        raise ValueError("El SHA-256 del modelo extensión no coincide.")
    if hash_full and _hash_archivo(modelo_full) != hash_full:
        raise ValueError("El SHA-256 del modelo full no coincide.")

    max_validado = int(datos.get("max_pedidos_rl_validado", 12))
    if max_validado <= 0:
        raise ValueError("max_pedidos_rl_validado debe ser > 0.")

    pref_hasta_10 = str(
        datos.get("preferencia_empate_hasta_10", "EXTENSION")
    ).upper()
    pref_desde_11 = str(
        datos.get("preferencia_empate_desde_11", "FULL")
    ).upper()
    fuentes_validas = {"EXTENSION", "FULL"}
    if pref_hasta_10 not in fuentes_validas:
        raise ValueError("preferencia_empate_hasta_10 inválida.")
    if pref_desde_11 not in fuentes_validas:
        raise ValueError("preferencia_empate_desde_11 inválida.")

    return ConfiguracionOperacionTemporalV4(
        version=version,
        modelo_extension=modelo_extension,
        modelo_full=modelo_full,
        sha256_extension=hash_extension,
        sha256_full=hash_full,
        max_pedidos_rl_validado=max_validado,
        usar_guardia_greedy=bool(datos.get("usar_guardia_greedy", True)),
        preferencia_empate_hasta_10=pref_hasta_10,
        preferencia_empate_desde_11=pref_desde_11,
    )


class RLTemporalV4OperationalPlanner:
    """
    Política de despliegue para el RL temporal v4.

    Genera planes con los dos checkpoints validados (extensión y full) y,
    opcionalmente, con Greedy como red de seguridad. Selecciona el mejor
    resultado por el criterio congelado del proyecto:

        pedidos tardíos -> tardanza total -> costo estimado.

    En empates exactos prioriza RL. Hasta diez pedidos prioriza la extensión;
    desde once pedidos prioriza el modelo full.
    """

    VERSION_PLANIFICADOR = "RL_TEMPORAL_V4_OPERACIONAL"

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
        planner_extension: PlanificadorCompatible | None = None,
        planner_full: PlanificadorCompatible | None = None,
        greedy_factory: Callable[..., PlanTurno] = generar_plan_greedy,
        configuracion_operacional: ConfiguracionOperacionTemporalV4 | None = None,
    ) -> None:
        if max_pedidos <= 0:
            raise ValueError("max_pedidos debe ser > 0.")

        self.configuracion = configuracion or ConfiguracionPlanificacion()
        self.proveedor_viaje = proveedor_viaje
        self.max_pedidos = max_pedidos
        self.deterministic = deterministic
        self.greedy_factory = greedy_factory

        if configuracion_operacional is None:
            if manifest_path is None:
                raise ValueError(
                    "Se requiere manifest_path o configuracion_operacional."
                )
            configuracion_operacional = cargar_configuracion_operacional(
                manifest_path
            )
        self.configuracion_operacional = configuracion_operacional

        if planner_extension is None or planner_full is None:
            from planner.rl.rl_temporal_v4_planner import (
                RLTemporalV4Planner,
            )

        self.planner_extension = planner_extension or RLTemporalV4Planner(
            model_path=configuracion_operacional.modelo_extension,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
            max_pedidos=max_pedidos,
            deterministic=deterministic,
        )
        self.planner_full = planner_full or RLTemporalV4Planner(
            model_path=configuracion_operacional.modelo_full,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
            max_pedidos=max_pedidos,
            deterministic=deterministic,
        )

        self.ultima_decision: DecisionOperacionTemporalV4 | None = None
        self.ultimo_detalle = "SIN_DECISION"

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        inicio = perf_counter()
        cantidad_pedidos = len(instancia.pedidos)
        matriz = construir_matriz_viaje(
            instancia,
            self.configuracion,
            self.proveedor_viaje,
        )

        candidatos: list[_Candidato] = []
        errores: dict[str, str] = {}

        if cantidad_pedidos <= self.configuracion_operacional.max_pedidos_rl_validado:
            self._agregar_candidato(
                candidatos,
                errores,
                fuente="EXTENSION",
                generar=lambda: self.planner_extension.generar_plan(instancia),
                instancia=instancia,
                matriz=matriz,
            )
            self._agregar_candidato(
                candidatos,
                errores,
                fuente="FULL",
                generar=lambda: self.planner_full.generar_plan(instancia),
                instancia=instancia,
                matriz=matriz,
            )
        else:
            errores["EXTENSION"] = (
                "OMITIDO_FUERA_RANGO_VALIDADO_"
                f"{self.configuracion_operacional.max_pedidos_rl_validado}"
            )
            errores["FULL"] = (
                "OMITIDO_FUERA_RANGO_VALIDADO_"
                f"{self.configuracion_operacional.max_pedidos_rl_validado}"
            )

        if self.configuracion_operacional.usar_guardia_greedy:
            self._agregar_candidato(
                candidatos,
                errores,
                fuente="GREEDY",
                generar=lambda: self.greedy_factory(
                    instancia,
                    configuracion=self.configuracion,
                    proveedor_viaje=self.proveedor_viaje,
                ),
                instancia=instancia,
                matriz=matriz,
            )

        if not candidatos:
            detalle = "; ".join(
                f"{fuente}: {mensaje}" for fuente, mensaje in errores.items()
            )
            raise RuntimeError(
                "No se pudo generar ningún plan operacional válido. " + detalle
            )

        ganador = min(candidatos, key=lambda item: item.metricas.clave)
        ganador.plan.costo_estimado = ganador.estimacion.costo_total
        ganador.plan.tiempo_computo_ms = (
            perf_counter() - inicio
        ) * 1000.0
        ganador.plan.warnings.append(self.VERSION_PLANIFICADOR)
        ganador.plan.warnings.append(f"FUENTE_OPERACIONAL={ganador.fuente}")

        decision = DecisionOperacionTemporalV4(
            cantidad_pedidos=cantidad_pedidos,
            fuente_seleccionada=ganador.fuente,
            metricas_seleccionadas=ganador.metricas,
            metricas_por_fuente={
                item.fuente: item.metricas for item in candidatos
            },
            errores_por_fuente=errores,
            tiempo_total_ms=ganador.plan.tiempo_computo_ms,
        )
        self.ultima_decision = decision
        self.ultimo_detalle = decision.serializar()
        return ganador.plan

    def _agregar_candidato(
        self,
        candidatos: list[_Candidato],
        errores: dict[str, str],
        *,
        fuente: str,
        generar: Callable[[], PlanTurno],
        instancia: InstanciaTurno,
        matriz: Any,
    ) -> None:
        try:
            plan = generar()
            validacion = validar_plan(instancia, plan)
            if not validacion.valido:
                raise RuntimeError(" | ".join(validacion.errores))

            estimacion = evaluar_plan_estimado(
                instancia,
                plan,
                matriz,
                self.configuracion,
            )
            pedidos_tardios = int(
                getattr(
                    estimacion,
                    "pedidos_tardios",
                    0 if estimacion.tardanza_total_min <= TOLERANCIA else 1,
                )
            )
            metricas = MetricasCandidatoOperacional(
                fuente=fuente,
                pedidos_tardios=pedidos_tardios,
                tardanza_total_min=float(estimacion.tardanza_total_min),
                costo_total=float(estimacion.costo_total),
                prioridad_empate=self._prioridad_empate(
                    fuente,
                    len(instancia.pedidos),
                ),
            )
            if (
                metricas.pedidos_tardios < 0
                or not isfinite(metricas.tardanza_total_min)
                or metricas.tardanza_total_min < 0.0
                or not isfinite(metricas.costo_total)
                or metricas.costo_total < 0.0
            ):
                raise RuntimeError("Métricas operacionales inválidas.")

            candidatos.append(
                _Candidato(
                    fuente=fuente,
                    plan=plan,
                    estimacion=estimacion,
                    metricas=metricas,
                )
            )
        except Exception as exc:  # noqa: BLE001 - se registra por candidato
            errores[fuente] = f"{type(exc).__name__}: {exc}"

    def _prioridad_empate(self, fuente: str, cantidad_pedidos: int) -> int:
        preferida = (
            self.configuracion_operacional.preferencia_empate_hasta_10
            if cantidad_pedidos <= 10
            else self.configuracion_operacional.preferencia_empate_desde_11
        )
        otra_rl = "FULL" if preferida == "EXTENSION" else "EXTENSION"
        orden = {
            preferida: 0,
            otra_rl: 1,
            "GREEDY": 2,
        }
        return orden.get(fuente, 99)
