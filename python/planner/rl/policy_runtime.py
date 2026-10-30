"""Módulo de ejecución productiva (runtime) para políticas de Aprendizaje por Refuerzo."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import loads
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Protocol

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.routing.objective import EstimacionPlan, evaluar_plan_estimado
from planner.routing.travel import ProveedorViaje, construir_matriz_viaje

__all__ = [
    "VERSION_RUNTIME",
    "FUENTE_RL_UNICA",
    "PlanificadorCompatible",
    "ConfiguracionOperacionRL",
    "MetricasOperacionRL",
    "DecisionOperacionRL",
    "cargar_configuracion_operacional",
    "PlanificadorOperativoRL",
]

VERSION_RUNTIME = "pedemonte-rl-single-policy-v1"
FUENTE_RL_UNICA = "POLITICA_UNICA"
TOLERANCIA = 1e-9


class PlanificadorCompatible(Protocol):
    """Protocolo compatible para inferencia de planes mediante modelos RL."""

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno: ...


@dataclass(frozen=True)
class ConfiguracionOperacionRL:
    """Metadatos y parámetros de validación para un modelo RL en producción."""

    version: str
    modelo: Path
    sha256_modelo: str
    max_pedidos_rl_validado: int
    usar_mascara_temporal_dura: bool
    ordenar_slots_canonicamente: bool = False
    num_rollouts: int = 1


@dataclass(frozen=True)
class MetricasOperacionRL:
    """Métricas operacionales de la solución generada por la política RL."""

    pedidos_tardios: int
    tardanza_total_min: float
    costo_total: float


@dataclass(frozen=True)
class DecisionOperacionRL:
    """Auditoría de decisión e inferencia de la política de Aprendizaje por Refuerzo."""

    cantidad_pedidos: int
    fuente_seleccionada: str
    metricas: MetricasOperacionRL
    tiempo_total_ms: float
    mascara_temporal_dura: bool

    def serializar(self) -> str:
        """Serializa la auditoría de inferencia en formato clave=valor."""
        mascara = "DURA" if self.mascara_temporal_dura else "NORMAL"
        return (
            f"version={VERSION_RUNTIME}"
            f"|arquitectura=RL_PURO_POLITICA_UNICA"
            f"|pedidos={self.cantidad_pedidos}"
            f"|fuente={self.fuente_seleccionada}"
            f"|tardios={self.metricas.pedidos_tardios}"
            f"|tardanza={self.metricas.tardanza_total_min:.6f}"
            f"|costo={self.metricas.costo_total:.6f}"
            f"|mascara_temporal={mascara}"
            f"|errores=NINGUNO"
            f"|tiempo_total_ms={self.tiempo_total_ms:.3f}"
        )


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
) -> ConfiguracionOperacionRL:
    """Carga y valida el manifiesto JSON de una política RL productiva."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"No existe el manifiesto de la política RL: {manifest}")

    datos = loads(manifest.read_text(encoding="utf-8"))
    version = str(datos.get("version", ""))
    if version != VERSION_RUNTIME:
        raise ValueError(
            f"Versión de manifiesto RL no soportada: {version!r}. Esperada: {VERSION_RUNTIME!r}."
        )

    modelo = _resolver_ruta(manifest.parent, str(datos["modelo"]))
    if not modelo.is_file():
        raise FileNotFoundError(f"No existe el modelo RL: {modelo}")

    hash_modelo = str(datos.get("sha256_modelo", "")).lower()
    if hash_modelo and _hash_archivo(modelo) != hash_modelo:
        raise ValueError("El SHA-256 del modelo RL no coincide.")

    max_validado = int(datos.get("max_pedidos_rl_validado", 12))
    if max_validado <= 0:
        raise ValueError("max_pedidos_rl_validado debe ser > 0.")

    mascara_dura = datos.get("usar_mascara_temporal_dura", True)
    if not isinstance(mascara_dura, bool):
        raise ValueError("usar_mascara_temporal_dura debe ser booleano.")

    ordenar_canonico = bool(datos.get("ordenar_slots_canonicamente", False))
    num_rollouts = max(1, int(datos.get("num_rollouts", 1)))

    return ConfiguracionOperacionRL(
        version=version,
        modelo=modelo,
        sha256_modelo=hash_modelo,
        max_pedidos_rl_validado=max_validado,
        usar_mascara_temporal_dura=mascara_dura,
        ordenar_slots_canonicamente=ordenar_canonico,
        num_rollouts=num_rollouts,
    )


class PlanificadorOperativoRL:
    """Planificador RL productivo con una única política de inferencia.

    La política genera el plan sin ejecutar heurísticas adicionales.
    El manifiesto fija el modelo, su SHA-256, el rango técnico validado
    y la configuración de máscara temporal utilizada durante la inferencia.
    """

    VERSION_PLANIFICADOR = "RL_POLITICA_UNICA"

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
        planner_rl: PlanificadorCompatible | None = None,
        configuracion_operacional: ConfiguracionOperacionRL | None = None,
    ) -> None:
        if max_pedidos <= 0:
            raise ValueError("max_pedidos debe ser > 0.")

        self.configuracion = configuracion or ConfiguracionPlanificacion()
        self.proveedor_viaje = proveedor_viaje
        self.max_pedidos = max_pedidos
        self.deterministic = deterministic

        if configuracion_operacional is None:
            if manifest_path is None:
                raise ValueError("Se requiere manifest_path o configuracion_operacional.")
            configuracion_operacional = cargar_configuracion_operacional(manifest_path)
        self.configuracion_operacional = configuracion_operacional

        if planner_rl is None:
            from planner.rl.policy_planner import PlanificadorPoliticaRL

            planner_rl = PlanificadorPoliticaRL(
                model_path=configuracion_operacional.modelo,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
                configuracion_temporal=ConfiguracionPoliticaRL(
                    usar_mascara_temporal_dura=configuracion_operacional.usar_mascara_temporal_dura,
                    ordenar_slots_canonicamente=configuracion_operacional.ordenar_slots_canonicamente,
                ),
                max_pedidos=max_pedidos,
                deterministic=deterministic,
                num_rollouts=configuracion_operacional.num_rollouts,
            )

        self.planner_rl = planner_rl
        self.ultima_decision: DecisionOperacionRL | None = None
        self.ultimo_detalle = "SIN_DECISION"

    @property
    def ultimas_permutaciones(self) -> tuple[tuple[str, ...], ...]:
        return tuple(getattr(self.planner_rl, "ultimas_permutaciones", ()))

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        """Genera un plan de turno validado mediante la política neuronal RL."""
        cantidad_pedidos = len(instancia.pedidos)
        maximo = self.configuracion_operacional.max_pedidos_rl_validado
        if cantidad_pedidos > maximo:
            raise RuntimeError(
                f"La política RL única está fuera del rango técnico validado: pedidos={cantidad_pedidos}, máximo={maximo}."
            )

        inicio = perf_counter()

        try:
            plan = self.planner_rl.generar_plan(instancia)
        except Exception as exc:
            raise RuntimeError(
                f"La política RL única no pudo generar un plan ejecutable: {type(exc).__name__}: {exc}"
            ) from exc

        validacion = validar_plan(instancia, plan)
        if not validacion.valido:
            raise RuntimeError(
                "La política RL única produjo un plan inválido: " + " | ".join(validacion.errores)
            )

        matriz = construir_matriz_viaje(
            instancia,
            self.configuracion,
            self.proveedor_viaje,
        )
        estimacion: EstimacionPlan = evaluar_plan_estimado(
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
        metricas = MetricasOperacionRL(
            pedidos_tardios=pedidos_tardios,
            tardanza_total_min=float(estimacion.tardanza_total_min),
            costo_total=float(estimacion.costo_total),
        )

        if (
            metricas.pedidos_tardios < 0
            or not isfinite(metricas.tardanza_total_min)
            or metricas.tardanza_total_min < 0.0
            or not isfinite(metricas.costo_total)
            or metricas.costo_total < 0.0
        ):
            raise RuntimeError("La política RL única produjo métricas inválidas.")

        plan.costo_estimado = metricas.costo_total
        plan.tiempo_computo_ms = (perf_counter() - inicio) * 1000.0
        plan.warnings.append(self.VERSION_PLANIFICADOR)
        plan.warnings.append(f"FUENTE_RL={FUENTE_RL_UNICA}")

        decision = DecisionOperacionRL(
            cantidad_pedidos=cantidad_pedidos,
            fuente_seleccionada=FUENTE_RL_UNICA,
            metricas=metricas,
            tiempo_total_ms=plan.tiempo_computo_ms,
            mascara_temporal_dura=self.configuracion_operacional.usar_mascara_temporal_dura,
        )
        self.ultima_decision = decision
        self.ultimo_detalle = decision.serializar()
        return plan
