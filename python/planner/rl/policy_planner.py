"""Integración de inferencia con modelos MaskablePPO entrenados para planificación."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from gymnasium import spaces

# pyrefly: ignore [missing-import]
from sb3_contrib import MaskablePPO

from planner.core.base import PlanificadorTurno
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.policy_env import EntornoPlanificacionRL
from planner.routing.travel import ProveedorViaje

__all__ = ["PlanificadorPoliticaRL"]


def _asegurar_compatibilidad_numpy() -> None:
    """Garantiza compatibilidad entre checkpoints serializados con NumPy 2.x y entornos NumPy 1.x."""
    if hasattr(np, "core"):
        import importlib

        core = getattr(np, "core")
        sys.modules.setdefault("numpy._core", core)
        for submodulo in (
            "numeric",
            "multiarray",
            "umath",
            "_multiarray_umath",
            "fromnumeric",
        ):
            try:
                mod = importlib.import_module(f"numpy.core.{submodulo}")
                sys.modules.setdefault(f"numpy._core.{submodulo}", mod)
            except ImportError:
                pass

    try:
        from numpy.random import _pickle as np_pickle  # type: ignore[attr-defined]

        bit_generators = getattr(np_pickle, "BitGenerators", None)
        if isinstance(bit_generators, dict):
            for cls in list(bit_generators.values()):
                bit_generators.setdefault(cls, cls)
    except Exception:
        pass


def _detectar_dimensiones_checkpoint(
    ruta_zip: Path,
    max_pedidos_default: int,
) -> tuple[int, int]:
    """Inspecciona `policy.pth` dentro del checkpoint .zip de Stable-Baselines3.

    Obtiene las dimensiones reales `(max_pedidos_checkpoint, dim_obs_checkpoint)`
    con las que fue entrenado el modelo, permitiendo cargar tanto modelos estándar
    (30 slots) como variantes mayores sin discrepancia de tensores.
    """
    dim_obs_default = (
        max_pedidos_default * EntornoPlanificacionRL.FEATURES_POR_PEDIDO
        + EntornoPlanificacionRL.FEATURES_GLOBALES
    )
    try:
        import io
        import zipfile

        import torch

        with zipfile.ZipFile(ruta_zip, "r") as zf:
            if "policy.pth" not in zf.namelist():
                return max_pedidos_default, dim_obs_default
            estado = torch.load(
                io.BytesIO(zf.read("policy.pth")),
                map_location="cpu",
            )
        if not isinstance(estado, dict):
            return max_pedidos_default, dim_obs_default

        bias_acciones = estado.get("action_net.bias")
        pesos_entrada = estado.get("mlp_extractor.policy_net.0.weight")
        n_acciones = (
            int(bias_acciones.shape[0])
            if hasattr(bias_acciones, "shape") and len(bias_acciones.shape) == 1
            else max_pedidos_default
        )
        dim_obs = (
            int(pesos_entrada.shape[1])
            if hasattr(pesos_entrada, "shape") and len(pesos_entrada.shape) == 2
            else (
                n_acciones * EntornoPlanificacionRL.FEATURES_POR_PEDIDO
                + EntornoPlanificacionRL.FEATURES_GLOBALES
            )
        )
        return n_acciones, dim_obs
    except Exception:
        return max_pedidos_default, dim_obs_default


class PlanificadorPoliticaRL(PlanificadorTurno):
    """Planificador de turnos basado en inferencia neuronal guiada por máscaras de acción.

    Utiliza una política MaskablePPO para seleccionar secuencialmente el siguiente pedido
    a despachar, garantizando que ninguna acción viole restricciones duras de turno.
    """

    VERSION_PLANIFICADOR = "RL_POLITICA"

    def __init__(
        self,
        model_path: str | Path,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        configuracion_temporal: ConfiguracionPoliticaRL | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
        num_rollouts: int = 1,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo RL de la política RL: {self.model_path}")

        self.configuracion = configuracion or ConfiguracionPlanificacion()
        self.proveedor_viaje = proveedor_viaje
        self.configuracion_temporal = configuracion_temporal or ConfiguracionPoliticaRL()
        self.deterministic = deterministic
        self.num_rollouts = max(1, int(num_rollouts))
        _asegurar_compatibilidad_numpy()
        max_pedidos_ckpt, dim_obs = _detectar_dimensiones_checkpoint(
            self.model_path,
            max_pedidos_default=max_pedidos,
        )
        self.max_pedidos = max_pedidos_ckpt
        custom_objects = {
            "observation_space": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(dim_obs,),
                dtype=np.float32,
            ),
            "action_space": spaces.Discrete(self.max_pedidos),
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.2,
        }
        self.model = MaskablePPO.load(
            str(self.model_path),
            custom_objects=custom_objects,
        )
        self.ultima_permutacion: tuple[str, ...] = ()
        self.ultimas_permutaciones: tuple[tuple[str, ...], ...] = ()
        self.ultima_info_temporal: dict[str, Any] = {}

    def _ejecutar_un_episodio(
        self,
        env: EntornoPlanificacionRL,
        instancia: InstanciaTurno,
        seed_episodio: int,
        determinista: bool,
    ) -> tuple[PlanTurno, tuple[str, ...], dict[str, Any]]:
        if not determinista and hasattr(self.model, "set_random_seed"):
            self.model.set_random_seed(seed_episodio)

        observacion, _ = env.reset(seed=seed_episodio)
        terminado = False
        info_final: dict[str, Any] = {}

        while not terminado:
            mascara = env.action_masks()
            accion, _ = self.model.predict(
                observacion,
                action_masks=mascara,
                deterministic=determinista,
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
                raise RuntimeError("La política RL de la política RL produjo un episodio truncado.")

        plan = env.ultimo_plan
        if plan is None:
            raise RuntimeError("El entorno de la política RL finalizó sin producir un plan.")

        validacion = validar_plan(instancia, plan)
        if not validacion.valido:
            raise RuntimeError(
                "La política RL de la política RL produjo un plan inválido: "
                + " | ".join(validacion.errores)
            )

        return plan, env.permutacion_actual, info_final

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        """Genera un plan de despacho resolviendo el episodio con la política neuronal."""
        inicio_computo = perf_counter()
        env = EntornoPlanificacionRL(
            instancia=instancia,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
            configuracion_temporal=self.configuracion_temporal,
            max_pedidos=self.max_pedidos,
        )

        try:
            forma_modelo = getattr(self.model.observation_space, "shape", None)
            forma_entorno = env.observation_space.shape

            if forma_modelo != forma_entorno:
                raise RuntimeError(
                    f"El modelo no es compatible con el entorno temporal actual. "
                    f"Observación modelo={forma_modelo}, entorno={forma_entorno}."
                )

            seed_rl = (
                instancia.seed_ejecucion
                if instancia.seed_ejecucion > 0
                else instancia.seed_escenario
            )

            candidatos: list[
                tuple[tuple[int, float, float], PlanTurno, tuple[str, ...], dict[str, Any]]
            ] = []

            for k in range(self.num_rollouts):
                usar_det = self.deterministic if k == 0 else False
                plan_k, perm_k, info_k = self._ejecutar_un_episodio(
                    env=env,
                    instancia=instancia,
                    seed_episodio=seed_rl + k * 997,
                    determinista=usar_det,
                )
                tardios_k = int(info_k.get("pedidos_tardios_terminal", 0))
                tardanza_k = float(info_k.get("tardanza_total_terminal_min", 0.0))
                clave_k = (tardios_k, tardanza_k, float(plan_k.costo_estimado))
                candidatos.append((clave_k, plan_k, perm_k, info_k))

            candidatos.sort(key=lambda item: item[0])
            _, mejor_plan, mejor_perm, mejor_info = candidatos[0]

            perms_unicas: list[tuple[str, ...]] = []
            vistas: set[tuple[str, ...]] = set()
            for _, _, perm_c, _ in candidatos:
                if perm_c not in vistas:
                    vistas.add(perm_c)
                    perms_unicas.append(perm_c)

            mejor_plan.tiempo_computo_ms = (perf_counter() - inicio_computo) * 1000.0
            mejor_plan.warnings.append(self.VERSION_PLANIFICADOR)
            self.ultima_permutacion = mejor_perm
            self.ultimas_permutaciones = tuple(perms_unicas)
            self.ultima_info_temporal = mejor_info
            return mejor_plan
        finally:
            env.close()
