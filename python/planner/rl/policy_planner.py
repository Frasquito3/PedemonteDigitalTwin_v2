from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

# pyrefly: ignore [missing-import]
from sb3_contrib import MaskablePPO

from planner.core.base import PlanificadorTurno
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.domain.validator import validar_plan
from planner.rl.policy_config import ConfiguracionTemporalV4RL
from planner.rl.policy_env import PedemonteTemporalV4PlanEnv
from planner.routing.travel import ProveedorViaje


class RLTemporalV4Planner(PlanificadorTurno):
    VERSION_PLANIFICADOR = "RL_TEMPORAL_V4"

    def __init__(
        self,
        model_path: str | Path,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        configuracion_temporal: ConfiguracionTemporalV4RL | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "No existe el modelo RL temporal v4: "
                f"{self.model_path}"
            )

        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )
        self.proveedor_viaje = proveedor_viaje
        self.configuracion_temporal = (
            configuracion_temporal
            if configuracion_temporal is not None
            else ConfiguracionTemporalV4RL()
        )
        self.max_pedidos = max_pedidos
        self.deterministic = deterministic
        self.model = MaskablePPO.load(str(self.model_path))
        self.ultima_permutacion: tuple[str, ...] = ()
        self.ultima_info_temporal: dict[str, Any] = {}

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        inicio_computo = perf_counter()
        env = PedemonteTemporalV4PlanEnv(
            instancia=instancia,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
            configuracion_temporal=self.configuracion_temporal,
            max_pedidos=self.max_pedidos,
        )

        try:
            forma_modelo = getattr(
                self.model.observation_space,
                "shape",
                None,
            )
            forma_entorno = env.observation_space.shape

            if forma_modelo != forma_entorno:
                raise RuntimeError(
                    "El modelo no es compatible con el entorno temporal "
                    f"v4. Observación modelo={forma_modelo}, "
                    f"entorno={forma_entorno}."
                )

            observacion, _ = env.reset(seed=instancia.seed_escenario)
            terminado = False
            info_final: dict[str, Any] = {}

            while not terminado:
                mascara = env.action_masks()
                accion, _ = self.model.predict(
                    observacion,
                    action_masks=mascara,
                    deterministic=self.deterministic,
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
                        "La política RL temporal v4 produjo un episodio "
                        "truncado."
                    )

            plan = env.ultimo_plan
            if plan is None:
                raise RuntimeError(
                    "El entorno temporal v4 finalizó sin producir un plan."
                )

            validacion = validar_plan(instancia, plan)
            if not validacion.valido:
                raise RuntimeError(
                    "La política RL temporal v4 produjo un plan inválido: "
                    + " | ".join(validacion.errores)
                )

            plan.tiempo_computo_ms = (
                perf_counter() - inicio_computo
            ) * 1000.0
            plan.warnings.append(self.VERSION_PLANIFICADOR)
            self.ultima_permutacion = env.permutacion_actual
            self.ultima_info_temporal = info_final
            return plan
        finally:
            env.close()
