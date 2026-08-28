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
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.policy_env import EntornoPlanificacionRL
from planner.routing.travel import ProveedorViaje


class PlanificadorPoliticaRL(PlanificadorTurno):
    VERSION_PLANIFICADOR = "RL_POLITICA"

    def __init__(
        self,
        model_path: str | Path,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        configuracion_temporal: ConfiguracionPoliticaRL | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "No existe el modelo RL de la política RL: "
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
            else ConfiguracionPoliticaRL()
        )
        self.max_pedidos = max_pedidos
        self.deterministic = deterministic
        self.model = MaskablePPO.load(str(self.model_path))
        self.ultima_permutacion: tuple[str, ...] = ()
        self.ultima_info_temporal: dict[str, Any] = {}

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        inicio_computo = perf_counter()
        env = EntornoPlanificacionRL(
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
                    f"actual. Observación modelo={forma_modelo}, "
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
                        "La política RL de la política RL produjo un episodio "
                        "truncado."
                    )

            plan = env.ultimo_plan
            if plan is None:
                raise RuntimeError(
                    "El entorno de la política RL finalizó sin producir un plan."
                )

            validacion = validar_plan(instancia, plan)
            if not validacion.valido:
                raise RuntimeError(
                    "La política RL de la política RL produjo un plan inválido: "
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
