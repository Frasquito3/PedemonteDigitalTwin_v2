from pathlib import Path
from time import perf_counter

import numpy as np

# pyrefly: ignore [missing-import]
from sb3_contrib import MaskablePPO

from planner.core.base import PlanificadorTurno

from planner.core.config import (
    ConfiguracionPlanificacion,
)

from planner.rl.rl_env import PedemontePlanEnv

from planner.routing.travel import ProveedorViaje

from planner.core.schema import (
    InstanciaTurno,
    PlanTurno,
)

from planner.domain.validator import validar_plan


class RLPlanner(
    PlanificadorTurno
):
    def __init__(
        self,
        model_path: str | Path,
        configuracion:
            ConfiguracionPlanificacion
            | None = None,
        proveedor_viaje:
            ProveedorViaje
            | None = None,
        max_pedidos: int = 30,
        deterministic: bool = True,
    ) -> None:
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "No existe el modelo RL: "
                f"{self.model_path}"
            )

        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )

        self.proveedor_viaje = proveedor_viaje

        self.max_pedidos = max_pedidos

        self.deterministic = deterministic

        self.model = MaskablePPO.load(
            str(
                self.model_path
            )
        )

        self.ultima_permutacion: tuple[str, ...] = ()

    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        inicio_computo = perf_counter()

        env = PedemontePlanEnv(
            instancia=instancia,

            configuracion=(
                self.configuracion
            ),

            proveedor_viaje=(
                self.proveedor_viaje
            ),

            max_pedidos=(
                self.max_pedidos
            ),
        )

        observacion, _ = env.reset(
            seed=instancia.seed_escenario
        )

        terminado = False

        while not terminado:
            mascara = env.action_masks()

            accion, _ = self.model.predict(
                observacion,

                action_masks=mascara,

                deterministic=(
                    self.deterministic
                ),
            )

            accion_entera = int(
                np.asarray(
                    accion
                ).item()
            )

            (
                observacion,
                _,
                terminado,
                truncado,
                _,
            ) = env.step(
                accion_entera
            )

            if truncado:
                raise RuntimeError(
                    "La política RL produjo un "
                    "episodio truncado."
                )

        plan = env.ultimo_plan

        if plan is None:
            raise RuntimeError(
                "El entorno finalizó sin producir "
                "un PlanTurno."
            )

        validacion = validar_plan(
            instancia,
            plan,
        )

        if not validacion.valido:
            raise RuntimeError(
                "La política RL produjo un plan "
                "inválido: "
                + " | ".join(
                    validacion.errores
                )
            )

        plan.tiempo_computo_ms = (
            perf_counter()
            - inicio_computo
        ) * 1000.0

        self.ultima_permutacion = (
            env.permutacion_actual
        )

        env.close()

        return plan