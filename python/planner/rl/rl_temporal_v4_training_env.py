from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.rl.rl_reward import ConfiguracionRewardRL
from planner.rl.rl_temporal_v4_config import ConfiguracionTemporalV4RL
from planner.rl.rl_temporal_v4_env import PedemonteTemporalV4PlanEnv
from planner.routing.objective import EstimacionPlan
from planner.routing.travel import ProveedorViaje


class GeneradorInstanciasV4Protocol(Protocol):
    def generar(self, seed: int) -> InstanciaTurno:
        ...


class PedemonteTemporalV4TrainingEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        generador: GeneradorInstanciasV4Protocol,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        seed_base: int = 164_000,
        semillas_fijas: Sequence[int] | None = None,
        max_pedidos: int = 30,
        escala_reward: float = 100.0,
        configuracion_reward: ConfiguracionRewardRL | None = None,
        configuracion_temporal: ConfiguracionTemporalV4RL | None = None,
    ) -> None:
        super().__init__()

        if max_pedidos <= 0:
            raise ValueError("max_pedidos debe ser > 0.")

        self.generador = generador
        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )
        self.proveedor_viaje = proveedor_viaje
        self.seed_base = int(seed_base)
        self.semillas_fijas = (
            tuple(int(seed) for seed in semillas_fijas)
            if semillas_fijas is not None
            else None
        )

        if self.semillas_fijas is not None and not self.semillas_fijas:
            raise ValueError("semillas_fijas no puede estar vacío.")

        self.max_pedidos = max_pedidos
        self.escala_reward = escala_reward
        self.configuracion_reward = configuracion_reward
        self.configuracion_temporal = configuracion_temporal
        self._indice_episodio = 0
        self.seed_instancia_actual = self._seed_para_indice(0)

        instancia_inicial = self.generador.generar(
            self.seed_instancia_actual
        )
        self._env = self._crear_entorno(instancia_inicial)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _seed_para_indice(self, indice: int) -> int:
        if indice < 0:
            raise ValueError("indice no puede ser negativo.")

        if self.semillas_fijas is not None:
            return int(
                self.semillas_fijas[indice % len(self.semillas_fijas)]
            )

        return self.seed_base + indice

    def _crear_entorno(
        self,
        instancia: InstanciaTurno,
    ) -> PedemonteTemporalV4PlanEnv:
        return PedemonteTemporalV4PlanEnv(
            instancia=instancia,
            configuracion=self.configuracion,
            proveedor_viaje=self.proveedor_viaje,
            max_pedidos=self.max_pedidos,
            escala_reward=self.escala_reward,
            configuracion_reward=self.configuracion_reward,
            configuracion_temporal=self.configuracion_temporal,
        )

    @property
    def instancia(self) -> InstanciaTurno:
        return self._env.instancia

    @property
    def ultimo_plan(self) -> PlanTurno | None:
        return self._env.ultimo_plan

    @property
    def ultima_estimacion(self) -> EstimacionPlan | None:
        return self._env.ultima_estimacion

    @property
    def permutacion_actual(self) -> tuple[str, ...]:
        return self._env.permutacion_actual

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self._indice_episodio = 0
            if self.semillas_fijas is None:
                self.seed_base = int(seed)

        seed_instancia = self._seed_para_indice(self._indice_episodio)
        self._indice_episodio += 1
        self.seed_instancia_actual = seed_instancia

        self._env.close()
        instancia = self.generador.generar(seed_instancia)
        nuevo_env = self._crear_entorno(instancia)

        if nuevo_env.action_space != self.action_space:
            nuevo_env.close()
            raise RuntimeError(
                "El action_space cambió entre instancias temporales v4."
            )

        if nuevo_env.observation_space != self.observation_space:
            nuevo_env.close()
            raise RuntimeError(
                "El observation_space cambió entre instancias temporales v4."
            )

        self._env = nuevo_env
        observacion, info = self._env.reset(
            seed=seed_instancia,
            options=options,
        )
        info = dict(info)
        info["seed_instancia"] = seed_instancia
        info["instancia_id"] = instancia.instancia_id
        info["version_entorno_rl"] = (
            PedemonteTemporalV4PlanEnv.VERSION_ENTORNO
        )
        return observacion, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observacion, reward, terminated, truncated, info = self._env.step(
            action
        )
        info = dict(info)
        info["seed_instancia"] = self.seed_instancia_actual
        info["instancia_id"] = self.instancia.instancia_id
        return observacion, reward, terminated, truncated, info

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()
