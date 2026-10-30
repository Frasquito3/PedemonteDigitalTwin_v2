"""Entorno Gym de episodios continuos para entrenamiento con Stable-Baselines3.

Este módulo provee un wrapper estándar de Gymnasium (`EntornoEntrenamientoRL`) que
re-genera dinámicamente instancias de planificación en cada invocación de `reset()`,
garantizando espacios de observación y acción estables entre episodios.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.policy_env import EntornoPlanificacionRL
from planner.rl.reward import ConfiguracionRewardRL
from planner.routing.objective import EstimacionPlan
from planner.routing.travel import ProveedorViaje

__all__ = [
    "EntornoEntrenamientoRL",
    "GeneradorEntrenamientoProtocol",
]


class GeneradorEntrenamientoProtocol(Protocol):
    """Protocolo estructural para generadores de instancias basados en semilla."""

    def generar(self, seed: int) -> InstanciaTurno:
        """Genera una instancia de turno para la semilla dada."""
        ...


class EntornoEntrenamientoRL(gym.Env[np.ndarray, int]):
    """Entorno de entrenamiento compatible con SB3 sobre secuencias de instancias.

    Re-instancia internamente `EntornoPlanificacionRL` al finalizar cada episodio,
    manteniendo la compatibilidad del espacio vectorial de estado y máscaras de acción.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        generador: GeneradorEntrenamientoProtocol,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        seed_base: int = 164_000,
        semillas_fijas: Sequence[int] | None = None,
        max_pedidos: int = 30,
        escala_reward: float = 100.0,
        configuracion_reward: ConfiguracionRewardRL | None = None,
        configuracion_temporal: ConfiguracionPoliticaRL | None = None,
    ) -> None:
        super().__init__()

        if max_pedidos <= 0:
            raise ValueError("max_pedidos debe ser > 0.")

        self.generador = generador
        self.configuracion = (
            configuracion if configuracion is not None else ConfiguracionPlanificacion()
        )
        self.proveedor_viaje = proveedor_viaje
        self.seed_base = int(seed_base)
        self.semillas_fijas = (
            tuple(int(seed) for seed in semillas_fijas) if semillas_fijas is not None else None
        )

        if self.semillas_fijas is not None and not self.semillas_fijas:
            raise ValueError("semillas_fijas no puede estar vacío.")

        self.max_pedidos = max_pedidos
        self.escala_reward = escala_reward
        self.configuracion_reward = configuracion_reward
        self.configuracion_temporal = configuracion_temporal
        self._indice_episodio = 0
        self.seed_instancia_actual = self._seed_para_indice(0)

        instancia_inicial = self.generador.generar(self.seed_instancia_actual)
        self._env = self._crear_entorno(instancia_inicial)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _seed_para_indice(self, indice: int) -> int:
        if indice < 0:
            raise ValueError("indice no puede ser negativo.")

        if self.semillas_fijas is not None:
            return int(self.semillas_fijas[indice % len(self.semillas_fijas)])

        return self.seed_base + indice

    def _crear_entorno(self, instancia: InstanciaTurno) -> EntornoPlanificacionRL:
        return EntornoPlanificacionRL(
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
        """Instancia del turno actualmente activo en el episodio."""
        return self._env.instancia

    @property
    def ultimo_plan(self) -> PlanTurno | None:
        """Último plan generado por el entorno interno."""
        return self._env.ultimo_plan

    @property
    def ultima_estimacion(self) -> EstimacionPlan | None:
        """Última estimación de métricas y costos calculada."""
        return self._env.ultima_estimacion

    @property
    def permutacion_actual(self) -> tuple[str, ...]:
        """Secuencia acumulada de IDs de pedidos despachados en el episodio."""
        return self._env.permutacion_actual

    def action_masks(self) -> np.ndarray:
        """Máscara booleana binaria de acciones válidas."""
        return self._env.action_masks()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reinicia el entorno, generando la siguiente instancia del curriculum."""
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
            raise RuntimeError("El action_space cambió entre instancias de entrenamiento.")

        if nuevo_env.observation_space != self.observation_space:
            nuevo_env.close()
            raise RuntimeError("El observation_space cambió entre instancias de entrenamiento.")

        self._env = nuevo_env
        observacion, info = self._env.reset(
            seed=seed_instancia,
            options=options,
        )
        info = dict(info)
        info["seed_instancia"] = seed_instancia
        info["instancia_id"] = instancia.instancia_id
        info["version_entorno_rl"] = EntornoPlanificacionRL.VERSION_ENTORNO
        return observacion, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Ejecuta un paso de simulación despachando un pedido."""
        observacion, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)
        info["seed_instancia"] = self.seed_instancia_actual
        info["instancia_id"] = self.instancia.instancia_id
        return observacion, reward, terminated, truncated, info

    def render(self) -> None:
        """Renderiza el estado del entorno si aplica."""
        self._env.render()

    def close(self) -> None:
        """Cierra el entorno interno y libera recursos asociados."""
        self._env.close()
