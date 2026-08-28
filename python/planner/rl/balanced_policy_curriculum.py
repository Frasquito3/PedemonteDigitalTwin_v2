from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EtapaExtensionTemporalV4RL:
    """Etapa diagnóstica para extender la política v4 al rango 9-12."""

    nombre: str
    min_pedidos_finales: int
    max_pedidos_finales: int
    timesteps: int
    eval_freq: int
    checkpoint_freq: int
    probabilidad_ventana_especifica: float
    probabilidad_patron_conflictivo: float
    probabilidad_replay_3_8: float
    probabilidad_replay_9_10: float
    probabilidad_volcador: float
    probabilidad_pedido_grande: float
    ancho_ventana_min: int
    ancho_ventana_max: int

    def __post_init__(self) -> None:
        if not self.nombre.strip():
            raise ValueError("nombre no puede estar vacío.")
        if self.min_pedidos_finales <= 0:
            raise ValueError("min_pedidos_finales debe ser > 0.")
        if self.max_pedidos_finales < self.min_pedidos_finales:
            raise ValueError("Rango de pedidos inválido.")

        for nombre, valor in {
            "timesteps": self.timesteps,
            "eval_freq": self.eval_freq,
            "checkpoint_freq": self.checkpoint_freq,
            "ancho_ventana_min": self.ancho_ventana_min,
            "ancho_ventana_max": self.ancho_ventana_max,
        }.items():
            if valor <= 0:
                raise ValueError(f"{nombre} debe ser > 0.")

        if self.ancho_ventana_max < self.ancho_ventana_min:
            raise ValueError(
                "ancho_ventana_max no puede ser menor que el mínimo."
            )

        probabilidades = {
            "probabilidad_ventana_especifica": (
                self.probabilidad_ventana_especifica
            ),
            "probabilidad_patron_conflictivo": (
                self.probabilidad_patron_conflictivo
            ),
            "probabilidad_replay_3_8": self.probabilidad_replay_3_8,
            "probabilidad_replay_9_10": self.probabilidad_replay_9_10,
            "probabilidad_volcador": self.probabilidad_volcador,
            "probabilidad_pedido_grande": self.probabilidad_pedido_grande,
        }
        for nombre, valor in probabilidades.items():
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nombre} debe estar entre 0 y 1.")

        suma_replay = (
            self.probabilidad_replay_3_8
            + self.probabilidad_replay_9_10
        )
        if suma_replay > 1.0:
            raise ValueError(
                "La suma de probabilidades de replay no puede superar 1."
            )

    @property
    def probabilidad_banda_actual(self) -> float:
        return 1.0 - (
            self.probabilidad_replay_3_8
            + self.probabilidad_replay_9_10
        )


def crear_curriculum_extension_temporal_v4_diagnostico(
) -> list[EtapaExtensionTemporalV4RL]:
    """
    Currículo intermedio de 48.000 pasos nominales.

    No sustituye al currículo completo de 350.000 pasos. Su único objetivo
    es comprobar si la formulación v4 puede extenderse a 9-12 pedidos sin
    perder B04, volcador, split ni la mejora ya observada en 3-8 pedidos.
    """

    return [
        EtapaExtensionTemporalV4RL(
            nombre="stage_04_focus_9_10",
            min_pedidos_finales=9,
            max_pedidos_finales=10,
            timesteps=16_000,
            eval_freq=4_000,
            checkpoint_freq=8_000,
            probabilidad_ventana_especifica=0.90,
            probabilidad_patron_conflictivo=0.75,
            probabilidad_replay_3_8=0.25,
            probabilidad_replay_9_10=0.0,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        ),
        EtapaExtensionTemporalV4RL(
            nombre="stage_05_focus_11_12",
            min_pedidos_finales=11,
            max_pedidos_finales=12,
            timesteps=32_000,
            eval_freq=4_000,
            checkpoint_freq=8_000,
            probabilidad_ventana_especifica=0.90,
            probabilidad_patron_conflictivo=0.75,
            probabilidad_replay_3_8=0.20,
            probabilidad_replay_9_10=0.30,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        ),
    ]
