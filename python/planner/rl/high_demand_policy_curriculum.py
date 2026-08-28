from __future__ import annotations

from dataclasses import dataclass


TIMESTEPS_BASE_EXTENSION_V4 = 68_288
TIMESTEPS_COMPLETOS_ADICIONALES_V4 = 280_000
TIMESTEPS_ACUMULADOS_MAXIMOS_V4 = (
    TIMESTEPS_BASE_EXTENSION_V4 + TIMESTEPS_COMPLETOS_ADICIONALES_V4
)


@dataclass(frozen=True)
class EtapaEntrenamientoCompletoTemporalV4RL:
    """Etapa de continuación completa enfocada en 11-12 pedidos."""

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
    probabilidad_replay_general_11_12: float
    probabilidad_replay_exactos_12: float
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
            "probabilidad_replay_general_11_12": (
                self.probabilidad_replay_general_11_12
            ),
            "probabilidad_replay_exactos_12": (
                self.probabilidad_replay_exactos_12
            ),
            "probabilidad_volcador": self.probabilidad_volcador,
            "probabilidad_pedido_grande": self.probabilidad_pedido_grande,
        }
        for nombre, valor in probabilidades.items():
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nombre} debe estar entre 0 y 1.")

        if self.probabilidad_replay_total > 1.0:
            raise ValueError(
                "La suma de probabilidades de replay no puede superar 1."
            )
        if self.probabilidad_banda_actual <= 0.0:
            raise ValueError(
                "La banda actual debe conservar probabilidad positiva."
            )

    @property
    def probabilidad_replay_total(self) -> float:
        return (
            self.probabilidad_replay_3_8
            + self.probabilidad_replay_9_10
            + self.probabilidad_replay_general_11_12
            + self.probabilidad_replay_exactos_12
        )

    @property
    def probabilidad_banda_actual(self) -> float:
        return 1.0 - self.probabilidad_replay_total


def crear_curriculum_entrenamiento_completo_temporal_v4(
) -> list[EtapaEntrenamientoCompletoTemporalV4RL]:
    """
    Continuación de 280.000 pasos desde el checkpoint de 68.288 pasos.

    El máximo acumulado nominal es 348.288 pasos; el checkpoint seleccionado puede quedar antes por selección externa. La observación, el reward y la
    máscara temporal permanecen sin cambios. El currículo ataca primero los
    escenarios generales de 11-12 pedidos, después los casos de exactamente
    12 pedidos y termina con una consolidación balanceada de 9-12.
    """

    return [
        EtapaEntrenamientoCompletoTemporalV4RL(
            nombre="stage_06_general_11_12",
            min_pedidos_finales=11,
            max_pedidos_finales=12,
            timesteps=80_000,
            eval_freq=10_000,
            checkpoint_freq=20_000,
            probabilidad_ventana_especifica=0.90,
            probabilidad_patron_conflictivo=0.15,
            probabilidad_replay_3_8=0.15,
            probabilidad_replay_9_10=0.15,
            probabilidad_replay_general_11_12=0.0,
            probabilidad_replay_exactos_12=0.15,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        ),
        EtapaEntrenamientoCompletoTemporalV4RL(
            nombre="stage_07_exact_12_balanced",
            min_pedidos_finales=12,
            max_pedidos_finales=12,
            timesteps=120_000,
            eval_freq=15_000,
            checkpoint_freq=30_000,
            probabilidad_ventana_especifica=0.90,
            probabilidad_patron_conflictivo=0.35,
            probabilidad_replay_3_8=0.10,
            probabilidad_replay_9_10=0.15,
            probabilidad_replay_general_11_12=0.25,
            probabilidad_replay_exactos_12=0.0,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        ),
        EtapaEntrenamientoCompletoTemporalV4RL(
            nombre="stage_08_consolidation_9_12",
            min_pedidos_finales=9,
            max_pedidos_finales=12,
            timesteps=80_000,
            eval_freq=10_000,
            checkpoint_freq=20_000,
            probabilidad_ventana_especifica=0.90,
            probabilidad_patron_conflictivo=0.35,
            probabilidad_replay_3_8=0.15,
            probabilidad_replay_9_10=0.15,
            probabilidad_replay_general_11_12=0.20,
            probabilidad_replay_exactos_12=0.25,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        ),
    ]
