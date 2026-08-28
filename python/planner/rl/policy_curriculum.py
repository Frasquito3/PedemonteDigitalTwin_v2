from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EtapaCurriculumTemporalV4RL:
    nombre: str
    min_pedidos_finales: int
    max_pedidos_finales: int
    timesteps: int
    eval_freq: int
    checkpoint_freq: int
    probabilidad_ventana_especifica: float
    probabilidad_patron_conflictivo: float
    probabilidad_replay_core: float
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

        for nombre, valor in {
            "probabilidad_ventana_especifica": (
                self.probabilidad_ventana_especifica
            ),
            "probabilidad_patron_conflictivo": (
                self.probabilidad_patron_conflictivo
            ),
            "probabilidad_replay_core": self.probabilidad_replay_core,
            "probabilidad_volcador": self.probabilidad_volcador,
            "probabilidad_pedido_grande": self.probabilidad_pedido_grande,
        }.items():
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nombre} debe estar entre 0 y 1.")


def crear_curriculum_temporal_v4(
) -> list[EtapaCurriculumTemporalV4RL]:
    return [
        EtapaCurriculumTemporalV4RL(
            nombre="stage_01_second_order_core_3_5",
            min_pedidos_finales=3,
            max_pedidos_finales=5,
            timesteps=50_000,
            eval_freq=10_000,
            checkpoint_freq=25_000,
            probabilidad_ventana_especifica=1.0,
            probabilidad_patron_conflictivo=0.95,
            probabilidad_replay_core=1.0,
            probabilidad_volcador=0.0,
            probabilidad_pedido_grande=0.0,
            ancho_ventana_min=45,
            ancho_ventana_max=90,
        ),
        EtapaCurriculumTemporalV4RL(
            nombre="stage_02_second_order_mixed_4_8",
            min_pedidos_finales=4,
            max_pedidos_finales=8,
            timesteps=100_000,
            eval_freq=20_000,
            checkpoint_freq=50_000,
            probabilidad_ventana_especifica=0.95,
            probabilidad_patron_conflictivo=0.85,
            probabilidad_replay_core=0.30,
            probabilidad_volcador=0.10,
            probabilidad_pedido_grande=0.0,
            ancho_ventana_min=45,
            ancho_ventana_max=120,
        ),
        EtapaCurriculumTemporalV4RL(
            nombre="stage_03_second_order_operational_4_12",
            min_pedidos_finales=4,
            max_pedidos_finales=12,
            timesteps=200_000,
            eval_freq=40_000,
            checkpoint_freq=100_000,
            probabilidad_ventana_especifica=0.85,
            probabilidad_patron_conflictivo=0.70,
            probabilidad_replay_core=0.20,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        ),
    ]


def crear_curriculum_temporal_v4_rapido(
) -> list[EtapaCurriculumTemporalV4RL]:
    return [
        EtapaCurriculumTemporalV4RL(
            nombre="stage_01_second_order_core_3_5",
            min_pedidos_finales=3,
            max_pedidos_finales=5,
            timesteps=3_000,
            eval_freq=1_000,
            checkpoint_freq=1_000,
            probabilidad_ventana_especifica=1.0,
            probabilidad_patron_conflictivo=0.95,
            probabilidad_replay_core=1.0,
            probabilidad_volcador=0.0,
            probabilidad_pedido_grande=0.0,
            ancho_ventana_min=45,
            ancho_ventana_max=90,
        ),
        EtapaCurriculumTemporalV4RL(
            nombre="stage_02_second_order_mixed_4_8",
            min_pedidos_finales=4,
            max_pedidos_finales=8,
            timesteps=6_000,
            eval_freq=2_000,
            checkpoint_freq=2_000,
            probabilidad_ventana_especifica=0.95,
            probabilidad_patron_conflictivo=0.85,
            probabilidad_replay_core=0.30,
            probabilidad_volcador=0.10,
            probabilidad_pedido_grande=0.0,
            ancho_ventana_min=45,
            ancho_ventana_max=120,
        ),
        EtapaCurriculumTemporalV4RL(
            nombre="stage_03_second_order_operational_4_12",
            min_pedidos_finales=4,
            max_pedidos_finales=12,
            timesteps=12_000,
            eval_freq=4_000,
            checkpoint_freq=4_000,
            probabilidad_ventana_especifica=0.85,
            probabilidad_patron_conflictivo=0.70,
            probabilidad_replay_core=0.20,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_ventana_min=45,
            ancho_ventana_max=150,
        ),
    ]
