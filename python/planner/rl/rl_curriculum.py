from dataclasses import dataclass


@dataclass(frozen=True)
class EtapaCurriculumRL:
    nombre: str

    min_pedidos_finales: int

    max_pedidos_finales: int

    timesteps: int

    eval_freq: int

    checkpoint_freq: int

    def __post_init__(self) -> None:
        if not self.nombre.strip():
            raise ValueError(
                "nombre no puede estar vacío."
            )

        if self.min_pedidos_finales <= 0:
            raise ValueError(
                "min_pedidos_finales debe ser > 0."
            )

        if (
            self.max_pedidos_finales
            < self.min_pedidos_finales
        ):
            raise ValueError(
                "Rango de pedidos inválido."
            )

        if self.timesteps <= 0:
            raise ValueError(
                "timesteps debe ser > 0."
            )

        if self.eval_freq <= 0:
            raise ValueError(
                "eval_freq debe ser > 0."
            )

        if self.checkpoint_freq <= 0:
            raise ValueError(
                "checkpoint_freq debe ser > 0."
            )


def crear_curriculum_fase9c(
) -> list[EtapaCurriculumRL]:
    return [
        EtapaCurriculumRL(
            nombre="stage_01_4_6",

            min_pedidos_finales=4,

            max_pedidos_finales=6,

            timesteps=20_000,

            eval_freq=5_000,

            checkpoint_freq=10_000,
        ),

        EtapaCurriculumRL(
            nombre="stage_02_4_8",

            min_pedidos_finales=4,

            max_pedidos_finales=8,

            timesteps=40_000,

            eval_freq=10_000,

            checkpoint_freq=20_000,
        ),

        EtapaCurriculumRL(
            nombre="stage_03_4_12",

            min_pedidos_finales=4,

            max_pedidos_finales=12,

            timesteps=80_000,

            eval_freq=20_000,

            checkpoint_freq=40_000,
        ),
    ]


def crear_curriculum_rapido_fase9c(
) -> list[EtapaCurriculumRL]:
    return [
        EtapaCurriculumRL(
            nombre="stage_01_4_6",

            min_pedidos_finales=4,

            max_pedidos_finales=6,

            timesteps=2_000,

            eval_freq=1_000,

            checkpoint_freq=1_000,
        ),

        EtapaCurriculumRL(
            nombre="stage_02_4_8",

            min_pedidos_finales=4,

            max_pedidos_finales=8,

            timesteps=4_000,

            eval_freq=2_000,

            checkpoint_freq=2_000,
        ),

        EtapaCurriculumRL(
            nombre="stage_03_4_12",

            min_pedidos_finales=4,

            max_pedidos_finales=12,

            timesteps=8_000,

            eval_freq=4_000,

            checkpoint_freq=4_000,
        ),
    ]