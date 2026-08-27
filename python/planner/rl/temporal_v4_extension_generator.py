from __future__ import annotations

from random import Random
from typing import Protocol

from planner.core.schema import InstanciaTurno


class GeneradorExtensionProtocol(Protocol):
    def generar(self, seed: int) -> InstanciaTurno:
        ...


class GeneradorReplayMultibandaTemporalV4RL:
    """
    Mezcla la banda objetivo con dos memorias de replay.

    La selección es determinista para una semilla dada. Las semillas se
    transforman por fuente para evitar que dos generadores reciban la misma
    secuencia interna cuando se cambia de banda.
    """

    def __init__(
        self,
        *,
        generador_actual: GeneradorExtensionProtocol,
        generador_replay_3_8: GeneradorExtensionProtocol,
        generador_replay_9_10: GeneradorExtensionProtocol,
        probabilidad_replay_3_8: float,
        probabilidad_replay_9_10: float,
    ) -> None:
        for nombre, valor in {
            "probabilidad_replay_3_8": probabilidad_replay_3_8,
            "probabilidad_replay_9_10": probabilidad_replay_9_10,
        }.items():
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nombre} debe estar entre 0 y 1.")

        if probabilidad_replay_3_8 + probabilidad_replay_9_10 > 1.0:
            raise ValueError(
                "La suma de probabilidades de replay no puede superar 1."
            )

        self.generador_actual = generador_actual
        self.generador_replay_3_8 = generador_replay_3_8
        self.generador_replay_9_10 = generador_replay_9_10
        self.probabilidad_replay_3_8 = probabilidad_replay_3_8
        self.probabilidad_replay_9_10 = probabilidad_replay_9_10

    def seleccionar_fuente(self, seed: int) -> str:
        rng = Random(int(seed) ^ 0x16D6)
        valor = rng.random()
        if valor < self.probabilidad_replay_3_8:
            return "REPLAY_3_8"
        if valor < (
            self.probabilidad_replay_3_8
            + self.probabilidad_replay_9_10
        ):
            return "REPLAY_9_10"
        return "BANDA_ACTUAL"

    def generar(self, seed: int) -> InstanciaTurno:
        fuente = self.seleccionar_fuente(seed)
        if fuente == "REPLAY_3_8":
            return self.generador_replay_3_8.generar(
                int(seed) ^ 0x380000
            )
        if fuente == "REPLAY_9_10":
            return self.generador_replay_9_10.generar(
                int(seed) ^ 0x910000
            )
        return self.generador_actual.generar(int(seed))
