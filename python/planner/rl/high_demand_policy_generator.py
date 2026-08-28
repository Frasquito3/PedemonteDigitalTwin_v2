from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from planner.core.schema import InstanciaTurno


class GeneradorTemporalV4Protocol(Protocol):
    def generar(self, seed: int) -> InstanciaTurno:
        ...


@dataclass(frozen=True)
class FuenteReplayTemporalV4:
    nombre: str
    generador: GeneradorTemporalV4Protocol
    probabilidad: float
    mascara_seed: int

    def __post_init__(self) -> None:
        if not self.nombre.strip():
            raise ValueError("El nombre de la fuente no puede estar vacío.")
        if not 0.0 <= self.probabilidad <= 1.0:
            raise ValueError("La probabilidad debe estar entre 0 y 1.")
        if self.mascara_seed < 0:
            raise ValueError("mascara_seed no puede ser negativa.")


class GeneradorMezclaCompletaTemporalV4RL:
    """
    Mezcla una banda actual con fuentes de replay nombradas.

    La fuente elegida es determinista para una semilla dada. Cada replay
    recibe una transformación XOR distinta para desacoplar sus secuencias
    internas de la banda actual y de las demás fuentes.
    """

    MASCARA_SELECCION = 0x16D8

    def __init__(
        self,
        *,
        generador_actual: GeneradorTemporalV4Protocol,
        fuentes_replay: tuple[FuenteReplayTemporalV4, ...],
    ) -> None:
        nombres = [fuente.nombre for fuente in fuentes_replay]
        if len(nombres) != len(set(nombres)):
            raise ValueError("Los nombres de las fuentes deben ser únicos.")

        suma = sum(fuente.probabilidad for fuente in fuentes_replay)
        if suma > 1.0 + 1e-12:
            raise ValueError(
                "La suma de probabilidades de replay no puede superar 1."
            )
        if 1.0 - suma <= 0.0:
            raise ValueError(
                "La banda actual debe conservar probabilidad positiva."
            )

        self.generador_actual = generador_actual
        self.fuentes_replay = fuentes_replay
        self.probabilidad_banda_actual = 1.0 - suma

    def seleccionar_fuente(self, seed: int) -> str:
        valor = Random(int(seed) ^ self.MASCARA_SELECCION).random()
        acumulada = 0.0
        for fuente in self.fuentes_replay:
            acumulada += fuente.probabilidad
            if valor < acumulada:
                return fuente.nombre
        return "BANDA_ACTUAL"

    def generar(self, seed: int) -> InstanciaTurno:
        fuente_elegida = self.seleccionar_fuente(seed)
        if fuente_elegida == "BANDA_ACTUAL":
            return self.generador_actual.generar(int(seed))

        for fuente in self.fuentes_replay:
            if fuente.nombre == fuente_elegida:
                return fuente.generador.generar(
                    int(seed) ^ int(fuente.mascara_seed)
                )
        raise RuntimeError(f"Fuente desconocida: {fuente_elegida}.")
