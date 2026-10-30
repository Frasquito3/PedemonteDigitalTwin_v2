"""Generadores de instancias sintéticas avanzadas para el curriculum de RL.

Este módulo implementa generadores específicos para inyectar patrones conflictivos
de ventanas temporales (temprano / medio / tardío) y esquemas mixtos con replay de
instancias básicas para prevenir el olvido catastrófico durante el entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Protocol

from planner.core.schema import InstanciaTurno, PedidoInput
from planner.domain.validator import validar_instancia
from planner.rl.instance_generator import GeneradorInstanciasRL

__all__ = [
    "ConfiguracionGeneradorPoliticaRL",
    "GeneradorInstanciasPoliticaRL",
    "GeneradorInstanciasProtocol",
    "GeneradorMixtoPoliticaRL",
]


@dataclass(frozen=True)
class ConfiguracionGeneradorPoliticaRL:
    """Configuración para la inyección de patrones de conflicto temporal."""

    probabilidad_patron_ventanas_conflictivas: float = 0.75
    unidades_por_pedido_patron: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.probabilidad_patron_ventanas_conflictivas <= 1.0:
            raise ValueError("probabilidad_patron_ventanas_conflictivas debe estar entre 0 y 1.")
        if self.unidades_por_pedido_patron <= 0:
            raise ValueError("unidades_por_pedido_patron debe ser > 0.")


class GeneradorInstanciasProtocol(Protocol):
    """Protocolo de generador de instancias basado en semilla."""

    def generar(self, seed: int) -> InstanciaTurno:
        """Genera una instancia de turno a partir de una semilla numérica."""
        ...


class GeneradorInstanciasPoliticaRL:
    """Generador decorador que introduce patrones de ventanas horarias conflictivas.

    Selecciona tres pedidos sin volcador ni partición y les asigna ventanas ajustadas
    (temprano, medio, tardío) calculadas a partir del horizonte del turno, simulando
    situaciones donde una heurística voraz simple suele quedar atrapada en óptimos locales.
    """

    MARCA_PATRON = "PATRON_CONFLICTIVO_POLITICA_RL"

    def __init__(
        self,
        generador_base: GeneradorInstanciasRL,
        configuracion: ConfiguracionGeneradorPoliticaRL | None = None,
    ) -> None:
        self.generador_base = generador_base
        self.configuracion = (
            configuracion if configuracion is not None else ConfiguracionGeneradorPoliticaRL()
        )

    def generar(self, seed: int) -> InstanciaTurno:
        """Genera una instancia modificando pedidos candidatos con el patrón de conflicto."""
        instancia = self.generador_base.generar(seed)
        rng = Random(seed ^ 0x16D4)

        if rng.random() > self.configuracion.probabilidad_patron_ventanas_conflictivas:
            return instancia

        candidatos = [
            pedido
            for pedido in instancia.pedidos
            if pedido.total_partes == 1 and not pedido.requiere_volcador
        ]

        if len(candidatos) < 3:
            return instancia

        temprano = rng.choice(candidatos)
        restantes = [pedido for pedido in candidatos if pedido.pedido_id != temprano.pedido_id]
        tardio = min(restantes, key=lambda pedido: self._distancia_cuadrada(temprano, pedido))
        restantes_sin_tardio = [
            pedido for pedido in restantes if pedido.pedido_id != tardio.pedido_id
        ]
        medio = max(
            restantes_sin_tardio, key=lambda pedido: self._distancia_cuadrada(temprano, pedido)
        )

        horizonte = max(1, instancia.hora_fin_objetivo_min - instancia.hora_inicio_turno_min)
        inicio = instancia.hora_inicio_turno_min
        ventanas = {
            temprano.pedido_id: (
                inicio + round(0.05 * horizonte),
                inicio + round(0.25 * horizonte),
                "TEMPRANO",
            ),
            medio.pedido_id: (
                inicio + round(0.32 * horizonte),
                inicio + round(0.58 * horizonte),
                "MEDIO",
            ),
            tardio.pedido_id: (
                inicio + round(0.66 * horizonte),
                inicio + round(0.92 * horizonte),
                "TARDIO_TRAMPA",
            ),
        }
        unidades_patron = min(
            self.configuracion.unidades_por_pedido_patron,
            max(1, instancia.capacidad_camion // 3),
        )
        pedidos_nuevos: list[PedidoInput] = []

        for pedido in instancia.pedidos:
            ventana = ventanas.get(pedido.pedido_id)
            if ventana is None:
                pedidos_nuevos.append(pedido)
                continue

            hora_desde, hora_hasta, rol = ventana
            marca = f"{self.MARCA_PATRON};ROL={rol};SEED={seed}"
            observaciones = f"{pedido.observaciones}; {marca}" if pedido.observaciones else marca
            pedidos_nuevos.append(
                replace(
                    pedido,
                    unidades_capacidad=unidades_patron,
                    requiere_volcador=False,
                    tiene_ventana_especifica=True,
                    hora_desde_min=hora_desde,
                    hora_hasta_min=hora_hasta,
                    observaciones=observaciones,
                )
            )

        instancia_temporal = replace(
            instancia,
            instancia_id=f"{instancia.instancia_id}-POLITICA-RL",
            pedidos=pedidos_nuevos,
        )
        errores = validar_instancia(instancia_temporal)
        if errores:
            raise RuntimeError(
                "El generador de la política RL produjo una instancia inválida: "
                + " | ".join(errores)
            )

        return instancia_temporal

    @staticmethod
    def _distancia_cuadrada(origen: PedidoInput, destino: PedidoInput) -> float:
        delta_lat = origen.latitud - destino.latitud
        delta_lon = origen.longitud - destino.longitud
        return delta_lat * delta_lat + delta_lon * delta_lon


class GeneradorMixtoPoliticaRL:
    """Generador híbrido que intercala instancias de la etapa actual con replay de etapas base."""

    def __init__(
        self,
        generador_actual: GeneradorInstanciasProtocol,
        generador_core: GeneradorInstanciasProtocol,
        probabilidad_replay_core: float,
    ) -> None:
        if not 0.0 <= probabilidad_replay_core <= 1.0:
            raise ValueError("probabilidad_replay_core debe estar entre 0 y 1.")
        self.generador_actual = generador_actual
        self.generador_core = generador_core
        self.probabilidad_replay_core = probabilidad_replay_core

    def generar(self, seed: int) -> InstanciaTurno:
        """Genera una instancia de la etapa actual o una de repetición según la probabilidad."""
        rng = Random(seed ^ 0xC0E4)
        if rng.random() < self.probabilidad_replay_core:
            return self.generador_core.generar(seed ^ 0x400000)
        return self.generador_actual.generar(seed)
