from dataclasses import dataclass, replace
from random import Random

from planner.domain.preprocess import preprocesar_instancia

from planner.core.schema import (
    InstanciaTurno,
    PedidoInput,
    Turno,
)

from planner.domain.validator import validar_instancia


@dataclass(frozen=True)
class ConfiguracionGeneradorInstancias:
    min_pedidos_finales: int = 4

    max_pedidos_finales: int = 8

    capacidad_camion: int = 8

    cantidad_camiones: int = 2

    probabilidad_volcador: float = 0.15

    probabilidad_ventana_especifica: float = 0.65

    probabilidad_pedido_mayor_capacidad: float = 0.10

    max_unidades_pedido_grande: int = 16

    desplazamiento_max_grados: float = 0.035

    ancho_ventana_min: int = 60

    ancho_ventana_max: int = 150

    max_intentos_generacion: int = 500


    def __post_init__(self) -> None:
        if self.min_pedidos_finales <= 0:
            raise ValueError(
                "min_pedidos_finales debe ser > 0."
            )

        if (
            self.max_pedidos_finales
            < self.min_pedidos_finales
        ):
            raise ValueError(
                "max_pedidos_finales no puede ser "
                "menor que min_pedidos_finales."
            )

        if self.capacidad_camion <= 0:
            raise ValueError(
                "capacidad_camion debe ser > 0."
            )

        if self.cantidad_camiones <= 0:
            raise ValueError(
                "cantidad_camiones debe ser > 0."
            )

        probabilidades = (
            self.probabilidad_volcador,
            self.probabilidad_ventana_especifica,
            self.probabilidad_pedido_mayor_capacidad,
        )

        if any(
            probabilidad < 0.0
            or probabilidad > 1.0
            for probabilidad in probabilidades
        ):
            raise ValueError(
                "Las probabilidades deben estar "
                "entre 0 y 1."
            )

        if (
            self.max_unidades_pedido_grande
            <= self.capacidad_camion
        ):
            raise ValueError(
                "max_unidades_pedido_grande debe "
                "superar la capacidad del camión."
            )

        if self.desplazamiento_max_grados <= 0.0:
            raise ValueError(
                "desplazamiento_max_grados debe ser > 0."
            )

        if self.ancho_ventana_min <= 0:
            raise ValueError(
                "ancho_ventana_min debe ser > 0."
            )

        if (
            self.ancho_ventana_max
            < self.ancho_ventana_min
        ):
            raise ValueError(
                "ancho_ventana_max no puede ser "
                "menor que ancho_ventana_min."
            )


class GeneradorInstanciasRL:
    def __init__(
        self,
        configuracion:
            ConfiguracionGeneradorInstancias
            | None = None,
    ) -> None:
        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionGeneradorInstancias()
        )

    def generar(
        self,
        seed: int,
    ) -> InstanciaTurno:
        rng = Random(seed)

        for intento in range(
            self.configuracion
            .max_intentos_generacion
        ):
            instancia_raw = (
                self._generar_instancia_raw(
                    seed=seed,
                    intento=intento,
                    rng=rng,
                )
            )

            instancia = preprocesar_instancia(
                instancia_raw
            )

            cantidad_final = len(
                instancia.pedidos
            )

            if not (
                self.configuracion
                .min_pedidos_finales
                <= cantidad_final
                <= self.configuracion
                .max_pedidos_finales
            ):
                continue

            # Evita que el agente aprenda una relación
            # artificial entre el índice de acción y
            # el tipo de pedido.
            pedidos_mezclados = list(
                instancia.pedidos
            )

            rng.shuffle(
                pedidos_mezclados
            )

            instancia = replace(
                instancia,
                pedidos=pedidos_mezclados,
            )

            errores = validar_instancia(
                instancia
            )

            if errores:
                continue

            return instancia

        raise RuntimeError(
            "No fue posible generar una instancia "
            "válida después de "
            f"{self.configuracion.max_intentos_generacion} "
            "intentos."
        )

    def _generar_instancia_raw(
        self,
        seed: int,
        intento: int,
        rng: Random,
    ) -> InstanciaTurno:
        lat_corralon = -32.8495006
        lon_corralon = -60.722653

        turno = (
            Turno.MANANA
            if rng.random() < 0.5
            else Turno.TARDE
        )

        if turno == Turno.MANANA:
            inicio_turno = 450
            fin_objetivo = 720

        else:
            inicio_turno = 840
            fin_objetivo = 1020

        fin_tolerancia = (
            fin_objetivo + 15
        )

        cantidad_raw = rng.randint(
            self.configuracion
            .min_pedidos_finales,

            self.configuracion
            .max_pedidos_finales,
        )

        pedidos: list[PedidoInput] = []

        for indice in range(
            cantidad_raw
        ):
            pedido_id = (
                f"P{indice + 1:03d}"
            )

            unidades = (
                self._generar_unidades(
                    rng
                )
            )

            requiere_volcador = (
                rng.random()
                <
                self.configuracion
                .probabilidad_volcador
            )

            latitud, longitud = (
                self._generar_coordenadas(
                    lat_corralon,
                    lon_corralon,
                    rng,
                )
            )

            (
                tiene_ventana,
                hora_desde,
                hora_hasta,
            ) = self._generar_ventana(
                inicio_turno,
                fin_objetivo,
                rng,
            )

            pedidos.append(
                PedidoInput(
                    pedido_id=pedido_id,

                    pedido_original_id=pedido_id,

                    numero_parte=1,

                    total_partes=1,

                    turno=turno,

                    latitud=latitud,

                    longitud=longitud,

                    unidades_capacidad=unidades,

                    requiere_volcador=(
                        requiere_volcador
                    ),

                    tiene_ventana_especifica=(
                        tiene_ventana
                    ),

                    hora_desde_min=hora_desde,

                    hora_hasta_min=hora_hasta,
                )
            )

        return InstanciaTurno(
            instancia_id=(
                f"RL-GEN-{seed}-{intento}"
            ),

            fecha_operacion=(
                f"2026-09-"
                f"{seed % 28 + 1:02d}"
            ),

            turno=turno,

            pedidos=pedidos,

            lat_corralon=lat_corralon,

            lon_corralon=lon_corralon,

            capacidad_camion=(
                self.configuracion
                .capacidad_camion
            ),

            cantidad_camiones=(
                self.configuracion
                .cantidad_camiones
            ),

            hora_inicio_turno_min=(
                inicio_turno
            ),

            hora_fin_objetivo_min=(
                fin_objetivo
            ),

            hora_fin_tolerancia_min=(
                fin_tolerancia
            ),

            seed_escenario=seed,

            seed_ejecucion=(
                1_000_000 + seed
            ),
        )

    def _generar_unidades(
        self,
        rng: Random,
    ) -> int:
        if (
            rng.random()
            <
            self.configuracion
            .probabilidad_pedido_mayor_capacidad
        ):
            return rng.randint(
                self.configuracion
                .capacidad_camion + 1,

                self.configuracion
                .max_unidades_pedido_grande,
            )

        return rng.randint(
            1,
            self.configuracion
            .capacidad_camion,
        )

    def _generar_coordenadas(
        self,
        lat_corralon: float,
        lon_corralon: float,
        rng: Random,
    ) -> tuple[float, float]:
        max_desplazamiento = (
            self.configuracion
            .desplazamiento_max_grados
        )

        desplazamiento_lat = rng.uniform(
            -max_desplazamiento,
            max_desplazamiento,
        )

        desplazamiento_lon = rng.uniform(
            -max_desplazamiento,
            max_desplazamiento,
        )

        # Evita generar accidentalmente un pedido
        # prácticamente encima del depósito.
        if (
            abs(desplazamiento_lat)
            + abs(desplazamiento_lon)
            < 0.002
        ):
            desplazamiento_lat += 0.004

        return (
            lat_corralon
            + desplazamiento_lat,

            lon_corralon
            + desplazamiento_lon,
        )

    def _generar_ventana(
        self,
        inicio_turno: int,
        fin_turno: int,
        rng: Random,
    ) -> tuple[bool, int, int]:
        tiene_ventana = (
            rng.random()
            <
            self.configuracion
            .probabilidad_ventana_especifica
        )

        if not tiene_ventana:
            return (
                False,
                inicio_turno,
                fin_turno,
            )

        ancho_maximo = min(
            self.configuracion
            .ancho_ventana_max,

            fin_turno - inicio_turno,
        )

        ancho = rng.randint(
            self.configuracion
            .ancho_ventana_min,

            ancho_maximo,
        )

        hora_desde = rng.randint(
            inicio_turno,
            fin_turno - ancho,
        )

        hora_hasta = (
            hora_desde + ancho
        )

        return (
            True,
            hora_desde,
            hora_hasta,
        )