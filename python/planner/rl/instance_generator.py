from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from random import Random

from planner.core.schema import (
    InstanciaTurno,
    PedidoInput,
    Turno,
)
from planner.data.real_demand import (
    CatalogoDemandaReal,
    DivisionDemandaReal,
    ParticionDemandaReal,
    PuntoDemandaReal,
    SEED_DIVISION_DEMANDA_REAL_V1,
)
from planner.domain.preprocess import preprocesar_instancia
from planner.domain.validator import validar_instancia


LAT_CORRALON = -32.8495006
LON_CORRALON = -60.722653

RUTA_DEMANDA_REAL_POR_DEFECTO = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "demanda_geografica.csv"
)


class ModoDemandaGeografica(str, Enum):
    SINTETICA = "SINTETICA"
    REAL = "REAL"


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

    modo_demanda_geografica: ModoDemandaGeografica = (
        ModoDemandaGeografica.SINTETICA
    )

    ruta_demanda_real: str = ""

    muestreo_demanda_real_con_reemplazo: bool = False

    particion_demanda_real: ParticionDemandaReal | None = None

    seed_division_demanda_real: int = (
        SEED_DIVISION_DEMANDA_REAL_V1
    )

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

        if self.max_intentos_generacion <= 0:
            raise ValueError(
                "max_intentos_generacion debe ser > 0."
            )

        if not isinstance(
            self.modo_demanda_geografica,
            ModoDemandaGeografica,
        ):
            raise ValueError(
                "modo_demanda_geografica debe ser una "
                "instancia de ModoDemandaGeografica."
            )

        if (
            self.particion_demanda_real is not None
            and not isinstance(
                self.particion_demanda_real,
                ParticionDemandaReal,
            )
        ):
            raise ValueError(
                "particion_demanda_real debe ser None o una "
                "instancia de ParticionDemandaReal."
            )

        if (
            self.modo_demanda_geografica
            == ModoDemandaGeografica.SINTETICA
            and self.particion_demanda_real is not None
        ):
            raise ValueError(
                "particion_demanda_real sólo puede utilizarse "
                "cuando el modo de demanda geográfica es REAL."
            )


class GeneradorInstanciasRL:
    def __init__(
        self,
        configuracion:
            ConfiguracionGeneradorInstancias
            | None = None,
        catalogo_demanda_real:
            CatalogoDemandaReal
            | None = None,
    ) -> None:
        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionGeneradorInstancias()
        )

        self._catalogo_demanda_real: (
            CatalogoDemandaReal
            | None
        ) = None

        self._catalogo_demanda_real_completo: (
            CatalogoDemandaReal
            | None
        ) = None

        self._division_demanda_real: (
            DivisionDemandaReal
            | None
        ) = None

        self._ruta_demanda_real_resuelta: (
            Path
            | None
        ) = None

        if (
            self.configuracion
            .modo_demanda_geografica
            == ModoDemandaGeografica.REAL
        ):
            self._inicializar_demanda_real(
                catalogo_demanda_real
            )
        elif catalogo_demanda_real is not None:
            raise ValueError(
                "catalogo_demanda_real sólo puede "
                "utilizarse cuando el modo de demanda "
                "geográfica es REAL."
            )

    @property
    def catalogo_demanda_real(
        self,
    ) -> CatalogoDemandaReal | None:
        return self._catalogo_demanda_real

    @property
    def catalogo_demanda_real_completo(
        self,
    ) -> CatalogoDemandaReal | None:
        return self._catalogo_demanda_real_completo

    @property
    def division_demanda_real(
        self,
    ) -> DivisionDemandaReal | None:
        return self._division_demanda_real

    @property
    def particion_demanda_real(
        self,
    ) -> ParticionDemandaReal | None:
        return self.configuracion.particion_demanda_real

    @property
    def ruta_demanda_real_resuelta(
        self,
    ) -> Path | None:
        return self._ruta_demanda_real_resuelta

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

    def _inicializar_demanda_real(
        self,
        catalogo_recibido:
            CatalogoDemandaReal
            | None,
    ) -> None:
        if catalogo_recibido is not None:
            catalogo_completo = catalogo_recibido
            ruta_resuelta = (
                catalogo_recibido.ruta_fuente
            )
        else:
            ruta_resuelta = (
                self._resolver_ruta_demanda_real()
            )

            catalogo_completo = (
                CatalogoDemandaReal.desde_csv(
                    ruta_resuelta
                )
            )

        particion = (
            self.configuracion
            .particion_demanda_real
        )

        division: DivisionDemandaReal | None = None
        catalogo_efectivo = catalogo_completo

        if particion is not None:
            division = (
                catalogo_completo
                .dividir_por_direccion_fuente(
                    seed=(
                        self.configuracion
                        .seed_division_demanda_real
                    )
                )
            )

            division.validar_sin_fuga()

            catalogo_efectivo = (
                division.catalogo_para(
                    particion
                )
            )

        if (
            not self.configuracion
            .muestreo_demanda_real_con_reemplazo
            and self.configuracion
            .max_pedidos_finales
            > len(catalogo_efectivo)
        ):
            descripcion_particion = (
                particion.value
                if particion is not None
                else "CATALOGO_COMPLETO"
            )

            raise ValueError(
                "El catálogo de demanda real no tiene "
                "suficientes registros para muestrear "
                "sin reemplazo. "
                f"Partición={descripcion_particion}, "
                "registros disponibles="
                f"{len(catalogo_efectivo)}, "
                "máximo solicitado="
                f"{self.configuracion.max_pedidos_finales}."
            )

        self._catalogo_demanda_real_completo = (
            catalogo_completo
        )
        self._division_demanda_real = division
        self._catalogo_demanda_real = (
            catalogo_efectivo
        )
        self._ruta_demanda_real_resuelta = (
            ruta_resuelta
        )

    def _resolver_ruta_demanda_real(
        self,
    ) -> Path:
        ruta_configurada = (
            self.configuracion
            .ruta_demanda_real
            .strip()
        )

        if ruta_configurada:
            ruta = Path(
                ruta_configurada
            )

            if not ruta.is_absolute():
                ruta = (
                    Path(__file__)
                    .resolve()
                    .parents[2]
                    / ruta
                )
        else:
            ruta = (
                RUTA_DEMANDA_REAL_POR_DEFECTO
            )

        return ruta.resolve()

    def _generar_instancia_raw(
        self,
        seed: int,
        intento: int,
        rng: Random,
    ) -> InstanciaTurno:
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

        puntos_demanda_real = (
            self._muestrear_puntos_demanda_real(
                cantidad=cantidad_raw,
                rng=rng,
            )
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

            punto_demanda_real = (
                puntos_demanda_real[indice]
                if puntos_demanda_real
                is not None
                else None
            )

            (
                latitud,
                longitud,
                direccion,
                barrio,
                observaciones,
            ) = self._generar_ubicacion(
                punto_demanda_real=(
                    punto_demanda_real
                ),
                rng=rng,
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

                    direccion=direccion,

                    barrio=barrio,

                    observaciones=observaciones,
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

            lat_corralon=LAT_CORRALON,

            lon_corralon=LON_CORRALON,

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

    def _muestrear_puntos_demanda_real(
        self,
        cantidad: int,
        rng: Random,
    ) -> list[PuntoDemandaReal] | None:
        if (
            self.configuracion
            .modo_demanda_geografica
            == ModoDemandaGeografica.SINTETICA
        ):
            return None

        catalogo = (
            self._catalogo_demanda_real
        )

        if catalogo is None:
            raise RuntimeError(
                "El modo REAL requiere un catálogo "
                "de demanda inicializado."
            )

        return catalogo.muestrear(
            cantidad=cantidad,
            rng=rng,
            con_reemplazo=(
                self.configuracion
                .muestreo_demanda_real_con_reemplazo
            ),
        )

    def _generar_ubicacion(
        self,
        punto_demanda_real:
            PuntoDemandaReal
            | None,
        rng: Random,
    ) -> tuple[
        float,
        float,
        str,
        str,
        str,
    ]:
        if punto_demanda_real is None:
            latitud, longitud = (
                self._generar_coordenadas_sinteticas(
                    LAT_CORRALON,
                    LON_CORRALON,
                    rng,
                )
            )

            return (
                latitud,
                longitud,
                "",
                "",
                "",
            )

        observaciones = (
            "FUENTE_DEMANDA_REAL="
            f"{punto_demanda_real.registro_id}; "
            "CIUDAD="
            f"{punto_demanda_real.ciudad}; "
            "FRECUENCIA_FUENTE="
            f"{punto_demanda_real.frecuencia_direccion_fuente}"
        )

        return (
            punto_demanda_real.latitud,
            punto_demanda_real.longitud,
            punto_demanda_real.direccion_corta,
            punto_demanda_real.barrio,
            observaciones,
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

    def _generar_coordenadas_sinteticas(
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