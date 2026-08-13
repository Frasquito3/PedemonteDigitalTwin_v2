from dataclasses import dataclass
from random import Random
from time import perf_counter

from .base import PlanificadorTurno

from .config import (
    ConfiguracionPlanificacion,
)

from .greedy import (
    generar_plan_greedy,
)

from .objective import (
    evaluar_plan_estimado,
    tiempo_carga_estimado_min,
    tiempo_descarga_estimado_min,
)

from .schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    ViajePlan,
)

from .travel import (
    MatrizViaje,
    construir_matriz_viaje,
    tiempo_viaje_esperado_min,
)

from .validator import (
    validar_instancia,
    validar_plan,
)


Cromosoma = tuple[str, ...]


@dataclass(frozen=True)
class ConfiguracionGA:
    tamano_poblacion: int = 60

    generaciones: int = 100

    tamano_elite: int = 4

    tamano_torneo: int = 3

    probabilidad_crossover: float = 0.90

    probabilidad_mutacion_swap: float = 0.20

    probabilidad_mutacion_inversion: float = 0.10

    generaciones_sin_mejora_max: int = 30

    tolerancia_mejora: float = 1e-9


    def __post_init__(self) -> None:
        if self.tamano_poblacion < 2:
            raise ValueError(
                "tamano_poblacion debe ser >= 2."
            )

        if not (
            1
            <= self.tamano_elite
            < self.tamano_poblacion
        ):
            raise ValueError(
                "tamano_elite debe estar entre 1 "
                "y tamano_poblacion - 1."
            )

        if not (
            2
            <= self.tamano_torneo
            <= self.tamano_poblacion
        ):
            raise ValueError(
                "tamano_torneo fuera de rango."
            )

        if self.generaciones <= 0:
            raise ValueError(
                "generaciones debe ser > 0."
            )

        if self.generaciones_sin_mejora_max <= 0:
            raise ValueError(
                "generaciones_sin_mejora_max "
                "debe ser > 0."
            )

        probabilidades = (
            self.probabilidad_crossover,
            self.probabilidad_mutacion_swap,
            self.probabilidad_mutacion_inversion,
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


@dataclass(frozen=True)
class IndividuoEvaluado:
    cromosoma: Cromosoma

    costo: float


class GeneticAlgorithmPlanner(
    PlanificadorTurno
):
    def __init__(
        self,
        configuracion:
            ConfiguracionPlanificacion
            | None = None,
        configuracion_ga:
            ConfiguracionGA
            | None = None,
        seed: int | None = None,
    ) -> None:
        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )

        self.configuracion_ga = (
            configuracion_ga
            if configuracion_ga is not None
            else ConfiguracionGA()
        )

        self.seed = seed

        self.ultima_seed_utilizada: int | None = None

        self.generaciones_ejecutadas: int = 0

        self.mejor_costo_por_generacion: list[float] = []

    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        inicio_computo = perf_counter()

        errores_instancia = validar_instancia(
            instancia
        )

        if errores_instancia:
            raise ValueError(
                "Instancia inválida: "
                + " | ".join(
                    errores_instancia
                )
            )

        seed_efectiva = (
            self.seed
            if self.seed is not None
            else instancia.seed_escenario
            + 8001
        )

        self.ultima_seed_utilizada = (
            seed_efectiva
        )

        self.generaciones_ejecutadas = 0

        self.mejor_costo_por_generacion = []

        rng = Random(
            seed_efectiva
        )

        matriz = construir_matriz_viaje(
            instancia,
            self.configuracion,
        )

        pedidos_por_id = {
            pedido.pedido_id: pedido
            for pedido in instancia.pedidos
        }

        if not pedidos_por_id:
            plan_vacio = PlanTurno(
                instancia_id=(
                    instancia.instancia_id
                ),

                algoritmo=(
                    AlgoritmoPlanificacion.GA
                ),

                camiones=[
                    PlanCamion(
                        camion_id=camion_id
                    )
                    for camion_id in range(
                        instancia.cantidad_camiones
                    )
                ],
            )

            plan_vacio.costo_estimado = 0.0

            plan_vacio.tiempo_computo_ms = (
                perf_counter()
                - inicio_computo
            ) * 1000.0

            return plan_vacio

        poblacion = self._crear_poblacion_inicial(
            instancia,
            rng,
        )

        cache_costos: dict[
            Cromosoma,
            float,
        ] = {}

        evaluados = self._evaluar_poblacion(
            instancia,
            matriz,
            pedidos_por_id,
            poblacion,
            cache_costos,
        )

        mejor_global = min(
            evaluados,
            key=lambda individuo:
                individuo.costo,
        )

        generaciones_sin_mejora = 0

        self.mejor_costo_por_generacion.append(
            mejor_global.costo
        )

        for numero_generacion in range(
            1,
            self.configuracion_ga
            .generaciones
            + 1,
        ):
            nueva_poblacion = (
                self._crear_siguiente_generacion(
                    evaluados,
                    rng,
                )
            )

            evaluados = self._evaluar_poblacion(
                instancia,
                matriz,
                pedidos_por_id,
                nueva_poblacion,
                cache_costos,
            )

            mejor_generacion = min(
                evaluados,
                key=lambda individuo:
                    individuo.costo,
            )

            if (
                mejor_generacion.costo
                <
                mejor_global.costo
                -
                self.configuracion_ga
                .tolerancia_mejora
            ):
                mejor_global = (
                    mejor_generacion
                )

                generaciones_sin_mejora = 0

            else:
                generaciones_sin_mejora += 1

            self.mejor_costo_por_generacion.append(
                mejor_global.costo
            )

            self.generaciones_ejecutadas = (
                numero_generacion
            )

            if (
                generaciones_sin_mejora
                >=
                self.configuracion_ga
                .generaciones_sin_mejora_max
            ):
                break

        mejor_plan = self._decodificar_plan(
            instancia,
            matriz,
            pedidos_por_id,
            mejor_global.cromosoma,
        )

        validacion = validar_plan(
            instancia,
            mejor_plan,
        )

        if not validacion.valido:
            raise RuntimeError(
                "El GA produjo un plan inválido: "
                + " | ".join(
                    validacion.errores
                )
            )

        estimacion = evaluar_plan_estimado(
            instancia,
            mejor_plan,
            matriz,
            self.configuracion,
        )

        mejor_plan.costo_estimado = (
            estimacion.costo_total
        )

        mejor_plan.tiempo_computo_ms = (
            perf_counter()
            - inicio_computo
        ) * 1000.0

        return mejor_plan

    def _crear_poblacion_inicial(
        self,
        instancia: InstanciaTurno,
        rng: Random,
    ) -> list[Cromosoma]:
        pedido_ids = tuple(
            pedido.pedido_id
            for pedido in instancia.pedidos
        )

        poblacion: list[Cromosoma] = []

        vistos: set[Cromosoma] = set()

        # =================================================
        # SEMILLA 1: SOLUCIÓN GREEDY
        # =================================================
        #
        # Esto garantiza que el GA nunca termine peor
        # que el Greedy por pérdida accidental de una
        # buena solución inicial, ya que además usamos
        # elitismo.

        plan_greedy = generar_plan_greedy(
            instancia,
            configuracion=self.configuracion,
        )

        cromosoma_greedy = tuple(
            pedido_id

            for camion in plan_greedy.camiones
            for viaje in camion.viajes
            for pedido_id in viaje.pedido_ids
        )

        self._agregar_si_nuevo(
            poblacion,
            vistos,
            cromosoma_greedy,
        )

        # =================================================
        # SEMILLA 2: EARLIEST DUE DATE
        # =================================================

        cromosoma_edd = tuple(
            pedido.pedido_id
            for pedido in sorted(
                instancia.pedidos,

                key=lambda pedido: (
                    pedido.hora_hasta_min,

                    0
                    if pedido.requiere_volcador
                    else 1,

                    -pedido.unidades_capacidad,

                    pedido.pedido_id,
                ),
            )
        )

        self._agregar_si_nuevo(
            poblacion,
            vistos,
            cromosoma_edd,
        )

        # =================================================
        # SEMILLA 3: MAYOR CAPACIDAD PRIMERO
        # =================================================

        cromosoma_capacidad = tuple(
            pedido.pedido_id
            for pedido in sorted(
                instancia.pedidos,

                key=lambda pedido: (
                    -pedido.unidades_capacidad,

                    pedido.hora_hasta_min,

                    pedido.pedido_id,
                ),
            )
        )

        self._agregar_si_nuevo(
            poblacion,
            vistos,
            cromosoma_capacidad,
        )

        # =================================================
        # RESTO ALEATORIO
        # =================================================

        intentos_sin_nuevo = 0

        max_intentos_sin_nuevo = (
            self.configuracion_ga
            .tamano_poblacion
            * 20
        )

        while (
            len(poblacion)
            <
            self.configuracion_ga
            .tamano_poblacion
        ):
            genes = list(
                pedido_ids
            )

            rng.shuffle(
                genes
            )

            cromosoma = tuple(
                genes
            )

            agregado = self._agregar_si_nuevo(
                poblacion,
                vistos,
                cromosoma,
            )

            if agregado:
                intentos_sin_nuevo = 0

            else:
                intentos_sin_nuevo += 1

            # Para instancias muy pequeñas puede no
            # haber suficientes permutaciones distintas
            # para llenar una población grande.
            #
            # En ese caso permitimos duplicados.
            if (
                intentos_sin_nuevo
                >= max_intentos_sin_nuevo
            ):
                poblacion.append(
                    cromosoma
                )

        return poblacion

    @staticmethod
    def _agregar_si_nuevo(
        poblacion: list[Cromosoma],
        vistos: set[Cromosoma],
        cromosoma: Cromosoma,
    ) -> bool:
        if cromosoma in vistos:
            return False

        poblacion.append(
            cromosoma
        )

        vistos.add(
            cromosoma
        )

        return True

    def _evaluar_poblacion(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id:
            dict[str, PedidoInput],
        poblacion: list[Cromosoma],
        cache_costos:
            dict[Cromosoma, float],
    ) -> list[IndividuoEvaluado]:
        evaluados: list[
            IndividuoEvaluado
        ] = []

        for cromosoma in poblacion:
            costo = cache_costos.get(
                cromosoma
            )

            if costo is None:
                plan = self._decodificar_plan(
                    instancia,
                    matriz,
                    pedidos_por_id,
                    cromosoma,
                )

                estimacion = evaluar_plan_estimado(
                    instancia,
                    plan,
                    matriz,
                    self.configuracion,
                )

                costo = estimacion.costo_total

                cache_costos[
                    cromosoma
                ] = costo

            evaluados.append(
                IndividuoEvaluado(
                    cromosoma=cromosoma,
                    costo=costo,
                )
            )

        evaluados.sort(
            key=lambda individuo: (
                individuo.costo,
                individuo.cromosoma,
            )
        )

        return evaluados

    def _crear_siguiente_generacion(
        self,
        evaluados:
            list[IndividuoEvaluado],
        rng: Random,
    ) -> list[Cromosoma]:
        nueva_poblacion = [
            individuo.cromosoma

            for individuo
            in evaluados[
                :
                self.configuracion_ga
                .tamano_elite
            ]
        ]

        while (
            len(nueva_poblacion)
            <
            self.configuracion_ga
            .tamano_poblacion
        ):
            padre_a = self._seleccion_torneo(
                evaluados,
                rng,
            )

            padre_b = self._seleccion_torneo(
                evaluados,
                rng,
            )

            if (
                len(padre_a) >= 2
                and
                rng.random()
                <
                self.configuracion_ga
                .probabilidad_crossover
            ):
                hijo_a, hijo_b = (
                    self._crossover_ordenado(
                        padre_a,
                        padre_b,
                        rng,
                    )
                )

            else:
                hijo_a = padre_a
                hijo_b = padre_b

            hijo_a = self._mutar(
                hijo_a,
                rng,
            )

            hijo_b = self._mutar(
                hijo_b,
                rng,
            )

            nueva_poblacion.append(
                hijo_a
            )

            if (
                len(nueva_poblacion)
                <
                self.configuracion_ga
                .tamano_poblacion
            ):
                nueva_poblacion.append(
                    hijo_b
                )

        return nueva_poblacion

    def _seleccion_torneo(
        self,
        evaluados:
            list[IndividuoEvaluado],
        rng: Random,
    ) -> Cromosoma:
        participantes = [
            evaluados[
                rng.randrange(
                    len(evaluados)
                )
            ]

            for _ in range(
                self.configuracion_ga
                .tamano_torneo
            )
        ]

        ganador = min(
            participantes,

            key=lambda individuo: (
                individuo.costo,
                individuo.cromosoma,
            ),
        )

        return ganador.cromosoma

    @staticmethod
    def _crossover_ordenado(
        padre_a: Cromosoma,
        padre_b: Cromosoma,
        rng: Random,
    ) -> tuple[Cromosoma, Cromosoma]:
        cantidad_genes = len(
            padre_a
        )

        if cantidad_genes < 2:
            return padre_a, padre_b

        corte_1, corte_2 = sorted(
            rng.sample(
                range(cantidad_genes),
                2,
            )
        )

        # El extremo superior se incluye.
        corte_2 += 1

        hijo_a = GeneticAlgorithmPlanner._crear_hijo_ox(
            padre_a,
            padre_b,
            corte_1,
            corte_2,
        )

        hijo_b = GeneticAlgorithmPlanner._crear_hijo_ox(
            padre_b,
            padre_a,
            corte_1,
            corte_2,
        )

        return hijo_a, hijo_b

    @staticmethod
    def _crear_hijo_ox(
        padre_base: Cromosoma,
        padre_relleno: Cromosoma,
        inicio: int,
        fin: int,
    ) -> Cromosoma:
        cantidad_genes = len(
            padre_base
        )

        hijo: list[
            str | None
        ] = [
            None
            for _ in range(
                cantidad_genes
            )
        ]

        hijo[inicio:fin] = (
            padre_base[inicio:fin]
        )

        genes_usados = set(
            padre_base[inicio:fin]
        )

        genes_restantes = [
            gen
            for gen in padre_relleno
            if gen not in genes_usados
        ]

        posiciones_libres = [
            posicion
            for posicion, gen
            in enumerate(hijo)
            if gen is None
        ]

        for (
            posicion,
            gen,
        ) in zip(
            posiciones_libres,
            genes_restantes,
        ):
            hijo[posicion] = gen

        if any(
            gen is None
            for gen in hijo
        ):
            raise RuntimeError(
                "El crossover OX dejó genes vacíos."
            )

        return tuple(
            gen
            for gen in hijo
            if gen is not None
        )

    def _mutar(
        self,
        cromosoma: Cromosoma,
        rng: Random,
    ) -> Cromosoma:
        genes = list(
            cromosoma
        )

        if (
            len(genes) >= 2
            and
            rng.random()
            <
            self.configuracion_ga
            .probabilidad_mutacion_swap
        ):
            posicion_a, posicion_b = (
                rng.sample(
                    range(len(genes)),
                    2,
                )
            )

            genes[
                posicion_a
            ], genes[
                posicion_b
            ] = (
                genes[posicion_b],
                genes[posicion_a],
            )

        if (
            len(genes) >= 3
            and
            rng.random()
            <
            self.configuracion_ga
            .probabilidad_mutacion_inversion
        ):
            inicio, fin = sorted(
                rng.sample(
                    range(len(genes)),
                    2,
                )
            )

            fin += 1

            genes[inicio:fin] = reversed(
                genes[inicio:fin]
            )

        return tuple(
            genes
        )

    def _decodificar_plan(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id:
            dict[str, PedidoInput],
        cromosoma: Cromosoma,
    ) -> PlanTurno:
        self._validar_cromosoma(
            pedidos_por_id,
            cromosoma,
        )

        viajes = self._decodificar_viajes(
            instancia,
            pedidos_por_id,
            cromosoma,
        )

        planes_camion = [
            PlanCamion(
                camion_id=camion_id
            )
            for camion_id in range(
                instancia.cantidad_camiones
            )
        ]

        disponibilidad = [
            float(
                instancia
                .hora_inicio_turno_min
            )
            for _ in range(
                instancia.cantidad_camiones
            )
        ]

        for pedido_ids in viajes:
            camion_id = min(
                range(
                    instancia.cantidad_camiones
                ),

                key=lambda candidato_id: (
                    disponibilidad[
                        candidato_id
                    ],

                    len(
                        planes_camion[
                            candidato_id
                        ].viajes
                    ),

                    candidato_id,
                ),
            )

            numero_viaje = (
                len(
                    planes_camion[
                        camion_id
                    ].viajes
                )
                + 1
            )

            planes_camion[
                camion_id
            ].viajes.append(
                ViajePlan(
                    numero_viaje=numero_viaje,

                    pedido_ids=list(
                        pedido_ids
                    ),
                )
            )

            disponibilidad[
                camion_id
            ] = self._estimar_fin_viaje(
                pedido_ids,
                disponibilidad[camion_id],
                pedidos_por_id,
                matriz,
            )

        return PlanTurno(
            instancia_id=(
                instancia.instancia_id
            ),

            algoritmo=(
                AlgoritmoPlanificacion.GA
            ),

            camiones=planes_camion,
        )

    @staticmethod
    def _validar_cromosoma(
        pedidos_por_id:
            dict[str, PedidoInput],
        cromosoma: Cromosoma,
    ) -> None:
        ids_esperados = set(
            pedidos_por_id
        )

        ids_recibidos = set(
            cromosoma
        )

        if (
            len(cromosoma)
            != len(pedidos_por_id)
        ):
            raise ValueError(
                "Longitud incorrecta del cromosoma."
            )

        if (
            len(cromosoma)
            != len(ids_recibidos)
        ):
            raise ValueError(
                "El cromosoma contiene "
                "pedidos repetidos."
            )

        if ids_recibidos != ids_esperados:
            faltantes = sorted(
                ids_esperados
                - ids_recibidos
            )

            desconocidos = sorted(
                ids_recibidos
                - ids_esperados
            )

            raise ValueError(
                "Cromosoma incompatible. "
                f"Faltantes={faltantes}, "
                f"desconocidos={desconocidos}."
            )

    @staticmethod
    def _decodificar_viajes(
        instancia: InstanciaTurno,
        pedidos_por_id:
            dict[str, PedidoInput],
        cromosoma: Cromosoma,
    ) -> list[list[str]]:
        viajes: list[list[str]] = []

        viaje_actual: list[str] = []

        carga_actual = 0

        contiene_volcador = False

        def cerrar_viaje() -> None:
            nonlocal viaje_actual
            nonlocal carga_actual
            nonlocal contiene_volcador

            if viaje_actual:
                viajes.append(
                    viaje_actual
                )

            viaje_actual = []

            carga_actual = 0

            contiene_volcador = False

        for pedido_id in cromosoma:
            pedido = pedidos_por_id[
                pedido_id
            ]

            # No debería ocurrir porque un volcador
            # cierra el viaje de inmediato, pero se
            # mantiene como defensa.
            if contiene_volcador:
                cerrar_viaje()

            supera_capacidad = (
                carga_actual
                + pedido.unidades_capacidad
                >
                instancia.capacidad_camion
            )

            if supera_capacidad:
                cerrar_viaje()

            viaje_actual.append(
                pedido_id
            )

            carga_actual += (
                pedido.unidades_capacidad
            )

            if pedido.requiere_volcador:
                contiene_volcador = True

                # El volcador debe quedar último.
                cerrar_viaje()

        cerrar_viaje()

        return viajes

    def _estimar_fin_viaje(
        self,
        pedido_ids: list[str],
        minuto_inicio: float,
        pedidos_por_id:
            dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> float:
        unidades = sum(
            pedidos_por_id[
                pedido_id
            ].unidades_capacidad
            for pedido_id in pedido_ids
        )

        minuto_actual = (
            minuto_inicio
            + tiempo_carga_estimado_min(
                unidades,
                self.configuracion,
            )
        )

        nodo_actual = (
            self.configuracion
            .id_nodo_corralon
        )

        for pedido_id in pedido_ids:
            pedido = pedidos_por_id[
                pedido_id
            ]

            minuto_actual += (
                tiempo_viaje_esperado_min(
                    matriz,
                    nodo_actual,
                    pedido_id,
                    minuto_actual,
                    self.configuracion,
                )
            )

            if (
                minuto_actual
                < pedido.hora_desde_min
            ):
                minuto_actual = (
                    pedido.hora_desde_min
                )

            minuto_actual += (
                tiempo_descarga_estimado_min(
                    pedido,
                    self.configuracion,
                )
            )

            nodo_actual = pedido_id

        minuto_actual += (
            tiempo_viaje_esperado_min(
                matriz,
                nodo_actual,

                self.configuracion
                .id_nodo_corralon,

                minuto_actual,

                self.configuracion,
            )
        )

        return minuto_actual


def generar_plan_ga(
    instancia: InstanciaTurno,
    seed: int | None = None,
    configuracion:
        ConfiguracionPlanificacion
        | None = None,
    configuracion_ga:
        ConfiguracionGA
        | None = None,
) -> PlanTurno:
    return GeneticAlgorithmPlanner(
        configuracion=configuracion,

        configuracion_ga=(
            configuracion_ga
        ),

        seed=seed,
    ).generar_plan(instancia)