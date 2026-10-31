"""Algoritmo genético (GA) basado en permutaciones para optimización global de turnos."""

from dataclasses import dataclass
from random import Random
from time import perf_counter

from planner.algorithms.greedy import generar_plan_greedy
from planner.core.base import PlanificadorTurno
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
)
from planner.domain.validator import validar_instancia, validar_plan
from planner.routing.decoder import (
    Cromosoma,
    decodificar_plan_permutacion,
    validar_permutacion,
)
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import (
    MatrizViaje,
    ProveedorViaje,
    construir_matriz_viaje,
)

__all__ = [
    "ConfiguracionGA",
    "GeneticAlgorithmPlanner",
    "generar_plan_ga",
]


@dataclass(frozen=True)
class ConfiguracionGA:
    """Hiperparámetros de control para la metaheurística del Algoritmo Genético.

    Attributes:
        tamano_poblacion: Cantidad de individuos en cada generación.
        generaciones: Límite máximo de iteraciones evolutivas.
        tamano_elite: Individuos con mejor aptitud preservados intactos para la siguiente generación.
        tamano_torneo: Cantidad de individuos en competencia para selección por torneo.
        probabilidad_crossover: Probabilidad de aplicar cruce ordenado (OX) a una pareja de padres.
        probabilidad_mutacion_swap: Probabilidad de mutación por intercambio de dos posiciones.
        probabilidad_mutacion_inversion: Probabilidad de mutación por inversión de un segmento.
        generaciones_sin_mejora_max: Criterio de parada temprana (paciencia) si no hay mejora en el costo.
        tolerancia_mejora: Umbral mínimo absoluto para considerar que una solución superó a la anterior.
    """

    tamano_poblacion: int = 60
    generaciones: int = 100
    tamano_elite: int = 4
    tamano_torneo: int = 3
    probabilidad_crossover: float = 0.90
    probabilidad_mutacion_swap: float = 0.20
    probabilidad_mutacion_inversion: float = 0.10
    generaciones_sin_mejora_max: int = 30
    tolerancia_mejora: float = 1e-9
    usar_tasas_adaptativas: bool = True
    probabilidad_crossover_min: float = 0.65
    probabilidad_mutacion_swap_max: float = 0.45
    probabilidad_mutacion_inversion_max: float = 0.30

    def __post_init__(self) -> None:
        if self.tamano_poblacion < 2:
            raise ValueError("tamano_poblacion debe ser >= 2.")

        if not (1 <= self.tamano_elite < self.tamano_poblacion):
            raise ValueError("tamano_elite debe estar entre 1 y tamano_poblacion - 1.")

        if not (2 <= self.tamano_torneo <= self.tamano_poblacion):
            raise ValueError("tamano_torneo fuera de rango.")

        if self.generaciones <= 0:
            raise ValueError("generaciones debe ser > 0.")

        if self.generaciones_sin_mejora_max <= 0:
            raise ValueError("generaciones_sin_mejora_max debe ser > 0.")

        probabilidades = (
            self.probabilidad_crossover,
            self.probabilidad_mutacion_swap,
            self.probabilidad_mutacion_inversion,
            self.probabilidad_crossover_min,
            self.probabilidad_mutacion_swap_max,
            self.probabilidad_mutacion_inversion_max,
        )
        if any(p < 0.0 or p > 1.0 for p in probabilidades):
            raise ValueError("Las probabilidades deben estar entre 0 y 1.")


@dataclass(frozen=True)
class IndividuoEvaluado:
    """Estructura interna que asocia un cromosoma a su costo de función objetivo evaluado."""

    cromosoma: Cromosoma
    costo: float


class GeneticAlgorithmPlanner(PlanificadorTurno):
    """Optimizador metaheurístico evolutivo que opera sobre permutaciones de pedidos.

    Evoluciona soluciones mediante cruce ordenado (OX1), operadores duales de mutación
    (swap e inversión de subsecuencia) y selección por torneo con elitismo. Permite la
    inyección de semillas iniciales determinísticas (Voraz, EDD, heurísticas de capacidad)
    o externas (políticas de Reinforcement Learning).
    """

    def __init__(
        self,
        configuracion: ConfiguracionPlanificacion | None = None,
        configuracion_ga: ConfiguracionGA | None = None,
        seed: int | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        semillas_iniciales: tuple[Cromosoma, ...] | None = None,
        incluir_semilla_greedy: bool = True,
        fraccion_variantes_semilla: float = 0.0,
    ) -> None:
        self.configuracion = (
            configuracion if configuracion is not None else ConfiguracionPlanificacion()
        )
        self.configuracion_ga = (
            configuracion_ga if configuracion_ga is not None else ConfiguracionGA()
        )
        self.seed = seed
        self.proveedor_viaje = proveedor_viaje
        self.semillas_iniciales = tuple(semillas_iniciales or ())
        self.incluir_semilla_greedy = bool(incluir_semilla_greedy)

        if not 0.0 <= fraccion_variantes_semilla <= 1.0:
            raise ValueError("fraccion_variantes_semilla debe estar entre 0 y 1.")

        self.fraccion_variantes_semilla = fraccion_variantes_semilla
        self.semillas_iniciales_utilizadas: tuple[Cromosoma, ...] = ()
        self.ultima_seed_utilizada: int | None = None
        self.generaciones_ejecutadas: int = 0
        self.mejor_costo_por_generacion: list[float] = []

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        """Ejecuta el ciclo evolutivo completo y retorna el plan óptimo decodificado.

        Args:
            instancia: Escenario del turno a resolver.

        Returns:
            PlanTurno óptimo hallado con costo y tiempo de cómputo registrados.

        Raises:
            ValueError: Si la instancia incumple restricciones estructurales.
            RuntimeError: Si la decodificación del mejor individuo resulta inviable.
        """
        inicio_computo = perf_counter()

        errores_instancia = validar_instancia(instancia)
        if errores_instancia:
            raise ValueError("Instancia inválida: " + " | ".join(errores_instancia))

        seed_efectiva = self.seed if self.seed is not None else instancia.seed_escenario + 8001
        self.ultima_seed_utilizada = seed_efectiva
        self.generaciones_ejecutadas = 0
        self.mejor_costo_por_generacion = []

        rng = Random(seed_efectiva)
        matriz = construir_matriz_viaje(
            instancia, self.configuracion, proveedor=self.proveedor_viaje
        )
        pedidos_por_id = {p.pedido_id: p for p in instancia.pedidos}

        if not pedidos_por_id:
            plan_vacio = PlanTurno(
                instancia_id=instancia.instancia_id,
                algoritmo=AlgoritmoPlanificacion.GA,
                camiones=[
                    PlanCamion(camion_id=c_id) for c_id in range(instancia.cantidad_camiones)
                ],
            )
            plan_vacio.costo_estimado = 0.0
            plan_vacio.tiempo_computo_ms = (perf_counter() - inicio_computo) * 1000.0
            return plan_vacio

        poblacion = self._crear_poblacion_inicial(instancia, rng)
        cache_costos: dict[Cromosoma, float] = {}

        evaluados = self._evaluar_poblacion(
            instancia, matriz, pedidos_por_id, poblacion, cache_costos
        )
        mejor_global = min(evaluados, key=lambda ind: ind.costo)
        generaciones_sin_mejora = 0
        self.mejor_costo_por_generacion.append(mejor_global.costo)

        for numero_generacion in range(1, self.configuracion_ga.generaciones + 1):
            nueva_poblacion = self._crear_siguiente_generacion(
                evaluados,
                rng,
                numero_generacion=numero_generacion,
                generaciones_sin_mejora=generaciones_sin_mejora,
            )
            evaluados = self._evaluar_poblacion(
                instancia, matriz, pedidos_por_id, nueva_poblacion, cache_costos
            )

            mejor_generacion = min(evaluados, key=lambda ind: ind.costo)

            if (
                mejor_generacion.costo
                < mejor_global.costo - self.configuracion_ga.tolerancia_mejora
            ):
                mejor_global = mejor_generacion
                generaciones_sin_mejora = 0
            else:
                generaciones_sin_mejora += 1

            self.mejor_costo_por_generacion.append(mejor_global.costo)
            self.generaciones_ejecutadas = numero_generacion

            if generaciones_sin_mejora >= self.configuracion_ga.generaciones_sin_mejora_max:
                break

        mejor_plan = self._decodificar_plan(
            instancia, matriz, pedidos_por_id, mejor_global.cromosoma
        )

        validacion = validar_plan(instancia, mejor_plan)
        if not validacion.valido:
            raise RuntimeError("El GA produjo un plan inválido: " + " | ".join(validacion.errores))

        estimacion = evaluar_plan_estimado(instancia, mejor_plan, matriz, self.configuracion)
        mejor_plan.costo_estimado = estimacion.costo_total
        mejor_plan.tiempo_computo_ms = (perf_counter() - inicio_computo) * 1000.0

        return mejor_plan

    def _crear_poblacion_inicial(
        self,
        instancia: InstanciaTurno,
        rng: Random,
    ) -> list[Cromosoma]:
        """Siembra la población con heurísticas y genera el resto mediante permutaciones aleatorias."""
        pedido_ids = tuple(p.pedido_id for p in instancia.pedidos)
        poblacion: list[Cromosoma] = []
        vistos: set[Cromosoma] = set()

        # 1. Semillas externas inyectadas (ej. provenientes de RL)
        semillas_validas: list[Cromosoma] = []
        pedidos_dict = {p.pedido_id: p for p in instancia.pedidos}

        for c_semilla in self.semillas_iniciales:
            validar_permutacion(pedidos_dict, c_semilla)
            if self._agregar_si_nuevo(poblacion, vistos, c_semilla):
                semillas_validas.append(c_semilla)

        self.semillas_iniciales_utilizadas = tuple(semillas_validas)

        # 2. Variantes mutadas de las semillas externas
        objetivo_variantes = int(
            round(self.configuracion_ga.tamano_poblacion * self.fraccion_variantes_semilla)
        )
        if semillas_validas and objetivo_variantes > 0:
            intentos_variantes = 0
            max_intentos = max(20, objetivo_variantes * 20)
            while len(poblacion) < objetivo_variantes and intentos_variantes < max_intentos:
                base = semillas_validas[intentos_variantes % len(semillas_validas)]
                variante = self._crear_variante_semilla(base, rng)
                self._agregar_si_nuevo(poblacion, vistos, variante)
                intentos_variantes += 1

        # 3. Semilla Voraz (Greedy)
        if self.incluir_semilla_greedy:
            plan_greedy = generar_plan_greedy(
                instancia,
                configuracion=self.configuracion,
                proveedor_viaje=self.proveedor_viaje,
            )
            c_greedy = tuple(
                p_id for cam in plan_greedy.camiones for v in cam.viajes for p_id in v.pedido_ids
            )
            self._agregar_si_nuevo(poblacion, vistos, c_greedy)

        # 4. Semilla por Urgencia (Earliest Due Date - EDD)
        c_edd = tuple(
            p.pedido_id
            for p in sorted(
                instancia.pedidos,
                key=lambda p: (
                    p.hora_hasta_min,
                    0 if p.requiere_volcador else 1,
                    -p.unidades_capacidad,
                    p.pedido_id,
                ),
            )
        )
        self._agregar_si_nuevo(poblacion, vistos, c_edd)

        # 5. Semilla por Capacidad Mayor Primero
        c_capacidad = tuple(
            p.pedido_id
            for p in sorted(
                instancia.pedidos,
                key=lambda p: (-p.unidades_capacidad, p.hora_hasta_min, p.pedido_id),
            )
        )
        self._agregar_si_nuevo(poblacion, vistos, c_capacidad)

        # 6. Complemento con permutaciones aleatorias
        intentos_sin_nuevo = 0
        max_intentos = self.configuracion_ga.tamano_poblacion * 20

        while len(poblacion) < self.configuracion_ga.tamano_poblacion:
            genes = list(pedido_ids)
            rng.shuffle(genes)
            c_aleatorio = tuple(genes)

            if self._agregar_si_nuevo(poblacion, vistos, c_aleatorio):
                intentos_sin_nuevo = 0
            else:
                intentos_sin_nuevo += 1

            if intentos_sin_nuevo >= max_intentos:
                poblacion.append(c_aleatorio)

        return poblacion

    @staticmethod
    def _crear_variante_semilla(cromosoma: Cromosoma, rng: Random) -> Cromosoma:
        """Crea una ligera perturbación de un cromosoma base invirtiendo o intercambiando genes."""
        genes = list(cromosoma)
        if len(genes) < 2:
            return cromosoma

        if len(genes) >= 3 and rng.random() < 0.5:
            inicio, fin = sorted(rng.sample(range(len(genes)), 2))
            genes[inicio : fin + 1] = reversed(genes[inicio : fin + 1])
        else:
            pos_a, pos_b = rng.sample(range(len(genes)), 2)
            genes[pos_a], genes[pos_b] = genes[pos_b], genes[pos_a]

        return tuple(genes)

    @staticmethod
    def _agregar_si_nuevo(
        poblacion: list[Cromosoma],
        vistos: set[Cromosoma],
        cromosoma: Cromosoma,
    ) -> bool:
        """Agrega un cromosoma a la lista únicamente si no fue visto previamente."""
        if cromosoma in vistos:
            return False
        poblacion.append(cromosoma)
        vistos.add(cromosoma)
        return True

    def _evaluar_poblacion(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id: dict[str, PedidoInput],
        poblacion: list[Cromosoma],
        cache_costos: dict[Cromosoma, float],
    ) -> list[IndividuoEvaluado]:
        """Decodifica y evalúa cada cromosoma registrando en caché para evitar cómputos redundantes."""
        evaluados: list[IndividuoEvaluado] = []

        for cromosoma in poblacion:
            costo = cache_costos.get(cromosoma)
            if costo is None:
                plan = self._decodificar_plan(instancia, matriz, pedidos_por_id, cromosoma)
                estimacion = evaluar_plan_estimado(instancia, plan, matriz, self.configuracion)
                costo = estimacion.costo_total
                cache_costos[cromosoma] = costo

            evaluados.append(IndividuoEvaluado(cromosoma=cromosoma, costo=costo))

        evaluados.sort(key=lambda ind: (ind.costo, ind.cromosoma))
        return evaluados

    def _calcular_tasas_generacion(
        self,
        numero_generacion: int = 1,
        generaciones_sin_mejora: int = 0,
    ) -> tuple[float, float, float]:
        """Calcula las tasas efectivas de crossover y mutación según progreso y estancamiento."""
        cfg = self.configuracion_ga
        if not cfg.usar_tasas_adaptativas:
            return (
                cfg.probabilidad_crossover,
                cfg.probabilidad_mutacion_swap,
                cfg.probabilidad_mutacion_inversion,
            )

        progreso_gen = (max(1, numero_generacion) - 1) / max(1, cfg.generaciones - 1)
        progreso_estanc = max(0, generaciones_sin_mejora) / max(1, cfg.generaciones_sin_mejora_max)
        alpha = min(1.0, max(0.0, max(progreso_gen, progreso_estanc)))

        p_cross_min = min(cfg.probabilidad_crossover, cfg.probabilidad_crossover_min)
        p_swap_max = max(cfg.probabilidad_mutacion_swap, cfg.probabilidad_mutacion_swap_max)
        p_inv_max = max(
            cfg.probabilidad_mutacion_inversion, cfg.probabilidad_mutacion_inversion_max
        )

        p_cross = cfg.probabilidad_crossover - alpha * (cfg.probabilidad_crossover - p_cross_min)
        p_swap = cfg.probabilidad_mutacion_swap + alpha * (
            p_swap_max - cfg.probabilidad_mutacion_swap
        )
        p_inv = cfg.probabilidad_mutacion_inversion + alpha * (
            p_inv_max - cfg.probabilidad_mutacion_inversion
        )
        return p_cross, p_swap, p_inv

    def _crear_siguiente_generacion(
        self,
        evaluados: list[IndividuoEvaluado],
        rng: Random,
        numero_generacion: int = 1,
        generaciones_sin_mejora: int = 0,
    ) -> list[Cromosoma]:
        """Aplica elitismo, selección por torneo, cruce OX y mutación para formar la nueva población."""
        p_cross, p_swap, p_inv = self._calcular_tasas_generacion(
            numero_generacion=numero_generacion,
            generaciones_sin_mejora=generaciones_sin_mejora,
        )
        nueva_poblacion = [ind.cromosoma for ind in evaluados[: self.configuracion_ga.tamano_elite]]

        while len(nueva_poblacion) < self.configuracion_ga.tamano_poblacion:
            padre_a = self._seleccion_torneo(evaluados, rng)
            padre_b = self._seleccion_torneo(evaluados, rng)

            if len(padre_a) >= 2 and rng.random() < p_cross:
                hijo_a, hijo_b = self._crossover_ordenado(padre_a, padre_b, rng)
            else:
                hijo_a, hijo_b = padre_a, padre_b

            hijo_a = self._mutar(hijo_a, rng, prob_swap=p_swap, prob_inversion=p_inv)
            hijo_b = self._mutar(hijo_b, rng, prob_swap=p_swap, prob_inversion=p_inv)

            nueva_poblacion.append(hijo_a)
            if len(nueva_poblacion) < self.configuracion_ga.tamano_poblacion:
                nueva_poblacion.append(hijo_b)

        return nueva_poblacion

    def _seleccion_torneo(
        self,
        evaluados: list[IndividuoEvaluado],
        rng: Random,
    ) -> Cromosoma:
        """Selecciona el individuo de menor costo entre una muestra aleatoria de tamaño K."""
        participantes = [
            evaluados[rng.randrange(len(evaluados))]
            for _ in range(self.configuracion_ga.tamano_torneo)
        ]
        ganador = min(participantes, key=lambda ind: (ind.costo, ind.cromosoma))
        return ganador.cromosoma

    @staticmethod
    def _crossover_ordenado(
        padre_a: Cromosoma,
        padre_b: Cromosoma,
        rng: Random,
    ) -> tuple[Cromosoma, Cromosoma]:
        """Aplica el operador de Cruce Ordenado (OX1) entre dos permutaciones."""
        cantidad_genes = len(padre_a)
        if cantidad_genes < 2:
            return padre_a, padre_b

        corte_1, corte_2 = sorted(rng.sample(range(cantidad_genes), 2))
        corte_2 += 1

        hijo_a = GeneticAlgorithmPlanner._crear_hijo_ox(padre_a, padre_b, corte_1, corte_2)
        hijo_b = GeneticAlgorithmPlanner._crear_hijo_ox(padre_b, padre_a, corte_1, corte_2)
        return hijo_a, hijo_b

    @staticmethod
    def _crear_hijo_ox(
        padre_base: Cromosoma,
        padre_relleno: Cromosoma,
        inicio: int,
        fin: int,
    ) -> Cromosoma:
        """Construye un hijo preservando el segmento central del padre base y rellenando con el otro."""
        cantidad_genes = len(padre_base)
        hijo: list[str | None] = [None for _ in range(cantidad_genes)]
        hijo[inicio:fin] = padre_base[inicio:fin]

        genes_usados = set(padre_base[inicio:fin])
        genes_restantes = [g for g in padre_relleno if g not in genes_usados]
        posiciones_libres = [i for i, g in enumerate(hijo) if g is None]

        for pos, gen in zip(posiciones_libres, genes_restantes):
            hijo[pos] = gen

        if any(g is None for g in hijo):
            raise RuntimeError("El crossover OX dejó genes vacíos.")

        return tuple(g for g in hijo if g is not None)

    def _mutar(
        self,
        cromosoma: Cromosoma,
        rng: Random,
        prob_swap: float | None = None,
        prob_inversion: float | None = None,
    ) -> Cromosoma:
        """Aplica mutaciones estocásticas por intercambio (swap) y/o inversión de subcadena."""
        genes = list(cromosoma)
        p_swap = (
            self.configuracion_ga.probabilidad_mutacion_swap if prob_swap is None else prob_swap
        )
        p_inv = (
            self.configuracion_ga.probabilidad_mutacion_inversion
            if prob_inversion is None
            else prob_inversion
        )

        if len(genes) >= 2 and rng.random() < p_swap:
            pos_a, pos_b = rng.sample(range(len(genes)), 2)
            genes[pos_a], genes[pos_b] = genes[pos_b], genes[pos_a]

        if len(genes) >= 3 and rng.random() < p_inv:
            inicio, fin = sorted(rng.sample(range(len(genes)), 2))
            genes[inicio : fin + 1] = reversed(genes[inicio : fin + 1])

        return tuple(genes)

    def _decodificar_plan(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id: dict[str, PedidoInput],
        cromosoma: Cromosoma,
    ) -> PlanTurno:
        """Decodifica un cromosoma en un PlanTurno completo."""
        _ = pedidos_por_id
        return decodificar_plan_permutacion(
            instancia=instancia,
            matriz=matriz,
            configuracion=self.configuracion,
            cromosoma=cromosoma,
            algoritmo=AlgoritmoPlanificacion.GA,
        )


def generar_plan_ga(
    instancia: InstanciaTurno,
    seed: int | None = None,
    configuracion: ConfiguracionPlanificacion | None = None,
    configuracion_ga: ConfiguracionGA | None = None,
    proveedor_viaje: ProveedorViaje | None = None,
    semillas_iniciales: tuple[Cromosoma, ...] | None = None,
    incluir_semilla_greedy: bool = True,
    fraccion_variantes_semilla: float = 0.0,
) -> PlanTurno:
    """Función de conveniencia para ejecutar el planificador genético."""
    return GeneticAlgorithmPlanner(
        configuracion=configuracion,
        configuracion_ga=configuracion_ga,
        seed=seed,
        proveedor_viaje=proveedor_viaje,
        semillas_iniciales=semillas_iniciales,
        incluir_semilla_greedy=incluir_semilla_greedy,
        fraccion_variantes_semilla=fraccion_variantes_semilla,
    ).generar_plan(instancia)
