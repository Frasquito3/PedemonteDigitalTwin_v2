"""Planificador estocástico de soluciones factibles para benchmarking y líneas de base."""

from random import Random
from time import perf_counter

from planner.core.base import PlanificadorTurno
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PlanCamion,
    PlanTurno,
    ViajePlan,
)
from planner.domain.validator import validar_instancia, validar_plan
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import ProveedorViaje, construir_matriz_viaje

__all__ = ["RandomFeasiblePlanner", "generar_plan_random"]


class RandomFeasiblePlanner(PlanificadorTurno):
    """Generador estocástico de planes factibles que respetan capacidades y restricciones duras.

    Agrupa pedidos de forma aleatoria garantizando que ningún viaje supere la capacidad
    nominal del camión y que los pedidos con descarga por volcador se ubiquen siempre al final.
    Distribuye los viajes entre los vehículos buscando balancear la cantidad de salidas,
    sin optimizar tiempos ni distancias.
    """

    def __init__(
        self,
        configuracion: ConfiguracionPlanificacion | None = None,
        seed: int | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
    ) -> None:
        self.configuracion = (
            configuracion if configuracion is not None else ConfiguracionPlanificacion()
        )
        self.seed = seed
        self.proveedor_viaje = proveedor_viaje
        self.ultima_seed_utilizada: int | None = None

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        """Construye un plan estocástico válido y computa su función objetivo estimada.

        Args:
            instancia: Escenario del turno con pedidos y restricciones de flota.

        Returns:
            PlanTurno válido con costo y tiempo de cómputo registrados.

        Raises:
            ValueError: Si la instancia incumple validaciones estructurales.
            RuntimeError: Si el plan generado no supera las reglas duras de factibilidad.
        """
        inicio_computo = perf_counter()

        errores_instancia = validar_instancia(instancia)
        if errores_instancia:
            raise ValueError("Instancia inválida: " + " | ".join(errores_instancia))

        seed_efectiva = self.seed if self.seed is not None else instancia.seed_escenario + 7001
        self.ultima_seed_utilizada = seed_efectiva
        rng = Random(seed_efectiva)

        matriz = construir_matriz_viaje(
            instancia, self.configuracion, proveedor=self.proveedor_viaje
        )
        viajes = self._construir_viajes_aleatorios(instancia, rng)
        plan = self._asignar_viajes_a_camiones(instancia, viajes, rng)

        validacion = validar_plan(instancia, plan)
        if not validacion.valido:
            raise RuntimeError(
                "RandomFeasible generó un plan inválido: " + " | ".join(validacion.errores)
            )

        estimacion = evaluar_plan_estimado(instancia, plan, matriz, self.configuracion)
        plan.costo_estimado = estimacion.costo_total
        plan.tiempo_computo_ms = (perf_counter() - inicio_computo) * 1000.0

        return plan

    def _construir_viajes_aleatorios(
        self,
        instancia: InstanciaTurno,
        rng: Random,
    ) -> list[list[str]]:
        """Genera grupos aleatorios de pedidos respetando capacidad y regla de volcador."""
        normales = [p for p in instancia.pedidos if not p.requiere_volcador]
        volcadores = [p for p in instancia.pedidos if p.requiere_volcador]

        rng.shuffle(normales)
        rng.shuffle(volcadores)
        viajes: list[list[str]] = []

        # 1. Viajes con volcador: un volcador por viaje, ubicado obligatoriamente al final
        for pedido_volcador in volcadores:
            seleccionados: list[str] = []
            carga = pedido_volcador.unidades_capacidad

            while True:
                indices_factibles = [
                    i
                    for i, p in enumerate(normales)
                    if carga + p.unidades_capacidad <= instancia.capacidad_camion
                ]
                if not indices_factibles:
                    break

                indice_elegido = rng.choice(indices_factibles)
                elegido = normales.pop(indice_elegido)
                seleccionados.append(elegido.pedido_id)
                carga += elegido.unidades_capacidad

            rng.shuffle(seleccionados)
            seleccionados.append(pedido_volcador.pedido_id)
            viajes.append(seleccionados)

        # 2. Viajes de pedidos normales restantes
        while normales:
            ancla = normales.pop(rng.randrange(len(normales)))
            viaje = [ancla.pedido_id]
            carga = ancla.unidades_capacidad

            while True:
                indices_factibles = [
                    i
                    for i, p in enumerate(normales)
                    if carga + p.unidades_capacidad <= instancia.capacidad_camion
                ]
                if not indices_factibles:
                    break

                indice_elegido = rng.choice(indices_factibles)
                elegido = normales.pop(indice_elegido)
                viaje.append(elegido.pedido_id)
                carga += elegido.unidades_capacidad

            rng.shuffle(viaje)
            viajes.append(viaje)

        rng.shuffle(viajes)
        return viajes

    def _asignar_viajes_a_camiones(
        self,
        instancia: InstanciaTurno,
        viajes: list[list[str]],
        rng: Random,
    ) -> PlanTurno:
        """Asigna viajes aleatoriamente entre los camiones con menor cantidad de salidas."""
        planes_camion = [PlanCamion(camion_id=c_id) for c_id in range(instancia.cantidad_camiones)]
        cantidad_viajes = [0 for _ in range(instancia.cantidad_camiones)]

        for pedido_ids in viajes:
            menor_cantidad = min(cantidad_viajes)
            camiones_candidatos = [
                c_id for c_id, cant in enumerate(cantidad_viajes) if cant == menor_cantidad
            ]
            camion_id = rng.choice(camiones_candidatos)

            numero_viaje = len(planes_camion[camion_id].viajes) + 1
            planes_camion[camion_id].viajes.append(
                ViajePlan(numero_viaje=numero_viaje, pedido_ids=list(pedido_ids))
            )
            cantidad_viajes[camion_id] += 1

        return PlanTurno(
            instancia_id=instancia.instancia_id,
            algoritmo=AlgoritmoPlanificacion.RANDOM,
            camiones=planes_camion,
        )


def generar_plan_random(
    instancia: InstanciaTurno,
    seed: int | None = None,
    configuracion: ConfiguracionPlanificacion | None = None,
    proveedor_viaje: ProveedorViaje | None = None,
) -> PlanTurno:
    """Función de conveniencia para generar un plan aleatorio factible."""
    return RandomFeasiblePlanner(
        configuracion=configuracion,
        seed=seed,
        proveedor_viaje=proveedor_viaje,
    ).generar_plan(instancia)
