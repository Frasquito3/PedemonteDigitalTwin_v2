from random import Random
from time import perf_counter

from .base import PlanificadorTurno

from .config import (
    ConfiguracionPlanificacion,
)

from .objective import (
    evaluar_plan_estimado,
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
    construir_matriz_viaje,
)

from .validator import (
    validar_instancia,
    validar_plan,
)


class RandomFeasiblePlanner(
    PlanificadorTurno
):
    def __init__(
        self,
        configuracion:
            ConfiguracionPlanificacion
            | None = None,
        seed: int | None = None,
    ) -> None:
        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )

        self.seed = seed

        self.ultima_seed_utilizada: int | None = None

    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        inicio_computo = perf_counter()

        errores_instancia = (
            validar_instancia(instancia)
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
            + 7001
        )

        self.ultima_seed_utilizada = (
            seed_efectiva
        )

        rng = Random(
            seed_efectiva
        )

        matriz = construir_matriz_viaje(
            instancia,
            self.configuracion,
        )

        viajes = (
            self._construir_viajes_aleatorios(
                instancia,
                rng,
            )
        )

        plan = (
            self._asignar_viajes_a_camiones(
                instancia,
                viajes,
                rng,
            )
        )

        validacion = validar_plan(
            instancia,
            plan,
        )

        if not validacion.valido:
            raise RuntimeError(
                "RandomFeasible generó un "
                "plan inválido: "
                + " | ".join(
                    validacion.errores
                )
            )

        estimacion = evaluar_plan_estimado(
            instancia,
            plan,
            matriz,
            self.configuracion,
        )

        plan.costo_estimado = (
            estimacion.costo_total
        )

        plan.tiempo_computo_ms = (
            perf_counter()
            - inicio_computo
        ) * 1000.0

        return plan

    def _construir_viajes_aleatorios(
        self,
        instancia: InstanciaTurno,
        rng: Random,
    ) -> list[list[str]]:
        normales = [
            pedido
            for pedido in instancia.pedidos
            if not pedido.requiere_volcador
        ]

        volcadores = [
            pedido
            for pedido in instancia.pedidos
            if pedido.requiere_volcador
        ]

        # Trabajamos sobre listas nuevas.
        # No modificamos instancia.pedidos.
        rng.shuffle(normales)
        rng.shuffle(volcadores)

        viajes: list[list[str]] = []

        # =================================================
        # VIAJES CON VOLCADOR
        # =================================================
        #
        # Cada volcador genera un viaje independiente.
        # Se pueden agregar pedidos normales antes,
        # pero el volcador queda siempre último.

        for pedido_volcador in volcadores:
            seleccionados: list[str] = []

            carga = (
                pedido_volcador
                .unidades_capacidad
            )

            while True:
                indices_factibles = [
                    indice
                    for indice, pedido
                    in enumerate(normales)
                    if (
                        carga
                        + pedido.unidades_capacidad
                        <= instancia.capacidad_camion
                    )
                ]

                if not indices_factibles:
                    break

                indice_elegido = rng.choice(
                    indices_factibles
                )

                elegido = normales.pop(
                    indice_elegido
                )

                seleccionados.append(
                    elegido.pedido_id
                )

                carga += (
                    elegido.unidades_capacidad
                )

            # El orden de los normales también
            # puede variar.
            rng.shuffle(
                seleccionados
            )

            # Restricción dura:
            # el volcador siempre se agrega al final.
            seleccionados.append(
                pedido_volcador.pedido_id
            )

            viajes.append(
                seleccionados
            )

        # =================================================
        # VIAJES DE PEDIDOS NORMALES RESTANTES
        # =================================================

        while normales:
            indice_ancla = rng.randrange(
                len(normales)
            )

            ancla = normales.pop(
                indice_ancla
            )

            viaje = [
                ancla.pedido_id
            ]

            carga = (
                ancla.unidades_capacidad
            )

            while True:
                indices_factibles = [
                    indice
                    for indice, pedido
                    in enumerate(normales)
                    if (
                        carga
                        + pedido.unidades_capacidad
                        <= instancia.capacidad_camion
                    )
                ]

                if not indices_factibles:
                    break

                indice_elegido = rng.choice(
                    indices_factibles
                )

                elegido = normales.pop(
                    indice_elegido
                )

                viaje.append(
                    elegido.pedido_id
                )

                carga += (
                    elegido.unidades_capacidad
                )

            rng.shuffle(
                viaje
            )

            viajes.append(
                viaje
            )

        # También aleatorizamos el orden en el que
        # los viajes se asignarán a los camiones.
        rng.shuffle(
            viajes
        )

        return viajes

    def _asignar_viajes_a_camiones(
        self,
        instancia: InstanciaTurno,
        viajes: list[list[str]],
        rng: Random,
    ) -> PlanTurno:
        planes_camion = [
            PlanCamion(
                camion_id=camion_id
            )
            for camion_id in range(
                instancia.cantidad_camiones
            )
        ]

        cantidad_viajes = [
            0
            for _ in range(
                instancia.cantidad_camiones
            )
        ]

        for pedido_ids in viajes:
            menor_cantidad = min(
                cantidad_viajes
            )

            camiones_candidatos = [
                camion_id
                for camion_id, cantidad
                in enumerate(cantidad_viajes)
                if cantidad == menor_cantidad
            ]

            # Aleatorio entre los camiones que
            # actualmente tienen menos viajes.
            #
            # Esto evita un baseline degenerado
            # que asigne todo a un solo camión,
            # pero no utiliza distancia ni costo.
            camion_id = rng.choice(
                camiones_candidatos
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

            cantidad_viajes[
                camion_id
            ] += 1

        return PlanTurno(
            instancia_id=(
                instancia.instancia_id
            ),

            algoritmo=(
                AlgoritmoPlanificacion
                .RANDOM
            ),

            camiones=planes_camion,
        )


def generar_plan_random(
    instancia: InstanciaTurno,
    seed: int | None = None,
    configuracion:
        ConfiguracionPlanificacion
        | None = None,
) -> PlanTurno:
    return RandomFeasiblePlanner(
        configuracion=configuracion,
        seed=seed,
    ).generar_plan(instancia)