from time import perf_counter

from planner.core.base import PlanificadorTurno

from planner.core.config import (
    ConfiguracionPlanificacion,
)

from planner.routing.objective import (
    evaluar_plan_estimado,
    tiempo_carga_estimado_min,
    tiempo_descarga_estimado_min,
)

from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    ViajePlan,
)

from planner.routing.travel import (
    MatrizViaje,
    construir_matriz_viaje,
    tiempo_viaje_esperado_min,
)

from planner.domain.validator import (
    validar_instancia,
    validar_plan,
)


class GreedyFeasiblePlanner(
    PlanificadorTurno
):
    def __init__(
        self,
        configuracion:
            ConfiguracionPlanificacion
            | None = None,
    ) -> None:
        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )

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

        matriz = construir_matriz_viaje(
            instancia,
            self.configuracion,
        )

        pedidos_por_id = {
            pedido.pedido_id: pedido
            for pedido in instancia.pedidos
        }

        viajes_candidatos = (
            self._construir_viajes(
                instancia,
                matriz,
                pedidos_por_id,
            )
        )

        plan = (
            self._asignar_viajes_a_camiones(
                instancia,
                matriz,
                pedidos_por_id,
                viajes_candidatos,
            )
        )

        validacion = validar_plan(
            instancia,
            plan,
        )

        if not validacion.valido:
            raise RuntimeError(
                "GreedyFeasible generó un "
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

    def _construir_viajes(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id:
            dict[str, PedidoInput],
    ) -> list[list[str]]:
        normales = [
            pedido
            for pedido in instancia.pedidos
            if not pedido.requiere_volcador
        ]

        volcadores = sorted(
            (
                pedido
                for pedido
                in instancia.pedidos
                if pedido.requiere_volcador
            ),
            key=lambda pedido: (
                pedido.hora_hasta_min,

                matriz.distancia(
                    self.configuracion
                    .id_nodo_corralon,

                    pedido.pedido_id,
                ),

                pedido.pedido_id,
            ),
        )

        viajes: list[list[str]] = []

        # =================================================
        # VIAJES QUE CONTIENEN VOLCADOR
        # =================================================
        #
        # Primero reservamos un viaje factible para cada
        # volcador. Los normales pueden ir antes, pero
        # nunca después.

        for pedido_volcador in volcadores:
            seleccionados: list[str] = []

            carga = (
                pedido_volcador
                .unidades_capacidad
            )

            while True:
                candidatos = [
                    pedido
                    for pedido in normales
                    if (
                        carga
                        + pedido.unidades_capacidad
                        <= instancia
                        .capacidad_camion
                    )
                ]

                if not candidatos:
                    break

                # Best fit:
                # elegimos el pedido que deje el menor
                # espacio vacío en el camión.
                elegido = min(
                    candidatos,
                    key=lambda pedido: (
                        instancia.capacidad_camion
                        - (
                            carga
                            + pedido
                            .unidades_capacidad
                        ),

                        pedido.hora_hasta_min,

                        matriz.distancia(
                            self.configuracion
                            .id_nodo_corralon,

                            pedido.pedido_id,
                        ),

                        pedido.pedido_id,
                    ),
                )

                seleccionados.append(
                    elegido.pedido_id
                )

                normales.remove(elegido)

                carga += (
                    elegido.unidades_capacidad
                )

            secuencia = (
                self._secuenciar_normales(
                    seleccionados,

                    instancia
                    .hora_inicio_turno_min,

                    pedidos_por_id,

                    matriz,
                )
            )

            # Restricción dura:
            # el volcador se agrega siempre al final.
            secuencia.append(
                pedido_volcador.pedido_id
            )

            viajes.append(secuencia)

        # =================================================
        # PEDIDOS NORMALES RESTANTES
        # =================================================

        while normales:
            ancla = min(
                normales,
                key=lambda pedido: (
                    pedido.hora_hasta_min,

                    -pedido.unidades_capacidad,

                    matriz.distancia(
                        self.configuracion
                        .id_nodo_corralon,

                        pedido.pedido_id,
                    ),

                    pedido.pedido_id,
                ),
            )

            seleccionados = [
                ancla.pedido_id
            ]

            normales.remove(ancla)

            carga = (
                ancla.unidades_capacidad
            )

            while True:
                candidatos = [
                    pedido
                    for pedido in normales
                    if (
                        carga
                        + pedido.unidades_capacidad
                        <= instancia
                        .capacidad_camion
                    )
                ]

                if not candidatos:
                    break

                elegido = min(
                    candidatos,
                    key=lambda pedido: (
                        instancia.capacidad_camion
                        - (
                            carga
                            + pedido
                            .unidades_capacidad
                        ),

                        pedido.hora_hasta_min,

                        matriz.distancia(
                            self.configuracion
                            .id_nodo_corralon,

                            pedido.pedido_id,
                        ),

                        pedido.pedido_id,
                    ),
                )

                seleccionados.append(
                    elegido.pedido_id
                )

                normales.remove(elegido)

                carga += (
                    elegido.unidades_capacidad
                )

            viajes.append(
                self._secuenciar_normales(
                    seleccionados,

                    instancia
                    .hora_inicio_turno_min,

                    pedidos_por_id,

                    matriz,
                )
            )

        # Orden estable:
        # 1. ventana más urgente;
        # 2. viajes con volcador;
        # 3. mayor ocupación;
        # 4. IDs como desempate.

        viajes.sort(
            key=lambda viaje: (
                min(
                    pedidos_por_id[
                        pedido_id
                    ].hora_hasta_min
                    for pedido_id in viaje
                ),

                (
                    0
                    if any(
                        pedidos_por_id[
                            pedido_id
                        ].requiere_volcador
                        for pedido_id in viaje
                    )
                    else 1
                ),

                -sum(
                    pedidos_por_id[
                        pedido_id
                    ].unidades_capacidad
                    for pedido_id in viaje
                ),

                tuple(viaje),
            )
        )

        return viajes

    def _secuenciar_normales(
        self,
        pedido_ids: list[str],
        minuto_inicio: float,
        pedidos_por_id:
            dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> list[str]:
        pendientes = [
            pedidos_por_id[pedido_id]
            for pedido_id in pedido_ids
        ]

        secuencia: list[str] = []

        nodo_actual = (
            self.configuracion
            .id_nodo_corralon
        )

        minuto_actual = minuto_inicio

        while pendientes:
            elegido = min(
                pendientes,
                key=lambda pedido:
                    self._puntaje_siguiente(
                        pedido,
                        nodo_actual,
                        minuto_actual,
                        matriz,
                    ),
            )

            secuencia.append(
                elegido.pedido_id
            )

            pendientes.remove(elegido)

            minuto_actual += (
                tiempo_viaje_esperado_min(
                    matriz,
                    nodo_actual,
                    elegido.pedido_id,
                    minuto_actual,
                    self.configuracion,
                )
            )

            if (
                minuto_actual
                < elegido.hora_desde_min
            ):
                minuto_actual = (
                    elegido.hora_desde_min
                )

            minuto_actual += (
                tiempo_descarga_estimado_min(
                    elegido,
                    self.configuracion,
                )
            )

            nodo_actual = elegido.pedido_id

        return secuencia

    def _puntaje_siguiente(
        self,
        pedido: PedidoInput,
        nodo_actual: str,
        minuto_actual: float,
        matriz: MatrizViaje,
    ) -> tuple:
        llegada_estimada = (
            minuto_actual
            + tiempo_viaje_esperado_min(
                matriz,
                nodo_actual,
                pedido.pedido_id,
                minuto_actual,
                self.configuracion,
            )
        )

        tardanza_estimada = max(
            0.0,
            llegada_estimada
            - pedido.hora_hasta_min,
        )

        return (
            # Primero preferimos no llegar tarde.
            (
                1
                if tardanza_estimada > 0.0
                else 0
            ),

            tardanza_estimada,

            # Después, ventana más urgente.
            pedido.hora_hasta_min,

            # Luego, menor distancia.
            matriz.distancia(
                nodo_actual,
                pedido.pedido_id,
            ),

            # Desempate determinista.
            pedido.pedido_id,
        )

    def _asignar_viajes_a_camiones(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id:
            dict[str, PedidoInput],
        viajes_candidatos:
            list[list[str]],
    ) -> PlanTurno:
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

        for pedido_ids in viajes_candidatos:
            camion_id = min(
                range(
                    instancia
                    .cantidad_camiones
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

                disponibilidad[
                    camion_id
                ],

                pedidos_por_id,

                matriz,
            )

        return PlanTurno(
            instancia_id=(
                instancia.instancia_id
            ),

            algoritmo=(
                AlgoritmoPlanificacion
                .GREEDY
            ),

            camiones=planes_camion,
        )

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


def generar_plan_greedy(
    instancia: InstanciaTurno,

    configuracion:
        ConfiguracionPlanificacion
        | None = None,
) -> PlanTurno:
    return GreedyFeasiblePlanner(
        configuracion=configuracion,
    ).generar_plan(instancia)