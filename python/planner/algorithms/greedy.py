"""Heurística constructiva voraz (Greedy) para empaquetado de viajes y ruteo."""

from time import perf_counter

from planner.core.base import PlanificadorTurno
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PedidoInput,
    PlanCamion,
    PlanTurno,
    ViajePlan,
)
from planner.domain.validator import validar_instancia, validar_plan
from planner.routing.objective import (
    estimar_espera_cliente,
    evaluar_plan_estimado,
    tiempo_carga_estimado_min,
    tiempo_descarga_estimado_min,
)
from planner.routing.operations import simular_plan_operativo_estimado
from planner.routing.travel import (
    MatrizViaje,
    ProveedorViaje,
    construir_matriz_viaje,
    tiempo_viaje_esperado_min,
)

__all__ = ["GreedyFeasiblePlanner", "generar_plan_greedy"]


class GreedyFeasiblePlanner(PlanificadorTurno):
    """Planificador voraz determinístico guiado por urgencia de ventanas y empaquetado óptimo.

    Construye viajes aplicando la heurística *Best-Fit Decreasing* sobre la capacidad vehicular,
    priorizando pedidos urgentes y garantizando la ubicación final de descargas por volcador.
    La asignación a vehículos se realiza mediante una política tipo LPT (Longest Processing Time)
    que minimiza el desbalance de horario entre camiones.
    """

    def __init__(
        self,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
    ) -> None:
        self.configuracion = (
            configuracion if configuracion is not None else ConfiguracionPlanificacion()
        )
        self.proveedor_viaje = proveedor_viaje

    def generar_plan(self, instancia: InstanciaTurno) -> PlanTurno:
        """Genera un plan de distribución voraz completamente determinístico.

        Args:
            instancia: Escenario del turno a planificar.

        Returns:
            PlanTurno válido con métricas de costo y tiempo de cálculo asociadas.

        Raises:
            ValueError: Si la instancia incumple restricciones estructurales.
            RuntimeError: Si el plan resultante viola alguna restricción de factibilidad.
        """
        inicio_computo = perf_counter()

        errores_instancia = validar_instancia(instancia)
        if errores_instancia:
            raise ValueError("Instancia inválida: " + " | ".join(errores_instancia))

        matriz = construir_matriz_viaje(
            instancia, self.configuracion, proveedor=self.proveedor_viaje
        )
        pedidos_por_id = {p.pedido_id: p for p in instancia.pedidos}

        viajes_candidatos = self._construir_viajes(instancia, matriz, pedidos_por_id)
        plan = self._asignar_viajes_a_camiones(instancia, matriz, pedidos_por_id, viajes_candidatos)

        validacion = validar_plan(instancia, plan)
        if not validacion.valido:
            raise RuntimeError(
                "GreedyFeasible generó un plan inválido: " + " | ".join(validacion.errores)
            )

        estimacion = evaluar_plan_estimado(instancia, plan, matriz, self.configuracion)
        plan.costo_estimado = estimacion.costo_total
        plan.tiempo_computo_ms = (perf_counter() - inicio_computo) * 1000.0

        return plan

    def _construir_viajes_volcador(
        self,
        instancia: InstanciaTurno,
        normales: list[PedidoInput],
        volcadores: list[PedidoInput],
        pedidos_por_id: dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> list[list[str]]:
        """Construye viajes para pedidos con volcador, rellenando con pedidos normales por Best-Fit."""
        viajes: list[list[str]] = []

        for pedido_volcador in volcadores:
            seleccionados: list[str] = []
            carga = pedido_volcador.unidades_capacidad

            while True:
                candidatos = [
                    p
                    for p in normales
                    if carga + p.unidades_capacidad <= instancia.capacidad_camion
                ]
                if not candidatos:
                    break

                elegido = min(
                    candidatos,
                    key=lambda p: (
                        instancia.capacidad_camion - (carga + p.unidades_capacidad),
                        p.hora_hasta_min,
                        matriz.distancia(self.configuracion.id_nodo_corralon, p.pedido_id),
                        p.pedido_id,
                    ),
                )
                seleccionados.append(elegido.pedido_id)
                normales.remove(elegido)
                carga += elegido.unidades_capacidad

            secuencia = self._secuenciar_normales(
                seleccionados, instancia.hora_inicio_turno_min, pedidos_por_id, matriz
            )
            secuencia.append(pedido_volcador.pedido_id)
            viajes.append(secuencia)

        return viajes

    def _construir_viajes_normales(
        self,
        instancia: InstanciaTurno,
        normales: list[PedidoInput],
        pedidos_por_id: dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> list[list[str]]:
        """Agrupa los pedidos convencionales restantes maximizando la ocupación del camión."""
        viajes: list[list[str]] = []

        while normales:
            ancla = min(
                normales,
                key=lambda p: (
                    p.hora_hasta_min,
                    -p.unidades_capacidad,
                    matriz.distancia(self.configuracion.id_nodo_corralon, p.pedido_id),
                    p.pedido_id,
                ),
            )
            seleccionados = [ancla.pedido_id]
            normales.remove(ancla)
            carga = ancla.unidades_capacidad

            while True:
                candidatos = [
                    p
                    for p in normales
                    if carga + p.unidades_capacidad <= instancia.capacidad_camion
                ]
                if not candidatos:
                    break

                elegido = min(
                    candidatos,
                    key=lambda p: (
                        instancia.capacidad_camion - (carga + p.unidades_capacidad),
                        p.hora_hasta_min,
                        matriz.distancia(self.configuracion.id_nodo_corralon, p.pedido_id),
                        p.pedido_id,
                    ),
                )
                seleccionados.append(elegido.pedido_id)
                normales.remove(elegido)
                carga += elegido.unidades_capacidad

            viajes.append(
                self._secuenciar_normales(
                    seleccionados, instancia.hora_inicio_turno_min, pedidos_por_id, matriz
                )
            )

        return viajes

    def _ordenar_viajes_por_prioridad(
        self,
        viajes: list[list[str]],
        instancia: InstanciaTurno,
        pedidos_por_id: dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> list[list[str]]:
        """Ordena los viajes para su despacho: urgencia horaria, volcador, ocupación y LPT."""
        viajes.sort(
            key=lambda viaje: (
                min(pedidos_por_id[p_id].hora_hasta_min for p_id in viaje),
                0 if any(pedidos_por_id[p_id].requiere_volcador for p_id in viaje) else 1,
                -sum(pedidos_por_id[p_id].unidades_capacidad for p_id in viaje),
                -self._duracion_referencia_viaje(viaje, instancia, pedidos_por_id, matriz),
                tuple(viaje),
            )
        )
        return viajes

    def _construir_viajes(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id: dict[str, PedidoInput],
    ) -> list[list[str]]:
        """Construye y prioriza todos los viajes del turno."""
        normales = [p for p in instancia.pedidos if not p.requiere_volcador]
        volcadores = sorted(
            [p for p in instancia.pedidos if p.requiere_volcador],
            key=lambda p: (
                p.hora_hasta_min,
                matriz.distancia(self.configuracion.id_nodo_corralon, p.pedido_id),
                p.pedido_id,
            ),
        )

        viajes = self._construir_viajes_volcador(
            instancia, normales, volcadores, pedidos_por_id, matriz
        )
        viajes.extend(self._construir_viajes_normales(instancia, normales, pedidos_por_id, matriz))
        return self._ordenar_viajes_por_prioridad(viajes, instancia, pedidos_por_id, matriz)

    def _duracion_referencia_viaje(
        self,
        pedido_ids: list[str],
        instancia: InstanciaTurno,
        pedidos_por_id: dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> float:
        """Calcula la duración estimada aislada de un viaje a efectos de ordenamiento LPT."""
        inicio = float(instancia.hora_inicio_turno_min)
        fin = self._estimar_fin_viaje(pedido_ids, inicio, pedidos_por_id, matriz)
        return max(0.0, fin - inicio)

    def _secuenciar_normales(
        self,
        pedido_ids: list[str],
        minuto_inicio: float,
        pedidos_por_id: dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> list[str]:
        """Ordena internamente las paradas de un viaje vorazmente por costo de inserción."""
        if len(pedido_ids) <= 1:
            return list(pedido_ids)

        pendientes = set(pedido_ids)
        secuencia: list[str] = []
        nodo_actual = self.configuracion.id_nodo_corralon
        minuto_actual = minuto_inicio

        while pendientes:
            candidatos = [pedidos_por_id[p_id] for p_id in pendientes]
            elegido = min(
                candidatos,
                key=lambda p: (
                    self._puntaje_siguiente(nodo_actual, p, minuto_actual, matriz),
                    p.hora_hasta_min,
                    matriz.distancia(nodo_actual, p.pedido_id),
                    p.pedido_id,
                ),
            )

            secuencia.append(elegido.pedido_id)
            pendientes.remove(elegido.pedido_id)

            tiempo_viaje = tiempo_viaje_esperado_min(
                matriz, nodo_actual, elegido.pedido_id, minuto_actual, self.configuracion
            )
            minuto_actual += tiempo_viaje
            espera = estimar_espera_cliente(elegido, minuto_actual, self.configuracion)
            minuto_actual = espera.minuto_inicio_descarga
            minuto_actual += tiempo_descarga_estimado_min(elegido, self.configuracion)
            nodo_actual = elegido.pedido_id

        return secuencia

    def _puntaje_siguiente(
        self,
        nodo_actual: str,
        pedido: PedidoInput,
        minuto_actual: float,
        matriz: MatrizViaje,
    ) -> float:
        """Puntaje ponderado de cercanía y penalización por tardanza para la siguiente parada."""
        tiempo_viaje = tiempo_viaje_esperado_min(
            matriz, nodo_actual, pedido.pedido_id, minuto_actual, self.configuracion
        )
        llegada_estimada = minuto_actual + tiempo_viaje
        tardanza = max(0.0, llegada_estimada - pedido.hora_hasta_min)

        return tiempo_viaje + (tardanza * 5.0)

    def _asignar_viajes_a_camiones(
        self,
        instancia: InstanciaTurno,
        matriz: MatrizViaje,
        pedidos_por_id: dict[str, PedidoInput],
        viajes: list[list[str]],
    ) -> PlanTurno:
        """Asigna secuencialmente los viajes al camión con menor tiempo proyectado de finalización."""
        planes_camion = [PlanCamion(camion_id=c_id) for c_id in range(instancia.cantidad_camiones)]
        disponibilidad = [
            float(instancia.hora_inicio_turno_min) for _ in range(instancia.cantidad_camiones)
        ]

        for pedido_ids in viajes:
            camion_id = min(
                range(instancia.cantidad_camiones),
                key=lambda c_id: (
                    disponibilidad[c_id],
                    len(planes_camion[c_id].viajes),
                    c_id,
                ),
            )

            numero_viaje = len(planes_camion[camion_id].viajes) + 1
            planes_camion[camion_id].viajes.append(
                ViajePlan(numero_viaje=numero_viaje, pedido_ids=list(pedido_ids))
            )

            plan_parcial = PlanTurno(
                instancia_id=instancia.instancia_id,
                algoritmo=AlgoritmoPlanificacion.GREEDY,
                camiones=planes_camion,
            )
            resultado_operativo = simular_plan_operativo_estimado(
                instancia, plan_parcial, matriz, self.configuracion
            )
            disponibilidad = list(resultado_operativo.finales_camiones_min)

        return PlanTurno(
            instancia_id=instancia.instancia_id,
            algoritmo=AlgoritmoPlanificacion.GREEDY,
            camiones=planes_camion,
        )

    def _estimar_fin_viaje(
        self,
        pedido_ids: list[str],
        minuto_inicio: float,
        pedidos_por_id: dict[str, PedidoInput],
        matriz: MatrizViaje,
    ) -> float:
        """Estima el minuto de finalización de un viaje individual para cálculos de ordenamiento."""
        unidades = sum(pedidos_por_id[p_id].unidades_capacidad for p_id in pedido_ids)
        minuto_actual = minuto_inicio + tiempo_carga_estimado_min(unidades, self.configuracion)
        nodo_actual = self.configuracion.id_nodo_corralon

        for pedido_id in pedido_ids:
            pedido = pedidos_por_id[pedido_id]
            minuto_actual += tiempo_viaje_esperado_min(
                matriz, nodo_actual, pedido_id, minuto_actual, self.configuracion
            )
            espera = estimar_espera_cliente(pedido, minuto_actual, self.configuracion)
            minuto_actual = espera.minuto_inicio_descarga
            minuto_actual += tiempo_descarga_estimado_min(pedido, self.configuracion)
            nodo_actual = pedido_id

        minuto_actual += tiempo_viaje_esperado_min(
            matriz,
            nodo_actual,
            self.configuracion.id_nodo_corralon,
            minuto_actual,
            self.configuracion,
        )
        return minuto_actual


def generar_plan_greedy(
    instancia: InstanciaTurno,
    configuracion: ConfiguracionPlanificacion | None = None,
    proveedor_viaje: ProveedorViaje | None = None,
) -> PlanTurno:
    """Función de conveniencia para ejecutar el planificador voraz determinístico."""
    return GreedyFeasiblePlanner(
        configuracion=configuracion,
        proveedor_viaje=proveedor_viaje,
    ).generar_plan(instancia)
