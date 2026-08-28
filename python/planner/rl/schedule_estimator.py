from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Iterable
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PedidoInput
from planner.routing.operations import estimar_espera_cliente, tiempo_carga_estimado_min, tiempo_descarga_estimado_min
from planner.routing.travel import MatrizViaje, tiempo_viaje_esperado_min

@dataclass(frozen=True)
class RegistroTemporalPedido:
    pedido_id: str
    camion_id: int
    numero_viaje: int
    orden_visita: int
    minuto_inicio_carga: float
    minuto_salida_viaje: float
    minuto_llegada: float
    espera_apertura_min: float
    espera_respuesta_min: float
    minuto_inicio_descarga: float
    minuto_fin_descarga: float
    holgura_llegada_min: float
    holgura_fin_descarga_min: float
    tardanza_llegada_min: float

    @property
    def llegada_tardia(self) -> bool:
        return self.tardanza_llegada_min > 1e-09

@dataclass(frozen=True)
class ResumenTemporalPrefijo:
    prefijo: tuple[str, ...]
    registros: tuple[RegistroTemporalPedido, ...]
    finales_camiones_min: tuple[float, ...]
    minuto_referencia: float
    tardanza_total_min: float
    espera_apertura_total_min: float
    pedidos_tardios: int

    def registro_de(self, pedido_id: str) -> RegistroTemporalPedido | None:
        for registro in self.registros:
            if registro.pedido_id == pedido_id:
                return registro
        return None

@dataclass(frozen=True)
class ProyeccionTemporalAccion:
    pedido_id: str
    registro: RegistroTemporalPedido

def _validar_prefijo(instancia: InstanciaTurno, prefijo: tuple[str, ...]) -> None:
    ids_validos = {pedido.pedido_id for pedido in instancia.pedidos}
    if len(prefijo) != len(set(prefijo)):
        raise ValueError('El prefijo temporal contiene pedidos repetidos.')
    desconocidos = [pedido_id for pedido_id in prefijo if pedido_id not in ids_validos]
    if desconocidos:
        raise ValueError(f'El prefijo temporal contiene pedidos desconocidos: {desconocidos}.')

def _agrupar_viajes_prefijo(instancia: InstanciaTurno, pedidos_por_id: dict[str, PedidoInput], prefijo: tuple[str, ...]) -> list[list[str]]:
    viajes: list[list[str]] = []
    viaje_actual: list[str] = []
    carga_actual = 0

    def cerrar_viaje() -> None:
        nonlocal viaje_actual
        nonlocal carga_actual
        if viaje_actual:
            viajes.append(list(viaje_actual))
        viaje_actual = []
        carga_actual = 0
    for pedido_id in prefijo:
        pedido = pedidos_por_id[pedido_id]
        if carga_actual + pedido.unidades_capacidad > instancia.capacidad_camion:
            cerrar_viaje()
        viaje_actual.append(pedido_id)
        carga_actual += pedido.unidades_capacidad
        if pedido.requiere_volcador:
            cerrar_viaje()
    cerrar_viaje()
    return viajes

def analizar_prefijo_temporal(instancia: InstanciaTurno, matriz: MatrizViaje, configuracion: ConfiguracionPlanificacion, prefijo: Iterable[str]) -> ResumenTemporalPrefijo:
    """
    Estima cronológicamente un prefijo de permutación.

    Es un estimador ligero para observaciones y shaping durante el
    entrenamiento. Conserva las reglas de capacidad, cierre por
    volcador, asignación al camión disponible y tiempos esperados de
    viaje/cliente. No reemplaza la evaluación operativa final ni la
    simulación estocástica de AnyLogic.
    """
    prefijo_tuple = tuple(prefijo)
    _validar_prefijo(instancia, prefijo_tuple)
    pedidos_por_id = {pedido.pedido_id: pedido for pedido in instancia.pedidos}
    finales_camiones = [float(instancia.hora_inicio_turno_min) for _ in range(instancia.cantidad_camiones)]
    cantidad_viajes_camion = [0 for _ in range(instancia.cantidad_camiones)]
    registros: list[RegistroTemporalPedido] = []
    for pedido_ids in _agrupar_viajes_prefijo(instancia, pedidos_por_id, prefijo_tuple):
        camion_id = min(range(instancia.cantidad_camiones), key=lambda candidato: (finales_camiones[candidato], cantidad_viajes_camion[candidato], candidato))
        numero_viaje = cantidad_viajes_camion[camion_id] + 1
        cantidad_viajes_camion[camion_id] += 1
        unidades = sum((pedidos_por_id[pedido_id].unidades_capacidad for pedido_id in pedido_ids))
        minuto_inicio_carga = finales_camiones[camion_id]
        minuto_actual = minuto_inicio_carga + tiempo_carga_estimado_min(unidades, configuracion)
        minuto_salida = minuto_actual
        nodo_actual = configuracion.id_nodo_corralon
        for orden_visita, pedido_id in enumerate(pedido_ids, start=1):
            pedido = pedidos_por_id[pedido_id]
            minuto_actual += tiempo_viaje_esperado_min(matriz, nodo_actual, pedido_id, minuto_actual, configuracion)
            minuto_llegada = minuto_actual
            tardanza = max(0.0, minuto_llegada - pedido.hora_hasta_min)
            espera = estimar_espera_cliente(pedido, minuto_llegada, configuracion)
            minuto_inicio_descarga = espera.minuto_inicio_descarga
            minuto_fin_descarga = minuto_inicio_descarga + tiempo_descarga_estimado_min(pedido, configuracion)
            registros.append(RegistroTemporalPedido(pedido_id=pedido_id, camion_id=camion_id, numero_viaje=numero_viaje, orden_visita=orden_visita, minuto_inicio_carga=minuto_inicio_carga, minuto_salida_viaje=minuto_salida, minuto_llegada=minuto_llegada, espera_apertura_min=espera.tiempo_espera_ventana_min, espera_respuesta_min=espera.tiempo_espera_respuesta_min, minuto_inicio_descarga=minuto_inicio_descarga, minuto_fin_descarga=minuto_fin_descarga, holgura_llegada_min=pedido.hora_hasta_min - minuto_llegada, holgura_fin_descarga_min=pedido.hora_hasta_min - minuto_fin_descarga, tardanza_llegada_min=tardanza))
            minuto_actual = minuto_fin_descarga
            nodo_actual = pedido_id
        minuto_actual += tiempo_viaje_esperado_min(matriz, nodo_actual, configuracion.id_nodo_corralon, minuto_actual, configuracion)
        finales_camiones[camion_id] = minuto_actual
    minuto_referencia = max(finales_camiones, default=float(instancia.hora_inicio_turno_min))
    tardanza_total = sum((registro.tardanza_llegada_min for registro in registros))
    espera_total = sum((registro.espera_apertura_min for registro in registros))
    pedidos_tardios = sum((1 for registro in registros if registro.llegada_tardia))
    valores = (*finales_camiones, minuto_referencia, tardanza_total, espera_total)
    if not all((isfinite(valor) for valor in valores)):
        raise RuntimeError('El estimador temporal produjo valores no finitos.')
    return ResumenTemporalPrefijo(prefijo=prefijo_tuple, registros=tuple(registros), finales_camiones_min=tuple(finales_camiones), minuto_referencia=minuto_referencia, tardanza_total_min=tardanza_total, espera_apertura_total_min=espera_total, pedidos_tardios=pedidos_tardios)

def proyectar_acciones_pendientes(instancia: InstanciaTurno, matriz: MatrizViaje, configuracion: ConfiguracionPlanificacion, prefijo: Iterable[str]) -> dict[str, ProyeccionTemporalAccion]:
    prefijo_tuple = tuple(prefijo)
    seleccionados = set(prefijo_tuple)
    proyecciones: dict[str, ProyeccionTemporalAccion] = {}
    for pedido in instancia.pedidos:
        if pedido.pedido_id in seleccionados:
            continue
        resumen = analizar_prefijo_temporal(instancia, matriz, configuracion, (*prefijo_tuple, pedido.pedido_id))
        registro = resumen.registro_de(pedido.pedido_id)
        if registro is None:
            raise RuntimeError(f'No se pudo recuperar la proyección temporal de {pedido.pedido_id}.')
        proyecciones[pedido.pedido_id] = ProyeccionTemporalAccion(pedido_id=pedido.pedido_id, registro=registro)
    return proyecciones
