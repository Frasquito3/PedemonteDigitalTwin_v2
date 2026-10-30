"""Parámetros operacionales, cinemáticos y ponderaciones de costos del planificador."""

from dataclasses import dataclass

__all__ = ["ConfiguracionPlanificacion"]


@dataclass(frozen=True)
class ConfiguracionPlanificacion:
    """Configuración global de parámetros operativos, tiempos y penalizaciones.

    Centraliza las constantes de velocidad, factores de congestión horaria,
    modelos de carga/descarga y pesos de la función objetivo utilizados
    tanto en la generación heurística como en la evaluación del plan.

    Attributes:
        id_nodo_corralon: Identificador del depósito central de operaciones.
        factor_urbano_distancia: Coeficiente de corrección sobre la distancia
            rectilínea para aproximar la red vial urbana.
        velocidad_base_kmh: Velocidad promedio de circulación fuera de horas pico.
        personas_carga_estimadas: Dotación fija de referencia para estimaciones
            estáticas y heurísticas constructivas (la simulación final evalúa
            la concurrencia dinámica real de operarios).
        cantidad_empleados_corralon: Dotación física total de operarios disponibles
            en el corralón para tareas simultáneas de carga y recepción de proveedores.
        carga_setup_min: Tiempo fijo de preparación previo a la carga de un viaje.
        carga_min_por_unidad_1_persona: Minutos base para cargar una unidad con un operario.
        carga_eficiencia_persona_adicional: Rendimiento marginal por cada operario adicional
            asignado a la misma carga (modelo de rendimientos decrecientes).
        cliente_espera_respuesta_min: Tiempo mínimo de espera hasta ser atendido en obra.
        cliente_espera_respuesta_moda: Tiempo modal más probable de espera en obra.
        cliente_espera_respuesta_max: Tiempo máximo de espera tolerado antes de iniciar descarga.
        descarga_setup_min: Tiempo fijo de posicionamiento y apertura en el cliente.
        descarga_min_por_unidad: Minutos requeridos para descargar cada unidad de pedido.
        trafico_factor_normal: Factor de ajuste de velocidad en horario habitual.
        trafico_factor_pico_manana: Multiplicador de congestión en horario pico matutino.
        trafico_pico_manana_inicio_min: Minuto del día de inicio del pico matutino.
        trafico_pico_manana_fin_min: Minuto del día de finalización del pico matutino.
        trafico_factor_pico_tarde: Multiplicador de congestión en horario pico vespertino.
        trafico_pico_tarde_inicio_min: Minuto del día de inicio del pico vespertino.
        trafico_pico_tarde_fin_min: Minuto del día de finalización del pico vespertino.
        costo_tarea_no_entregada: Penalización fija por cada tarea o entrega omitida.
        costo_pedido_original_incompleto: Penalización por dejar un pedido original
            con entregas parciales pendientes.
        costo_por_min_tardanza: Penalización por minuto de retraso respecto a la ventana pactada.
        costo_por_min_exceso_tolerancia: Penalización por minuto excedido sobre la tolerancia del turno.
        costo_por_min_operacion: Costo por minuto activo de trabajo de la flota.
        costo_por_km: Costo variable asociado al kilometraje recorrido.
        costo_por_viaje: Costo fijo incurrido al iniciar cada nuevo viaje desde el depósito.
        costo_por_min_desbalance_fin: Penalización por diferencia de horario de finalización
            entre el primer y el último camión en culminar operaciones.
    """

    id_nodo_corralon: str = "DEPOT"

    factor_urbano_distancia: float = 1.30
    velocidad_base_kmh: float = 25.0

    personas_carga_estimadas: int = 2
    cantidad_empleados_corralon: int = 2
    carga_setup_min: float = 2.0
    carga_min_por_unidad_1_persona: float = 2.5
    carga_eficiencia_persona_adicional: float = 0.70

    cliente_espera_respuesta_min: float = 0.0
    cliente_espera_respuesta_moda: float = 2.0
    cliente_espera_respuesta_max: float = 8.0

    descarga_setup_min: float = 1.0
    descarga_min_por_unidad: float = 1.2

    trafico_factor_normal: float = 1.0

    trafico_factor_pico_manana: float = 1.20
    trafico_pico_manana_inicio_min: int = 450
    trafico_pico_manana_fin_min: int = 540

    trafico_factor_pico_tarde: float = 1.20
    trafico_pico_tarde_inicio_min: int = 960
    trafico_pico_tarde_fin_min: int = 1020

    costo_tarea_no_entregada: float = 10000.0
    costo_pedido_original_incompleto: float = 5000.0
    costo_por_min_tardanza: float = 100.0
    costo_por_min_exceso_tolerancia: float = 500.0
    costo_por_min_operacion: float = 1.0
    costo_por_km: float = 2.0
    costo_por_viaje: float = 5.0
    costo_por_min_desbalance_fin: float = 0.5

    def __post_init__(self) -> None:
        """Valida que los parámetros numéricos y de rango sean operativamente válidos.

        Raises:
            ValueError: Si alguna cota, velocidad o parámetro de tiempo resulta
                inconsistente o infringe las restricciones de dominio.
        """
        if not self.id_nodo_corralon.strip():
            raise ValueError("id_nodo_corralon no puede estar vacío.")

        if self.factor_urbano_distancia <= 0.0:
            raise ValueError("factor_urbano_distancia debe ser > 0.")

        if self.velocidad_base_kmh <= 0.0:
            raise ValueError("velocidad_base_kmh debe ser > 0.")

        if self.personas_carga_estimadas <= 0:
            raise ValueError("personas_carga_estimadas debe ser > 0.")

        if self.cantidad_empleados_corralon < 0:
            raise ValueError("cantidad_empleados_corralon no puede ser negativa.")

        if self.carga_setup_min < 0.0:
            raise ValueError("carga_setup_min no puede ser negativa.")

        if self.carga_min_por_unidad_1_persona < 0.0:
            raise ValueError("carga_min_por_unidad_1_persona no puede ser negativa.")

        if self.carga_eficiencia_persona_adicional < 0.0:
            raise ValueError("carga_eficiencia_persona_adicional no puede ser negativa.")

        if self.cliente_espera_respuesta_min < 0.0:
            raise ValueError("cliente_espera_respuesta_min no puede ser negativa.")

        if self.cliente_espera_respuesta_moda < 0.0:
            raise ValueError("cliente_espera_respuesta_moda no puede ser negativa.")

        if self.cliente_espera_respuesta_max < 0.0:
            raise ValueError("cliente_espera_respuesta_max no puede ser negativa.")

        if not (
            self.cliente_espera_respuesta_min
            <= self.cliente_espera_respuesta_moda
            <= self.cliente_espera_respuesta_max
        ):
            raise ValueError("La espera triangular del cliente debe cumplir min <= moda <= max.")
