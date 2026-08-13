from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionPlanificacion:
    id_nodo_corralon: str = "DEPOT"

    factor_urbano_distancia: float = 1.30
    velocidad_base_kmh: float = 25.0

    personas_carga_estimadas: int = 2
    carga_setup_min: float = 2.0
    carga_min_por_unidad_1_persona: float = 2.5
    carga_eficiencia_persona_adicional: float = 0.70

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
        if not self.id_nodo_corralon.strip():
            raise ValueError(
                "id_nodo_corralon no puede estar vacío."
            )

        if self.factor_urbano_distancia <= 0.0:
            raise ValueError(
                "factor_urbano_distancia debe ser > 0."
            )

        if self.velocidad_base_kmh <= 0.0:
            raise ValueError(
                "velocidad_base_kmh debe ser > 0."
            )

        if self.personas_carga_estimadas <= 0:
            raise ValueError(
                "personas_carga_estimadas debe ser > 0."
            )

        if self.carga_eficiencia_persona_adicional < 0.0:
            raise ValueError(
                "carga_eficiencia_persona_adicional "
                "no puede ser negativa."
            )