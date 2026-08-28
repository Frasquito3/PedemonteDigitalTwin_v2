from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionPoliticaRL:
    """
    Parámetros de la política RL productiva.

    La señal densa usa arrepentimiento local de segundo orden: cada acción
    candidata se completa con una heurística temporal común y se compara con
    la mejor alternativa disponible en el mismo estado.

    La recompensa terminal separa primero la factibilidad temporal y luego
    utiliza un término de costo acotado. La máscara temporal dura forma parte
    de la configuración validada del checkpoint productivo actual.
    """

    # Escalas de normalización de las consecuencias temporales.
    escala_tardanza_min: float = 60.0
    escala_perdida_holgura_min: float = 120.0
    escala_espera_min: float = 120.0
    escala_duracion_min: float = 300.0

    # Pesos del arrepentimiento local normalizado. Deben sumar 1.
    peso_arrepentimiento_pedidos_tardios: float = 0.55
    peso_arrepentimiento_tardanza: float = 0.25
    peso_arrepentimiento_nuevos_riesgos: float = 0.10
    peso_arrepentimiento_holgura: float = 0.07
    peso_arrepentimiento_espera: float = 0.03

    penalizacion_arrepentimiento_max: float = 0.20
    bonificacion_mejor_accion_local: float = 0.02

    # Bandas terminales. Se mantienen deliberadamente separadas.
    bonificacion_terminal_factible: float = 20.0
    penalizacion_terminal_no_factible: float = 20.0
    penalizacion_terminal_por_pedido_tardio: float = 15.0
    penalizacion_terminal_tardanza_max: float = 2.0
    peso_terminal_costo_acotado: float = 0.50

    # La máscara dura elimina acciones con más pedidos tardíos proyectados
    # que la mejor alternativa y nunca deja el espacio de acciones vacío.
    usar_mascara_temporal_dura: bool = True

    # Tolerancias numéricas.
    epsilon_tiempo: float = 1e-9
    epsilon_mejor_accion: float = 1e-9

    def __post_init__(self) -> None:
        positivos = {
            "escala_tardanza_min": self.escala_tardanza_min,
            "escala_perdida_holgura_min": (
                self.escala_perdida_holgura_min
            ),
            "escala_espera_min": self.escala_espera_min,
            "escala_duracion_min": self.escala_duracion_min,
            "bonificacion_terminal_factible": (
                self.bonificacion_terminal_factible
            ),
            "penalizacion_terminal_no_factible": (
                self.penalizacion_terminal_no_factible
            ),
            "epsilon_tiempo": self.epsilon_tiempo,
            "epsilon_mejor_accion": self.epsilon_mejor_accion,
        }

        for nombre, valor in positivos.items():
            if valor <= 0.0:
                raise ValueError(f"{nombre} debe ser > 0.")

        no_negativos = {
            "peso_arrepentimiento_pedidos_tardios": (
                self.peso_arrepentimiento_pedidos_tardios
            ),
            "peso_arrepentimiento_tardanza": (
                self.peso_arrepentimiento_tardanza
            ),
            "peso_arrepentimiento_nuevos_riesgos": (
                self.peso_arrepentimiento_nuevos_riesgos
            ),
            "peso_arrepentimiento_holgura": (
                self.peso_arrepentimiento_holgura
            ),
            "peso_arrepentimiento_espera": (
                self.peso_arrepentimiento_espera
            ),
            "penalizacion_arrepentimiento_max": (
                self.penalizacion_arrepentimiento_max
            ),
            "bonificacion_mejor_accion_local": (
                self.bonificacion_mejor_accion_local
            ),
            "penalizacion_terminal_por_pedido_tardio": (
                self.penalizacion_terminal_por_pedido_tardio
            ),
            "penalizacion_terminal_tardanza_max": (
                self.penalizacion_terminal_tardanza_max
            ),
            "peso_terminal_costo_acotado": (
                self.peso_terminal_costo_acotado
            ),
        }

        for nombre, valor in no_negativos.items():
            if valor < 0.0:
                raise ValueError(f"{nombre} no puede ser negativo.")

        suma_pesos = (
            self.peso_arrepentimiento_pedidos_tardios
            + self.peso_arrepentimiento_tardanza
            + self.peso_arrepentimiento_nuevos_riesgos
            + self.peso_arrepentimiento_holgura
            + self.peso_arrepentimiento_espera
        )

        if abs(suma_pesos - 1.0) > 1e-9:
            raise ValueError(
                "Los pesos del arrepentimiento local deben sumar 1."
            )
