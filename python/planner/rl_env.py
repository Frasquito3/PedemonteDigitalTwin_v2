from __future__ import annotations

from typing import Any

# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np

# pyrefly: ignore [missing-import]
from gymnasium import spaces

from .config import ConfiguracionPlanificacion

from .decoder import (
    Cromosoma,
    decodificar_plan_permutacion,
)

from .objective import (
    EstimacionPlan,
    evaluar_plan_estimado,
)

from .schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PlanTurno,
)

from .travel import (
    MatrizViaje,
    construir_matriz_viaje,
)

from .validator import validar_instancia


class PedemontePlanEnv(
    gym.Env[np.ndarray, int]
):
    metadata = {
        "render_modes": [],
    }

    FEATURES_POR_PEDIDO = 9

    FEATURES_GLOBALES = 6

    def __init__(
        self,
        instancia: InstanciaTurno,
        configuracion:
            ConfiguracionPlanificacion
            | None = None,
        max_pedidos: int = 30,
        escala_reward: float = 100.0,
        penalizacion_accion_invalida:
            float = 1.0,
        max_acciones_invalidas: int = 10,
    ) -> None:
        super().__init__()

        self.configuracion = (
            configuracion
            if configuracion is not None
            else ConfiguracionPlanificacion()
        )

        self.instancia = instancia

        errores = validar_instancia(
            instancia
        )

        if errores:
            raise ValueError(
                "Instancia inválida para RL: "
                + " | ".join(errores)
            )

        if not instancia.pedidos:
            raise ValueError(
                "El entorno RL requiere al menos "
                "un pedido."
            )

        if max_pedidos <= 0:
            raise ValueError(
                "max_pedidos debe ser > 0."
            )

        if (
            len(instancia.pedidos)
            > max_pedidos
        ):
            raise ValueError(
                "La instancia contiene "
                f"{len(instancia.pedidos)} pedidos, "
                "pero max_pedidos="
                f"{max_pedidos}."
            )

        if escala_reward <= 0.0:
            raise ValueError(
                "escala_reward debe ser > 0."
            )

        if max_acciones_invalidas <= 0:
            raise ValueError(
                "max_acciones_invalidas debe ser > 0."
            )

        self.max_pedidos = max_pedidos

        self.escala_reward = escala_reward

        self.penalizacion_accion_invalida = (
            penalizacion_accion_invalida
        )

        self.max_acciones_invalidas = (
            max_acciones_invalidas
        )

        self.pedidos = list(
            instancia.pedidos
        )

        self.cantidad_pedidos = len(
            self.pedidos
        )

        self.indice_por_pedido_id = {
            pedido.pedido_id: indice

            for indice, pedido
            in enumerate(self.pedidos)
        }

        self.matriz: MatrizViaje = (
            construir_matriz_viaje(
                instancia,
                self.configuracion,
            )
        )

        self.max_distancia_metros = max(
            (
                distancia

                for fila
                in self.matriz.distancia_metros

                for distancia in fila
            ),

            default=1.0,
        )

        if self.max_distancia_metros <= 0.0:
            self.max_distancia_metros = 1.0

        self.action_space = spaces.Discrete(
            self.max_pedidos
        )

        self.dimension_observacion = (
            self.max_pedidos
            * self.FEATURES_POR_PEDIDO
            + self.FEATURES_GLOBALES
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,

            shape=(
                self.dimension_observacion,
            ),

            dtype=np.float32,
        )

        self.ultimo_plan: PlanTurno | None = None

        self.ultima_estimacion: EstimacionPlan | None = None

        self._reiniciar_estado()

    def _reiniciar_estado(self) -> None:
        self._seleccionados = np.zeros(
            self.cantidad_pedidos,
            dtype=np.bool_,
        )

        self._orden_seleccion = np.full(
            self.cantidad_pedidos,
            -1,
            dtype=np.int32,
        )

        self._permutacion: list[str] = []

        self._viaje_actual: list[str] = []

        self._carga_viaje_actual = 0

        self._viajes_cerrados: list[list[str]] = []

        self._nodo_actual_id = (
            self.configuracion
            .id_nodo_corralon
        )

        self._acciones_invalidas = 0

        self._episodio_finalizado = False

        self.ultimo_plan = None

        self.ultima_estimacion = None

    @property
    def cantidad_seleccionados(
        self,
    ) -> int:
        return len(
            self._permutacion
        )

    @property
    def permutacion_actual(
        self,
    ) -> Cromosoma:
        return tuple(
            self._permutacion
        )

    def accion_de_pedido_id(
        self,
        pedido_id: str,
    ) -> int:
        try:
            return self.indice_por_pedido_id[
                pedido_id
            ]

        except KeyError as exc:
            raise ValueError(
                "Pedido inexistente en el "
                f"entorno RL: {pedido_id}"
            ) from exc

    def pedido_id_de_accion(
        self,
        accion: int,
    ) -> str:
        if not (
            0
            <= accion
            < self.cantidad_pedidos
        ):
            raise ValueError(
                f"Acción sin pedido real: {accion}"
            )

        return self.pedidos[
            accion
        ].pedido_id

    def action_masks(
        self,
    ) -> np.ndarray:
        mascara = np.zeros(
            self.max_pedidos,
            dtype=np.bool_,
        )

        if self._episodio_finalizado:
            return mascara

        mascara[
            :self.cantidad_pedidos
        ] = ~self._seleccionados

        return mascara

    def reset(
        self,
        *,
        seed: int | None = None,
        options:
            dict[str, Any] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        super().reset(
            seed=seed
        )

        _ = options

        self._reiniciar_estado()

        observacion = (
            self._construir_observacion()
        )

        info = {
            "cantidad_pedidos":
                self.cantidad_pedidos,

            "acciones_validas":
                int(
                    self.action_masks().sum()
                ),
        }

        return observacion, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self._episodio_finalizado:
            raise RuntimeError(
                "El episodio ya terminó. "
                "Debe llamarse reset()."
            )

        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"Acción fuera del espacio: {action}"
            )

        accion = int(
            action
        )

        mascara = self.action_masks()

        if not bool(
            mascara[accion]
        ):
            self._acciones_invalidas += 1

            truncado = (
                self._acciones_invalidas
                >= self.max_acciones_invalidas
            )

            if truncado:
                self._episodio_finalizado = True

            observacion = (
                self._construir_observacion()
            )

            info = {
                "accion_valida": False,

                "motivo": (
                    "PEDIDO_YA_SELECCIONADO"
                    if accion
                    < self.cantidad_pedidos
                    else "ACCION_PADDING"
                ),

                "acciones_invalidas":
                    self._acciones_invalidas,
            }

            return (
                observacion,

                -float(
                    self.penalizacion_accion_invalida
                ),

                False,

                truncado,

                info,
            )

        pedido = self.pedidos[
            accion
        ]

        posicion = len(
            self._permutacion
        )

        self._seleccionados[
            accion
        ] = True

        self._orden_seleccion[
            accion
        ] = posicion

        self._permutacion.append(
            pedido.pedido_id
        )

        self._aplicar_pedido_a_estado_parcial(
            pedido_id=pedido.pedido_id,
            unidades=pedido.unidades_capacidad,
            requiere_volcador=(
                pedido.requiere_volcador
            ),
        )

        terminado = (
            len(self._permutacion)
            == self.cantidad_pedidos
        )

        recompensa = 0.0

        info: dict[str, Any] = {
            "accion_valida": True,

            "pedido_id":
                pedido.pedido_id,

            "seleccionados":
                len(self._permutacion),
        }

        if terminado:
            recompensa, info_terminal = (
                self._finalizar_episodio()
            )

            info.update(
                info_terminal
            )

        observacion = (
            self._construir_observacion()
        )

        return (
            observacion,

            float(recompensa),

            terminado,

            False,

            info,
        )

    def _aplicar_pedido_a_estado_parcial(
        self,
        pedido_id: str,
        unidades: int,
        requiere_volcador: bool,
    ) -> None:
        supera_capacidad = (
            self._carga_viaje_actual
            + unidades
            > self.instancia
            .capacidad_camion
        )

        if supera_capacidad:
            self._cerrar_viaje_parcial()

        self._viaje_actual.append(
            pedido_id
        )

        self._carga_viaje_actual += (
            unidades
        )

        self._nodo_actual_id = (
            pedido_id
        )

        if requiere_volcador:
            self._cerrar_viaje_parcial()

    def _cerrar_viaje_parcial(
        self,
    ) -> None:
        if self._viaje_actual:
            self._viajes_cerrados.append(
                list(
                    self._viaje_actual
                )
            )

        self._viaje_actual = []

        self._carga_viaje_actual = 0

        self._nodo_actual_id = (
            self.configuracion
            .id_nodo_corralon
        )

    def _finalizar_episodio(
        self,
    ) -> tuple[
        float,
        dict[str, Any],
    ]:
        self._cerrar_viaje_parcial()

        cromosoma = tuple(
            self._permutacion
        )

        plan = decodificar_plan_permutacion(
            instancia=self.instancia,

            matriz=self.matriz,

            configuracion=self.configuracion,

            cromosoma=cromosoma,

            algoritmo=(
                AlgoritmoPlanificacion.RL
            ),
        )

        estimacion = evaluar_plan_estimado(
            self.instancia,
            plan,
            self.matriz,
            self.configuracion,
        )

        plan.costo_estimado = (
            estimacion.costo_total
        )

        self.ultimo_plan = plan

        self.ultima_estimacion = (
            estimacion
        )

        self._episodio_finalizado = True

        recompensa = (
            -estimacion.costo_total
            / self.escala_reward
        )

        info = {
            "plan_valido": True,

            "costo_estimado":
                estimacion.costo_total,

            "cantidad_viajes":
                estimacion.viajes_totales,

            "reward_terminal":
                recompensa,

            "permutacion":
                cromosoma,
        }

        return recompensa, info

    def _construir_observacion(
        self,
    ) -> np.ndarray:
        observacion = np.zeros(
            self.dimension_observacion,
            dtype=np.float32,
        )

        horizonte = max(
            1.0,

            self.instancia
            .hora_fin_tolerancia_min
            -
            self.instancia
            .hora_inicio_turno_min,
        )

        for indice in range(
            self.cantidad_pedidos
        ):
            pedido = self.pedidos[
                indice
            ]

            base = (
                indice
                * self.FEATURES_POR_PEDIDO
            )

            pendiente = not bool(
                self._seleccionados[
                    indice
                ]
            )

            posicion = (
                self._orden_seleccion[
                    indice
                ]
            )

            orden_normalizado = (
                0.0
                if posicion < 0
                else (
                    posicion + 1
                )
                / self.cantidad_pedidos
            )

            desde_normalizado = (
                pedido.hora_desde_min
                -
                self.instancia
                .hora_inicio_turno_min
            ) / horizonte

            hasta_normalizado = (
                pedido.hora_hasta_min
                -
                self.instancia
                .hora_inicio_turno_min
            ) / horizonte

            distancia_actual = (
                self.matriz.distancia(
                    self._nodo_actual_id,
                    pedido.pedido_id,
                )
            )

            distancia_depot = (
                self.matriz.distancia(
                    self.configuracion
                    .id_nodo_corralon,

                    pedido.pedido_id,
                )
            )

            observacion[
                base
            ] = 1.0

            observacion[
                base + 1
            ] = float(
                pendiente
            )

            observacion[
                base + 2
            ] = float(
                orden_normalizado
            )

            observacion[
                base + 3
            ] = (
                pedido.unidades_capacidad
                /
                self.instancia
                .capacidad_camion
            )

            observacion[
                base + 4
            ] = float(
                pedido.requiere_volcador
            )

            observacion[
                base + 5
            ] = float(
                desde_normalizado
            )

            observacion[
                base + 6
            ] = float(
                hasta_normalizado
            )

            observacion[
                base + 7
            ] = (
                distancia_actual
                / self.max_distancia_metros
            )

            observacion[
                base + 8
            ] = (
                distancia_depot
                / self.max_distancia_metros
            )

        base_global = (
            self.max_pedidos
            * self.FEATURES_POR_PEDIDO
        )

        observacion[
            base_global
        ] = (
            len(self._permutacion)
            / self.cantidad_pedidos
        )

        observacion[
            base_global + 1
        ] = (
            self._carga_viaje_actual
            /
            self.instancia.capacidad_camion
        )

        observacion[
            base_global + 2
        ] = (
            len(self._viaje_actual)
            / self.cantidad_pedidos
        )

        observacion[
            base_global + 3
        ] = min(
            1.0,

            len(self._viajes_cerrados)
            / self.cantidad_pedidos,
        )

        observacion[
            base_global + 4
        ] = float(
            self._nodo_actual_id
            ==
            self.configuracion
            .id_nodo_corralon
        )

        observacion[
            base_global + 5
        ] = (
            self._acciones_invalidas
            / self.max_acciones_invalidas
        )

        np.clip(
            observacion,
            0.0,
            1.0,
            out=observacion,
        )

        return observacion

    def render(self) -> None:
        print(
            "Permutación actual: "
            f"{self._permutacion}"
        )

        print(
            "Viajes cerrados: "
            f"{self._viajes_cerrados}"
        )

        print(
            "Viaje abierto: "
            f"{self._viaje_actual}"
        )