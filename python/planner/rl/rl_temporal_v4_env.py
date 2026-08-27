from __future__ import annotations

from typing import Any

import numpy as np

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno
from planner.rl.rl_env import PedemontePlanEnv
from planner.rl.rl_reward import ConfiguracionRewardRL
from planner.rl.rl_temporal_v4_config import ConfiguracionTemporalV4RL
from planner.rl.temporal_estimator import (
    ResumenTemporalPrefijo,
    analizar_prefijo_temporal,
)
from planner.rl.temporal_v4_estimator import (
    ConsecuenciaTemporalAccionV4,
    ResultadoArrepentimientoLocalV4,
    calcular_arrepentimiento_local_v4,
    calcular_reward_terminal_v4,
    proyectar_consecuencias_segundo_orden_v4,
    seleccionar_mejor_consecuencia_v4,
)
from planner.routing.travel import ProveedorViaje


class PedemonteTemporalV4PlanEnv(PedemontePlanEnv):
    """Entorno v4 con consecuencias de segundo orden y reward jerárquico."""

    FEATURES_POR_PEDIDO = 23
    FEATURES_GLOBALES = 12
    VERSION_ENTORNO = "pedemonte-rl-temporal-v4"

    def __init__(
        self,
        instancia: InstanciaTurno,
        configuracion: ConfiguracionPlanificacion | None = None,
        proveedor_viaje: ProveedorViaje | None = None,
        max_pedidos: int = 30,
        escala_reward: float = 100.0,
        configuracion_reward: ConfiguracionRewardRL | None = None,
        configuracion_temporal: ConfiguracionTemporalV4RL | None = None,
        penalizacion_accion_invalida: float = 1.0,
        max_acciones_invalidas: int = 10,
    ) -> None:
        self.configuracion_temporal_v4 = (
            configuracion_temporal
            if configuracion_temporal is not None
            else ConfiguracionTemporalV4RL()
        )
        self._cache_prefijo_v4: tuple[str, ...] | None = None
        self._cache_resumen_v4: ResumenTemporalPrefijo | None = None
        self._cache_consecuencias_v4: (
            dict[str, ConsecuenciaTemporalAccionV4] | None
        ) = None
        self._cache_arrepentimientos_v4: (
            dict[str, ResultadoArrepentimientoLocalV4] | None
        ) = None

        super().__init__(
            instancia=instancia,
            configuracion=configuracion,
            proveedor_viaje=proveedor_viaje,
            max_pedidos=max_pedidos,
            escala_reward=escala_reward,
            configuracion_reward=configuracion_reward,
            penalizacion_accion_invalida=penalizacion_accion_invalida,
            max_acciones_invalidas=max_acciones_invalidas,
        )

    def _invalidar_cache_v4(self) -> None:
        self._cache_prefijo_v4 = None
        self._cache_resumen_v4 = None
        self._cache_consecuencias_v4 = None
        self._cache_arrepentimientos_v4 = None

    def _reiniciar_estado(self) -> None:
        super()._reiniciar_estado()
        self._invalidar_cache_v4()

    def _aplicar_pedido_a_estado_parcial(
        self,
        pedido_id: str,
        unidades: int,
        requiere_volcador: bool,
    ) -> None:
        super()._aplicar_pedido_a_estado_parcial(
            pedido_id=pedido_id,
            unidades=unidades,
            requiere_volcador=requiere_volcador,
        )
        self._invalidar_cache_v4()

    def _obtener_estado_v4(
        self,
    ) -> tuple[
        ResumenTemporalPrefijo,
        dict[str, ConsecuenciaTemporalAccionV4],
        dict[str, ResultadoArrepentimientoLocalV4],
    ]:
        prefijo = tuple(self._permutacion)

        if (
            self._cache_prefijo_v4 == prefijo
            and self._cache_resumen_v4 is not None
            and self._cache_consecuencias_v4 is not None
            and self._cache_arrepentimientos_v4 is not None
        ):
            return (
                self._cache_resumen_v4,
                self._cache_consecuencias_v4,
                self._cache_arrepentimientos_v4,
            )

        resumen = analizar_prefijo_temporal(
            self.instancia,
            self.matriz,
            self.configuracion,
            prefijo,
        )
        consecuencias = proyectar_consecuencias_segundo_orden_v4(
            self.instancia,
            self.matriz,
            self.configuracion,
            prefijo,
        )
        arrepentimientos = {
            pedido_id: calcular_arrepentimiento_local_v4(
                consecuencias,
                pedido_id,
                self.configuracion_temporal_v4,
            )
            for pedido_id in consecuencias
        }

        self._cache_prefijo_v4 = prefijo
        self._cache_resumen_v4 = resumen
        self._cache_consecuencias_v4 = consecuencias
        self._cache_arrepentimientos_v4 = arrepentimientos

        return resumen, consecuencias, arrepentimientos

    @property
    def resumen_temporal_actual(self) -> ResumenTemporalPrefijo:
        return self._obtener_estado_v4()[0]

    @property
    def consecuencias_temporales_actuales(
        self,
    ) -> dict[str, ConsecuenciaTemporalAccionV4]:
        return dict(self._obtener_estado_v4()[1])

    def action_masks(self) -> np.ndarray:
        mascara = super().action_masks()

        if (
            not self.configuracion_temporal_v4.usar_mascara_temporal_dura
            or self._episodio_finalizado
        ):
            return mascara

        _, consecuencias, _ = self._obtener_estado_v4()
        indices_validos = [
            indice
            for indice in range(self.cantidad_pedidos)
            if bool(mascara[indice])
        ]

        if len(indices_validos) <= 1 or not consecuencias:
            return mascara

        minimo_tardios = min(
            consecuencia.pedidos_tardios_finales
            for consecuencia in consecuencias.values()
        )

        for indice in indices_validos:
            pedido_id = self.pedidos[indice].pedido_id
            if (
                consecuencias[pedido_id].pedidos_tardios_finales
                > minimo_tardios
            ):
                mascara[indice] = False

        if not bool(mascara[: self.cantidad_pedidos].any()):
            return super().action_masks()

        return mascara

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        accion = int(action)
        mascara_antes = self.action_masks()
        resultado_local: ResultadoArrepentimientoLocalV4 | None = None

        if (
            0 <= accion < self.cantidad_pedidos
            and bool(mascara_antes[accion])
        ):
            pedido_id = self.pedidos[accion].pedido_id
            _, _, arrepentimientos = self._obtener_estado_v4()
            resultado_local = arrepentimientos[pedido_id]

        (
            observacion,
            reward_base,
            terminado,
            truncado,
            info,
        ) = super().step(accion)
        info = dict(info)

        if not bool(info.get("accion_valida", False)):
            return observacion, reward_base, terminado, truncado, info

        if resultado_local is None:
            raise RuntimeError(
                "No se calculó el arrepentimiento de una acción válida."
            )

        resumen_despues, _, _ = self._obtener_estado_v4()
        reward_terminal = 0.0
        componente_factibilidad = 0.0
        componente_costo = 0.0
        factible_terminal: bool | None = None

        if terminado:
            terminal = calcular_reward_terminal_v4(
                resumen_despues,
                float(reward_base),
                self.configuracion_temporal_v4,
            )
            reward_terminal = terminal.reward_terminal_total
            componente_factibilidad = terminal.componente_factibilidad
            componente_costo = terminal.componente_costo_acotado
            factible_terminal = terminal.factible_temporalmente

        reward_total = resultado_local.reward_local + reward_terminal

        info.update(
            {
                "version_entorno_rl": self.VERSION_ENTORNO,
                "reward_costo_base_no_usado_directamente": float(
                    reward_base
                ),
                "reward_arrepentimiento_local": (
                    resultado_local.reward_local
                ),
                "arrepentimiento_local_normalizado": (
                    resultado_local.arrepentimiento_normalizado
                ),
                "mejor_accion_local_id": resultado_local.mejor_pedido_id,
                "accion_elegida_es_mejor_local": (
                    resultado_local.es_mejor_accion
                ),
                "reward_terminal_v4": reward_terminal,
                "componente_terminal_factibilidad": (
                    componente_factibilidad
                ),
                "componente_terminal_costo_acotado": componente_costo,
                "factible_temporal_terminal": factible_terminal,
                "reward_total": reward_total,
                "pedidos_tardios_prefijo": resumen_despues.pedidos_tardios,
                "tardanza_prefijo_min": (
                    resumen_despues.tardanza_total_min
                ),
            }
        )

        return observacion, reward_total, terminado, truncado, info

    def _construir_observacion(self) -> np.ndarray:
        observacion = super()._construir_observacion()
        resumen, consecuencias, arrepentimientos = self._obtener_estado_v4()

        horizonte = max(
            1.0,
            self.instancia.hora_fin_tolerancia_min
            - self.instancia.hora_inicio_turno_min,
        )
        cantidad_total = max(1, self.cantidad_pedidos)
        seleccionados = {
            registro.pedido_id: registro
            for registro in resumen.registros
        }

        for indice in range(self.cantidad_pedidos):
            pedido = self.pedidos[indice]
            base = indice * self.FEATURES_POR_PEDIDO
            consecuencia = consecuencias.get(pedido.pedido_id)
            registro = seleccionados.get(pedido.pedido_id)

            if consecuencia is not None:
                registro = consecuencia.registro_inmediato

            if registro is None:
                continue

            observacion[base + 9] = float(
                (registro.minuto_llegada - self.instancia.hora_inicio_turno_min)
                / horizonte
            )
            observacion[base + 10] = float(
                registro.espera_apertura_min / horizonte
            )
            observacion[base + 11] = float(
                max(0.0, registro.holgura_llegada_min) / horizonte
            )
            observacion[base + 12] = float(
                registro.tardanza_llegada_min / horizonte
            )
            observacion[base + 13] = float(
                pedido.tiene_ventana_especifica
            )

            if consecuencia is None:
                continue

            arrepentimiento = arrepentimientos[pedido.pedido_id]
            observacion[base + 14] = float(
                consecuencia.pedidos_tardios_finales / cantidad_total
            )
            observacion[base + 15] = float(
                consecuencia.tardanza_total_final_min / horizonte
            )
            observacion[base + 16] = float(
                consecuencia.pedidos_nuevos_en_riesgo / cantidad_total
            )
            observacion[base + 17] = float(
                consecuencia.perdida_holgura_total_min
                / self.configuracion_temporal_v4.escala_perdida_holgura_min
            )
            observacion[base + 18] = float(
                max(0.0, consecuencia.holgura_minima_final_min) / horizonte
            )
            observacion[base + 19] = float(
                consecuencia.espera_apertura_total_final_min
                / self.configuracion_temporal_v4.escala_espera_min
            )
            observacion[base + 20] = float(
                consecuencia.duracion_operacion_final_min
                / self.configuracion_temporal_v4.escala_duracion_min
            )
            observacion[base + 21] = float(
                arrepentimiento.arrepentimiento_normalizado
            )
            observacion[base + 22] = float(
                arrepentimiento.es_mejor_accion
            )

        base_global = self.max_pedidos * self.FEATURES_POR_PEDIDO
        pendientes = max(1, self.cantidad_pedidos - len(self._permutacion))

        observacion[base_global + 6] = float(
            (resumen.minuto_referencia - self.instancia.hora_inicio_turno_min)
            / horizonte
        )
        observacion[base_global + 7] = float(
            resumen.pedidos_tardios / cantidad_total
        )
        observacion[base_global + 8] = float(
            resumen.tardanza_total_min / horizonte
        )

        if consecuencias:
            mejor = seleccionar_mejor_consecuencia_v4(consecuencias)
            sin_riesgo = sum(
                1
                for consecuencia in consecuencias.values()
                if consecuencia.sin_riesgo_final
            )
            observacion[base_global + 9] = float(
                mejor.pedidos_tardios_finales / cantidad_total
            )
            observacion[base_global + 10] = float(
                mejor.tardanza_total_final_min / horizonte
            )
            observacion[base_global + 11] = float(
                sin_riesgo / pendientes
            )

        np.clip(observacion, 0.0, 1.0, out=observacion)
        return observacion
