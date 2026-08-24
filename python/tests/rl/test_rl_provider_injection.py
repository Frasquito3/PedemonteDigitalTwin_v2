from __future__ import annotations

import unittest
from dataclasses import dataclass

from planner.core.config import ConfiguracionPlanificacion
from planner.rl.rl_env import PedemontePlanEnv
from planner.rl.rl_reward import (
    ConfiguracionRewardRL,
    ModoRewardRL,
)
from planner.routing.travel import (
    Coordenada,
    FuenteMatrizViaje,
    ResultadoTramoViaje,
)
from tests.fixtures import crear_instancia_demo


@dataclass
class ProveedorRLPrueba:
    llamadas: int = 0

    @property
    def fuente(self) -> FuenteMatrizViaje:
        return FuenteMatrizViaje.VIAL_CACHE

    @property
    def version(self) -> str:
        return "proveedor-rl-prueba-v1"

    def calcular_tramo(
        self,
        origen: Coordenada,
        destino: Coordenada,
        configuracion: ConfiguracionPlanificacion,
    ) -> ResultadoTramoViaje:
        _ = origen
        _ = destino
        _ = configuracion

        self.llamadas += 1

        return ResultadoTramoViaje(
            distancia_metros=2_000.0,
            tiempo_base_min=5.0,
            fuente=self.fuente,
        )


class InyeccionProveedorRLTest(unittest.TestCase):
    def test_entorno_rl_usa_proveedor_inyectado(self) -> None:
        instancia = crear_instancia_demo()
        proveedor = ProveedorRLPrueba()

        env = PedemontePlanEnv(
            instancia=instancia,
            proveedor_viaje=proveedor,
        )

        self.assertEqual(
            env.matriz.fuente,
            FuenteMatrizViaje.VIAL_CACHE,
        )
        self.assertEqual(
            env.matriz.version_fuente,
            "proveedor-rl-prueba-v1",
        )
        self.assertGreater(proveedor.llamadas, 0)

        env.close()

    def test_reward_relativo_comparte_proveedor_con_greedy(self) -> None:
        instancia = crear_instancia_demo()
        proveedor = ProveedorRLPrueba()

        env = PedemontePlanEnv(
            instancia=instancia,
            proveedor_viaje=proveedor,
            configuracion_reward=ConfiguracionRewardRL(
                modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
            ),
        )

        cantidad_nodos = len(instancia.pedidos) + 1
        llamadas_por_matriz = cantidad_nodos * (cantidad_nodos - 1)

        self.assertEqual(
            proveedor.llamadas,
            llamadas_por_matriz * 2,
        )
        self.assertIsNotNone(env.costo_greedy_referencia)

        env.close()


if __name__ == "__main__":
    unittest.main()
