import unittest

from planner.rl.instance_generator import (
    GeneradorInstanciasRL,
)

from planner.domain.validator import (
    validar_instancia,
)


def firma_instancia(
    instancia,
) -> tuple:
    return (
        instancia.turno,

        tuple(
            (
                pedido.pedido_id,
                pedido.pedido_original_id,
                pedido.unidades_capacidad,
                pedido.requiere_volcador,
                pedido.hora_desde_min,
                pedido.hora_hasta_min,
                pedido.latitud,
                pedido.longitud,
            )

            for pedido in instancia.pedidos
        ),
    )


class GeneradorInstanciasRLTest(
    unittest.TestCase
):
    def test_misma_seed_reproduce_instancia(
        self,
    ) -> None:
        generador = (
            GeneradorInstanciasRL()
        )

        instancia_a = generador.generar(
            91_001
        )

        instancia_b = generador.generar(
            91_001
        )

        self.assertEqual(
            firma_instancia(instancia_a),
            firma_instancia(instancia_b),
        )

    def test_distintas_seeds_generan_diversidad(
        self,
    ) -> None:
        generador = (
            GeneradorInstanciasRL()
        )

        firmas = {
            firma_instancia(
                generador.generar(seed)
            )

            for seed in range(
                91_000,
                91_010,
            )
        }

        self.assertGreater(
            len(firmas),
            1,
        )

    def test_instancias_generadas_son_validas(
        self,
    ) -> None:
        generador = (
            GeneradorInstanciasRL()
        )

        for seed in range(
            91_000,
            91_100,
        ):
            instancia = generador.generar(
                seed
            )

            errores = validar_instancia(
                instancia
            )

            self.assertFalse(
                errores,

                msg=(
                    f"Seed={seed}: "
                    + " | ".join(errores)
                ),
            )

            self.assertGreaterEqual(
                len(instancia.pedidos),
                4,
            )

            self.assertLessEqual(
                len(instancia.pedidos),
                8,
            )

            for pedido in instancia.pedidos:
                self.assertGreater(
                    pedido.unidades_capacidad,
                    0,
                )

                self.assertLessEqual(
                    pedido.unidades_capacidad,
                    instancia.capacidad_camion,
                )

                self.assertGreaterEqual(
                    pedido.hora_desde_min,
                    instancia.hora_inicio_turno_min,
                )

                self.assertLessEqual(
                    pedido.hora_hasta_min,
                    instancia.hora_fin_objetivo_min,
                )

                self.assertLess(
                    pedido.hora_desde_min,
                    pedido.hora_hasta_min,
                )


if __name__ == "__main__":
    unittest.main()