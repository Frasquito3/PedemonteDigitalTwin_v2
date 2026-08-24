from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from planner.integration import (
    pypeline_selector_bridge as bridge,
)
from planner.routing.travel import (
    FuenteMatrizViaje,
    ProveedorHaversineAjustado,
)
from planner.routing.vial_cache import (
    ProveedorVialCachePersistente,
)


ENCABEZADO = (
    "version_cache,lat_origen,lon_origen,"
    "lat_destino,lon_destino,distancia_metros,"
    "tiempo_base_min,fuente_distancia,fuente_tiempo\n"
)


class SelectorFalso:
    construcciones = 0

    def __init__(
        self,
        *,
        model_path_rl,
        max_pedidos,
        deterministic,
        proveedor_viaje,
    ) -> None:
        type(self).construcciones += 1
        self.model_path_rl = model_path_rl
        self.max_pedidos = max_pedidos
        self.deterministic = deterministic
        self.proveedor_viaje = proveedor_viaje
        self.ultima_decision = None
        self.precargado = False

    def precargar_rl(self) -> None:
        self.precargado = True


class PypelineSelectorBridgeVialTest(
    unittest.TestCase
):
    def setUp(self) -> None:
        bridge.reiniciar()
        SelectorFalso.construcciones = 0
        self.temporal = tempfile.TemporaryDirectory()
        self.raiz = Path(self.temporal.name)
        self.modelo = self.raiz / "modelo_rl.zip"
        self.modelo.write_bytes(b"modelo-prueba")
        self.cache = self.raiz / "cache_vial_v1.csv"
        self._escribir_cache(dos_tramos=True)

    def tearDown(self) -> None:
        bridge.reiniciar()
        self.temporal.cleanup()

    def _escribir_cache(
        self,
        *,
        dos_tramos: bool,
    ) -> None:
        filas = [
            (
                "pedemonte-vial-v1,-32.849501,-60.722653,"
                "-32.831000,-60.719000,3203.462000,,"
                "ANYLOGIC_ROUTE_PROVIDER,VELOCIDAD_BASE_CONFIG\n"
            )
        ]

        if dos_tramos:
            filas.append(
                "pedemonte-vial-v1,-32.831000,-60.719000,"
                "-32.849501,-60.722653,4617.467000,,"
                "ANYLOGIC_ROUTE_PROVIDER,VELOCIDAD_BASE_CONFIG\n"
            )

        self.cache.write_text(
            ENCABEZADO + "".join(filas),
            encoding="utf-8",
        )

    @patch.object(
        bridge,
        "SelectorPlanificadores",
        SelectorFalso,
    )
    def test_inicializa_cache_vial_estricta(
        self,
    ) -> None:
        estado = bridge.inicializar(
            str(self.modelo),
            cache_vial_path=str(self.cache),
            permitir_fallback_vial=False,
        )

        self.assertIn("OK|CARGADO", estado)
        self.assertIn("proveedor=VIAL_CACHE", estado)
        self.assertIn("cache_tramos=2", estado)
        self.assertIn("fallback=ESTRICTO", estado)

        selector = bridge._selector
        self.assertIsNotNone(selector)
        assert selector is not None

        selector_falso = cast(
            SelectorFalso,
            selector,
        )

        self.assertIsInstance(
            selector_falso.proveedor_viaje,
            ProveedorVialCachePersistente,
        )
        self.assertTrue(
            selector_falso.precargado
        )

    @patch.object(
        bridge,
        "SelectorPlanificadores",
        SelectorFalso,
    )
    def test_reutiliza_si_modelo_y_cache_no_cambian(
        self,
    ) -> None:
        bridge.inicializar(
            str(self.modelo),
            cache_vial_path=str(self.cache),
        )

        estado = bridge.inicializar(
            str(self.modelo),
            cache_vial_path=str(self.cache),
        )

        self.assertIn("OK|REUTILIZADO", estado)
        self.assertEqual(
            SelectorFalso.construcciones,
            1,
        )

    @patch.object(
        bridge,
        "SelectorPlanificadores",
        SelectorFalso,
    )
    def test_recarga_si_cambia_huella_cache(
        self,
    ) -> None:
        bridge.inicializar(
            str(self.modelo),
            cache_vial_path=str(self.cache),
        )

        self._escribir_cache(
            dos_tramos=False,
        )

        estado = bridge.inicializar(
            str(self.modelo),
            cache_vial_path=str(self.cache),
        )

        self.assertIn("OK|CARGADO", estado)
        self.assertIn("cache_tramos=1", estado)
        self.assertEqual(
            SelectorFalso.construcciones,
            2,
        )

    @patch.object(
        bridge,
        "SelectorPlanificadores",
        SelectorFalso,
    )
    def test_sin_cache_conserva_haversine(
        self,
    ) -> None:
        estado = bridge.inicializar(
            str(self.modelo),
        )

        self.assertIn(
            "proveedor=HAVERSINE_AJUSTADA",
            estado,
        )

        selector = bridge._selector
        self.assertIsNotNone(selector)
        assert selector is not None

        self.assertIsInstance(
            selector.proveedor_viaje,
            ProveedorHaversineAjustado,
        )
        self.assertEqual(
            selector.proveedor_viaje.fuente,
            FuenteMatrizViaje.HAVERSINE_AJUSTADA,
        )

    @patch.object(
        bridge,
        "SelectorPlanificadores",
        SelectorFalso,
    )
    def test_rechaza_cache_inexistente(
        self,
    ) -> None:
        with self.assertRaises(
            FileNotFoundError
        ):
            bridge.inicializar(
                str(self.modelo),
                cache_vial_path=str(
                    self.raiz / "no_existe.csv"
                ),
            )


if __name__ == "__main__":
    unittest.main()
