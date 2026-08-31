from __future__ import annotations

from dataclasses import replace

import pytest

from planner.integration.simulated_comparison import (
    ComparacionSimulada,
    ESTADO_ERROR,
    ESTADO_FINALIZADO,
    ESTADO_OMITIDO,
    ResultadoMetodoSimulado,
)
from planner.integration.simulated_execution import (
    ResultadoEjecucionSimulada,
)
from planner.integration import simulated_comparison_bridge as bridge


FIRMA = "a" * 64


def _resultado(
    algoritmo: str = "RL",
    costo: float = 100.0,
) -> ResultadoEjecucionSimulada:
    return ResultadoEjecucionSimulada(
        version=1,
        instancia_id="UI-1",
        fecha_operacion="2026-08-30",
        algoritmo_aplicado=algoritmo,
        seed_escenario=6001,
        seed_ejecucion=1006001,
        costo_total=costo,
        tareas_entregadas=5,
        tareas_no_entregadas=0,
        viajes_totales=3,
        duracion_simulada_min=150.0,
        distancia_total_km=32.0,
        tardanza_total_min=0.0,
        diferencia_fin_camiones_min=9.0,
        ocupacion_global_pct=75.0,
        costo_tareas_no_entregadas=0.0,
        costo_pedidos_originales_incompletos=0.0,
        costo_tardanza=0.0,
        costo_exceso_tolerancia=0.0,
        costo_operacion=150.0,
        costo_distancia=64.0,
        costo_viajes=15.0,
        costo_desbalance=costo - 229.0,
        mensaje="EJECUCIÓN FINALIZADA",
        estado_motor="FINISHED",
    )


def _comparacion() -> ComparacionSimulada:
    metodos = ("RL", "HIBRIDO", "GREEDY", "RANDOM", "GA")
    algoritmos = ("RL", "GA", "GREEDY", "RANDOM", "GA")
    filas = []

    for indice, (metodo, algoritmo) in enumerate(
        zip(metodos, algoritmos)
    ):
        filas.append(
            ResultadoMetodoSimulado(
                metodo_solicitado=metodo,
                estado=ESTADO_FINALIZADO,
                resultado=_resultado(
                    algoritmo=algoritmo,
                    costo=230.0 + indice,
                ),
                error="",
                tiempo_motor_segundos=2.0 + indice,
            )
        )

    return ComparacionSimulada(
        version=1,
        instancia_id="UI-1",
        fecha_operacion="2026-08-30",
        firma_instancia=FIRMA,
        seed_escenario=6001,
        seed_ejecucion=1006001,
        proveedores_habilitados=True,
        resultados=tuple(filas),
        tiempo_total_segundos=20.0,
    )


def test_codifica_cabecera_y_longitud_estable() -> None:
    vector = bridge.codificar_comparacion_simulada(
        _comparacion()
    )

    assert vector[:4] == [1.0, 5.0, 21.0, 20.0]
    assert len(vector) == 4 + 5 * 21


def test_codifica_campos_de_un_metodo_finalizado() -> None:
    vector = bridge.codificar_comparacion_simulada(
        _comparacion()
    )

    bloque_rl = vector[4 : 4 + 21]

    assert bloque_rl == [
        0.0,  # método RL
        1.0,  # FINALIZADO
        0.0,  # algoritmo RL
        230.0,
        32.0,
        150.0,
        3.0,
        0.0,
        9.0,
        5.0,
        0.0,
        75.0,
        0.0,
        0.0,
        0.0,
        0.0,
        150.0,
        64.0,
        15.0,
        1.0,
        2.0,
    ]


def test_codifica_error_con_metricas_no_disponibles() -> None:
    comparacion = _comparacion()
    filas = list(comparacion.resultados)
    filas[2] = ResultadoMetodoSimulado(
        metodo_solicitado="GREEDY",
        estado=ESTADO_ERROR,
        resultado=None,
        error="falló",
        tiempo_motor_segundos=3.5,
    )
    comparacion = replace(
        comparacion,
        resultados=tuple(filas),
    )

    vector = bridge.codificar_comparacion_simulada(
        comparacion
    )
    base = 4 + 2 * 21
    bloque = vector[base : base + 21]

    assert bloque[:3] == [2.0, 2.0, -1.0]
    assert bloque[3:20] == [-1.0] * 17
    assert bloque[20] == 3.5


def test_codifica_omitido() -> None:
    comparacion = _comparacion()
    filas = list(comparacion.resultados)
    filas[4] = ResultadoMetodoSimulado(
        metodo_solicitado="GA",
        estado=ESTADO_OMITIDO,
        resultado=None,
        error="sin plan",
        tiempo_motor_segundos=0.0,
    )

    vector = bridge.codificar_comparacion_simulada(
        replace(comparacion, resultados=tuple(filas))
    )
    base = 4 + 4 * 21

    assert vector[base : base + 3] == [4.0, 3.0, -1.0]


def test_comparar_usa_los_planes_estimados_y_la_misma_identidad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instancia_vector = [1.0, 2.0, 3.0]
    firma = bridge.firmar_instancia_vector(
        instancia_vector,
        6001,
        1006001,
    )
    planes = {
        metodo: [1.0, float(indice + 1)]
        for indice, metodo in enumerate(
            bridge.METODOS_COMPARACION_SIMULADA
        )
    }
    capturado: dict[str, object] = {}
    comparacion = replace(
        _comparacion(),
        firma_instancia=firma,
    )

    monkeypatch.setattr(
        bridge.selector_bridge,
        "obtener_firma_comparacion_estimada",
        lambda: firma,
    )
    monkeypatch.setattr(
        bridge.selector_bridge,
        "obtener_plan_comparacion_vector",
        lambda metodo: planes[metodo],
    )

    def ejecutar_falso(**kwargs: object) -> ComparacionSimulada:
        capturado.update(kwargs)
        return comparacion

    monkeypatch.setattr(
        bridge,
        "ejecutar_comparacion_simulada",
        ejecutar_falso,
    )

    vector = bridge.comparar_simulado_vector(
        instancia_vector,
        6001,
        1006001,
        "modelo.zip",
        ".",
        "IDMAP1|0",
        "UI-1",
        "2026-08-30",
        True,
        240,
        600.0,
    )

    assert vector[:3] == [1.0, 5.0, 21.0]
    assert capturado["planes_por_metodo"] == planes
    assert capturado["instancia_id"] == "UI-1"
    assert capturado["fecha_operacion"] == "2026-08-30"
    assert capturado["seed_escenario"] == 6001
    assert capturado["seed_ejecucion"] == 1006001
    assert capturado["horizonte_simulacion_min"] == 600.0


def test_comparar_rechaza_si_no_hay_comparacion_estimada(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge.selector_bridge,
        "obtener_firma_comparacion_estimada",
        lambda: "SIN_COMPARACION",
    )

    with pytest.raises(
        RuntimeError,
        match="No existe una comparación estimada",
    ):
        bridge.comparar_simulado_vector(
            [1.0],
            1,
            2,
            "modelo.zip",
            ".",
            "IDMAP1|0",
            "UI-1",
            "2026-08-30",
        )


def test_comparar_rechaza_firma_distinta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge.selector_bridge,
        "obtener_firma_comparacion_estimada",
        lambda: "b" * 64,
    )

    with pytest.raises(
        RuntimeError,
        match="firma de la instancia no coincide",
    ):
        bridge.comparar_simulado_vector(
            [1.0],
            1,
            2,
            "modelo.zip",
            ".",
            "IDMAP1|0",
            "UI-1",
            "2026-08-30",
        )


def test_resumen_estado_y_limpieza(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge,
        "_ultima_comparacion_simulada",
        _comparacion(),
    )

    assert bridge.obtener_resumen_comparacion_simulada().startswith(
        "OK|version=1"
    )
    assert bridge.obtener_firma_comparacion_simulada() == FIRMA
    assert (
        bridge.obtener_estado_metodo_comparacion_simulada(
            "HÍBRIDO"
        )
        == "metodo=HIBRIDO|estado=FINALIZADO|algoritmo=GA|error="
    )

    assert (
        bridge.limpiar_comparacion_simulada()
        == "OK|COMPARACION_SIMULADA_LIMPIA"
    )
    assert (
        bridge.obtener_resumen_comparacion_simulada()
        == "SIN_COMPARACION_SIMULADA"
    )
