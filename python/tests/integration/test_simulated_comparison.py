from __future__ import annotations

import pytest

from planner.integration.simulated_comparison import (
    ESTADO_ERROR,
    ESTADO_FINALIZADO,
    ESTADO_OMITIDO,
    METODOS_COMPARACION_SIMULADA,
    ejecutar_comparacion_simulada,
    normalizar_metodo_simulado,
)
from planner.integration.simulated_execution import (
    ResultadoEjecucionSimulada,
)


def _resultado(
    algoritmo: str,
    *,
    costo: float = 100.0,
) -> ResultadoEjecucionSimulada:
    return ResultadoEjecucionSimulada(
        version=1,
        instancia_id="UI-2026-08-30-6001",
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
        diferencia_fin_camiones_min=8.0,
        ocupacion_global_pct=75.0,
        costo_tareas_no_entregadas=0.0,
        costo_pedidos_originales_incompletos=0.0,
        costo_tardanza=0.0,
        costo_exceso_tolerancia=0.0,
        costo_operacion=150.0,
        costo_distancia=64.0,
        costo_viajes=15.0,
        costo_desbalance=4.0,
        mensaje="EJECUCIÓN FINALIZADA",
        estado_motor="FINISHED",
    )


def _planes() -> dict[str, list[float]]:
    return {
        metodo: [1.0, float(indice + 1)]
        for indice, metodo
        in enumerate(METODOS_COMPARACION_SIMULADA)
    }


def _argumentos() -> dict:
    return {
        "modelo_exportado": "modelo.zip",
        "raiz_python": "python",
        "instancia_vector": [1.0, 2.0, 3.0],
        "planes_por_metodo": _planes(),
        "identificadores_pedidos": "IDMAP1|1|WA==:WA==",
        "instancia_id": "UI-2026-08-30-6001",
        "fecha_operacion": "2026-08-30",
        "seed_escenario": 6001,
        "seed_ejecucion": 1006001,
        "proveedores_habilitados": True,
        "horizonte_simulacion_min": 600.0,
    }


def test_ejecuta_cinco_motores_en_orden_y_con_misma_semilla():
    llamadas: list[dict] = []

    def runner(**kwargs):
        llamadas.append(kwargs)
        metodo = kwargs["log_id"].split("-")[-1].upper()
        algoritmo = "GA" if metodo == "HIBRIDO" else metodo
        return _resultado(algoritmo)

    comparacion = ejecutar_comparacion_simulada(
        **_argumentos(),
        runner=runner,
    )

    assert comparacion.completa
    assert comparacion.cantidad_finalizados == 5
    assert [
        resultado.metodo_solicitado
        for resultado in comparacion.resultados
    ] == list(METODOS_COMPARACION_SIMULADA)
    assert len(llamadas) == 5
    assert {
        llamada["seed_ejecucion"]
        for llamada in llamadas
    } == {1006001}
    assert {
        llamada["horizonte_simulacion_min"]
        for llamada in llamadas
    } == {600.0}
    assert [
        llamada["log_id"]
        for llamada in llamadas
    ] == [
        "simulated-comparison-rl",
        "simulated-comparison-hibrido",
        "simulated-comparison-greedy",
        "simulated-comparison-random",
        "simulated-comparison-ga",
    ]


def test_cada_metodo_recibe_su_plan_almacenado():
    llamadas: list[tuple[str, list[float]]] = []

    def runner(**kwargs):
        llamadas.append(
            (kwargs["log_id"], kwargs["plan_vector"])
        )
        return _resultado("GA")

    ejecutar_comparacion_simulada(
        **_argumentos(),
        runner=runner,
    )

    assert llamadas == [
        (
            f"simulated-comparison-{metodo.lower()}",
            [1.0, float(indice + 1)],
        )
        for indice, metodo
        in enumerate(METODOS_COMPARACION_SIMULADA)
    ]


def test_error_individual_no_impide_metodos_siguientes():
    ejecutados: list[str] = []

    def runner(**kwargs):
        metodo = kwargs["log_id"].split("-")[-1].upper()
        ejecutados.append(metodo)

        if metodo == "GREEDY":
            raise RuntimeError("fallo controlado")

        return _resultado(metodo)

    comparacion = ejecutar_comparacion_simulada(
        **_argumentos(),
        runner=runner,
        continuar_ante_error=True,
    )

    assert ejecutados == list(METODOS_COMPARACION_SIMULADA)
    assert not comparacion.completa
    greedy = comparacion.obtener_resultado("GREEDY")
    assert greedy.estado == ESTADO_ERROR
    assert "fallo controlado" in greedy.error
    assert comparacion.obtener_resultado("GA").finalizado


def test_plan_ausente_se_marca_omitido_sin_invocar_motor():
    argumentos = _argumentos()
    argumentos["planes_por_metodo"].pop("RANDOM")
    llamados: list[str] = []

    def runner(**kwargs):
        llamados.append(kwargs["log_id"])
        return _resultado("GA")

    comparacion = ejecutar_comparacion_simulada(
        **argumentos,
        runner=runner,
    )

    random = comparacion.obtener_resultado("RANDOM")
    assert random.estado == ESTADO_OMITIDO
    assert random.resultado is None
    assert len(llamados) == 4
    assert not comparacion.completa


def test_serializacion_conserva_resultados_por_metodo():
    def runner(**kwargs):
        metodo = kwargs["log_id"].split("-")[-1].upper()
        return _resultado(
            metodo,
            costo=100.0 + len(metodo),
        )

    comparacion = ejecutar_comparacion_simulada(
        **_argumentos(),
        runner=runner,
    )

    payload = comparacion.como_dict()

    assert payload["version"] == 1
    assert payload["completa"] is True
    assert payload["cantidad_finalizados"] == 5
    assert len(payload["resultados"]) == 5
    assert payload["resultados"][0]["estado"] == ESTADO_FINALIZADO
    assert payload["resultados"][0]["resultado"]["instancia_id"] == (
        "UI-2026-08-30-6001"
    )
    assert comparacion.resumen().startswith("OK|version=1")


def test_normaliza_hibrido_con_tilde_y_rechaza_desconocido():
    assert normalizar_metodo_simulado("HÍBRIDO") == "HIBRIDO"
    assert normalizar_metodo_simulado(" rl ") == "RL"

    with pytest.raises(ValueError, match="no reconocido"):
        normalizar_metodo_simulado("OTRO")
