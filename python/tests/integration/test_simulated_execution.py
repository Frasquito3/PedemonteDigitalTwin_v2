from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from planner.integration.simulated_execution import (
    ErrorEjecucionSimulada,
    codificar_identificadores_pedidos,
    construir_resultado_simulado,
    ejecutar_plan_en_modelo_exportado,
)


OBSERVACION_FINAL = {
    "error": False,
    "mensaje": "EJECUCIÓN FINALIZADA",
    "configurado": True,
    "instanciaAceptada": True,
    "planAceptado": True,
    "ejecucionFinalizada": True,
    "resultadoDisponible": True,
    "instanciaId": "UI-2026-08-30-6001",
    "fechaOperacionResultado": "2026-08-30",
    "algoritmoAplicado": "GA",
    "seedEscenarioResultado": 6001,
    "seedEjecucionResultado": 1006001,
    "costoTotal": 235.0,
    "tareasEntregadas": 5,
    "tareasNoEntregadas": 0,
    "viajesTotales": 3,
    "tiempoSimuladoMin": 151.0,
    "distanciaTotalKm": 32.0,
    "tardanzaTotalMin": 0.0,
    "diferenciaFinCamionesMin": 8.0,
    "ocupacionGlobalPct": 75.0,
    "costoTareasNoEntregadas": 0.0,
    "costoPedidosOriginalesIncompletos": 0.0,
    "costoTardanza": 0.0,
    "costoExcesoTolerancia": 0.0,
    "costoOperacion": 151.0,
    "costoDistancia": 64.0,
    "costoViajes": 15.0,
    "costoDesbalance": 5.0,
}


@dataclass
class EstadoFalso:
    observation: dict
    state: str
    stop: bool


class SimFalso:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.configuracion = None
        self.accion = None
        self.cierres = 0

    def reset(self, **kwargs):
        self.configuracion = kwargs
        return EstadoFalso(
            observation={
                "error": False,
                "mensaje": "ESPERANDO ACCIÓN",
                "configurado": True,
            },
            state="PAUSED",
            stop=False,
        )

    def take_action(self, **kwargs):
        self.accion = kwargs
        return EstadoFalso(
            observation=dict(OBSERVACION_FINAL),
            state="FINISHED",
            stop=True,
        )

    def _quit_app(self):
        self.cierres += 1


def test_construye_resultado_final_consistente():
    resultado = construir_resultado_simulado(
        OBSERVACION_FINAL,
        estado_motor="FINISHED",
        stop=True,
    )

    assert resultado.instancia_id == "UI-2026-08-30-6001"
    assert resultado.fecha_operacion == "2026-08-30"
    assert resultado.seed_escenario == 6001
    assert resultado.seed_ejecucion == 1006001
    assert resultado.algoritmo_aplicado == "GA"
    assert resultado.viajes_totales == 3
    assert resultado.costo_total == 235.0
    assert resultado.ocupacion_global_pct == 75.0
    assert resultado.resumen().startswith("OK|version=1")


def test_repara_mojibake_del_mensaje():
    observacion = dict(OBSERVACION_FINAL)
    observacion["mensaje"] = "EJECUCIÃ“N FINALIZADA"

    resultado = construir_resultado_simulado(
        observacion,
        estado_motor="FINISHED",
        stop=True,
    )

    assert resultado.mensaje == "EJECUCIÓN FINALIZADA"


def test_rechaza_costo_inconsistente():
    observacion = dict(OBSERVACION_FINAL)
    observacion["costoTotal"] = 999.0

    with pytest.raises(
        ErrorEjecucionSimulada,
        match="no coincide con su desglose",
    ):
        construir_resultado_simulado(
            observacion,
            estado_motor="FINISHED",
            stop=True,
        )


def test_rechaza_resultado_no_disponible():
    observacion = dict(OBSERVACION_FINAL)
    observacion["resultadoDisponible"] = False

    with pytest.raises(
        ErrorEjecucionSimulada,
        match="resultado disponible",
    ):
        construir_resultado_simulado(
            observacion,
            estado_motor="FINISHED",
            stop=True,
        )


def test_rechaza_ocupacion_superior_a_cien():
    observacion = dict(OBSERVACION_FINAL)
    observacion["ocupacionGlobalPct"] = 101.0

    with pytest.raises(
        ErrorEjecucionSimulada,
        match="no puede superar 100",
    ):
        construir_resultado_simulado(
            observacion,
            estado_motor="FINISHED",
            stop=True,
        )


def test_envia_identidad_configuracion_y_accion_correctas(tmp_path: Path):
    modelo = tmp_path / "modelo.zip"
    modelo.write_bytes(b"zip")
    raiz = tmp_path / "python"
    raiz.mkdir()

    creado: list[SimFalso] = []

    def fabrica(*args, **kwargs):
        sim = SimFalso(*args, **kwargs)
        creado.append(sim)
        return sim

    resultado = ejecutar_plan_en_modelo_exportado(
        modelo_exportado=modelo,
        raiz_python=raiz,
        instancia_vector=[1.0, 2.0],
        plan_vector=[1.0, 1.0, 2.0],
        identificadores_pedidos="IDMAP1|1|VEFSRUE=:T1JJR0lOQUw=",
        instancia_id="UI-2026-08-30-6001",
        fecha_operacion="2026-08-30",
        seed_escenario=6001,
        seed_ejecucion=1006001,
        proveedores_habilitados=True,
        timeout_segundos=99,
        sim_factory=fabrica,
    )

    sim = creado[0]
    assert sim.configuracion == {
        "instanciaId": "UI-2026-08-30-6001",
        "fechaOperacion": "2026-08-30",
        "identificadoresPedidos": "IDMAP1|1|VEFSRUE=:T1JJR0lOQUw=",
        "seedEscenario": 6001,
        "seedEjecucion": 1006001,
        "instanciaVector": [1.0, 2.0],
        "rutaPythonProyectoPypeline": str(raiz.resolve()),
        "proveedoresHabilitados": True,
    }
    assert sim.accion == {
        "accionCodigo": 2,
        "planVector": [1.0, 1.0, 2.0],
    }
    assert sim.kwargs["auto_finish"] is True
    assert sim.kwargs["lock_defaults"] == {"timeout": 99}
    assert sim.kwargs["engine_overrides"] == {"stop_time": 600.0}
    assert sim.cierres == 1
    assert resultado.instancia_id == "UI-2026-08-30-6001"


def test_rechaza_identidad_distinta(tmp_path: Path):
    modelo = tmp_path / "modelo.zip"
    modelo.write_bytes(b"zip")
    raiz = tmp_path / "python"
    raiz.mkdir()

    class SimIdentidadIncorrecta(SimFalso):
        def take_action(self, **kwargs):
            observacion = dict(OBSERVACION_FINAL)
            observacion["seedEscenarioResultado"] = 9999
            return EstadoFalso(
                observation=observacion,
                state="FINISHED",
                stop=True,
            )

    with pytest.raises(
        ErrorEjecucionSimulada,
        match="no preservó la identidad",
    ):
        ejecutar_plan_en_modelo_exportado(
            modelo_exportado=modelo,
            raiz_python=raiz,
            instancia_vector=[1.0],
            plan_vector=[1.0],
            identificadores_pedidos="IDMAP1|1|VEFSRUE=:T1JJR0lOQUw=",
            instancia_id="UI-2026-08-30-6001",
            fecha_operacion="2026-08-30",
            seed_escenario=6001,
            seed_ejecucion=1006001,
            proveedores_habilitados=True,
            sim_factory=SimIdentidadIncorrecta,
        )


def test_acepta_motor_finished_con_resultado_completo_aunque_stop_sea_false(
    tmp_path: Path,
):
    modelo = tmp_path / "modelo.zip"
    modelo.write_bytes(b"zip")
    raiz = tmp_path / "python"
    raiz.mkdir()
    creado: list[SimFalso] = []

    class SimFinishedSinStop(SimFalso):
        def take_action(self, **kwargs):
            self.accion = kwargs
            return EstadoFalso(
                observation=dict(OBSERVACION_FINAL),
                state="FINISHED",
                stop=False,
            )

    def fabrica(*args, **kwargs):
        sim = SimFinishedSinStop(*args, **kwargs)
        creado.append(sim)
        return sim

    resultado = ejecutar_plan_en_modelo_exportado(
        modelo_exportado=modelo,
        raiz_python=raiz,
        instancia_vector=[1.0],
        plan_vector=[1.0],
        identificadores_pedidos="IDMAP1|1|WA==:WA==",
        instancia_id="UI-2026-08-30-6001",
        fecha_operacion="2026-08-30",
        seed_escenario=6001,
        seed_ejecucion=1006001,
        proveedores_habilitados=True,
        sim_factory=fabrica,
    )

    assert resultado.estado_motor == "FINISHED"
    assert resultado.costo_total == 235.0
    assert creado[0].cierres == 1


def test_recupera_estado_terminal_si_take_action_devuelve_paused_obsoleto(
    tmp_path: Path,
):
    modelo = tmp_path / "modelo.zip"
    modelo.write_bytes(b"zip")
    raiz = tmp_path / "python"
    raiz.mkdir()
    creado = []

    @dataclass
    class EstadoConSecuencia:
        observation: dict
        state: str
        stop: bool
        sequence_id: int

    class SimEstadoObsoleto(SimFalso):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.consultas_estado = 0

        def reset(self, **kwargs):
            self.configuracion = kwargs
            return EstadoConSecuencia(
                observation={
                    "error": False,
                    "mensaje": "ESPERANDO ACCIÓN",
                    "configurado": True,
                },
                state="PAUSED",
                stop=False,
                sequence_id=1,
            )

        def take_action(self, **kwargs):
            self.accion = kwargs
            return EstadoConSecuencia(
                observation={
                    "error": False,
                    "configurado": True,
                    "ejecucionFinalizada": False,
                    "resultadoDisponible": False,
                },
                state="PAUSED",
                stop=False,
                sequence_id=1,
            )

        def status(self):
            self.consultas_estado += 1
            return EstadoConSecuencia(
                observation=dict(OBSERVACION_FINAL),
                state="FINISHED",
                stop=False,
                sequence_id=2,
            )

    def fabrica(*args, **kwargs):
        sim = SimEstadoObsoleto(*args, **kwargs)
        creado.append(sim)
        return sim

    resultado = ejecutar_plan_en_modelo_exportado(
        modelo_exportado=modelo,
        raiz_python=raiz,
        instancia_vector=[1.0],
        plan_vector=[1.0],
        identificadores_pedidos="IDMAP1|1|WA==:WA==",
        instancia_id="UI-2026-08-30-6001",
        fecha_operacion="2026-08-30",
        seed_escenario=6001,
        seed_ejecucion=1006001,
        proveedores_habilitados=True,
        timeout_segundos=1,
        sim_factory=fabrica,
    )

    assert resultado.estado_motor == "FINISHED"
    assert resultado.costo_total == 235.0
    assert creado[0].consultas_estado >= 1
    assert creado[0].cierres == 1


def test_cierra_motor_si_el_resultado_terminal_nunca_se_publica(
    tmp_path: Path,
):
    modelo = tmp_path / "modelo.zip"
    modelo.write_bytes(b"zip")
    raiz = tmp_path / "python"
    raiz.mkdir()
    creado: list[SimFalso] = []

    observacion_no_terminal = {
        "error": False,
        "configurado": True,
        "ejecucionFinalizada": False,
        "resultadoDisponible": False,
    }

    class SimPausadoSinResultado(SimFalso):
        def take_action(self, **kwargs):
            self.accion = kwargs
            return EstadoFalso(
                observation=dict(observacion_no_terminal),
                state="PAUSED",
                stop=False,
            )

        def status(self):
            return EstadoFalso(
                observation=dict(observacion_no_terminal),
                state="PAUSED",
                stop=False,
            )

    def fabrica(*args, **kwargs):
        sim = SimPausadoSinResultado(*args, **kwargs)
        creado.append(sim)
        return sim

    with pytest.raises(
        ErrorEjecucionSimulada,
        match="no publicó un resultado terminal",
    ):
        ejecutar_plan_en_modelo_exportado(
            modelo_exportado=modelo,
            raiz_python=raiz,
            instancia_vector=[1.0],
            plan_vector=[1.0],
            identificadores_pedidos="IDMAP1|1|WA==:WA==",
            instancia_id="UI-2026-08-30-6001",
            fecha_operacion="2026-08-30",
            seed_escenario=6001,
            seed_ejecucion=1006001,
            proveedores_habilitados=True,
            timeout_segundos=0.01,
            sim_factory=fabrica,
        )

    assert creado[0].cierres == 1


def test_rechaza_horizonte_simulado_no_positivo(tmp_path: Path):
    modelo = tmp_path / "modelo.zip"
    modelo.write_bytes(b"zip")
    raiz = tmp_path / "python"
    raiz.mkdir()

    with pytest.raises(
        ValueError,
        match="horizonte_simulacion_min",
    ):
        ejecutar_plan_en_modelo_exportado(
            modelo_exportado=modelo,
            raiz_python=raiz,
            instancia_vector=[1.0],
            plan_vector=[1.0],
            identificadores_pedidos="IDMAP1|1|WA==:WA==",
            instancia_id="UI-2026-08-30-6001",
            fecha_operacion="2026-08-30",
            seed_escenario=6001,
            seed_ejecucion=1006001,
            proveedores_habilitados=True,
            horizonte_simulacion_min=0.0,
            sim_factory=SimFalso,
        )


def test_codifica_identificadores_en_orden():
    @dataclass
    class Pedido:
        pedido_id: str
        pedido_original_id: str

    contrato = codificar_identificadores_pedidos(
        [
            Pedido("XLS03-P1", "XLS03"),
            Pedido("XLS03-P2", "XLS03"),
        ]
    )

    assert contrato == (
        "IDMAP1|2|WExTMDMtUDE=:WExTMDM="
        "|WExTMDMtUDI=:WExTMDM="
    )
