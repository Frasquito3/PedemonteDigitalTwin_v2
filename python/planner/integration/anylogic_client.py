from __future__ import annotations

import os
import shutil

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim

from planner.core.schema import (
    InstanciaTurno,
    PlanTurno,
)

from .alpyne_codec import (
    PROTOCOL_VERSION,
    codificar_instancia_alpyne,
    codificar_plan_alpyne,
)


EXPECTED_CONFIGURATION_FIELDS = {
    "seedEjecucion",
    "instanciaVector",
}

EXPECTED_ACTION_FIELDS = {
    "accionCodigo",
    "planVector",
}

EXPECTED_OBSERVATION_FIELDS = {
    "protocoloVersion",
    "configurado",
    "accionRecibida",
    "instanciaAceptada",
    "planAceptado",
    "cantidadPedidos",
    "ejecucionEnCurso",
    "ejecucionFinalizada",
    "error",
    "mensaje",
    "costoTotal",
    "tareasEntregadas",
    "tareasNoEntregadas",
    "viajesTotales",
    "tiempoSimuladoMin",
}


@dataclass(frozen=True)
class ResultadoEjecucionAnyLogic:
    modelo: str

    java: str

    seed_ejecucion: int

    schema: dict[
        str,
        list[str],
    ]

    estado_inicial: str

    estado_final: str

    stop_condition: bool

    observacion_inicial: dict[
        str,
        Any,
    ]

    observacion_final: dict[
        str,
        Any,
    ]

    instancia_vector: list[float]

    plan_vector: list[float]

    def a_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "modelo": self.modelo,
            "java": self.java,
            "seed_ejecucion": (
                self.seed_ejecucion
            ),
            "schema": self.schema,
            "estado_inicial": (
                self.estado_inicial
            ),
            "estado_final": (
                self.estado_final
            ),
            "stop_condition": (
                self.stop_condition
            ),
            "observacion_inicial": (
                self.observacion_inicial
            ),
            "observacion_final": (
                self.observacion_final
            ),
            "instancia_vector": (
                self.instancia_vector
            ),
            "plan_vector": (
                self.plan_vector
            ),
        }


class AnyLogicDynamicClient:
    """
    Cliente para ejecutar un plan completo mediante el
    contrato dinámico Alpyne de la Fase 10C.
    """

    def __init__(
        self,
        model_path: str | Path,
        java_exe: str | Path | None = None,
        timeout_segundos: int = 120,
        max_server_await_time: float = 30.0,
        limite_ple_min: float = 59.0,
        log_id: str = "phase10d",
        habilitar_logs: bool = True,
    ) -> None:
        self.model_path = Path(
            model_path
        ).expanduser().resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "No existe el ZIP exportado "
                "de AnyLogic: "
                f"{self.model_path}"
            )

        if (
            self.model_path
            .suffix.lower()
            != ".zip"
        ):
            raise ValueError(
                "model_path debe apuntar al "
                "ZIP exportado desde el "
                "experimento RL."
            )

        self.java_path = (
            resolver_java_anylogic(
                java_exe
            )
        )

        if timeout_segundos <= 0:
            raise ValueError(
                "timeout_segundos debe ser > 0."
            )

        if max_server_await_time <= 0.0:
            raise ValueError(
                "max_server_await_time debe ser > 0."
            )

        if limite_ple_min <= 0.0:
            raise ValueError(
                "limite_ple_min debe ser > 0."
            )

        self.timeout_segundos = (
            timeout_segundos
        )

        self.max_server_await_time = (
            max_server_await_time
        )

        self.limite_ple_min = (
            limite_ple_min
        )

        self.log_id = log_id

        self.habilitar_logs = (
            habilitar_logs
        )

    def ejecutar(
        self,
        instancia: InstanciaTurno,
        plan: PlanTurno,
    ) -> ResultadoEjecucionAnyLogic:
        instancia_vector = (
            codificar_instancia_alpyne(
                instancia
            )
        )

        plan_vector = (
            codificar_plan_alpyne(
                instancia,
                plan,
            )
        )

        sim = AnyLogicSim(
            model_path=str(
                self.model_path
            ),
            java_exe=str(
                self.java_path
            ),
            auto_lock=True,
            auto_finish=True,
            py_log_level=(
                self.habilitar_logs
            ),
            java_log_level=(
                self.habilitar_logs
            ),
            log_id=self.log_id,
            lock_defaults={
                "flag": (
                    EngineState.PAUSED
                    |
                    EngineState.FINISHED
                    |
                    EngineState.ERROR
                ),
                "timeout": (
                    self.timeout_segundos
                ),
            },
            max_server_await_time=(
                self.max_server_await_time
            ),
        )

        schema = self._validar_schema(
            sim
        )

        status_inicial = (
            self._exigir_status(
                sim.reset(
                    seedEjecucion=(
                        instancia
                        .seed_ejecucion
                    ),
                    instanciaVector=(
                        instancia_vector
                    ),
                ),
                "reset",
            )
        )

        observacion_inicial = (
            self._validar_inicial(
                status_inicial
            )
        )

        status_final = (
            self._exigir_status(
                sim.take_action(
                    accionCodigo=2,
                    planVector=(
                        plan_vector
                    ),
                ),
                "take_action",
            )
        )

        observacion_final = (
            self._validar_final(
                status=status_final,
                cantidad_pedidos=(
                    len(
                        instancia.pedidos
                    )
                ),
                cantidad_viajes=(
                    sum(
                        len(
                            camion.viajes
                        )
                        for camion
                        in plan.camiones
                    )
                ),
            )
        )

        return ResultadoEjecucionAnyLogic(
            modelo=str(
                self.model_path
            ),
            java=str(
                self.java_path
            ),
            seed_ejecucion=(
                instancia
                .seed_ejecucion
            ),
            schema=schema,
            estado_inicial=str(
                status_inicial.state
            ),
            estado_final=str(
                status_final.state
            ),
            stop_condition=bool(
                status_final.stop
            ),
            observacion_inicial=(
                observacion_inicial
            ),
            observacion_final=(
                observacion_final
            ),
            instancia_vector=(
                instancia_vector
            ),
            plan_vector=(
                plan_vector
            ),
        )

    def _validar_schema(
        self,
        sim: AnyLogicSim,
    ) -> dict[str, list[str]]:
        schema = sim.schema

        if schema is None:
            raise RuntimeError(
                "Alpyne no devolvió el schema."
            )

        configuration = set(
            schema.configuration.keys()
        )

        action = set(
            schema.action.keys()
        )

        observation = set(
            schema.observation.keys()
        )

        self._exigir_campos_schema(
            "Configuration",
            configuration,
            EXPECTED_CONFIGURATION_FIELDS,
        )

        self._exigir_campos_schema(
            "Action",
            action,
            EXPECTED_ACTION_FIELDS,
        )

        self._exigir_campos_schema(
            "Observation",
            observation,
            EXPECTED_OBSERVATION_FIELDS,
        )

        return {
            "configuration": sorted(
                configuration
            ),
            "action": sorted(
                action
            ),
            "observation": sorted(
                observation
            ),
        }

    def _validar_inicial(
        self,
        status: Any,
    ) -> dict[str, Any]:
        if self._contiene_estado(
            status,
            EngineState.ERROR,
        ):
            raise RuntimeError(
                "AnyLogic quedó en ERROR "
                "durante reset: "
                f"{status.message}"
            )

        if not self._contiene_estado(
            status,
            EngineState.PAUSED,
        ):
            raise RuntimeError(
                "Después de reset se esperaba "
                "EngineState.PAUSED, pero se "
                f"recibió {status.state}."
            )

        observacion = dict(
            status.observation
        )

        if observacion["error"]:
            raise RuntimeError(
                "AnyLogic reportó un error "
                "inicial: "
                f"{observacion['mensaje']}"
            )

        if (
            int(
                observacion[
                    "protocoloVersion"
                ]
            )
            != PROTOCOL_VERSION
        ):
            raise RuntimeError(
                "Versión de protocolo "
                "inesperada: "
                f"{observacion['protocoloVersion']}"
            )

        if not observacion[
            "configurado"
        ]:
            raise RuntimeError(
                "La observación inicial indica "
                "configurado=false."
            )

        if observacion[
            "accionRecibida"
        ]:
            raise RuntimeError(
                "La acción figura recibida "
                "antes de take_action."
            )

        return observacion

    def _validar_final(
        self,
        status: Any,
        cantidad_pedidos: int,
        cantidad_viajes: int,
    ) -> dict[str, Any]:
        observacion = dict(
            status.observation
        )

        if self._contiene_estado(
            status,
            EngineState.ERROR,
        ):
            raise RuntimeError(
                "AnyLogic terminó con ERROR: "
                f"{status.message}"
            )

        if not self._contiene_estado(
            status,
            EngineState.FINISHED,
        ):
            raise RuntimeError(
                "Estado final inesperado: "
                f"{status.state}"
            )

        if not status.stop:
            raise RuntimeError(
                "La condición terminal no "
                "devolvió true."
            )

        if observacion["error"]:
            raise RuntimeError(
                "AnyLogic rechazó la instancia "
                "o el plan: "
                f"{observacion['mensaje']}"
            )

        if not observacion[
            "instanciaAceptada"
        ]:
            raise RuntimeError(
                "AnyLogic no aceptó la instancia."
            )

        if not observacion[
            "planAceptado"
        ]:
            raise RuntimeError(
                "AnyLogic no aceptó el plan."
            )

        if observacion[
            "ejecucionEnCurso"
        ]:
            raise RuntimeError(
                "La ejecución continúa activa "
                "después del estado final."
            )

        if not observacion[
            "ejecucionFinalizada"
        ]:
            raise RuntimeError(
                "ejecucionFinalizada continúa "
                "en false."
            )

        cantidad_observada = int(
            observacion[
                "cantidadPedidos"
            ]
        )

        if (
            cantidad_observada
            != cantidad_pedidos
        ):
            raise RuntimeError(
                "Cantidad de pedidos distinta "
                "entre Python y AnyLogic. "
                f"Python={cantidad_pedidos}, "
                f"AnyLogic={cantidad_observada}."
            )

        entregadas = int(
            observacion[
                "tareasEntregadas"
            ]
        )

        no_entregadas = int(
            observacion[
                "tareasNoEntregadas"
            ]
        )

        if (
            entregadas < 0
            or no_entregadas < 0
            or (
                entregadas
                + no_entregadas
            )
            != cantidad_pedidos
        ):
            raise RuntimeError(
                "Balance de tareas inválido. "
                f"Entregadas={entregadas}, "
                "no entregadas="
                f"{no_entregadas}, "
                "planificadas="
                f"{cantidad_pedidos}."
            )

        viajes_observados = int(
            observacion[
                "viajesTotales"
            ]
        )

        if (
            viajes_observados
            != cantidad_viajes
        ):
            raise RuntimeError(
                "Cantidad de viajes distinta "
                "entre el plan y AnyLogic. "
                f"Plan={cantidad_viajes}, "
                f"AnyLogic={viajes_observados}."
            )

        costo = float(
            observacion[
                "costoTotal"
            ]
        )

        if (
            not isfinite(
                costo
            )
            or costo < 0.0
        ):
            raise RuntimeError(
                f"Costo operativo inválido: {costo}."
            )

        duracion = float(
            observacion[
                "tiempoSimuladoMin"
            ]
        )

        if (
            not isfinite(
                duracion
            )
            or duracion <= 0.0
        ):
            raise RuntimeError(
                "Duración operativa inválida: "
                f"{duracion}."
            )

        if duracion >= self.limite_ple_min:
            raise RuntimeError(
                "La ejecución llegó o superó "
                "el límite preventivo para PLE: "
                f"{duracion:.3f} min."
            )

        return observacion

    @staticmethod
    def _exigir_status(
        status: Any,
        operacion: str,
    ) -> Any:
        if status is None:
            raise RuntimeError(
                f"{operacion} no devolvió status."
            )

        return status

    @staticmethod
    def _contiene_estado(
        status: Any,
        estado: EngineState,
    ) -> bool:
        return bool(
            status.state
            & estado
        )

    @staticmethod
    def _exigir_campos_schema(
        nombre: str,
        encontrados: set[str],
        esperados: set[str],
    ) -> None:
        faltantes = (
            esperados
            - encontrados
        )

        adicionales = (
            encontrados
            - esperados
        )

        if faltantes:
            raise RuntimeError(
                f"Faltan campos en {nombre}: "
                f"{sorted(faltantes)}"
            )

        if adicionales:
            raise RuntimeError(
                f"Sobran campos en {nombre}: "
                f"{sorted(adicionales)}"
            )


def resolver_java_anylogic(
    java_exe: str | Path | None = None,
) -> Path:
    if java_exe is not None:
        texto = str(
            java_exe
        ).strip()

        if texto:
            candidato = Path(
                texto
            ).expanduser().resolve()

            java_path = (
                _normalizar_java(
                    candidato
                )
            )

            if java_path is None:
                raise FileNotFoundError(
                    "La ruta indicada no contiene "
                    "un java.exe válido: "
                    f"{candidato}"
                )

            return java_path

    java_home = os.environ.get(
        "JAVA_HOME"
    )

    if java_home:
        java_path = _normalizar_java(
            Path(
                java_home
            )
        )

        if java_path is not None:
            return java_path

    java_en_path = shutil.which(
        "java"
    )

    if java_en_path:
        java_path = _normalizar_java(
            Path(
                java_en_path
            )
        )

        if java_path is not None:
            return java_path

    raices: list[Path] = []

    for variable in (
        "ProgramFiles",
        "ProgramFiles(x86)",
        "LOCALAPPDATA",
    ):
        valor = os.environ.get(
            variable
        )

        if valor:
            raices.append(
                Path(
                    valor
                )
            )

    directorios_anylogic: list[
        Path
    ] = []

    for raiz in raices:
        if not raiz.is_dir():
            continue

        try:
            directorios_anylogic.extend(
                ruta
                for ruta in raiz.glob(
                    "AnyLogic*"
                )
                if ruta.is_dir()
            )

        except OSError:
            continue

    for directorio in (
        directorios_anylogic
    ):
        java_path = _normalizar_java(
            directorio
        )

        if java_path is not None:
            return java_path

    for directorio in (
        directorios_anylogic
    ):
        try:
            for candidato in (
                directorio.rglob(
                    "java.exe"
                )
            ):
                if candidato.is_file():
                    return (
                        candidato.resolve()
                    )

        except OSError:
            continue

    raise FileNotFoundError(
        "No se encontró el Java de AnyLogic. "
        "Indicá su ruta mediante java_exe."
    )


def _normalizar_java(
    candidato: Path,
) -> Path | None:
    if candidato.is_file():
        if (
            candidato.name.lower()
            == "java.exe"
        ):
            return candidato.resolve()

        return None

    if not candidato.is_dir():
        return None

    candidatos = (
        candidato
        / "bin"
        / "java.exe",

        candidato
        / "jre"
        / "bin"
        / "java.exe",

        candidato
        / "jdk"
        / "bin"
        / "java.exe",

        candidato
        / "runtime"
        / "bin"
        / "java.exe",
    )

    for java_path in candidatos:
        if java_path.is_file():
            return java_path.resolve()

    return None