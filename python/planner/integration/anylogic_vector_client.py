from __future__ import annotations

import os
import shutil

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim


PROTOCOL_VERSION = 1

EXPECTED_CONFIGURATION_FIELDS = {
    "seedEjecucion",
    "instanciaVector",
    "rutaPythonProyectoPypeline",
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
class ResultadoEjecucionVectoresAnyLogic:
    modelo: str
    java: str
    seed_ejecucion: int
    identificador_corrida: str
    schema: dict[str, list[str]]
    estado_inicial: str
    estado_final: str
    stop_condition: bool
    observacion_inicial: dict[str, Any]
    observacion_final: dict[str, Any]
    instancia_vector: list[float]
    plan_vector: list[float]

    def a_dict(self) -> dict[str, Any]:
        return {
            "modelo": self.modelo,
            "java": self.java,
            "seed_ejecucion": self.seed_ejecucion,
            "identificador_corrida": self.identificador_corrida,
            "schema": self.schema,
            "estado_inicial": self.estado_inicial,
            "estado_final": self.estado_final,
            "stop_condition": self.stop_condition,
            "observacion_inicial": self.observacion_inicial,
            "observacion_final": self.observacion_final,
            "instancia_vector": self.instancia_vector,
            "plan_vector": self.plan_vector,
        }


class AnyLogicVectorClient:
    """
    Ejecuta vectores ya preparados mediante el contrato dinámico Alpyne.

    Cada llamada crea una instancia nueva del modelo exportado. Esto evita
    que un algoritmo herede estado interno, agentes, RNG o métricas de la
    corrida anterior y permite aplicar Common Random Numbers usando la misma
    seed de ejecución en todas las alternativas.
    """

    def __init__(
        self,
        model_path: str | Path,
        java_exe: str | Path | None = None,
        python_root: str | Path | None = None,
        timeout_segundos: int = 180,
        max_server_await_time: float = 45.0,
        limite_ple_min: float = 300.0,
        log_id_base: str = "phase16b",
        habilitar_logs: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "No existe el ZIP exportado de AnyLogic: "
                f"{self.model_path}"
            )

        if self.model_path.suffix.lower() != ".zip":
            raise ValueError(
                "model_path debe apuntar al ZIP exportado desde "
                "AlpyneExperiment."
            )

        self.java_path = resolver_java_anylogic(java_exe)
        self.python_root = resolver_raiz_python_proyecto(python_root)

        if timeout_segundos <= 0:
            raise ValueError("timeout_segundos debe ser > 0.")

        if max_server_await_time <= 0.0:
            raise ValueError("max_server_await_time debe ser > 0.")

        if limite_ple_min <= 0.0:
            raise ValueError("limite_ple_min debe ser > 0.")

        if not log_id_base.strip():
            raise ValueError("log_id_base no puede estar vacío.")

        self.timeout_segundos = timeout_segundos
        self.max_server_await_time = max_server_await_time
        self.limite_ple_min = limite_ple_min
        self.log_id_base = log_id_base.strip()
        self.habilitar_logs = habilitar_logs

    def ejecutar_vectores(
        self,
        *,
        instancia_vector: Sequence[float],
        plan_vector: Sequence[float],
        seed_ejecucion: int,
        cantidad_pedidos: int,
        cantidad_viajes: int,
        identificador_corrida: str,
    ) -> ResultadoEjecucionVectoresAnyLogic:
        instancia = _normalizar_vector(
            instancia_vector,
            nombre="instancia_vector",
        )
        plan = _normalizar_vector(
            plan_vector,
            nombre="plan_vector",
        )

        _validar_cabecera_instancia(
            instancia,
            cantidad_pedidos=cantidad_pedidos,
        )
        _validar_cabecera_plan(
            plan,
            cantidad_pedidos=cantidad_pedidos,
        )

        if seed_ejecucion < 0:
            raise ValueError("seed_ejecucion no puede ser negativa.")

        if cantidad_viajes < 0:
            raise ValueError("cantidad_viajes no puede ser negativa.")

        corrida = _normalizar_identificador(identificador_corrida)
        log_id = f"{self.log_id_base}_{corrida}"

        sim = AnyLogicSim(
            model_path=str(self.model_path),
            java_exe=str(self.java_path),
            auto_lock=True,
            auto_finish=True,
            py_log_level=self.habilitar_logs,
            java_log_level=self.habilitar_logs,
            log_id=log_id,
            lock_defaults={
                "flag": (
                    EngineState.PAUSED
                    | EngineState.FINISHED
                    | EngineState.ERROR
                ),
                "timeout": self.timeout_segundos,
            },
            max_server_await_time=self.max_server_await_time,
        )

        schema = self._validar_schema(sim)

        status_inicial = self._exigir_status(
            sim.reset(
                seedEjecucion=seed_ejecucion,
                instanciaVector=instancia,
                rutaPythonProyectoPypeline=str(self.python_root),
            ),
            "reset",
        )

        observacion_inicial = self._validar_inicial(status_inicial)

        status_final = self._exigir_status(
            sim.take_action(
                accionCodigo=2,
                planVector=plan,
            ),
            "take_action",
        )

        observacion_final = self._validar_final(
            status=status_final,
            cantidad_pedidos=cantidad_pedidos,
            cantidad_viajes=cantidad_viajes,
        )

        return ResultadoEjecucionVectoresAnyLogic(
            modelo=str(self.model_path),
            java=str(self.java_path),
            seed_ejecucion=seed_ejecucion,
            identificador_corrida=corrida,
            schema=schema,
            estado_inicial=str(status_inicial.state),
            estado_final=str(status_final.state),
            stop_condition=bool(status_final.stop),
            observacion_inicial=observacion_inicial,
            observacion_final=observacion_final,
            instancia_vector=instancia,
            plan_vector=plan,
        )

    def _validar_schema(
        self,
        sim: AnyLogicSim,
    ) -> dict[str, list[str]]:
        schema = sim.schema

        if schema is None:
            raise RuntimeError("Alpyne no devolvió el schema.")

        configuration = set(schema.configuration.keys())
        action = set(schema.action.keys())
        observation = set(schema.observation.keys())

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
            "configuration": sorted(configuration),
            "action": sorted(action),
            "observation": sorted(observation),
        }

    def _validar_inicial(self, status: Any) -> dict[str, Any]:
        if self._contiene_estado(status, EngineState.ERROR):
            raise RuntimeError(
                "AnyLogic quedó en ERROR durante reset: "
                f"{status.message}"
            )

        if not self._contiene_estado(status, EngineState.PAUSED):
            raise RuntimeError(
                "Después de reset se esperaba EngineState.PAUSED, "
                f"pero se recibió {status.state}."
            )

        observacion = dict(status.observation)

        if bool(observacion["error"]):
            raise RuntimeError(
                "AnyLogic reportó un error inicial: "
                f"{observacion['mensaje']}"
            )

        if int(observacion["protocoloVersion"]) != PROTOCOL_VERSION:
            raise RuntimeError(
                "Versión de protocolo inesperada: "
                f"{observacion['protocoloVersion']}"
            )

        if not bool(observacion["configurado"]):
            raise RuntimeError(
                "La observación inicial indica configurado=false."
            )

        if bool(observacion["accionRecibida"]):
            raise RuntimeError(
                "La acción figura recibida antes de take_action."
            )

        return observacion

    def _validar_final(
        self,
        *,
        status: Any,
        cantidad_pedidos: int,
        cantidad_viajes: int,
    ) -> dict[str, Any]:
        observacion = dict(status.observation)

        if self._contiene_estado(status, EngineState.ERROR):
            raise RuntimeError(
                "AnyLogic terminó con ERROR: "
                f"{status.message}"
            )

        if not self._contiene_estado(status, EngineState.FINISHED):
            raise RuntimeError(
                "Estado final inesperado: "
                f"{status.state}"
            )

        if not bool(status.stop):
            raise RuntimeError(
                "La condición terminal no devolvió true."
            )

        if bool(observacion["error"]):
            raise RuntimeError(
                "AnyLogic rechazó la instancia o el plan: "
                f"{observacion['mensaje']}"
            )

        if not bool(observacion["instanciaAceptada"]):
            raise RuntimeError("AnyLogic no aceptó la instancia.")

        if not bool(observacion["planAceptado"]):
            raise RuntimeError("AnyLogic no aceptó el plan.")

        if bool(observacion["ejecucionEnCurso"]):
            raise RuntimeError(
                "La ejecución continúa activa después del estado final."
            )

        if not bool(observacion["ejecucionFinalizada"]):
            raise RuntimeError(
                "ejecucionFinalizada continúa en false."
            )

        cantidad_observada = int(observacion["cantidadPedidos"])
        if cantidad_observada != cantidad_pedidos:
            raise RuntimeError(
                "Cantidad de pedidos distinta entre Python y AnyLogic. "
                f"Python={cantidad_pedidos}, "
                f"AnyLogic={cantidad_observada}."
            )

        entregadas = int(observacion["tareasEntregadas"])
        no_entregadas = int(observacion["tareasNoEntregadas"])

        if (
            entregadas < 0
            or no_entregadas < 0
            or entregadas + no_entregadas != cantidad_pedidos
        ):
            raise RuntimeError(
                "Balance de tareas inválido. "
                f"Entregadas={entregadas}, "
                f"no entregadas={no_entregadas}, "
                f"planificadas={cantidad_pedidos}."
            )

        viajes_observados = int(observacion["viajesTotales"])
        if viajes_observados != cantidad_viajes:
            raise RuntimeError(
                "Cantidad de viajes distinta entre el plan y AnyLogic. "
                f"Plan={cantidad_viajes}, "
                f"AnyLogic={viajes_observados}."
            )

        costo = float(observacion["costoTotal"])
        if not isfinite(costo) or costo < 0.0:
            raise RuntimeError(
                f"Costo operativo inválido: {costo}."
            )

        duracion = float(observacion["tiempoSimuladoMin"])
        if not isfinite(duracion) or duracion <= 0.0:
            raise RuntimeError(
                f"Duración operativa inválida: {duracion}."
            )

        if duracion >= self.limite_ple_min:
            raise RuntimeError(
                "La ejecución llegó o superó el límite preventivo "
                f"configurado: {duracion:.3f} min."
            )

        return observacion

    @staticmethod
    def _exigir_status(status: Any, operacion: str) -> Any:
        if status is None:
            raise RuntimeError(f"{operacion} no devolvió status.")
        return status

    @staticmethod
    def _contiene_estado(status: Any, estado: EngineState) -> bool:
        return bool(status.state & estado)

    @staticmethod
    def _exigir_campos_schema(
        nombre: str,
        encontrados: set[str],
        esperados: set[str],
    ) -> None:
        faltantes = esperados - encontrados
        adicionales = encontrados - esperados

        if faltantes:
            raise RuntimeError(
                f"Faltan campos en {nombre}: {sorted(faltantes)}"
            )

        if adicionales:
            raise RuntimeError(
                f"Sobran campos en {nombre}: {sorted(adicionales)}"
            )


def _normalizar_vector(
    valores: Sequence[float],
    *,
    nombre: str,
) -> list[float]:
    if valores is None:
        raise ValueError(f"{nombre} no puede ser null.")

    resultado = [float(valor) for valor in valores]

    if not resultado:
        raise ValueError(f"{nombre} no puede estar vacío.")

    for indice, valor in enumerate(resultado):
        if not isfinite(valor):
            raise ValueError(
                f"{nombre}[{indice}] no es finito: {valor}."
            )

    return resultado


def _validar_cabecera_instancia(
    vector: Sequence[float],
    *,
    cantidad_pedidos: int,
) -> None:
    if cantidad_pedidos <= 0:
        raise ValueError("cantidad_pedidos debe ser > 0.")

    if len(vector) < 8:
        raise ValueError(
            "instancia_vector no contiene la cabecera completa."
        )

    version = _entero_exacto(vector[0], "instancia.version")
    pedidos_vector = _entero_exacto(
        vector[2],
        "instancia.cantidadPedidos",
    )

    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"Versión de instancia no soportada: {version}."
        )

    if pedidos_vector != cantidad_pedidos:
        raise ValueError(
            "Cantidad de pedidos inconsistente en instancia_vector. "
            f"Cabecera={pedidos_vector}, esperada={cantidad_pedidos}."
        )

    longitud_esperada = 8 + cantidad_pedidos * 10
    if len(vector) != longitud_esperada:
        raise ValueError(
            "Longitud incorrecta de instancia_vector. "
            f"Esperada={longitud_esperada}, recibida={len(vector)}."
        )


def _validar_cabecera_plan(
    vector: Sequence[float],
    *,
    cantidad_pedidos: int,
) -> None:
    if len(vector) < 5:
        raise ValueError("plan_vector no contiene la cabecera completa.")

    version = _entero_exacto(vector[0], "plan.version")
    asignaciones = _entero_exacto(
        vector[1],
        "plan.cantidadAsignaciones",
    )

    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"Versión de plan no soportada: {version}."
        )

    if asignaciones != cantidad_pedidos:
        raise ValueError(
            "Cantidad de asignaciones inconsistente en plan_vector. "
            f"Cabecera={asignaciones}, esperada={cantidad_pedidos}."
        )

    longitud_esperada = 5 + cantidad_pedidos * 4
    if len(vector) != longitud_esperada:
        raise ValueError(
            "Longitud incorrecta de plan_vector. "
            f"Esperada={longitud_esperada}, recibida={len(vector)}."
        )


def _entero_exacto(valor: float, nombre: str) -> int:
    entero = int(round(float(valor)))
    if abs(float(valor) - entero) > 1e-9:
        raise ValueError(f"{nombre} debe ser entero: {valor}.")
    return entero


def _normalizar_identificador(valor: str) -> str:
    texto = str(valor).strip()
    if not texto:
        raise ValueError("identificador_corrida no puede estar vacío.")

    seguro = "".join(
        caracter if caracter.isalnum() else "_"
        for caracter in texto
    ).strip("_")

    if not seguro:
        raise ValueError(
            "identificador_corrida no contiene caracteres válidos."
        )

    return seguro[:80]


def resolver_raiz_python_proyecto(
    python_root: str | Path | None = None,
) -> Path:
    """
    Resuelve la carpeta ``python`` del proyecto que contiene ``planner`` y
    ``pyrefly.toml``. La ruta se envía explícitamente al experimento Alpyne
    porque el modelo exportado se ejecuta desde un directorio temporal.
    """
    if python_root is None:
        candidato = Path(__file__).resolve().parents[2]
    else:
        texto = str(python_root).strip()
        if not texto:
            raise ValueError("python_root no puede estar vacío.")
        candidato = Path(texto).expanduser().resolve()

    planner = candidato / "planner"
    pyrefly = candidato / "pyrefly.toml"
    cache_vial = candidato / "data" / "routing" / "cache_vial_v1.csv"

    faltantes: list[str] = []
    if not planner.is_dir():
        faltantes.append(str(planner))
    if not pyrefly.is_file():
        faltantes.append(str(pyrefly))
    if not cache_vial.is_file():
        faltantes.append(str(cache_vial))

    if faltantes:
        raise FileNotFoundError(
            "La raíz Python no contiene los recursos requeridos: "
            + "; ".join(faltantes)
        )

    return candidato


def resolver_java_anylogic(
    java_exe: str | Path | None = None,
) -> Path:
    if java_exe is not None:
        texto = str(java_exe).strip()
        if texto:
            candidato = Path(texto).expanduser().resolve()
            java_path = _normalizar_java(candidato)
            if java_path is None:
                raise FileNotFoundError(
                    "La ruta indicada no contiene un java.exe válido: "
                    f"{candidato}"
                )
            return java_path

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java_path = _normalizar_java(Path(java_home))
        if java_path is not None:
            return java_path

    java_en_path = shutil.which("java")
    if java_en_path:
        java_path = _normalizar_java(Path(java_en_path))
        if java_path is not None:
            return java_path

    raices: list[Path] = []
    for variable in (
        "ProgramFiles",
        "ProgramFiles(x86)",
        "LOCALAPPDATA",
    ):
        valor = os.environ.get(variable)
        if valor:
            raices.append(Path(valor))

    directorios_anylogic: list[Path] = []
    for raiz in raices:
        if not raiz.is_dir():
            continue
        try:
            directorios_anylogic.extend(
                ruta
                for ruta in raiz.glob("AnyLogic*")
                if ruta.is_dir()
            )
        except OSError:
            continue

    for directorio in directorios_anylogic:
        java_path = _normalizar_java(directorio)
        if java_path is not None:
            return java_path

    for directorio in directorios_anylogic:
        try:
            for candidato in directorio.rglob("java.exe"):
                if candidato.is_file():
                    return candidato.resolve()
        except OSError:
            continue

    raise FileNotFoundError(
        "No se encontró el Java de AnyLogic. Usá --java con la "
        "ruta a java.exe."
    )


def _normalizar_java(candidato: Path) -> Path | None:
    candidato = candidato.expanduser()

    if candidato.is_file():
        if candidato.name.lower() in {"java", "java.exe"}:
            return candidato.resolve()
        return None

    if not candidato.is_dir():
        return None

    candidatos = (
        candidato / "bin" / "java.exe",
        candidato / "bin" / "java",
        candidato / "jre" / "bin" / "java.exe",
        candidato / "jre" / "bin" / "java",
        candidato / "jdk" / "bin" / "java.exe",
        candidato / "jdk" / "bin" / "java",
        candidato / "runtime" / "bin" / "java.exe",
        candidato / "runtime" / "bin" / "java",
    )

    for java_path in candidatos:
        if java_path.is_file():
            return java_path.resolve()

    return None
