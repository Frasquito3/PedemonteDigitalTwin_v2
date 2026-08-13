from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

from pathlib import Path
from typing import Any

from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PYTHON_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PYTHON_ROOT
    / "anylogic_export"
    / "phase10b_handshake"
    / "PedemonteDigitalTwin_v2.zip"
)

DEFAULT_OUTPUT_PATH = (
    PYTHON_ROOT
    / "rl_artifacts"
    / "phase10b_handshake"
    / "handshake_result.json"
)


# ============================================================
# CONTRATO ESPERADO
# ============================================================

EXPECTED_CONFIGURATION_FIELDS = {
    "seedEjecucion",
}

EXPECTED_ACTION_FIELDS = {
    "accionCodigo",
}

EXPECTED_OBSERVATION_FIELDS = {
    "configurado",
    "accionRecibida",
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


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prueba mínima de comunicación entre "
            "Python, Alpyne y AnyLogic."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(
            DEFAULT_MODEL_PATH
        ),
        help=(
            "Ruta al ZIP exportado desde el "
            "experimento Reinforcement Learning."
        ),
    )

    parser.add_argument(
        "--java",
        type=str,
        default="",
        help=(
            "Ruta opcional al java.exe. "
            "Se recomienda usar el Java incluido "
            "con AnyLogic."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2_001,
        help=(
            "Seed de ejecución enviada mediante "
            "la Configuration de Alpyne."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Archivo JSON donde se guardará "
            "el resultado del handshake."
        ),
    )

    return parser.parse_args()


# ============================================================
# RESOLUCIÓN DE RUTAS
# ============================================================

def resolver_ruta(
    ruta_texto: str,
) -> Path:
    ruta = Path(
        ruta_texto
    ).expanduser()

    if not ruta.is_absolute():
        ruta = (
            Path.cwd()
            / ruta
        )

    return ruta.resolve()


def normalizar_candidato_java(
    candidato: Path,
) -> Path | None:
    candidato = candidato.expanduser()

    if candidato.is_file():
        if (
            candidato.name.lower()
            == "java.exe"
        ):
            return candidato.resolve()

        return None

    if not candidato.is_dir():
        return None

    candidatos_directos = (
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

    for java_path in candidatos_directos:
        if java_path.is_file():
            return java_path.resolve()

    return None


def buscar_java_en_anylogic() -> Path | None:
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
                Path(valor)
            )

    directorios_anylogic: list[Path] = []

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

    for directorio in directorios_anylogic:
        candidatos_directos = (
            directorio
            / "jre"
            / "bin"
            / "java.exe",

            directorio
            / "jdk"
            / "bin"
            / "java.exe",

            directorio
            / "runtime"
            / "bin"
            / "java.exe",

            directorio
            / "bin"
            / "java.exe",
        )

        for candidato in candidatos_directos:
            if candidato.is_file():
                return candidato.resolve()

    for directorio in directorios_anylogic:
        try:
            for candidato in directorio.rglob(
                "java.exe"
            ):
                if candidato.is_file():
                    return candidato.resolve()

        except OSError:
            continue

    return None


def resolver_java(
    java_recibido: str,
) -> Path:
    # --------------------------------------------------------
    # 1. Ruta indicada explícitamente
    # --------------------------------------------------------

    if java_recibido.strip():
        candidato_explicito = (
            resolver_ruta(
                java_recibido
            )
        )

        java_explicito = (
            normalizar_candidato_java(
                candidato_explicito
            )
        )

        if java_explicito is None:
            raise FileNotFoundError(
                "La ruta recibida mediante --java "
                "no contiene un java.exe válido: "
                f"{candidato_explicito}"
            )

        return java_explicito

    # --------------------------------------------------------
    # 2. JAVA_HOME
    # --------------------------------------------------------

    java_home = os.environ.get(
        "JAVA_HOME"
    )

    if java_home:
        java_desde_home = (
            normalizar_candidato_java(
                Path(
                    java_home
                )
            )
        )

        if java_desde_home is not None:
            return java_desde_home

    # --------------------------------------------------------
    # 3. PATH del sistema
    # --------------------------------------------------------

    java_en_path = shutil.which(
        "java"
    )

    if java_en_path:
        candidato_path = Path(
            java_en_path
        )

        if candidato_path.is_file():
            return candidato_path.resolve()

    # --------------------------------------------------------
    # 4. Instalación de AnyLogic
    # --------------------------------------------------------

    java_anylogic = (
        buscar_java_en_anylogic()
    )

    if java_anylogic is not None:
        return java_anylogic

    raise FileNotFoundError(
        "No se encontró java.exe. "
        "Indicá manualmente el Java incluido con "
        "AnyLogic mediante:\n\n"
        '  --java "C:\\ruta\\a\\java.exe"\n\n'
        "Podés buscarlo con PowerShell usando:\n\n"
        "  Get-ChildItem "
        '"C:\\Program Files", '
        '"C:\\Program Files (x86)" '
        "-Recurse -Filter java.exe "
        "-ErrorAction SilentlyContinue"
    )


# ============================================================
# FUNCIONES DE ALPYNE
# ============================================================

def exigir_status(
    status: Any,
    contexto: str,
) -> Any:
    if status is None:
        raise RuntimeError(
            "Alpyne no devolvió un estado después de "
            f"{contexto}. Verificá auto_lock=True."
        )

    return status


def contiene_estado(
    status: Any,
    estado: EngineState,
) -> bool:
    return bool(
        status.state
        & estado
    )


def observacion_como_dict(
    status: Any,
) -> dict[str, Any]:
    return dict(
        status.observation
    )


def validar_campos_schema(
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
        print(
            f"Advertencia: {nombre} contiene "
            f"campos adicionales: "
            f"{sorted(adicionales)}"
        )


def validar_schema(
    sim: AnyLogicSim,
) -> dict[str, list[str]]:
    schema = sim.schema

    if schema is None:
        raise RuntimeError(
            "Alpyne no pudo obtener el schema "
            "del experimento."
        )

    configuration_fields = set(
        schema.configuration.keys()
    )

    action_fields = set(
        schema.action.keys()
    )

    observation_fields = set(
        schema.observation.keys()
    )

    validar_campos_schema(
        nombre="Configuration",
        encontrados=configuration_fields,
        esperados=(
            EXPECTED_CONFIGURATION_FIELDS
        ),
    )

    validar_campos_schema(
        nombre="Action",
        encontrados=action_fields,
        esperados=(
            EXPECTED_ACTION_FIELDS
        ),
    )

    validar_campos_schema(
        nombre="Observation",
        encontrados=observation_fields,
        esperados=(
            EXPECTED_OBSERVATION_FIELDS
        ),
    )

    return {
        "configuration": sorted(
            configuration_fields
        ),

        "action": sorted(
            action_fields
        ),

        "observation": sorted(
            observation_fields
        ),
    }


# ============================================================
# VALIDACIÓN DE OBSERVACIONES
# ============================================================

def validar_observacion_inicial(
    status: Any,
) -> dict[str, Any]:
    if contiene_estado(
        status,
        EngineState.ERROR,
    ):
        raise RuntimeError(
            "AnyLogic quedó en estado ERROR "
            "durante el reset."
        )

    if not contiene_estado(
        status,
        EngineState.PAUSED,
    ):
        raise RuntimeError(
            "Después del reset se esperaba "
            "EngineState.PAUSED, pero se recibió "
            f"{status.state}."
        )

    observacion = observacion_como_dict(
        status
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
            "La observación inicial indica que "
            "ya se recibió una acción."
        )

    if observacion[
        "ejecucionEnCurso"
    ]:
        raise RuntimeError(
            "La ejecución comenzó antes de "
            "enviar la acción."
        )

    if observacion[
        "ejecucionFinalizada"
    ]:
        raise RuntimeError(
            "La ejecución figura finalizada "
            "antes de enviar la acción."
        )

    if observacion[
        "error"
    ]:
        raise RuntimeError(
            "AnyLogic reportó un error inicial: "
            f"{observacion['mensaje']}"
        )

    return observacion


def validar_observacion_final(
    status: Any,
) -> dict[str, Any]:
    observacion = observacion_como_dict(
        status
    )

    if contiene_estado(
        status,
        EngineState.ERROR,
    ):
        raise RuntimeError(
            "AnyLogic terminó en estado ERROR: "
            f"{status.message}"
        )

    if not contiene_estado(
        status,
        EngineState.FINISHED,
    ):
        raise RuntimeError(
            "Después de la acción se esperaba "
            "EngineState.FINISHED, pero se recibió "
            f"{status.state}."
        )

    if not status.stop:
        raise RuntimeError(
            "La condición terminal del experimento "
            "no devolvió true."
        )

    if observacion[
        "error"
    ]:
        raise RuntimeError(
            "El modelo reportó un error: "
            f"{observacion['mensaje']}"
        )

    if not observacion[
        "accionRecibida"
    ]:
        raise RuntimeError(
            "La acción no quedó registrada."
        )

    if observacion[
        "ejecucionEnCurso"
    ]:
        raise RuntimeError(
            "La ejecución continúa activa después "
            "de finalizar el episodio."
        )

    if not observacion[
        "ejecucionFinalizada"
    ]:
        raise RuntimeError(
            "ejecucionFinalizada continúa en false."
        )

    costo_total = float(
        observacion[
            "costoTotal"
        ]
    )

    if not math.isfinite(
        costo_total
    ):
        raise RuntimeError(
            "El costo total no es finito: "
            f"{costo_total}"
        )

    if costo_total < 0.0:
        raise RuntimeError(
            "El costo total es negativo: "
            f"{costo_total}"
        )

    if observacion[
        "tareasEntregadas"
    ] != 1:
        raise RuntimeError(
            "Se esperaba una tarea entregada, "
            "pero se obtuvieron "
            f"{observacion['tareasEntregadas']}."
        )

    if observacion[
        "tareasNoEntregadas"
    ] != 0:
        raise RuntimeError(
            "Se esperaban cero tareas no entregadas, "
            "pero se obtuvieron "
            f"{observacion['tareasNoEntregadas']}."
        )

    if observacion[
        "viajesTotales"
    ] != 1:
        raise RuntimeError(
            "Se esperaba un viaje, pero se obtuvieron "
            f"{observacion['viajesTotales']}."
        )

    tiempo_simulado = float(
        observacion[
            "tiempoSimuladoMin"
        ]
    )

    if tiempo_simulado >= 59.0:
        raise RuntimeError(
            "El episodio llegó al límite artificial "
            "de 59 minutos en vez de finalizar por "
            "la lógica del modelo."
        )

    return observacion


# ============================================================
# SALIDA
# ============================================================

def guardar_resultado(
    output_path: Path,
    resultado: dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            resultado,
            archivo,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    args = parse_args()

    model_path = resolver_ruta(
        args.model
    )

    output_path = resolver_ruta(
        args.output
    )

    java_path = resolver_java(
        args.java
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "No existe el ZIP exportado: "
            f"{model_path}"
        )

    if model_path.suffix.lower() != ".zip":
        raise ValueError(
            "--model debe apuntar al ZIP exportado "
            "desde el experimento RL."
        )

    print("")
    print(
        "=== HANDSHAKE ANYLOGIC–ALPYNE ==="
    )

    print(
        f"Modelo: {model_path}"
    )

    print(
        f"Java: {java_path}"
    )

    print(
        f"Seed: {args.seed}"
    )

    print("")
    print(
        "Iniciando servidor Java..."
    )

    sim = AnyLogicSim(
        model_path=str(
            model_path
        ),

        java_exe=str(
            java_path
        ),

        auto_lock=True,

        auto_finish=True,

        py_log_level=True,

        java_log_level=True,

        log_id="phase10b",

        lock_defaults={
            "flag": (
                EngineState.PAUSED
                | EngineState.FINISHED
                | EngineState.ERROR
            ),

            "timeout": 120,
        },

        max_server_await_time=30.0,
    )

    schema_resultado = validar_schema(
        sim
    )

    print("")
    print(
        "Schema validado:"
    )

    print(
        "  Configuration: "
        f"{schema_resultado['configuration']}"
    )

    print(
        "  Action: "
        f"{schema_resultado['action']}"
    )

    print(
        "  Observation: "
        f"{schema_resultado['observation']}"
    )

    print("")
    print(
        "Enviando Configuration..."
    )

    status_inicial = exigir_status(
        sim.reset(
            seedEjecucion=(
                args.seed
            )
        ),

        contexto="reset",
    )

    observacion_inicial = (
        validar_observacion_inicial(
            status_inicial
        )
    )

    print(
        "Estado inicial: "
        f"{status_inicial.state}"
    )

    print(
        "Mensaje inicial: "
        f"{observacion_inicial['mensaje']}"
    )

    print("")
    print(
        "Enviando Action accionCodigo=1..."
    )

    status_final = exigir_status(
        sim.take_action(
            accionCodigo=1
        ),

        contexto="take_action",
    )

    observacion_final = (
        validar_observacion_final(
            status_final
        )
    )

    resultado = {
        "modelo": str(
            model_path
        ),

        "java": str(
            java_path
        ),

        "seed": (
            args.seed
        ),

        "schema": (
            schema_resultado
        ),

        "estado_inicial": str(
            status_inicial.state
        ),

        "observacion_inicial": (
            observacion_inicial
        ),

        "estado_final": str(
            status_final.state
        ),

        "stop_condition": bool(
            status_final.stop
        ),

        "observacion_final": (
            observacion_final
        ),
    }

    guardar_resultado(
        output_path=output_path,
        resultado=resultado,
    )

    print("")
    print(
        "=== RESULTADO FINAL ==="
    )

    print(
        "Estado: "
        f"{status_final.state}"
    )

    print(
        "Mensaje: "
        f"{observacion_final['mensaje']}"
    )

    print(
        "Tiempo simulado: "
        f"{observacion_final['tiempoSimuladoMin']:.3f} min"
    )

    print(
        "Tareas entregadas: "
        f"{observacion_final['tareasEntregadas']}"
    )

    print(
        "Tareas no entregadas: "
        f"{observacion_final['tareasNoEntregadas']}"
    )

    print(
        "Viajes totales: "
        f"{observacion_final['viajesTotales']}"
    )

    print(
        "Costo total: "
        f"{observacion_final['costoTotal']:.6f}"
    )

    print("")
    print(
        "HANDSHAKE ANYLOGIC–ALPYNE: OK"
    )

    print(
        f"Resultado JSON: {output_path}"
    )


if __name__ == "__main__":
    main()