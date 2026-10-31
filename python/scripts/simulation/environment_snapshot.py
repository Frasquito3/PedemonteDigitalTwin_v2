"""Registro del hardware y software de los estudios experimentales."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import metadata as package_metadata
from typing import Any


def _version_paquete(nombre: str) -> str | None:
    """Obtiene la version instalada de una dependencia.

    Args:
        nombre: Nombre del paquete Python.

    Returns:
        Version instalada o None si no se encuentra el paquete.
    """
    try:
        return package_metadata.version(nombre)
    except package_metadata.PackageNotFoundError:
        return None


def _version_java() -> str | None:
    """Obtiene la version Java accesible desde PATH.

    Returns:
        Version informada por Java o None si no esta disponible.
    """
    java = shutil.which("java")
    if java is None:
        return None

    try:
        proceso = subprocess.run(
            [java, "-version"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=8,
            check=False,
        )
        lineas = (proceso.stderr or proceso.stdout).strip().splitlines()
        return lineas[0] if lineas else None
    except (OSError, subprocess.SubprocessError):
        return None


def _hardware_windows() -> dict[str, Any]:
    """Consulta CPU, RAM y GPU sin recopilar identificadores del equipo.

    Returns:
        Propiedades de hardware disponibles en Windows.
    """
    if sys.platform != "win32":
        return {}

    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return {}

    consulta = (
        "$cpu = Get-CimInstance Win32_Processor | "
        "Select-Object -First 1; "
        "$pc = Get-CimInstance Win32_ComputerSystem; "
        "$gpu = @(Get-CimInstance Win32_VideoController | "
        "ForEach-Object { $_.Name }); "
        "[PSCustomObject]@{ "
        "cpu_modelo = $cpu.Name; "
        "nucleos_fisicos = $cpu.NumberOfCores; "
        "ram_total_bytes = $pc.TotalPhysicalMemory; "
        "gpu_modelos = $gpu "
        "} | ConvertTo-Json -Compress -Depth 3"
    )

    try:
        proceso = subprocess.run(
            [
                str(powershell),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                consulta,
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=12,
            check=True,
        )
        datos = json.loads(proceso.stdout)

        if not isinstance(datos, dict):
            return {}

        salida: dict[str, Any] = {}

        if datos.get("cpu_modelo"):
            salida["cpu_modelo"] = str(datos["cpu_modelo"]).strip()

        if datos.get("nucleos_fisicos") is not None:
            salida["nucleos_fisicos"] = int(datos["nucleos_fisicos"])

        if datos.get("ram_total_bytes") is not None:
            ram = int(datos["ram_total_bytes"])
            salida["ram_total_bytes"] = ram
            salida["ram_total_gib"] = round(ram / (1024**3), 2)

        gpu = datos.get("gpu_modelos") or []
        if isinstance(gpu, str):
            gpu = [gpu]

        salida["gpu_modelos"] = [str(nombre).strip() for nombre in gpu if str(nombre).strip()]

        return salida

    except (
        OSError,
        ValueError,
        TypeError,
        subprocess.SubprocessError,
    ):
        return {}


def capturar_entorno_ejecucion() -> dict[str, Any]:
    """Captura el entorno computacional al iniciar una sesion.

    Returns:
        Diccionario con hardware, sistema operativo, versiones de software
        y variables relevantes para documentar reproducibilidad.

    Notes:
        No incluye usuario, hostname ni identificadores de hardware.
        La captura no garantiza determinismo numerico absoluto.
    """
    ahora = datetime.now().astimezone()

    hardware: dict[str, Any] = {
        "cpu_modelo": platform.processor() or None,
        "nucleos_logicos": os.cpu_count(),
        "nucleos_fisicos": None,
        "ram_total_bytes": None,
        "ram_total_gib": None,
        "gpu_modelos": [],
    }

    hardware.update(_hardware_windows())

    return {
        "version": 1,
        "fecha_captura": ahora.isoformat(timespec="seconds"),
        "hardware": hardware,
        "sistema_operativo": {
            "nombre": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "arquitectura": platform.machine(),
            "zona_horaria": str(ahora.tzinfo),
        },
        "software": {
            "python": platform.python_version(),
            "implementacion_python": platform.python_implementation(),
            "java_en_path": _version_java(),
            "dependencias": {
                nombre: _version_paquete(nombre)
                for nombre in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "torch",
                    "stable-baselines3",
                    "gymnasium",
                    "anylogic-alpyne",
                )
            },
        },
        "configuracion_entorno": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "observaciones": [
            "La version Java corresponde al ejecutable en PATH.",
            "La GPU detectada no implica que se utilice.",
            "Este registro no garantiza determinismo absoluto.",
        ],
    }
