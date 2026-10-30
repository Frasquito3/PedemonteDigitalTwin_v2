"""Archivado y versionado automático de hojas de ruta en Excel.

Este módulo gestiona la persistencia de planes en la carpeta `exports/` del proyecto,
generando nombres con marca temporal y sufijos incrementales para evitar sobrescrituras.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PlanTurno
from planner.integration.route_excel_export import exportar_hoja_ruta_excel
from planner.routing.travel import MatrizViaje

__all__ = [
    "exportar_hoja_ruta_archivada",
]


def _obtener_ruta_disponible(carpeta: Path, fecha_hora: datetime) -> Path:
    marca = fecha_hora.strftime("%Y-%m-%d_%H-%M-%S")
    nombre_base = f"hoja_ruta_{marca}"

    for numero in range(1, 10001):
        sufijo = "" if numero == 1 else f"_{numero:02d}"
        ruta = carpeta / f"{nombre_base}{sufijo}.xlsx"
        if not ruta.exists():
            return ruta

    raise RuntimeError("No se encontró un nombre disponible para exportar la hoja de ruta.")


def exportar_hoja_ruta_archivada(
    instancia: InstanciaTurno,
    plan: PlanTurno,
    matriz: MatrizViaje,
    configuracion: ConfiguracionPlanificacion,
    *,
    modo_solicitado: str | None = None,
    raiz_proyecto: str | Path | None = None,
    fecha_hora: datetime | None = None,
) -> Path:
    """Exporta la hoja de ruta estimada a la carpeta exports/ del proyecto.

    Args:
        instancia: Instancia del turno planificado.
        plan: Plan de ruteo obtenido.
        matriz: Matriz de distancias y tiempos viales calculada.
        configuracion: Parámetros operativos y costos.
        modo_solicitado: Nombre del algoritmo o modo que generó el plan.
        raiz_proyecto: Directorio raíz opcional del proyecto.
        fecha_hora: Timestamp opcional para el nombre del archivo.

    Returns:
        Ruta absoluta al archivo Excel generado.
    """
    if raiz_proyecto is None:
        raiz = Path(__file__).resolve().parents[3]
    else:
        raiz = Path(raiz_proyecto).expanduser().resolve()

    carpeta = raiz / "exports"
    carpeta.mkdir(parents=True, exist_ok=True)

    momento = fecha_hora if fecha_hora is not None else datetime.now()
    destino = _obtener_ruta_disponible(carpeta, momento)

    return exportar_hoja_ruta_excel(
        instancia,
        plan,
        matriz,
        configuracion,
        destino,
        modo_solicitado=modo_solicitado,
    )
