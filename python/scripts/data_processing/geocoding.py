from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.location import Location

PYTHON_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = PYTHON_ROOT / "data" / "processed" / "extraccion" / "direcciones.json"
DEFAULT_OUTPUT_FILE = PYTHON_ROOT / "data" / "processed" / "geocoding" / "direcciones_geolocalizadas.csv"
DEFAULT_DISCARDED_FILE = PYTHON_ROOT / "data" / "processed" / "geocoding" / "direcciones_descartadas.csv"
DEFAULT_PROGRESS_FILE = PYTHON_ROOT / "data" / "processed" / "geocoding" / "progreso_geo.json"

CIUDADES_PRIORITARIAS = [
    "Granadero Baigorria",
    "Rosario",
    "Ibarlucea",
    "Capitan Bermudez",
    "Fray Luis Beltran",
]

ESPERA_ENTRE_REQUESTS = 1.2  # Nominatim exige máximo 1 req/seg


def normalizar_ciudad_osm(osm_address: str, ciudad_solicitada: str) -> str:
    """Extrae y normaliza el municipio detectado en el texto devuelto por OpenStreetMap."""
    m = re.search(r"Municipio de ([^,]+)", osm_address)
    if m:
        c = m.group(1).strip()
        if "Ybarlucea" in c or "Ibarlucea" in c:
            return "Ibarlucea"
        if "Capitán" in c or "Capitan" in c:
            return "Capitán Bermúdez"
        if "Beltrán" in c or "Beltran" in c:
            return "Fray Luis Beltrán"
        return c

    if "Ybarlucea" in osm_address or "Ibarlucea" in osm_address:
        return "Ibarlucea"
    if "Capitán Bermúdez" in osm_address or "Capitan Bermudez" in osm_address:
        return "Capitán Bermúdez"
    if "Fray Luis Beltrán" in osm_address or "Fray Luis Beltran" in osm_address:
        return "Fray Luis Beltrán"
    if "Granadero Baigorria" in osm_address:
        return "Granadero Baigorria"
    if "Rosario" in osm_address:
        return "Rosario"
    if "Funes" in osm_address:
        return "Funes"
    if "Roldán" in osm_address or "Roldan" in osm_address:
        return "Roldán"
    if "San Lorenzo" in osm_address:
        return "San Lorenzo"

    return ciudad_solicitada


def geolocalizar_direccion(
    geolocator: Nominatim,
    calle: str,
    altura: str | int | None,
    ciudades: list[str] = CIUDADES_PRIORITARIAS,
    espera: float = ESPERA_ENTRE_REQUESTS,
) -> dict[str, Any] | None:
    """Geocodifica un par calle-altura iterando por ciudades prioritarias en Santa Fe."""
    str_altura = str(altura).strip() if altura is not None else ""
    direccion_base = f"{calle} {str_altura}".strip() if str_altura else str(calle).strip()

    for ciudad in ciudades:
        query = f"{direccion_base}, {ciudad}, Santa Fe, Argentina"
        try:
            time.sleep(espera)
            location = geolocator.geocode(query, addressdetails=True, timeout=10)
            if isinstance(location, Location):
                detalles = location.raw.get("address", {}) if isinstance(location.raw, dict) else {}
                barrio = detalles.get(
                    "suburb",
                    detalles.get(
                        "neighbourhood",
                        detalles.get("residential", "No especificado"),
                    ),
                )
                osm = str(location.address)
                ciudad_real = normalizar_ciudad_osm(osm, ciudad)

                return {
                    "latitud": location.latitude,
                    "longitud": location.longitude,
                    "ciudad_detectada": ciudad_real,
                    "barrio": barrio,
                    "direccion_osm": location.address,
                }
        except Exception as e:
            print(f"  [!] Error consultando '{query}': {e}")

    return None


def cargar_progreso(archivo_progreso: Path) -> dict[str, Any]:
    """Carga el estado del progreso previo si el archivo existe."""
    if archivo_progreso.exists():
        with open(archivo_progreso, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"procesados": 0, "validos": [], "descartados": []}


def guardar_progreso(
    archivo_progreso: Path,
    procesados: int,
    validos: list[dict[str, Any]],
    descartados: list[dict[str, Any]],
) -> None:
    """Persiste el progreso actual de geocodificación."""
    archivo_progreso.parent.mkdir(parents=True, exist_ok=True)
    with open(archivo_progreso, "w", encoding="utf-8") as f:
        json.dump(
            {"procesados": procesados, "validos": validos, "descartados": descartados},
            f,
            ensure_ascii=False,
        )


def ejecutar_geocodificacion(
    archivo_entrada: Path = DEFAULT_INPUT_FILE,
    archivo_salida: Path = DEFAULT_OUTPUT_FILE,
    archivo_descartes: Path = DEFAULT_DISCARDED_FILE,
    archivo_progreso: Path = DEFAULT_PROGRESS_FILE,
    espera: float = ESPERA_ENTRE_REQUESTS,
    user_agent: str = "ruteo_logistica_reparto_v2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el ciclo de geocodificación sobre las direcciones JSON con recuperación de fallos."""
    if not archivo_entrada.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {archivo_entrada}")

    with open(archivo_entrada, "r", encoding="utf-8") as f:
        datos_crudos: list[dict[str, Any]] = json.load(f)

    total = len(datos_crudos)
    geolocator = Nominatim(user_agent=user_agent)

    estado = cargar_progreso(archivo_progreso)
    inicio = estado["procesados"]
    resultados_validos: list[dict[str, Any]] = estado["validos"]
    descartados: list[dict[str, Any]] = estado["descartados"]

    if inicio > 0:
        print(
            f"[Reanudando] {inicio}/{total} procesadas, "
            f"{len(resultados_validos)} encontradas, {len(descartados)} descartadas."
        )
    else:
        print(f"Iniciando geocodificación de {total} direcciones...")

    for i, item in enumerate(datos_crudos[inicio:], start=inicio + 1):
        calle = item.get("calle", "")
        altura = item.get("altura", "")

        print(f"[{i}/{total}] Buscando: {calle} {altura}...", end=" ", flush=True)
        resultado_geo = geolocalizar_direccion(
            geolocator=geolocator,
            calle=calle,
            altura=altura,
            espera=espera,
        )

        if resultado_geo:
            fila_completa = {**item, **resultado_geo}
            resultados_validos.append(fila_completa)
            print(f"OK -> {resultado_geo['ciudad_detectada']} / {resultado_geo['barrio']}")
        else:
            descartados.append(item)
            print("[X] No encontrada")

        guardar_progreso(archivo_progreso, i, resultados_validos, descartados)

    df_validos = pd.DataFrame(resultados_validos)
    df_descartados = pd.DataFrame(descartados)

    archivo_salida.parent.mkdir(parents=True, exist_ok=True)
    archivo_descartes.parent.mkdir(parents=True, exist_ok=True)

    df_validos.to_csv(archivo_salida, index=False, encoding="utf-8-sig")
    if not df_descartados.empty:
        df_descartados.to_csv(archivo_descartes, index=False, encoding="utf-8-sig")

    if archivo_progreso.exists():
        archivo_progreso.unlink()

    print(f"\n[OK] Geocodificación finalizada:")
    print(f"     Encontradas: {len(df_validos)} en {archivo_salida}")
    print(f"     Descartadas: {len(df_descartados)} en {archivo_descartes}")
    return df_validos, df_descartados


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Geocodificación de direcciones con Nominatim (OSM)")
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="Archivo JSON de direcciones")
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="CSV de salida geolocalizadas")
    p.add_argument("--discarded-file", type=Path, default=DEFAULT_DISCARDED_FILE, help="CSV de direcciones descartadas")
    p.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS_FILE, help="JSON de progreso")
    p.add_argument("--delay", type=float, default=ESPERA_ENTRE_REQUESTS, help="Espera en segundos entre peticiones")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        ejecutar_geocodificacion(
            archivo_entrada=args.input_file.resolve(),
            archivo_salida=args.output_file.resolve(),
            archivo_descartes=args.discarded_file.resolve(),
            archivo_progreso=args.progress_file.resolve(),
            espera=args.delay,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
