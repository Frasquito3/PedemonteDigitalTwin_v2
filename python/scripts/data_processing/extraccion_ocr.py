from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Resolución de rutas relativas al proyecto
PYTHON_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PYTHON_ROOT.parent

# Cargar automáticamente variables de entorno desde .env si existe
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
load_dotenv()

DEFAULT_INPUT_DIR = PYTHON_ROOT / "data" / "raw" / "hojas_pedidos"
DEFAULT_OUTPUT_FILE = PYTHON_ROOT / "data" / "processed" / "extraccion" / "direcciones.json"
DEFAULT_PROGRESS_FILE = PYTHON_ROOT / "data" / "processed" / "extraccion" / "progreso.json"
DEFAULT_MODEL = os.getenv("GEMINI_OCR_MODEL", "gemini-2.5-flash")

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

PROMPT_OCR = """
Actua como un OCR avanzado y experto en extraccion de datos estructurados.
Analiza la imagen adjunta, que contiene una lista manuscrita de direcciones de reparto.
Estas direcciones pertenecen PRINCIPALMENTE a Granadero Baigorria, y tambien a Ibarlucea,
Rosario, Capitan Bermudez y Fray Luis Beltran.

Instrucciones:
1. Extrae unicamente el nombre de la calle y la altura (numero).
2. Ignora nombres de personas, montos de dinero, simbolos raros y texto tachado.
3. Si una calle esta abreviada, intenta inferir su nombre completo segun el contexto.

Devuelve el resultado ESTRICTAMENTE como una lista JSON valida con este formato exacto,
sin texto adicional ni bloques de codigo markdown:
[{"calle": "Nombre", "altura": "123"}, {"calle": "Nombre 2", "altura": "456"}]
""".strip()


def obtener_cliente_genai(api_key: str | None = None) -> genai.Client:
    """Inicializa y devuelve el cliente de Google GenAI validando la existencia de la API Key.

    Args:
        api_key: Clave opcional. Si no se provee, se busca en GEMINI_API_KEY o GOOGLE_API_KEY.

    Returns:
        genai.Client configurado.

    Raises:
        ValueError: Si no se encuentra una API Key válida.
    """
    clave = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not clave or clave.strip() in ("", "API_KEY", "tu_api_key_aqui"):
        raise ValueError(
            "No se encontró una API Key válida de Google AI Studio. "
            "Por favor definí 'GEMINI_API_KEY' o 'GOOGLE_API_KEY' en tu archivo .env "
            "en la raíz del repositorio (usá .env.example como plantilla)."
        )
    return genai.Client(api_key=clave)


def _escribir_json_atomico(
    archivo_destino: Path,
    datos: Any,
    indent: int | None = 2,
) -> None:
    """Escribe un archivo JSON de forma atómica mediante archivo temporal y reemplazo."""
    archivo_destino.parent.mkdir(parents=True, exist_ok=True)
    fd, ruta_tmp_str = tempfile.mkstemp(
        prefix=f".{archivo_destino.stem}_",
        suffix=".tmp",
        dir=str(archivo_destino.parent),
        text=True,
    )
    ruta_tmp = Path(ruta_tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(ruta_tmp, archivo_destino)
    except BaseException:
        if ruta_tmp.exists():
            try:
                ruta_tmp.unlink()
            except OSError:
                pass
        raise


def cargar_progreso(archivo_progreso: Path) -> set[str]:
    """Carga el conjunto de nombres de archivos de imagen ya procesados."""
    if archivo_progreso.exists():
        with open(archivo_progreso, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def guardar_progreso(archivo_progreso: Path, procesadas: set[str]) -> None:
    """Guarda el conjunto de imágenes procesadas de forma atómica."""
    _escribir_json_atomico(archivo_progreso, sorted(list(procesadas)), indent=2)


def cargar_resultado(archivo_salida: Path) -> list[dict[str, Any]]:
    """Carga los registros acumulados de direcciones extraídas."""
    if archivo_salida.exists():
        with open(archivo_salida, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def guardar_resultado(archivo_salida: Path, datos: list[dict[str, Any]]) -> None:
    """Guarda la lista acumulada de direcciones extraídas de forma atómica."""
    _escribir_json_atomico(archivo_salida, datos, indent=4)


def extraer_direcciones(
    ruta_imagen: Path,
    client: genai.Client,
    modelo: str = DEFAULT_MODEL,
    max_reintentos: int = 4,
    espera_base: int = 10,
) -> list[dict[str, str]] | None:
    """Envía una imagen al modelo Gemini para extraer direcciones manuscritas.

    Aplica reintentos con backoff exponencial ante saturación (503) o cuota (429).

    Args:
        ruta_imagen: Ruta al archivo de imagen.
        client: Cliente inicializado de genai.
        modelo: Nombre del modelo a utilizar.
        max_reintentos: Cantidad máxima de reintentos por imagen.
        espera_base: Segundos base de espera para backoff.

    Returns:
        Lista de diccionarios `{"calle": ..., "altura": ...}`, lista vacía si falló
        el parseo sintáctico no recuperable, o None en caso de error de red/cuota.
    """
    ext = ruta_imagen.suffix.lower()
    mime_type = MIME_TYPES.get(ext, "image/jpeg")

    with open(ruta_imagen, "rb") as f:
        imagen_bytes = f.read()

    espera = espera_base
    for intento in range(1, max_reintentos + 1):
        try:
            respuesta = client.models.generate_content(
                model=modelo,
                contents=[
                    types.Part.from_bytes(data=imagen_bytes, mime_type=mime_type),
                    PROMPT_OCR,
                ],
            )
            texto_json = respuesta.text.strip() if respuesta.text else "[]"
            texto_json = texto_json.replace("```json", "").replace("```", "").strip()
            datos = json.loads(texto_json)
            if isinstance(datos, list):
                return datos
            return []

        except Exception as e:
            mensaje = str(e)
            if "503" in mensaje or "UNAVAILABLE" in mensaje:
                if intento < max_reintentos:
                    print(f"  [503] Servidor saturado. Reintento {intento}/{max_reintentos - 1} en {espera}s...")
                    time.sleep(espera)
                    espera *= 2
                else:
                    print(f"  [ERROR] Fallo definitivo tras {max_reintentos} intentos: {mensaje[:120]}")
                    return None
            elif "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje:
                if intento < max_reintentos:
                    print(f"  [429] Cuota agotada. Esperando {espera}s ({intento}/{max_reintentos - 1})...")
                    time.sleep(espera)
                    espera *= 2
                else:
                    print(f"  [429] Cuota agotada definitivamente.")
                    return None
            elif "JSON" in mensaje or "json" in mensaje:
                print(f"  [!] Error parseando JSON: {mensaje[:120]}")
                return []
            else:
                print(f"  [ERROR] {mensaje[:120]}")
                return None

    return None


def ejecutar_pipeline_extraccion(
    carpeta_imagenes: Path = DEFAULT_INPUT_DIR,
    archivo_salida: Path = DEFAULT_OUTPUT_FILE,
    archivo_progreso: Path = DEFAULT_PROGRESS_FILE,
    modelo: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Procesa el lote completo de imágenes y almacena resultados con persistencia de estado."""
    client = obtener_cliente_genai(api_key=api_key)

    if not carpeta_imagenes.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {carpeta_imagenes}")

    imagenes_procesadas = cargar_progreso(archivo_progreso)
    todas_las_direcciones = cargar_resultado(archivo_salida)

    if imagenes_procesadas:
        print(
            f"[Reanudando] {len(imagenes_procesadas)} imágenes ya procesadas, "
            f"{len(todas_las_direcciones)} direcciones guardadas."
        )

    archivos = sorted([
        f for f in carpeta_imagenes.iterdir()
        if f.is_file() and f.suffix.lower() in MIME_TYPES
    ])

    total = len(archivos)
    print(f"Encontradas {total} imágenes en {carpeta_imagenes}")

    for i, ruta_archivo in enumerate(archivos, 1):
        nombre = ruta_archivo.name
        if nombre in imagenes_procesadas:
            print(f"[{i}/{total}] Saltando (ya procesada): {nombre}")
            continue

        print(f"[{i}/{total}] Procesando: {nombre} con {modelo}...")
        datos = extraer_direcciones(ruta_archivo, client=client, modelo=modelo)

        if datos is None:
            print("       -> [PENDIENTE] Se reintentará en la próxima ejecución.")
            continue

        todas_las_direcciones.extend(datos)
        imagenes_procesadas.add(nombre)

        guardar_resultado(archivo_salida, todas_las_direcciones)
        guardar_progreso(archivo_progreso, imagenes_procesadas)

        print(f"       -> {len(datos)} direcciones encontradas (total acumulado: {len(todas_las_direcciones)})")

    print(f"\n[OK] Extracción completada: {len(todas_las_direcciones)} direcciones totales.")
    print(f"     Resultado guardado en: {archivo_salida}")
    return todas_las_direcciones


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extracción OCR de hojas de pedidos manuscritas con Gemini")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Carpeta de imágenes de entrada")
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Archivo JSON de salida")
    p.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS_FILE, help="Archivo JSON de progreso")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Nombre del modelo Gemini")
    p.add_argument("--api-key", type=str, default=None, help="API Key explícita (opcional)")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        ejecutar_pipeline_extraccion(
            carpeta_imagenes=args.input_dir.resolve(),
            archivo_salida=args.output_file.resolve(),
            archivo_progreso=args.progress_file.resolve(),
            modelo=args.model,
            api_key=args.api_key,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
