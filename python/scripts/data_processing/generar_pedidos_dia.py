"""
Generador de pedidos para nuevos días a partir de datos históricos en CSV.

Analiza la información espacial y de demanda de pedidos anteriores y genera
nuevos escenarios/días de pedidos (N veces) con atributos operativos calibrados
(unidades, volcador, ventanas horarias, turnos) compatibles con AnyLogic y el optimizador.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import time
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import pandas as pd

# Asegurar importación de módulos internos del proyecto
PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_DISPONIBLE = True
except ImportError:
    OPENPYXL_DISPONIBLE = False

# Rutas predeterminadas
DEFAULT_HISTORICO_CSV = PYTHON_ROOT / "data" / "processed" / "demanda_geografica.csv"
DEFAULT_FALLBACK_CSV = PYTHON_ROOT / "data" / "processed" / "geocoding" / "direcciones_geolocalizadas.csv"
DEFAULT_OUTPUT_DIR = PYTHON_ROOT / "data" / "processed" / "pedidos_generados"

# Horarios de turnos (en minutos desde las 00:00 y formato HH:MM)
HORARIOS_TURNOS = {
    "MANANA": {
        "inicio_min": 7 * 60 + 30,  # 07:30 -> 450
        "fin_min": 12 * 60,         # 12:00 -> 720
        "inicio_str": "07:30",
        "fin_str": "12:00",
    },
    "TARDE": {
        "inicio_min": 14 * 60,       # 14:00 -> 840
        "fin_min": 17 * 60,         # 17:00 -> 1020
        "inicio_str": "14:00",
        "fin_str": "17:00",
    },
}

ENCABEZADOS_PEDIDOS = [
    "pedido_id",
    "cliente",
    "direccion",
    "barrio",
    "ciudad",
    "unidades",
    "latitud",
    "longitud",
    "requiere_volcador",
    "unidades_volcador",
    "tiene_ventana",
    "hora_desde",
    "hora_hasta",
    "turno",
    "observaciones",
]


@dataclass(frozen=True)
class UbicacionHistorica:
    """Representa una ubicación histórica analizada desde el CSV."""
    calle: str
    altura: str
    barrio: str
    ciudad: str
    latitud: float
    longitud: float
    peso_frecuencia: float
    registro_id: str = ""

    @property
    def direccion_completa(self) -> str:
        if self.altura:
            return f"{self.calle} {self.altura}"
        return self.calle


@dataclass
class PedidoGenerado:
    """Pedido simulado generado para un día de operación."""
    pedido_id: str
    cliente: str
    direccion: str
    barrio: str
    ciudad: str
    unidades: int
    latitud: float
    longitud: float
    requiere_volcador: str  # "SI" o "NO"
    unidades_volcador: int
    tiene_ventana: str      # "SI" o "NO"
    hora_desde: str         # "HH:MM"
    hora_hasta: str         # "HH:MM"
    turno: str              # "MANANA" o "TARDE"
    observaciones: str

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def cargar_y_analizar_historico(ruta_csv: Path) -> list[UbicacionHistorica]:
    """
    Lee y analiza el CSV de pedidos históricos, extrayendo ubicaciones y calculando pesos de frecuencia.
    """
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo histórico en: {ruta_csv}")

    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")

    # Mapeo flexible de nombres de columnas
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    def obtener_columna(nombres_posibles: list[str], default: str = "") -> str:
        for n in nombres_posibles:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return default

    col_calle = obtener_columna(["calle", "street", "direccion"])
    col_altura = obtener_columna(["altura", "number", "numero"])
    col_barrio = obtener_columna(["barrio", "neighborhood", "zona"])
    col_ciudad = obtener_columna(["ciudad", "ciudad_detectada", "city", "municipio"])
    col_lat = obtener_columna(["latitud", "lat"])
    col_lon = obtener_columna(["longitud", "lon", "lng"])
    col_freq = obtener_columna(["frecuencia_direccion_fuente", "frecuencia", "frec", "cantidad_pedidos"])
    col_id = obtener_columna(["registro_id", "id", "clave_direccion_fuente"])

    if not col_lat or not col_lon:
        raise ValueError(
            f"El CSV {ruta_csv} debe contener columnas de coordenadas (latitud y longitud)."
        )

    ubicaciones: list[UbicacionHistorica] = []

    for _, fila in df.iterrows():
        try:
            lat = float(fila[col_lat])
            lon = float(fila[col_lon])
        except (ValueError, TypeError):
            continue

        calle = str(fila.get(col_calle, "")).strip() if col_calle else "Calle s/n"
        altura = str(fila.get(col_altura, "")).strip() if col_altura else ""
        if altura.lower() in ("nan", "none", "0"):
            altura = ""

        barrio = str(fila.get(col_barrio, "")).strip() if col_barrio else "No especificado"
        if barrio.lower() in ("nan", "none", ""):
            barrio = "No especificado"

        ciudad = str(fila.get(col_ciudad, "")).strip() if col_ciudad else "Granadero Baigorria"
        if ciudad.lower() in ("nan", "none", ""):
            ciudad = "Granadero Baigorria"

        peso = 1.0
        if col_freq and pd.notna(fila.get(col_freq)):
            try:
                peso = max(1.0, float(fila[col_freq]))
            except (ValueError, TypeError):
                peso = 1.0

        reg_id = str(fila.get(col_id, "")).strip() if col_id else ""

        ubicaciones.append(
            UbicacionHistorica(
                calle=calle,
                altura=altura,
                barrio=barrio,
                ciudad=ciudad,
                latitud=lat,
                longitud=lon,
                peso_frecuencia=peso,
                registro_id=reg_id,
            )
        )

    if not ubicaciones:
        raise ValueError(f"No se pudieron extraer ubicaciones válidas desde: {ruta_csv}")

    return ubicaciones


def _minutos_a_hora_str(minutos: int) -> str:
    """Convierte minutos del día a string formato 'HH:MM'."""
    h = minutos // 60
    m = minutos % 60
    return f"{h:02d}:{m:02d}"


def generar_pedidos_para_un_dia(
    ubicaciones: Sequence[UbicacionHistorica],
    cantidad_pedidos: int,
    turno: str = "MANANA",
    dia_numero: int = 1,
    prob_volcador: float = 0.15,
    prob_ventana: float = 0.65,
    prob_pedido_grande: float = 0.10,
    capacidad_camion: int = 8,
    rng: random.Random | None = None,
) -> list[PedidoGenerado]:
    """
    Genera la lista de pedidos para un día de operación específico,
    muestreando las ubicaciones históricas según su peso de frecuencia y asignando
    atributos operacionales coherentes con la logística de Pedemonte.
    """
    if rng is None:
        rng = random.Random()

    pesos = [u.peso_frecuencia for u in ubicaciones]

    # Muestreo con reemplazo ponderado por frecuencia histórica
    ubicaciones_seleccionadas = rng.choices(ubicaciones, weights=pesos, k=cantidad_pedidos)

    horario = HORARIOS_TURNOS.get(turno, HORARIOS_TURNOS["MANANA"])
    inicio_turno_min = horario["inicio_min"]
    fin_turno_min = horario["fin_min"]

    pedidos: list[PedidoGenerado] = []

    for idx, ubi in enumerate(ubicaciones_seleccionadas, start=1):
        pedido_id = f"PED-D{dia_numero:02d}-{idx:02d}"
        cliente = f"Cliente {ubi.barrio} #{idx:02d}"

        # Unidades de carga
        if rng.random() < prob_pedido_grande:
            # Pedido grande (> capacidad camión, ej. 9 a 16 bolsas/m3)
            unidades = rng.randint(capacidad_camion + 1, capacidad_camion * 2)
        else:
            # Pedido estándar dentro de la capacidad de un camión (1 a 8)
            unidades = rng.randint(1, capacidad_camion)

        # Requerimiento de volcador
        requiere_volcador_bool = rng.random() < prob_volcador
        requiere_volcador_str = "SI" if requiere_volcador_bool else "NO"
        unidades_volcador = unidades if requiere_volcador_bool else 0

        # Ventana horaria de entrega
        tiene_ventana_bool = rng.random() < prob_ventana
        tiene_ventana_str = "SI" if tiene_ventana_bool else "NO"

        if tiene_ventana_bool:
            # Ancho de ventana entre 45 y 120 minutos
            duracion_ventana = rng.randint(45, 120)
            max_inicio = fin_turno_min - duracion_ventana
            if max_inicio <= inicio_turno_min:
                desde_min = inicio_turno_min
                hasta_min = fin_turno_min
            else:
                desde_min = rng.randint(inicio_turno_min, max_inicio)
                # Redondear a múltiplos de 15 minutos para realismo humano
                desde_min = (desde_min // 15) * 15
                hasta_min = min(fin_turno_min, desde_min + duracion_ventana)
                hasta_min = ((hasta_min + 14) // 15) * 15
                hasta_min = min(fin_turno_min, max(hasta_min, desde_min + 30))

            hora_desde = _minutos_a_hora_str(desde_min)
            hora_hasta = _minutos_a_hora_str(hasta_min)
        else:
            hora_desde = horario["inicio_str"]
            hora_hasta = horario["fin_str"]

        observaciones = f"Histórico ref={ubi.registro_id or 'Demanda'}; Día {dia_numero}"

        pedidos.append(
            PedidoGenerado(
                pedido_id=pedido_id,
                cliente=cliente,
                direccion=ubi.direccion_completa,
                barrio=ubi.barrio,
                ciudad=ubi.ciudad,
                unidades=unidades,
                latitud=round(ubi.latitud, 6),
                longitud=round(ubi.longitud, 6),
                requiere_volcador=requiere_volcador_str,
                unidades_volcador=unidades_volcador,
                tiene_ventana=tiene_ventana_str,
                hora_desde=hora_desde,
                hora_hasta=hora_hasta,
                turno=turno,
                observaciones=observaciones,
            )
        )

    return pedidos


def exportar_a_csv(pedidos: list[PedidoGenerado], ruta_archivo: Path) -> None:
    """Exporta los pedidos generados a un archivo CSV estándar UTF-8."""
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_archivo, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=ENCABEZADOS_PEDIDOS)
        escritor.writeheader()
        for p in pedidos:
            escritor.writerow(p.a_dict())


def exportar_a_excel(pedidos: list[PedidoGenerado], ruta_archivo: Path) -> None:
    """
    Exporta los pedidos generados a un archivo Excel (.xlsx) estructurado
    y 100% compatible con AnyLogic y el importador `importar_pedidos_excel`.
    """
    if not OPENPYXL_DISPONIBLE:
        raise RuntimeError("openpyxl no está instalado; no se puede exportar a formato .xlsx.")

    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    if ws is None:
        ws = wb.create_sheet("Pedidos")
    else:
        ws.title = "Pedidos"

    # Encabezado estándar en fila 1
    ws.append(ENCABEZADOS_PEDIDOS)

    # Estilos para el encabezado
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="17324D", end_color="17324D", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")

    for col_idx in range(1, len(ENCABEZADOS_PEDIDOS) + 1):
        celda = ws.cell(row=1, column=col_idx)
        celda.font = header_font
        celda.fill = header_fill
        celda.alignment = center_align

    # Agregar filas de datos
    for p in pedidos:
        fila = [getattr(p, col) for col in ENCABEZADOS_PEDIDOS]
        ws.append(fila)

    # Ajustar ancho de columnas automáticamente
    for col_idx, col in enumerate(ws.columns, start=1):
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(ruta_archivo)
    wb.close()


def parsear_rango_pedidos(valor: str) -> tuple[int, int]:
    """Parsea un argumento de cantidad de pedidos (ej: '10' o '8-12')."""
    valor = valor.strip()
    if "-" in valor:
        partes = valor.split("-", 1)
        min_p = int(partes[0].strip())
        max_p = int(partes[1].strip())
        if min_p <= 0 or max_p < min_p:
            raise ValueError(f"Rango de pedidos inválido: {valor}")
        return min_p, max_p
    p = int(valor)
    if p <= 0:
        raise ValueError(f"Cantidad de pedidos debe ser > 0: {valor}")
    return p, p


def ejecutar_generacion(
    dias_n: int = 1,
    rango_pedidos: tuple[int, int] = (8, 12),
    ruta_historico: Path = DEFAULT_HISTORICO_CSV,
    directorio_salida: Path = DEFAULT_OUTPUT_DIR,
    formato: str = "ambos",
    turno: str = "MANANA",
    prefijo: str = "pedidos_dia",
    crear_consolidado: bool = False,
    seed: int | None = None,
) -> list[tuple[Path, list[PedidoGenerado]]]:
    """
    Función principal que ejecuta el pipeline de generación para N días.
    """
    if dias_n <= 0:
        raise ValueError(f"La cantidad de días (N) debe ser mayor a 0: {dias_n}")

    rng = random.Random(seed) if seed is not None else random.Random()

    # Si la ruta principal no existe, intentar fallback
    if not ruta_historico.exists() and DEFAULT_FALLBACK_CSV.exists():
        print(f"[AVISO] No se encontró {ruta_historico}. Usando fallback {DEFAULT_FALLBACK_CSV}")
        ruta_historico = DEFAULT_FALLBACK_CSV

    print(f"=== Analizando datos históricos desde: {ruta_historico} ===")
    ubicaciones = cargar_y_analizar_historico(ruta_historico)
    print(f"[OK] {len(ubicaciones)} registros históricos analizados con éxito.")

    directorio_salida.mkdir(parents=True, exist_ok=True)
    resultados: list[tuple[Path, list[PedidoGenerado]]] = []
    todos_los_pedidos: list[PedidoGenerado] = []

    print(f"\n=== Generando {dias_n} día(s) de nuevos pedidos ===")

    for i in range(1, dias_n + 1):
        cant = rng.randint(rango_pedidos[0], rango_pedidos[1])
        turno_efectivo = turno
        if turno == "MIXTO":
            turno_efectivo = "MANANA" if rng.random() < 0.5 else "TARDE"

        pedidos_dia = generar_pedidos_para_un_dia(
            ubicaciones=ubicaciones,
            cantidad_pedidos=cant,
            turno=turno_efectivo,
            dia_numero=i,
            rng=rng,
        )

        nombre_base = f"{prefijo}_{i:02d}"
        archivos_creados = []

        if formato in ("csv", "ambos"):
            ruta_csv = directorio_salida / f"{nombre_base}.csv"
            exportar_a_csv(pedidos_dia, ruta_csv)
            archivos_creados.append(ruta_csv.name)
            resultados.append((ruta_csv, pedidos_dia))

        if formato in ("xlsx", "ambos"):
            if OPENPYXL_DISPONIBLE:
                ruta_xlsx = directorio_salida / f"{nombre_base}.xlsx"
                exportar_a_excel(pedidos_dia, ruta_xlsx)
                archivos_creados.append(ruta_xlsx.name)
            else:
                print(f"[AVISO] No se pudo generar {nombre_base}.xlsx: openpyxl no está instalado.")

        todos_los_pedidos.extend(pedidos_dia)
        archivos_str = ", ".join(archivos_creados)
        print(f"  -> Día {i:02d}: {cant} pedidos ({turno_efectivo}) -> {archivos_str}")

    if crear_consolidado and dias_n > 1:
        if formato in ("csv", "ambos"):
            ruta_cons_csv = directorio_salida / f"{prefijo}_consolidado.csv"
            exportar_a_csv(todos_los_pedidos, ruta_cons_csv)
            print(f"  -> Consolidado CSV guardado en: {ruta_cons_csv.name}")

        if formato in ("xlsx", "ambos") and OPENPYXL_DISPONIBLE:
            ruta_cons_xlsx = directorio_salida / f"{prefijo}_consolidado.xlsx"
            exportar_a_excel(todos_los_pedidos, ruta_cons_xlsx)
            print(f"  -> Consolidado Excel guardado en: {ruta_cons_xlsx.name}")

    print(f"\n[OK] Generación completada con éxito. Archivos guardados en: {directorio_salida}")
    return resultados


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera nuevos pedidos para nuevos días analizando CSVs históricos de demanda."
    )
    p.add_argument(
        "-n",
        "--dias",
        type=int,
        default=1,
        help="Cantidad N de días (escenarios) a generar (por defecto: 1)",
    )
    p.add_argument(
        "-p",
        "--pedidos-por-dia",
        type=str,
        default="8-12",
        help="Cantidad de pedidos por día. Puede ser un número fijo (ej. 10) o un rango (ej. 8-12). Por defecto: 8-12",
    )
    p.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_HISTORICO_CSV,
        help=f"Ruta al CSV con pedidos históricos (por defecto: {DEFAULT_HISTORICO_CSV})",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida para los archivos generados (por defecto: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--formato",
        choices=["csv", "xlsx", "ambos"],
        default="ambos",
        help="Formato de exportación: csv, xlsx o ambos (por defecto: ambos)",
    )
    p.add_argument(
        "--turno",
        choices=["MANANA", "TARDE", "MIXTO"],
        default="MANANA",
        help="Turno asignado a los pedidos generados (por defecto: MANANA)",
    )
    p.add_argument(
        "--prefijo",
        type=str,
        default="pedidos_dia",
        help="Prefijo para los nombres de los archivos generados (por defecto: pedidos_dia)",
    )
    p.add_argument(
        "--consolidado",
        action="store_true",
        help="Generar además un archivo unificado con todos los pedidos de todos los días",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria para reproducibilidad estricta",
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        rango = parsear_rango_pedidos(args.pedidos_por_dia)
        ejecutar_generacion(
            dias_n=args.dias,
            rango_pedidos=rango,
            ruta_historico=args.input_file.resolve(),
            directorio_salida=args.output_dir.resolve(),
            formato=args.formato,
            turno=args.turno,
            prefijo=args.prefijo,
            crear_consolidado=args.consolidado,
            seed=args.seed,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

