# Módulo `scripts.data_processing` — Pipeline de Datos Históricos y Generador

Este submódulo implementa el flujo de extracción, limpieza, geolocalización, análisis estadístico y síntesis estocástica de pedidos para benchmarking.

---

## Archivos del Módulo

### `extraccion_ocr.py`
Envía imágenes de remitos manuscritos al modelo multimodal Gemini (Google AI Studio) para extraer estructuradamente pares de calle y número catastral, gestionando cuotas y reintentos.

### `geocoding.py`
Geocodifica las direcciones resultantes mediante el servicio Nominatim de OpenStreetMap, normalizando municipios de la zona operativa (Granadero Baigorria, Rosario, Ibarlucea, etc.) y guardando checkpoints de progreso.

### `estadistica_demanda.py`
Calcula la distribución empírica de frecuencias por localidad y subdivisión barrial, generando el archivo `probabilidades_demanda.csv` empleado por el generador de pedidos.

### `graficos_demanda.py`
Genera gráficos de análisis exploratorio (barras horizontales del Top 15 barrios y gráfico donut de demanda por ciudad a 150 DPI).

### `generar_pedidos_dia.py`
Generador estocástico de turnos y pedidos para benchmarking:
- Muestreo ponderado de puntos históricos con desplazamiento aleatorio (*jitter* de 40 a 145 m) para no repetir coordenadas exactas.
- Reparto volumétrico realista respetando la capacidad de la flota (2 camiones de 8 unidades).
- Asignación binaria de volcador y franjas horarias consistentes con la operación real.
- Exportación directa a CSV y planillas Excel compatibles con AnyLogic.

### `run_pipeline_demanda.py`
Script CLI orquestador que permite ejecutar individualmente o de forma encadenada las cinco fases del pipeline (`ocr`, `geocoding`, `estadistica`, `graficos`, `generar`).

