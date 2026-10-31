# Paquete `scripts` — Herramientas, Pipelines y Benchmarking

El paquete `scripts` reúne los scripts ejecutables y utilitarios para el procesamiento de datos históricos, la experimentación en AnyLogic y el entrenamiento de modelos de Aprendizaje por Refuerzo.

---

## Estructura del Paquete

```
python/scripts/
├── data_processing/   # Pipeline de datos: OCR, geocodificación y generador estocástico
├── simulation/        # Ejecución y benchmarking en simulación AnyLogic
├── training/          # Entrenamiento curricular y validación externa de políticas RL
└── rl/                # Transferencia de pesos y scripts de evolución (RL_V2)
```

---

## Contenido de las Áreas Principales

### 1. `data_processing` — Procesamiento de Datos y Demanda
Transforma remitos manuscritos históricos en datasets y turnos reproducibles:
- `extraccion_ocr.py`: Extracción de direcciones mediante visión multimodal de Google GenAI (Gemini).
- `geocoding.py`: Geolocalización y normalización de calles con OpenStreetMap (Nominatim).
- `estadistica_demanda.py`: Cálculo de frecuencias relativas y probabilidades por barrio.
- `graficos_demanda.py`: Visualizaciones exploratorias (Top 15 barrios y torta de municipios).
- `generar_pedidos_dia.py`: Generador estocástico de turnos con perturbación espacial (*jitter*), reparto volumétrico y asignación de volcadores.
- `run_pipeline_demanda.py`: Orquestador completo para ejecutar las 5 etapas secuencialmente.

### 2. `simulation` — Ejecución y Comparación Simulada
Automatiza las corridas de simulación con el modelo AnyLogic:
- `run_single_simulated_execution.py`: Ejecuta un único plan en el modelo exportado vía Alpyne y genera su reporte JSON.
- `run_simulated_comparison.py`: Evalúa los 5 algoritmos (`RL`, `HIBRIDO`, `GREEDY`, `RANDOM`, `GA`) en réplicas múltiples y turnos variados, consolidando métricas estimadas y simuladas en tablas CSV/JSON estructuradas en carpetas con semilla.

### 3. `training` y `rl` — Entrenamiento de Políticas RL
- `train_rl_policy.py`: Entrenamiento curricular con máscaras duras, selección lexicográfica y validación sobre benchmarks externos.
- `train_rl.py`: Pipeline unificado y reproducible de entrenamiento RL guiado por archivos JSON en `configs/`.
- `train_rl_v2.py`: Wrapper de compatibilidad que delega en `train_rl.py` con la configuración V2.

